# ADR-063: Splash as a second peer of record for the Qwen families — design

Status: design, 2026-11-03. CPU-only investigation (no model load, no server start, no GPU run).
Operator decision (2026-11-03): for Qwen3.8 and Qwen3.6, a v0.1.30 Win requires hf2q to finish the canonical
four-agent coding task at least 5% sooner than **both** the current peer and Splash, each in its best valid
configuration. This document gives the Splash facts that decision needs, the harness design, the sweep, the GPU
budget and the ADR-063 amendment text. `$V` = `/Users/robert/.local/state/hf2q/v0130`; Splash source paths are
relative to the pinned checkout.

## 1. Findings

| Question | Answer (source) |
|---|---|
| Pinned source | `incoai/splash` `031a25a6125af2d7acf7096580afe56eb33d65bc` (`git describe`: `1.5.0-29-g031a25a6`, 29 commits after tag `1.5.0`, merged 2026-10-31). Clean clone and build: §2. |
| Families | `install/families.py` `FAMILIES`: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B (Ornith), Qwen3.8-Flash-Next, Flint. **No Gemma 4 and no DeepSeek-V4** (no layout type, no draft, no chat format for them). Splash enters only the Qwen3.8 and Qwen3.6 verdicts. |
| hf2q's own GGUFs | Both pass Splash's GGUF screening, run with Splash's own code on the headers. The built engine's `model-check` accepts both with their drafts (`{"family":"Qwen3.8-27B"}`, `{"family":"Qwen3.6-35B-A3B"}`; §4). Splash serves a local GGUF file directly (`--model-dir FILE`), so **the same-artifact rule holds**: no unsloth file and no MLX conversion are needed. Only the native weight load is still unproven, and the fit run proves it. |
| Speculation | Always on. "Every model decodes with a DFlash2 draft" (`docs/models.md` Drafts). The engine has no switch that turns the draft off (native flags in `runtime/`: `--ane --cache-dir --decode-share --expert-memory --idle-release --kv-format --kv-spill --kv-window-trim --max-context --max-memory --persistent-cache --state-format --verify-rows …`). The only speculation knob is `--verify-rows auto|all`, and it changes behaviour only on the MoE target (Qwen3.6). A Splash no-speculation arm is **Unavailable**. |
| Drafts | Qwen3.8: `incoai/Qwen3.8-27B-DFlash2@015e7956`. Its `config.json` (sha256 `873e3556…`) and `model.safetensors` (3,848,817,896 B, LFS sha256 `67fc76d6…`) are **byte-identical** to `z-lab/Qwen3.8-27B-DFlash2@50307d4c`, which is already in the local HF cache for the TensorFold arm. Qwen3.6: `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b69`, `model.safetensors` 1,052,513,456 B, LFS sha256 `d1f16c0f…`. It is not local; it is a ~1.05 GB download (under the 5 GB ask threshold). |
| Concurrency | Compiled maximum of 4 lanes (`SPLASH_MAXIMUM_BATCH_WIDTH 4u`, `runtime/metal/abi/ExecutionGeometry.h:16`). This is exactly the four-agent workload. Admission beyond that is by memory, not by a fixed reservation per slot. |
| Context | `--max-context 256K` = 262,144 tokens (K = 1024), the native window, with no YaRN. It is a per-request limit. KV is a shared paged pool. This matches hf2q's shared-budget semantics (4 slots, 262,144 tokens per slot). |
| Prefix cache | Always on. It does exact-prefix reuse of KV pages and GDN states in RAM, plus an SSD tier (`--max-cache-disk auto`, temporary unlinked files under `$TMPDIR`). `--persistent-cache` is off by default and must stay off. |
| Readiness | `GET /ready` returns 200 `{"status":"ready"}` once the engine reports Ready, and 503 before. `/health` is liveness only. |
| Identity | `/status.identity.cache.build_id` = `src-<sha256>`, a deterministic digest of the production sources and flags (`dev/tools/build_identity.py`). It is also written to `build/engine/build-identity.json`. Splash's `model.json` records the GGUF's **SHA-256** (`install/assembly.py` `_digest`), and the draft's LFS blob sha256. `/status` also reports `identity.kv.target_model_sha256`, `maximum_context_tokens`, `vision`, `verify_rows`, `ane_ffn`, `streamed_experts`, `chat_template.later_system` and `thermal_state`. |
| Telemetry | No outbound calls in `server/`, `runtime/` or the local-model installer. A local `--model-dir` start "never contacts the Hub". |
| Expected strength | Splash's own figures (M5 Pro, its launch benchmark): 27B decode 74 tok/s single and 170 tok/s aggregate at 4 concurrent short prompts; 35B-A3B 210 and 357 tok/s. In Splash's own same-GGUF comparison, the 27B decode is several times the peer's. **[INFERENCE] Splash is likely the stronger of the two peers on both Qwen families.** The operator's "both peers" bar is then set by Splash, not by the current peer. This is the main schedule risk for Q4-QWEN38 and Q4-QWEN36. |

## 2. Pinned build and identity

Requirements (`docs/README.md`): Apple silicon with macOS 26.4 or later, Xcode 26 or later, Python 3.12–3.14, and
a Metal 4 compiler that supports `uint4b_format`. Host: M5 Max (40-core GPU), 128 GB, macOS 27.2 (26B5112a),
Xcode 27.2 (27B52), Metal 32023.922, Apple clang 21.0.0, SDK 27.2, Python 3.13.12.

Build recipe. The script is `$V/splash-peer/build_splash.sh` and its evidence goes to `$V/splash-peer/build-031a25a6/`.
It is CPU-only and stops if `HOST-LOCK` appears. Steps:

```sh
git clone --no-checkout /opt/splash /Users/robert/.local/share/hf2q/peers/splash-031a25a6
git -C …/splash-031a25a6 checkout --detach 031a25a6125af2d7acf7096580afe56eb33d65bc
cd …/splash-031a25a6
make -j4                      # build/splash, build/splash.metallib, build/host-step.metallib
make install-environment      # .venv from install/requirements.txt (pip --only-binary, PyPI)
```

Recorded identity, bound per run by the launch wrapper (§6.2):
- the commit and a clean tracked tree;
- the sha256 of `build/splash`, `build/splash.metallib` and `build/host-step.metallib`;
- `build/engine/build-identity.json` (its `build_id` must equal `/status.identity.cache.build_id`);
- the toolchain versions;
- `pip freeze` of `.venv` and the sha256 of `install/requirements.txt`.

Build result: §2.1. The build succeeded on 2026-11-03 while the host lock was absent.

`build/host-step.metallib` matters for fairness. It uses `#pragma METAL internals`. A newer Metal compiler that
refuses the pragma only warns, and the engine then runs "awake host steps as plain ones" (Makefile comment). Those
are slower host steps. Its presence is part of the identity. A missing file means the arm is not Splash's best build.

The packaged alternatives are the `brew install incoai/tap/splash` bottle and the release archive. Both are tag
`1.5.0` and lack the 29 later commits, which include the Apple10 (M5) 64-row prefill tile (`f4236596`) and the
nameless tool-call fix (`9d9c577f`). A source build at `031a25a6` is the better pin. A packaged 1.5.0 arm would be
a separate identity.

### 2.1 Build result

Built 2026-11-03, CPU only, `HOST-LOCK` absent. `make -j4` took 45.8 s wall and `make install-environment` 13.1 s.
Evidence: `$V/splash-peer/build-031a25a6/`. Checkout: `/Users/robert/.local/share/hf2q/peers/splash-031a25a6`, detached
at `031a25a6` with a clean tracked tree; only `build/`, `.venv/` and `.venv.install.lock` are ignored.

| Item | Value |
|---|---|
| `build/splash` | sha256 `05551d42d1ffad2ca0052f156d7e6a0eab00c3782fc12fd3534936aa4b8b73c0` (3,837,352 B) |
| `build/splash.metallib` | `7d804ede552a29f3d3bfb52463735f3d6a08f63cb840a96b7a56edf6cbb01409` (50,723,432 B) |
| `build/host-step.metallib` | `9f94af73a1a7ab440d381268281989cb80698b3921ebec95772e2b889fc54ffb` (built: awake host steps available) |
| `build_id` | `src-f7663ee765e357e6ecff82e45483f19866d459f0204afaf94f71737b0584f919` |
| `install/requirements.txt` | `3f6fd07c57837a9a0664f65cf61a0e851f40ccec20697008696e54a1c469067b` |
| `.venv` | Python 3.13.15 (`/Users/robert/.local/bin/python3.13`), `pip-freeze.txt` in the evidence dir |
| Toolchain | Xcode 27.2 (27B52), Metal 32023.922, Apple clang 21.0.0, SDK 27.2, macOS 27.2 (26B5112a) |

No compiler warnings; the build uses `-Werror`. The one "warning" string in `make.log` is the echoed recipe text of
the host-step fallback, which did not fire.

## 3. Server surface

### 3.1 Process model

Two processes run in one process group: the Python HTTP server (`server.server`) and the native engine
(`build/splash serve-native`), which the server spawns with plain `subprocess.Popen` (`server/runtime.py:520`).
The Q5 supervisor's group footprint sum (`q5_runtime.group_stats`, by pgid) therefore counts the engine. Restart
policy: a failed engine is relaunched, with a 5 s back-off after the second failure. It stops after the third, and
requests then get `500 engine_failed` (`docs/server.md`). The harness treats any relaunch as a failed run: the
closing audit checks `/status.transport` and the console's relaunch lines.

### 3.2 Flags relevant to the race

| Flag | Default | Race setting |
|---|---|---|
| `--model-dir PATH` + `--model OWNER/REPO` | — | hf2q GGUF file; API id `hf2q/<family>` (must be `OWNER/REPO` form, no spaces) |
| `--draft-model DIR` | family draft from HF cache | pinned local draft dir (§5) |
| `--host/--port` | 127.0.0.1:8000 | 127.0.0.1:`{PORT}` |
| `--max-context` | auto (256K if memory holds) | `256K` explicit, so a refusal is visible |
| `--max-memory` | GPU recommended working set less margin | auto; a cap only if Q5 requires one (§7.3) |
| `--idle-release` | `10m` | `off` (no weight release mid-campaign) |
| `--kv-format` | `int8` | sweep `int8`/`bf16` |
| `--state-format` | `fp32` | sweep `fp32`/`bf16` |
| `--disable-ane` | off: the 27B prefills with a GPU+ANE FFN split | sweep (27B only; MoE never splits) |
| `--verify-rows` | `auto` (MoE: cost-model rows; dense: all) | sweep `auto`/`all` (35B only) |
| `--decode-share` | `0.5` | sweep 0.25/0.5/1.0 |
| `--max-cache-disk` | `auto` (≤ ¼ of free space beyond a 10 GiB floor, ≤ 32 GiB) | sweep `auto`/`0` |
| `--persistent-cache` | off | off (required) |
| `--thinking-end-text` | `</think>` | hf2q's forced close (§6.3, decision D4) |
| `--no-webui` | page on | on (`--no-webui`) |
| `--language-only` / `--mmproj` | no projector with `--model-dir` unless `--mmproj` | neither, so text only |
| `--queue-size` | 32 | default |

### 3.3 Memory plan

The plan sizes everything within the Metal budget at start (`engine::planStreamedMemory`). KV grows in extents
on demand, and a spare reserve of 7% of RAM (2–7 GiB) plus 1 GiB is kept against cache growth
(`docs/internals.md` Admission and eviction).

KV sizes, derived from the GGUF geometry:
- **Qwen3.8**: 16 full-attention layers × K+V × 4 KV heads × 256 × 1 B (int8) ≈ 32 KiB per token, about 8 GiB at
  262,144 tokens per lane; bf16 doubles it.
- **Qwen3.6**: 10 layers × 2 × 2 × 256 ≈ 10 KiB per token, about 2.5 GiB per full lane.

Per-lane GDN state is 153 MiB for the 27B and 64 MiB for the 35B in fp32 (`docs/server.md` SSD cache). Weights are
roughly the GGUF minus the MTP block. The 35B's routed experts load whole where they fit (`--expert-memory auto`),
which they do at 128 GB; readiness requires `/status.streamed_experts == null`.

Q5 interaction **[INFERENCE, to be measured in the fit run]**: Splash's cache growth deliberately drives the host's
spare memory down to its reserve. The Qwen3.8 Q5 effective floor measured on 2026-10-30 was 13.2 GiB of Free plus
cached memory, above Splash's 7%+1 GiB (≈ 9.96 GiB) spare target. A long run could therefore trip the floor stop
while Splash is behaving as designed. The canonical client reaches about 2.8k prompt tokens per turn, so the risk
is small there and larger for the long-context variant. If the calibration trips the floor, the remedy is an
explicit `--max-memory` (decision D6), not a silent context cut.

### 3.4 ANE prefill (Qwen3.8 only)

A dense target's prefill FFN is split between GPU and ANE when calibration finds a ≥5% gain (it takes 8 of 34 units
on an M5 Max per Splash's notes). The calibration is remembered per Mac, model, macOS, build and unit count in
`$(getconf DARWIN_USER_CACHE_DIR)/splash-ane/remembered-*` (`runtime/ane/Program.mm:375`). That is cross-run state.
The harness records the directory listing and digests before and after every run, and `/status.ane_ffn` (`state`,
`share`, `minimum_rows`, `reason`, `reruns`).

The first start of a configuration calibrates (3–8 s, plus a program compile). It runs in the sweep preflight, never
inside a timed run. The split uses a private ANE interface. A Splash arm with the split is the engine's documented
default and is valid unless the quality gate fails. `--disable-ane` is a sweep point and part of the matched
diagnostic.

### 3.5 Prefix cache and determinism

Exact-prefix reuse is on in every configuration, which matches hf2q's cross-turn reuse. Usage reports
`prompt_tokens_details.cached_tokens` and `metrics.cache.deduplicated_tokens`.

Splash documents that a greedy request's output depends on batch composition and chunking. On the 35B under
`--verify-rows auto` it can differ from run to run even alone (`docs/api.md` Sampling). This does not invalidate a
makespan race, but transcripts are not reproducible. The audit records `verify_rows` per run. Exactness claims are
never made for Splash.

### 3.6 Files Splash writes (cleanup and sandbox)

| Path | When | Harness handling |
|---|---|---|
| `<models_dir>/.resolved/<sha>/` (symlinks to the GGUF and the draft, plus `model.json` and derived tokenizer/config) | every start | campaign-scoped `models_dir` under `$V/splash-peer/<campaign>/models`, never the checkout's `install/models` |
| `<models_dir>/.metadata` | first start of a GGUF | same dir; keyed by GGUF size and digest |
| `$TMPDIR/splash-cache-*` | SSD cache on | unlinked at creation; space returns at exit. Record `df` before and after |
| `~/Library/Caches/Splash/prefix-cache/` | every start prunes namespaces unused 14 days (persistent cache off: nothing written) | record listing before and after |
| `$DARWIN_USER_CACHE_DIR/splash-ane/remembered-*` | ANE calibration | record, keep (cross-run state, bound in identity) |
| `~/Library/Application Support/Splash/thinking.key` | server start | record; harmless (Messages API only) |
| `<checkout>/build/runtime` | server lock | ignored files in the pinned checkout |

Model files: Splash only **symlinks** the GGUF and the draft (`install/assembly.py:73`). It never modifies or hard
links them, and it reads the GGUF once per new assembly to digest it. Whether a symlink in a run directory is
acceptable under the "never link a model file" host rule is decision D8. The hf2q Qwen3.8 case's
`text_only_locator` already hard-links the same file per run.

Note: creating that hard link changes the GGUF's ctime. Splash keys its assembly on path, size, mtime and ctime, so
after every hf2q Qwen3.8 run the next Splash start re-hashes 16.8 GB (about 5–10 s, outside the makespan) and gets a
new `.resolved/<sha>` path. Runs therefore bind the recorded **digest**, never the assembly path.

## 4. Same-artifact check (hf2q's GGUFs)

Splash's Python screening (`install/gguf.py`, the code its installer runs before any download: `Metadata`,
`model_config`, `require_arrays`, `require_loadable`, `loaded_tensors`, `tokenizer_files`) was run on the two
headers with `$V/splash-peer/screen_gguf.py`. Only metadata was read.

| | Qwen3.8 `qwen38-stock/Qwen3.8-27B-vanilla.gguf` | Qwen3.6 `qwen36-stock-adr063/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf` |
|---|---|---|
| sha256 (hf2q record) | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` | `815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220` |
| arch / tokenizer | `qwen35`, gpt2/`qwen35`, `add_bos_token false` | `qwen35moe`, gpt2/`qwen35`, `add_bos_token false` |
| tensor types | Q4_K 439, Q6_K 67, F32 360 | Q4_K 379, Q6_K 64, F32 308, F16 2 (MTP block router only) |
| derived config | 64 layers (65 − 1 MTP), hidden 5120, 24/4 heads × 256, vocab 248320 | 40 layers (41 − 1 MTP), hidden 2048, 16/2 heads, 256 experts |
| screening | all checks `ok` | all checks `ok` |
| vs Splash's fixtures | scalar metadata equal to `dev/tests/fixtures/model-configs/qwen3.8-27b/gguf-metadata.json` except hf2q's extra `qwen35.nextn.use_dedicated_embeddings=0` | equal except `block_count 41`/`nextn_predict_layers 1` (MTP retained) |
| native `requireMetadata` | `block_count` must equal layers + `nextn_predict_layers` (`runtime/model/GgufImage.cpp:828`), so a retained MTP block is expected and never loaded | same |
| embedded template sha256 | `c3cf9e34…`, the reviewed hf2q Qwen3.8 template | `e84f32a2…`, the reviewed hf2q Qwen3.6 template |

Splash's only F16 restriction (no F16 MoE router) concerns `blk.40.*`, the MTP block, which Splash never loads. The
GDN convolution, decay and dt bias are F32, and alpha/beta are a Q4_K pair, as the loader requires. Splash derives
the tokenizer and template from the GGUF itself ("no cross-repository fallback"), so both engines render from the
same template bytes.

CPU-only proof taken: `build/splash model-check gguf none <config> <gguf-metadata> [<draft config>]`.
- Inputs are the configs and scalar metadata derived by Splash's `gguf.model_config`/`scalar_metadata` from the two
  headers, and the two drafts' `config.json`.
- Result `{"family":"Qwen3.8-27B"}` and `{"family":"Qwen3.6-35B-A3B"}`, exit 0, with and without the draft config.
- Evidence `$V/splash-peer/model-check/`. `model-check` only inspects configuration (`runtime/main.mm` `checkModel`).

Template and tokenizer probe, also CPU-only, with Splash's server code on the GGUF-derived tokenizer
(`$V/splash-peer/template-probe/probe.txt`):
- Templates: `validate_tokenizer` passes, and the templates are byte-identical to hf2q's reviewed ones (`c3cf9e34…`,
  `e84f32a2…`).
- `chat_template.later_system` is `patched` for both, so a later system message renders in place. The patch is kept
  only when ordinary conversations render byte-identically.
- Qwen3.8 reasoning: efforts `low`/`medium`/`xhigh`, with a default of `xhigh`. Qwen3.6 has no effort tiers.
- hf2q's forced close `"\nI need to answer now.</think>"` (`src/serve/api/qwen_thinking_policy.rs:238`) is accepted
  as `--thinking-end-text`. It tokenizes to `[198, 40, 1144, 310, 4087, 1381, 13, 248069]`.

Remaining proof, at the fit run: the native load, and `/status` with `vision false`, `maximum_context_tokens
262144` and `streamed_experts null`.

Implications:
- Splash runs on **the exact hf2q artifacts**, so it is a same-artifact peer, the same class as the current peer
  and unlike the ds4/TensorFold/oMLX native arms.
- Splash adds one separately pinned artifact, the DFlash2 drafter. ADR-063 F1 already requires "pin separate
  drafter weights", which is satisfied by digest binding (§5).
- Splash's GGUF loader repacks the weights into its own kernel layouts in memory at load ("no copy on disk"). The
  numerics differ from hf2q's (int8 KV by default, ANE W8A8 FFN split on the 27B, its own kernels). That is engine
  behaviour, judged by the quality gate.

## 5. Drafter pins

| Family | Draft dir (local `--draft-model`) | Pins |
|---|---|---|
| Qwen3.8 | `~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c…` (present) | `config.json` sha256 `873e3556…`, `model.safetensors` sha256 `67fc76d6…`; equal to `incoai/Qwen3.8-27B-DFlash2@015e7956` (HF API LFS sha and config bytes compared 2026-11-03) |
| Qwen3.6 | `~/.cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2/snapshots/51ef7b69…` (to fetch) | `model.safetensors` sha256 `d1f16c0f…`, 1.05 GB |

Fetch command, root-run, internal disk because timed runs read it at load:
`HF_HUB_CACHE=~/.cache/huggingface/hub hf download incoai/Qwen3.6-35B-A3B-DFlash2 --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 config.json model.safetensors`.
Splash refuses a draft whose config does not match the target (`model-check` with the draft config).

## 6. Harness integration

### 6.1 Role: a same-artifact peer of record, engine kind `splash`

The native-arm registry (`native_peers.py`) assumes "serve their own native artifact" and a role that is "never the
verdict of record". Splash is neither. It reuses the **adapter interface** (closed request/response envelopes,
usage, readiness, closing audit, identity, process names) and gets a new role:

- `coding_family.ENGINE_KINDS` gains `splash`. It goes in a new tuple `SAME_ARTIFACT_PEERS = ('splash',)` beside
  `NATIVE_KINDS`, so that every "native means not the verdict" branch stays correct. The adapter module is
  `native_peer_splash.py`, resolved by a registry function that accepts both tuples. `q3_role` returns "peer of
  record (same hf2q GGUF)", and `NATIVE_COMPARISON`/`SAME_GGUF_UNAVAILABLE` are `None`.
- `launch_problems` **requires** the case artifact sha256 to equal the hf2q arm's artifact sha256. This is the
  inverse of the native arms, which refuse an ambient hf2q artifact.
- `PROCESSES = ('splash',)` covers the engine binary. The Python server is identified by argv, `-m server.server`
  under the pinned `.venv`, in the competing-runtime inventories.

### 6.2 Launch wrapper (`native_peer_splash_launch.py`; serve.argv execs it, so `SERVER_PID` is the server)

1. Verify `SPLASH_DIR`: the clean detached checkout at `SPLASH_COMMIT`, the binary/metallib/host-step sha256s,
   `build-identity.json` `build_id == SPLASH_BUILD_ID`, and `.venv` pip-freeze sha256.
2. Verify the GGUF size and mtime against the case, plus `MODEL_SHA256` from the hf2q receipt. Verify the draft
   dir's blob names and sizes against the pins.
3. Snapshot the cross-run state listings (§3.6) to `{OUT}/splash/state-before.json`.
4. Call `install.local.prepare_local_model(MODEL, draft=DRAFT_DIR, models_dir=SPLASH_MODELS_DIR)` inside Splash's
   `.venv`. Assert the returned `model.json` records the GGUF `digest == MODEL_SHA256` and the draft safetensors
   digest equal to the pin. Write `{OUT}/splash/launch.json` with the assembly path, `model.json` sha256, argv and
   env. Its sha256 goes in the approval.
5. `execve` `{SPLASH_DIR}/.venv/bin/python -u -P -m server.server <assembly> --model hf2q/<family> --binary
   {SPLASH_DIR}/build/splash --host 127.0.0.1 --port {PORT} --max-context 256K --idle-release off --no-webui
   --thinking-end-text <D4> <config flags>`. The environment is `PYTHONPATH={SPLASH_DIR}`, `HF_HUB_OFFLINE=1`,
   and no `SPLASH_*` except those the case names.

### 6.3 Request parity (closed envelope; Splash ignores unknown Chat fields, `frontend.py` reads by `body.get`)

| Canonical hf2q field / behaviour | Splash request | Note |
|---|---|---|
| model, messages, tools, `tool_choice auto`, `stream`, `stream_options.include_usage`, `logprobs false` | as is | `logprobs:true` and `n≠1` would 400 |
| `max_completion_tokens 8192` | as is | |
| `hf2q_overflow_policy reject` | dropped | Splash lowers the limit to the context instead. The client refuses a turn whose prompt+8192 > 262144 (as for oMLX) |
| greedy | `temperature 0, top_p 1, top_k 0, min_p 0, presence 0, frequency 0` | `temperature == 0.0` is accepted as argmax (`protocol.py:718`) |
| Qwen thinking on, `preserve_thinking` | `chat_template_kwargs {"enable_thinking": true}`, `preserve_thinking true`; no `reasoning_effort` | template default effort, as hf2q renders (Qwen3.8 template default renders as `xhigh`) |
| thinking ceilings 2048 (fresh user turn) / 512 (tool continuation) | `thinking_token_budget` from `qwen_n4_peer.request_controls` | the forced close text is decision D4 |
| repetition 1.05 over a 64-token window seeded from the prompt tail | **no equivalent**. Splash's `repetition_penalty` counts every prompt and output token (`docs/api.md` Sampling) | decision D3 |
| hf2q `parallel_tool_calls` default false (lazy tool grammar) | `parallel_tool_calls false` turns on Splash's tool grammar; omitted means unconstrained output | decision D5 |

Response envelopes (`server/api_shapes.py:1147-1223`):
- unary `{id chatcmpl-…, object, created, model, choices[1], usage, metrics, timings}`;
- chunks `{id, object, created, model, choices[{index, delta, finish_reason}]}` with `metrics`/`timings` on the
  final chunk, then a `choices: []` usage chunk;
- `delta` fields `role`, `content`, `reasoning_content`, `tool_calls`. Tool-call ids are `call_<request>_<n>`.
  Arguments stream incrementally (the Qwen XML parser emits as it reads), unlike oMLX's whole calls;
- `usage.prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`;
- `timings.draft_n`/`draft_n_accepted`; `metrics.speculation` {cycles, …}; `metrics.cache.deduplicated_tokens`.

`check_unary`/`check_frame` freeze exactly these keys, from a recorded fixture of the fit run.

### 6.4 Readiness (q5 case) and closing audit

Readiness: `GET /ready` with `status == "ready"` (timeout 1800 s, covering the first-start digest and the ANE
calibration). Recorded: `/status`, `/v1/models`, `/metrics`. Checks:
- `/status`: `instance.model == MODEL_ID`, `maximum_context_tokens == 262144`, `vision false`,
  `identity.cache.build_id == SPLASH_BUILD_ID`, `identity.cache.dtype`/`identity.kv.format` equal to the config,
  `verify_rows` equal to the config, `ane_ffn.state` consistent with the config, `streamed_experts null`,
  `requests.submitted == 0`, `transport.recovering false`;
- `/v1/models`: `max_model_len == 262144`;
- the startup lines (context and bound, Neural Engine split, SSD cache, verify rows, chat-template probe) parsed
  into `native_readiness.startup`.

Closing (outside the makespan):
- `/status` counters equal the client's turn count (`requests.completed`, `failed == 0`, `cancelled == 0`), with
  no engine relaunch;
- every turn's rendered prompt is re-rendered through `POST /apply-template`. Its bytes must equal hf2q's
  rendering of the same messages (byte parity), and `/tokenize` of it must equal `usage.prompt_tokens`;
- each reply is bound to its `Done` console line (tokens, cached, speculation).

### 6.5 Q5 cases (to be added with the adapter, not before)

`q5_cases/splash-qwen38.json` and `splash-qwen36.json`. Schema as `native-omlx-qwen36.json`, with these values:
- workload id `adr063-q5-splash-<family>-n4-coding-ctx262144`;
- `binary {SPLASH_DIR}/build/splash`;
- artifact `{MODEL}`/`{MODEL_SHA256}`, the **same** values as `qwen38.json`/`qwen36.json`, required non-empty;
- variables `SPLASH_DIR`, `SPLASH_COMMIT`, `SPLASH_BUILD_ID`, `SPLASH_BIN_SHA256`, `SPLASH_VENV_FREEZE_SHA256`,
  `DRAFT_DIR`, `DRAFT_SHA256`, `SPLASH_MODELS_DIR` and `SPLASH_CONFIG` (a named sweep point, §7);
- `client.argv … --engine-kind splash`. The approval carries a `native_splash` block (identity, config, launch.json
  sha256) and the source freeze adds `native_peers.py`, `native_peer_splash*.py` and their test.

Sweep cases are derived per config by `splash_sweep.py derive-case --config <name>`, which changes only
`SPLASH_CONFIG`.

### 6.6 Q2 certification and Q3 code changes

Because Splash cohorts are verdict-bearing, they must be Q2-certified with the adversarial controls, unlike the
native arms (`q2_certify.py` refuses native cohorts today). Code changes:
- `q3_campaign.ARMS` gains `splash` (and an optional `splash_matched`), from `--splash-case`;
- `q3_stats.family_verdict` gains a `vs_splash` contrast with the same `PEER_MARGIN 0.95`, and Win requires both
  peer contrasts;
- `required_effects`/the stopping rule include `vs_splash`;
- `q3_report`/F2 render the Splash contrast beside the peer contrast;
- `quality_campaign.py --reference-kind splash`;
- the identity freeze binds the Splash identity in §6.2.

## 7. Best valid configuration

### 7.1 Validity

A configuration is valid when all of these hold:
- the run is Q5 `CalibrationPass`;
- it reports 4 lanes (compiled), `max_model_len 262144`, text only, and no engine relaunch;
- it completes the canonical four-agent task with every Q1/Q3 task and tool gate;
- its request parity is per §6.3;
- the arm passes the paired task-quality check against the hf2q candidate's reference (the same rule as the
  current peer's).

Engine-documented lossy defaults (int8 KV, the ANE W8A8 split, bf16 state) are valid unless that quality check
fails.

### 7.2 Sweep grid (one factor at a time from the default, 2 runs each, then confirmation)

| Point | Qwen3.8 | Qwen3.6 |
|---|---|---|
| S0 default | int8 KV, fp32 state, ANE split auto, decode-share 0.5, SSD cache auto | int8, fp32, verify-rows auto, 0.5, auto |
| S1 | `--kv-format bf16` | `--kv-format bf16` |
| S2 | `--state-format bf16` | `--state-format bf16` |
| S3 | `--disable-ane` | `--verify-rows all` |
| S4/S5 | `--decode-share 0.25` / `1.0` | same |
| S6 | `--max-cache-disk 0` | same |
| M (matched diagnostic, not the arm) | bf16 KV, fp32 state, `--disable-ane`, cache-disk 0 | bf16 KV, fp32 state, `--verify-rows all`, cache-disk 0 |

Selection uses the declared makespan rule (Q4-DS4 precedent): lowest median makespan among valid points. A best
within the noise of the runner-up is provisional, and both are re-run (3 runs each) before the campaign of record.
Combining two winning factors (e.g. S2+S5) is one extra confirmation point, not a full factorial.

Speculation on/off cannot be swept (§1). Disclose it as "speculation always on (DFlash2), no off switch".

### 7.3 Preflight per family (before S0)

1. `build/splash model-check` on the prepared assembly (CPU).
2. A first start, untimed. It digests the GGUF, derives the tokenizer and, for Qwen3.8, calibrates the ANE split.
3. Q5 calibration run of S0. If the free-plus-cached floor trips (§3.3), the operator chooses a `--max-memory` cap
   (D6) and S0 is re-run.

## 8. GPU time estimates

Basis: Q3 v11 Qwen3.8 (`$E/q3-campaign-v11/qwen38-all`) took 24 n4 runs (4 arms, warmup plus 5 blocks) in 44 min,
**about 1.85 min per run**, and 60 n1 runs in 49 min, about 0.8 min per run. Qwen3.6 repair1 Q5 runs: peer
makespan 36 s, hf2q 53 s, readiness 4 s. Splash startup is unmeasured. **[INFERENCE]** 0.5–1.5 min per start (17–22 GB
read and repacked, a GGUF re-digest after a ctime change, ANE program build), so a Splash run costs
**about 2.5 min (Qwen3.8) and 1.7 min (Qwen3.6)**.

| Item | Qwen3.8 | Qwen3.6 |
|---|---|---|
| Preflight (model-check, first start, Q5 S0) | ~10 min | ~8 min (+1 GB draft fetch, network only) |
| Sweep S0–S6 × 2 + confirmation 2×3 + M × 2 | 22 runs ≈ 55 min | 22 runs ≈ 40 min |
| **Fit + sweep total** | **≈ 1.1 h (range 0.8–1.6 h)** | **≈ 0.8 h (0.6–1.2 h)** |
| Paired quality gate vs Splash (same instance count as the peer gate; observed ~45 s/run on Qwen3.6, ~1.5 min on Qwen3.8; ~48 runs) | ≈ 1.2 h | ≈ 0.7 h |
| Q3 n4 primary, 5 arms (stock, experiment, peer, peer_matched, splash), warmup + 5–20 blocks | 30–105 runs ≈ 1.0–3.6 h | ≈ 0.7–2.4 h |
| Q3 n1 control, 5 arms, 5–20 blocks (v11 stopped at 14) | ≈ 0.4–1.6 h | ≈ 0.3–1.2 h |
| **Full paired race per family (n4 + n1)** | **≈ 1.4–5.2 h**; Splash's share ≈ 25% | **≈ 1.0–3.6 h** |

Two peer contrasts must both reach the 2.5-point half-width. Expect the race to need more blocks than v11's 5 when
the two peers' makespans differ a lot. The mixed 3+1/2+2/1+3 controls, if Splash runs them, add about the n4
figure again per stage (decision D7).

## 9. ADR-063 amendment text (proposed)

New section after "Native-engine peer arms":

> ### Splash peer of record for the Qwen families (operator decision 2026-11-03)
>
> For Qwen3.8 27B and Qwen3.6 35B-A3B, Splash (`incoai/splash`, source build `031a25a6`, engine kind `splash`) is a
> second same-artifact peer of record. Splash serves neither Gemma 4 nor DeepSeek-V4 (`install/families.py`), so
> those families keep one peer.
>
> - **Same artifact.** Splash serves the hf2q GGUF of the family's hf2q arm, the same path and sha256, read through
>   its local-file mode; its `model.json` digest must equal the hf2q artifact sha256. It decodes with its DFlash2
>   drafter, a separately pinned artifact (F1): Qwen3.8 `incoai/Qwen3.8-27B-DFlash2@015e7956`, byte-identical to
>   `z-lab/…@50307d4c`, and Qwen3.6 `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b69`, bound by digest. It serves text only,
>   with four lanes and a 262,144-token context.
> - **Best valid configuration.** Splash has no switch that turns speculation off. Its best valid configuration is
>   chosen per family by the declared makespan rule from a one-factor sweep of its documented knobs: KV and state
>   formats, Neural Engine prefill split (Qwen3.8), verify rows (Qwen3.6), decode share and the SSD cache tier.
>   Every sweep run passes Q5 and the task gates, and the choice is confirmed before the campaign of record. A
>   matched-settings diagnostic (bf16 KV, fp32 state, Neural Engine split off, every verify row, no SSD tier) is
>   kept and labeled. A Splash no-speculation arm is Unavailable.
> - **Q3.** Q3 campaigns for these families add the measured arm `splash` to the Williams blocks and report
>   experiment-versus-Splash beside experiment-versus-peer. Splash cohorts are Q2-certified like the peer's.
> - **Q4 win (Qwen families).** `family_verdict` is Win only when the hf2q/peer and the hf2q/Splash four-agent
>   makespan ratios both have 95% CI upper bounds ≤ 0.95, each against that engine's best valid configuration, and
>   every other Q4 gate holds. A missing, failed or disqualified Splash arm cannot yield a Win. Splash task failures
>   are recorded as quality findings.
> - **Quality gate.** The paired task-quality gate on the exact candidate binary also runs against Splash, bound to
>   the Splash readiness identity.
> - **Fairness.** Vs-Splash worst-slot TTFT ratios are disclosed and not gated. SSE gaps against Splash are not
>   comparable, because one emission can carry several speculative tokens.
> - **Recorded differences.** Request parity differs in the repetition penalty (D3), the tool grammar (D5) and the
>   forced thinking close (D4). Splash greedy output is not run-to-run exact on the MoE target under its default
>   verify rows. Its numerics include int8 KV and the Neural Engine W8A8 FFN split where the chosen configuration
>   uses them. Per-turn prompt bytes are proven equal to hf2q's through `/apply-template`.
> - **Identity.** The commit, the binary/metallib sha256s, the `build_id`, the venv freeze, the drafter digests,
>   the exact argv and the Neural Engine calibration state are frozen with the campaign identity.
>
> This decision establishes no speed win.

Amended sentences:
- **Q4 Outcome.** Append: "For Qwen3.8 and Qwen3.6, at least 5% sooner than both the peer and Splash, each in its
  best valid configuration."
- **Q4 Acceptance, the win bullet.** Add: "For Qwen3.8 and Qwen3.6, the hf2q/Splash ratio must meet the same bound."
- **Q4 Acceptance, the quality bullet.** Add: "and against Splash for Qwen3.8 and Qwen3.6."
- **Q4-QWEN38 and Q4-QWEN36 Outcomes.** Add: "and than Splash in its best valid configuration"; the per-stage
  attribution is extended to the Splash gap.
- **Predecessors of Q4-QWEN38/Q4-QWEN36.** Add "Splash fit and best-configuration record (Qwen family)".

## 10. Decisions needed (operator)

- **D1 — Campaign structure.**
  - (a) Recommended: one five-arm Williams campaign per family (stock, experiment, peer, peer_matched, splash),
    with same-session thermal conditions and one identity freeze.
  - (b) A separate experiment-vs-Splash campaign without a stock arm, bound to the record campaign's candidate
    binary by sha256, as in the Q4-DS4-RACE D1 pattern. It is cheaper per block, but needs the no-stock-arm mode
    and cross-campaign binding.
- **D2 — Pin.** Recommended: source build `031a25a6` (§2). Alternative: the `1.5.0` bottle, which is older and lacks
  the M5 prefill tile.
- **D3 — Repetition penalty.**
  - (a) Recommended: send none (1.0), as the TensorFold arm does, and record it.
  - (b) Send 1.05. It then applies over the whole prompt and output, a stronger and different penalty that Splash
    warns delays tool calls.
- **D4 — Forced thinking close.**
  - (a) Recommended: `--thinking-end-text $'\nI need to answer now.</think>'`, hf2q's text. The CPU probe
    accepted it as 8 tokens ending in `</think>` (§4).
  - (b) Splash's default `</think>`.
- **D5 — `parallel_tool_calls`.**
  - (a) Recommended: send `false`. This is hf2q's default, and Splash then constrains calls by its grammar.
  - (b) Omit it, for unconstrained output. The client refuses a second call either way.
- **D6 — Memory cap.** Default `--max-memory auto`. If the Q5 floor trips, allow an explicit cap, recorded as part
  of the configuration?
- **D7 — Controls.** Splash in n4 and n1 only (recommended), or also in the 3+1/2+2/1+3 mixed controls (about
  +1 n4-equivalent each).
- **D8 — Symlinks.** Is a symlink to the GGUF in a Splash run directory acceptable under the host rule "never
  link a model file"? Splash makes no hard link and does not touch the file.
- **D9 — Qwen3.6 workload.** The Splash case follows whatever Q4-QWEN36 declares, canonical or repair variant.
  Splash's Qwen3.6 arm waits for that decision.
- **D10 — Validity of lossy defaults.** Are int8 KV and the Neural Engine split valid unless the quality gate
  fails (recommended), or must the Splash arm match hf2q's KV precision?

## 11. Open risks

- The native weight load of the hf2q GGUFs is unproven until the fit run; screening and `model-check` pass (§4).
- The Q5 floor versus Splash's cache-growth target (§3.3).
- Splash relaunches its engine on failure. A relaunch mid-run is an invalid run, not a recovered one.
- Splash appears much faster than the current peer on both Qwen families **[INFERENCE from its published figures]**.
  If so, the Qwen Win bar becomes Splash's, and Q4-QWEN38's starting gap (−6.4% vs the peer, Q3 v11) widens
  substantially.
