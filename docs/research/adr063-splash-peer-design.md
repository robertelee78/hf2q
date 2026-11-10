# ADR-063: Splash as a second peer of record for the Qwen families — design

Status: design approved (operator, 2026-11-04), with decisions D1–D10 in §10 as amended by the independent review.
The investigation was CPU-only: no model load, no server start and no GPU run.
Operator decision (2026-11-03): for Qwen3.8 and Qwen3.6, a v0.1.30 Win requires hf2q to finish the canonical
four-agent coding task at least 5% sooner than **both** the current peer and Splash, each in its best valid
configuration. This document gives the Splash facts that decision needs, the harness design, the sweep, the GPU
budget and the ADR-063 amendment text. `$V` = `/Users/robert/.local/state/hf2q/v0130`; Splash source paths are
relative to the pinned checkout.

Operator amendment (2026-11-10): v0.1.30's 5% minimum now applies against
published hf2q v0.1.29; every valid peer, including Splash, is a non-inferiority
gate with a 95% CI upper ratio bound of 1.00. Qwen3.8's corrected Splash task
arm is not a valid performance denominator and is replaced by ADR-067's 5.304
accepted-plus-bonus token/lane-cycle mechanism gate. The original decision
above remains the historical input to this design; this amendment governs new
release campaigns.

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
on demand. Growth that no request needs, the prefix cache included, keeps a spare reserve of 7% of RAM clamped to
2–7 GiB, plus a 1 GiB margin (`docs/internals.md` Admission and eviction). On this 128 GB Mac that is 7 + 1 =
**8 GiB**.

KV sizes, derived from the GGUF geometry:
- **Qwen3.8**: 16 full-attention layers × K+V × 4 KV heads × 256 × 1 B (int8) ≈ 32 KiB per token, about 8 GiB at
  262,144 tokens per lane; bf16 doubles it.
- **Qwen3.6**: 10 layers × 2 × 2 × 256 ≈ 10 KiB per token, about 2.5 GiB per full lane.

Per-lane GDN state is 153 MiB for the 27B and 64 MiB for the 35B in fp32 (`docs/server.md` SSD cache). Weights are
roughly the GGUF minus the MTP block. The 35B's routed experts load whole where they fit (`--expert-memory auto`),
which they do at 128 GB; readiness requires `/status.streamed_experts == null`.

Q5 interaction **[INFERENCE, to be measured in the fit run]**:
- **Splash's target.** Splash lets its cache grow until the host's spare memory falls to its 8 GiB reserve. It
  counts purgeable pages and part of what compressing anonymous pages would free (up to a quarter of them) as spare.
- **Q5's measure.** Q5 measures free plus speculative plus file-backed memory (`q5_host.py`), which excludes both.
  Splash can therefore drive Q5's measure below 8 GiB while it is still above its own reserve.
- **Q5's floor.** The effective floor is at least 2 GiB, raised per run from that run's own fastest post-readiness
  drop and the stop latency (ADR-063 Q5). The 13.2 GiB measured on 2026-10-30 is hf2q's Qwen3.8 floor, not
  Splash's; Splash gets its own from its fit calibration.
- **Exposure.** A run can trip the floor while Splash behaves as designed. Long prompts, the long-context variant and
  bf16 KV (which doubles the KV) are most exposed. The canonical client reaches about 2.8k prompt tokens per turn.
- **Remedy (decision D6).** An explicit `--max-memory` cap, recorded as part of the configuration. It applies at
  every sweep point and in the campaign, never as a silent context cut.

### 3.4 ANE prefill (Qwen3.8 only)

A dense target's prefill FFN is split between GPU and ANE when calibration finds a ≥5% gain (it takes 8 of 34 units
on an M5 Max per Splash's notes). The calibration is remembered per Mac, model, macOS, build and unit count in
`$(getconf DARWIN_USER_CACHE_DIR)/splash-ane/remembered-*` (`runtime/ane/Program.mm:375`). That is cross-run state.
The harness records the directory listing and digests before and after every run, and `/status.ane_ffn` (`state`,
`share`, `minimum_rows`, `reason`, `reruns`).

The first start of a configuration calibrates (3–8 s, plus a program compile). It runs in the sweep preflight, never
inside a timed run. The split uses a private ANE interface. A Splash arm with the split is the engine's documented
default. Like every configuration, it is valid when Splash's own task outcomes pass (§7.1). `--disable-ane` is a
sweep point and part of the matched diagnostic.

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
| `<models_dir>/.resolved/<sha>/` (symlinks to the GGUF and the draft, plus `model.json` and derived tokenizer/config) | every start | per-run `models_dir` `{OUT}/splash/models`, never the checkout's `install/models`. `{OUT}` must be on APFS: on exFAT, Splash's `collect_garbage` stops on the `._*` files macOS writes (observed in the dry run); `prepare_run` and Q3 `launch_problems` refuse it |
| `<HF cache>/models--<draft>/refs/splash/<installation>/<commit>` | only for sources with a revision (`install/assembly.py` `pins`, `install/hub.py` `pin`) | the local paths the harness passes record revision `null` (dry run), so no pin is written; `prepare_run` refuses a run that adds one, and `cross_run_state` records the folder before and after each run |
| `<models_dir>/.metadata` | every start (per-run dir) | same per-run dir; keyed by GGUF size and digest |
| `$TMPDIR/splash-cache-*` | SSD cache on | unlinked at creation; space returns at exit. Record `df` before and after |
| `~/Library/Caches/Splash/prefix-cache/` | every start prunes namespaces unused 14 days (persistent cache off: nothing written) | record listing before and after |
| `$DARWIN_USER_CACHE_DIR/splash-ane/remembered-*` | ANE calibration | keep; recorded before and after each run (per-run record, not frozen in the campaign identity) |
| `~/Library/Application Support/Splash/thinking.key` | server start | record; harmless (Messages API only) |
| `<checkout>/build/runtime` | server lock | ignored files in the pinned checkout |

Model files: Splash only **symlinks** the GGUF and the draft (`install/assembly.py:73`). It never modifies or hard
links them, and it reads the GGUF once per new assembly to digest it. Whether a symlink in a run directory is
acceptable under the "never link a model file" host rule is decision D8. The hf2q Qwen3.8 case's
`text_only_locator` already hard-links the same file per run.

Note: every Splash start uses a fresh per-run `models_dir` (`{OUT}/splash/models`), so every start of either family
reads the whole GGUF once to compute its digest: 16.8 GB for Qwen3.8 and 21.7 GB for Qwen3.6. This happens before
readiness and outside the makespan (§8 start estimate). `model.json` records each linked file's path, size, mtime,
ctime and content digest, and the assembly directory is the sha256 of `model.json` (`install/assembly.py:61-78,
113-129`). The assembly path, `model.json` sha256 and argv therefore differ from run to run: the run directory
differs, and an hf2q Qwen3.8 run's hard link changes the GGUF's ctime. The campaign identity freezes none of those
(§6.2). It freezes an argv template with the assembly as a per-run value, the content digests, and the `model.json`
fields other than paths and times.

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
  behaviour. A configuration's validity rests on Splash's own task outcomes (§7.1); the hf2q-vs-Splash quality gate
  applies to the hf2q candidate.

## 5. Drafter pins

| Family | Draft dir (local `--draft-model`) | Pins |
|---|---|---|
| Qwen3.8 | `~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c…` (present) | `config.json` sha256 `873e3556…`, `model.safetensors` sha256 `67fc76d6…`; equal to `incoai/Qwen3.8-27B-DFlash2@015e7956` (HF API LFS sha and config bytes compared 2026-11-03) |
| Qwen3.6 | `~/.cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2/snapshots/51ef7b69…` (downloaded 2026-11-04) | `config.json` sha256 `86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59`, `model.safetensors` sha256 `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d` (re-hashed locally, equal to the HF LFS sha), 1,052,513,456 B |

Downloaded to the internal disk, because timed runs read the draft at load:
`HF_HUB_CACHE=~/.cache/huggingface/hub hf download incoai/Qwen3.6-35B-A3B-DFlash2 --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 config.json model.safetensors`.
Splash refuses a draft whose config does not match the target (`model-check` with the draft config).

## 6. Harness integration

### 6.1 Role: a same-artifact peer of record, engine kind `splash`

The native-arm registry (`native_peers.py`) assumes "serve their own native artifact" and a role that is "never the
verdict of record". Splash is neither. It reuses the **adapter interface** (closed request/response envelopes,
usage, readiness, closing audit, identity, process names) and gets a new role:

- `coding_family` gains `SAME_ARTIFACT_KINDS = ('splash',)` beside `NATIVE_KINDS`, with
  `ADAPTER_KINDS = NATIVE_KINDS + SAME_ARTIFACT_KINDS` for client adapter dispatch. Every "native means not the
  verdict" branch keeps testing `NATIVE_KINDS` (`native_peers.is_native`) and so stays correct.
- The adapter module is `native_peer_splash.py`, resolved by `native_peers.adapter`, which accepts `ADAPTER_KINDS`.
  `q3_role` returns "same-artifact peer of record", and `NATIVE_COMPARISON`/`SAME_GGUF_UNAVAILABLE` are `None`.
- `launch_problems` **requires** the case artifact sha256 to equal the hf2q arm's artifact sha256. This is the
  inverse of the native arms, which refuse an ambient hf2q artifact.
- `PROCESSES = ('splash',)` covers the engine binary. The Python server is identified by argv, `-m server.server`
  under the pinned `.venv`, in the competing-runtime inventories.

### 6.2 Launch wrapper (`native_peer_splash_launch.py`; serve.argv execs it, so `SERVER_PID` is the server)

1. Verify `SPLASH_DIR`:
   - the clean detached checkout at `SPLASH_COMMIT`;
   - `build-identity.json` `build_id == SPLASH_BUILD_ID`;
   - `build/splash` sha256 `== SPLASH_BIN_SHA256`, `build/splash.metallib` sha256 `== SPLASH_METALLIB_SHA256` and
     `build/host-step.metallib` sha256 `== SPLASH_HOST_STEP_SHA256` (present: awake host steps);
   - the toolchain recorded at build time (`build-031a25a6/toolchain.txt`: Xcode 27.2 27B52, Metal 32023.922, Apple
     clang 21.0.0, SDK 27.2, macOS 27.2 26B5112a) equals the host's current `xcodebuild -version`, `metal
     --version`, `clang++ --version` and `sw_vers`;
   - every `install/requirements.txt` pin present in `.venv`, and the dist-info freeze sha256
     `== SPLASH_VENV_FREEZE_SHA256`.
2. Verify the draft: `DRAFT_DIR` is the pinned HF cache snapshot, its `config.json` sha256 equals the pin, and its
   `model.safetensors` blob name equals the pinned LFS sha256.
3. Call `install.local.prepare_local_model(MODEL, draft=DRAFT_DIR, models_dir={OUT}/splash/models)` inside Splash's
   `.venv`. It is a per-run directory and makes no network request.
4. Check the returned `model.json`:
   - `family`, `target_format gguf`, `vision_format none`;
   - exactly one linked GGUF, whose content `digest == MODEL_SHA256`;
   - the draft weights' digest equal to the pin.
5. Write `{OUT}/splash/launch.json`. It holds the pid, the argv, the environment, the verified identity, the
   assembly (path, `model.json` sha256, family, digests) and the cross-run state before the run. It is per run: its
   sha256 is bound into that run's approval (`native_splash_launch_sha256`), never into the campaign identity.
6. `execve` the argv template with that run's assembly path:
   `{SPLASH_DIR}/.venv/bin/python -P -u -m server.server <assembly> --model hf2q/<family> --binary
   {SPLASH_DIR}/build/splash --host 127.0.0.1 --port {PORT} --max-context 256K --idle-release off --no-webui
   --thinking-end-text <D4> <config flags>`.
   The environment is `PYTHONPATH={SPLASH_DIR}`, `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`. No
   `SPLASH_*`, `PYTHON*`, `HF_HOME`, `HF_HUB_CACHE` or `HF_TOKEN` is inherited.

What the campaign identity freezes (q3_identity, re-checked before every run):
- the commit, `build_id`, the sha256s of `build/splash`, `build/splash.metallib` and `build/host-step.metallib`,
  the build toolchain versions and the venv freeze;
- the draft's revision and digests;
- `MODEL_SHA256`, through the same-model rule;
- `SPLASH_CONFIG`;
- the argv template.

It never freezes the assembly path, `model.json` bytes, `launch.json` or file times, which change whenever an hf2q
run touches the GGUF's ctime (§3.6). Per run, readiness re-derives the argv from the template and that run's
assembly, and requires every `model.json` field other than paths and times to match the frozen values.

### 6.3 Request parity (closed envelope; Splash ignores unknown Chat fields, `frontend.py` reads by `body.get`)

| Canonical hf2q field / behaviour | Splash request | Note |
|---|---|---|
| model, messages, tools, `tool_choice auto`, `stream`, `stream_options.include_usage`, `logprobs false` | as is, except each function's semantic default `strict:false` is omitted from the Splash wire and restored by the inverse | Splash renders an explicit `strict:false` into the prompt while hf2q omits it; removing the default is semantically equal and is required for prompt-byte parity. `logprobs:true` and `n≠1` would 400 |
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
- every turn's prompt is re-rendered through `POST /apply-template`, and `/tokenize` of it must equal
  `usage.prompt_tokens`; the rendered bytes are archived as `splash-apply-template.raw`;
- prompt byte parity with hf2q is checked offline after the run, by the hf2q renderer below. Until it runs, the
  run's parity is `PendingPostOracle`. A mismatch makes the run Invalid;
- each reply is bound to its `Done` console line (input, cached and output token counts).

**Offline hf2q renderer.** This is a Splash mode of the hf2q exact-prefix post-oracle. It is a new ignored test,
`splash_prompt_parity_post_oracle` (`src/serve/api/handlers_splash_prompt_parity_post_oracle_tests.rs`), which
shares the evidence binding of `qwen38_n1_exact_prefix_post_oracle`
(`docs/research/adr063-qwen38-prefix-post-oracle.md`). `splash_sweep.py parity-config RUN --template FILE
--output-dir DIR` writes its frozen config. The oracle is model-free and starts no server. For each archived Splash
request it:
1. takes the canonical hf2q form of the request (`native_peer_splash.canonical`: the same messages and tools,
   `enable_thinking` and `preserve_thinking` true, no effort);
2. renders it with hf2q's production request schema, Qwen template kwargs and renderer, using the frozen
   artifact's GGUF-embedded template and tokenizer;
3. compares the prompt bytes with `splash-apply-template.raw`.

Splash's later-system patch is kept only when ordinary conversations render byte-identically to the original
template, so a byte mismatch means a real rendering difference.

Post-sweep correction (2026-11-09): a parity-specific replay of the failed
Qwen3.8 S0 run found the first byte mismatch at offset 532. Splash rendered
`\"strict\": false` once per function definition; hf2q omitted that semantic
default, making Splash's turn-0 prompt 51 bytes longer (3,135 vs 3,084).
`native_peer_splash.request_controls` now omits only this false default on the
Splash wire, while `canonical` restores it for the canonical request and
refuses any other wire shape. The sweep predating this correction remains
invalid and is never relabelled.

A corrected Q5 replay then passed the byte post-oracle on all 20 completed
turns. The run remained `ClientFailed`: two agents emitted and tested the
correct `std::num::ParseIntError` edit and two retained `std::fmt::Error`,
below the 0.75 task floor. The correction therefore closes prompt rendering
for that run without converting the frozen sweep or corrected Q5 into valid
Splash evidence.

### 6.5 Q5 cases (to be added with the adapter, not before)

`q5_cases/splash-qwen38.json` and `splash-qwen36.json`. Schema as `native-omlx-qwen36.json`, with these values:
- workload id `adr063-q5-splash-<family>-n4-coding-ctx262144`;
- `binary {SPLASH_DIR}/build/splash`;
- artifact `{MODEL}`/`{MODEL_SHA256}`, the **same** values as `qwen38.json`/`qwen36.json`, required non-empty;
- variables `SPLASH_DIR`, `SPLASH_COMMIT`, `SPLASH_BUILD_ID`, `SPLASH_BIN_SHA256`, `SPLASH_METALLIB_SHA256`,
  `SPLASH_HOST_STEP_SHA256`, `SPLASH_VENV_FREEZE_SHA256`, `DRAFT_DIR` and `SPLASH_CONFIG` (§7);
- `client.argv … --engine-kind splash`. The approval carries a `native_splash` block (the launch inputs) and the
  per-run `launch.json` sha256. The source freeze adds `native_peers.py`, `native_peer_splash.py`,
  `native_peer_splash_ready.py` and their test.

Sweep cases are derived per config by `splash_sweep.py derive-case --config <name>`, which changes only
`SPLASH_CONFIG`.

### 6.6 Run audit and Q3/Q4 code changes

Splash cohorts are audited the way the peer's are: the dry `audit_agent` audit, plus `q2_run_audit` with the arm's
audit module (`native_peer_splash_audit.audit_run`: Done/Started lines, render-proof archive, no cancellation, error
or engine restart). They are not Q2-certified: `q2_certify.py` refuses every non-hf2q cohort, the peer's included,
because only hf2q has a server phase trace. Code changes:
- **Arms.** `q3_campaign.ARMS` gains `splash`, set with `--splash-case` (kind `splash` only), `--splash-var`. There
  is no `splash_matched` arm: the matched-settings diagnostic is sweep-only (§7.2, M × 2).
- **Stages (D7).** The `splash` arm runs only in the n4 and n1 stages. The 3+1/2+2/1+3 control stages leave it out
  of their measured arms and Williams orders.
- **Splash is required.**
  - For `qwen38` and `qwen36`, the default campaign always runs the `splash` arm. `--splash-case` defaults to
    `q5_cases/splash-<family>.json`. The arm is left out only with `--splash-unavailable-evidence`; a
    peer-unavailable campaign (already `Unavailable`) runs no Splash arm, and a controls-only campaign follows its
    primary of record.
  - `plan.splash_required` is set when the arm is planned or declared unavailable. Such a campaign without a
    runnable Splash arm (declared unavailable, or a load failure) gets `Unavailable` from `family_verdict`, never a
    Win against the peer alone. Campaigns planned before Splash carry no requirement and keep their verdicts of
    record.
  - Q3 `launch_problems` refuses, before any run, a Splash build, toolchain or venv that differs from the case's
    declared hashes and a campaign output that is not on APFS; each would otherwise make every launch refuse.
  - A Splash Q5 memory stop halts the campaign (D6), including one in the load and readiness window before the
    client (the receipt's `stop.reason` is a memory stop, or the server died of Metal out-of-memory).
- **Verdict state.** `state` gains `splash_required`, `splash_unavailable` (dict or None), `splash_failures` and
  `vs_splash`.
- **Check order** in `family_verdict`:
  1. experiment failures → `Failed`;
  2. quality loss → `Failed`;
  3. peer unavailable → `Unavailable`;
  4. Splash required and unavailable → `Unavailable` ("no runnable Splash arm");
  5. peer failures → `Failed`;
  6. Splash failures → `Failed` ("a failed Splash arm cannot yield a win");
  7. not complete → `Pending`;
  8. vs-peer ratio CI above 1 → `Loss`;
  9. vs-Splash ratio CI above 1 → `Loss` ("experiment makespan is longer than Splash's");
  10. the Win conditions, where new v0.1.30 plans require `vs_splash` to be precise with a CI upper bound ≤
      `PEER_PARITY 1.00`, beside `vs_stock` and `vs_peer`; historical plans retain their frozen 0.95 threshold.
- **Stopping rule.** `q3_report.required_effects` adds `vs_splash` while Splash is available and has not failed.
  `CONTRASTS` gains `("vs_splash", "splash")`.
- **Report.** `verdict.json` gains a `splash` block (unavailable, failures, label). Rows render
  experiment-vs-Splash with baseline `splash`, descriptive, task makespan only, never a win label. Fairness against
  Splash is reported and not gated.
- **Quality gate.** `quality_campaign.py --reference-kind splash` (reference engine kind `splash`). In
  `f2_q3_tier`: `REFERENCES["splash"] = "Splash"`; `required_references` adds `splash` for Qwen families whose
  record has a Splash arm; `gate_status` binds the reference runs to the campaign's `splash` arm binary and one
  Splash readiness identity. `--of-record` takes the Splash gate manifest beside the stock and peer ones.
- **Identity.** `q3_identity` adds the adapter's `campaign_identity` for the `splash` arm. The same-model rule
  applies to it, unlike native arms, and `launch_problems` requires its artifact sha256 to equal the
  experiment's.

## 7. Best valid configuration

### 7.1 Validity

A configuration is valid when **Splash's own runs** pass, with no hf2q arm involved:
- every run is Q5 `CalibrationPass`, with no engine relaunch;
- every run reports 4 lanes, `max_model_len 262144` and text only;
- every run passes the dry audit, `q2_run_audit` with the Splash audit module, and request parity per §6.3;
- across the configuration's runs, Splash's task pass fraction (agents whose task passed over agents run) is at
  least the Q1 numeric-policy floor `paired_min_pass_fraction` 0.75 (`quality_campaign.POLICY`), and every
  passing task passes the Q1/Q3 task and tool gates.

The paired task-quality gate between the hf2q candidate and Splash is a Q4 gate on the candidate (§6.6). It never
rules out a Splash configuration: that would discard exactly the configurations where Splash does better than hf2q.
Engine-documented lossy settings (int8 KV, which is the default; the Neural Engine W8A8 split, the default on
Qwen3.8; bf16 state if selected) are valid whenever these conditions hold.

### 7.2 Sweep grid (one factor at a time from the default, 2 runs each, then confirmation)

| Point | Qwen3.8 | Qwen3.6 |
|---|---|---|
| S0 default | int8 KV, fp32 state, ANE split auto, decode-share 0.5, SSD cache auto | int8, fp32, verify-rows auto, 0.5, auto |
| S1 | `--kv-format bf16` | `--kv-format bf16` |
| S2 | `--state-format bf16` | `--state-format bf16` |
| S3 | `--disable-ane` | `--verify-rows all` |
| S4/S5 | `--decode-share 0.25` / `1.0` | same |
| S6 | `--max-cache-disk 0` | same |
| M (matched diagnostic, sweep-only; not a campaign arm) | bf16 KV, fp32 state, `--disable-ane`, cache-disk 0 | bf16 KV, fp32 state, `--verify-rows all`, cache-disk 0 |

Selection uses the declared makespan rule (Q4-DS4 precedent): lowest median makespan among valid points. A best
within the noise of the runner-up is provisional, and both are re-run (3 runs each) before the campaign of record.
Combining two winning factors (e.g. S2+S5) is one extra confirmation point, not a full factorial.

Speculation on/off cannot be swept (§1). Disclose it as "speculation always on (DFlash2), no off switch".

### 7.3 Preflight per family (before S0)

1. `build/splash model-check` on the prepared assembly (CPU).
2. A first start, untimed. It digests the GGUF, derives the tokenizer and, for Qwen3.8, calibrates the ANE split.
3. Q5 calibration run of S0. Splash gets its own effective floor from this run (§3.3). If the floor trips here, at
   any later sweep point or in the campaign, the operator chooses a `--max-memory` cap (D6), and the affected
   configuration is re-run under the cap.

## 8. GPU time estimates

Basis: Q3 v11 Qwen3.8 (`$E/q3-campaign-v11/qwen38-all`) took 24 n4 runs (4 arms, warmup plus 5 blocks) in 44 min,
**about 1.85 min per run**, and 60 n1 runs in 49 min, about 0.8 min per run. Qwen3.6 repair1 Q5 runs: peer
makespan 36 s, hf2q 53 s, readiness 4 s. Splash startup is unmeasured. **[INFERENCE]** 0.5–1.5 min per start: the
per-run GGUF digest (16.8 GB for Qwen3.8 and 21.7 GB for Qwen3.6, every start, at roughly 2–3 GB/s about 6–11 s),
reading and repacking 17–22 GB of weights, and for Qwen3.8 the Neural Engine program build. A Splash run therefore
costs **about 2.5 min (Qwen3.8) and 1.7 min (Qwen3.6)**.

| Item | Qwen3.8 | Qwen3.6 |
|---|---|---|
| Preflight (model-check, first start, Q5 S0) | ~10 min | ~8 min |
| Sweep S0–S6 × 2 + confirmation 2×3 + M × 2 | 22 runs ≈ 55 min | 22 runs ≈ 40 min |
| **Fit + sweep total** | **≈ 1.1 h (range 0.8–1.6 h)** | **≈ 0.8 h (0.6–1.2 h)** |
| Paired quality gate vs Splash (same instance count as the peer gate; observed ~45 s/run on Qwen3.6, ~1.5 min on Qwen3.8; ~48 runs) | ≈ 1.2 h | ≈ 0.7 h |
| Q3 n4 primary, 5 arms (stock, experiment, peer, peer_matched, splash), warmup + 5–20 blocks | 30–105 runs ≈ 1.0–3.6 h | ≈ 0.7–2.4 h |
| Q3 n1 control, 5 arms, 5–20 blocks (v11 stopped at 14) | ≈ 0.4–1.6 h | ≈ 0.3–1.2 h |
| **Full paired race per family (n4 + n1)** | **≈ 1.4–5.2 h**; Splash's share ≈ 25% | **≈ 1.0–3.6 h** |

Two peer contrasts must both reach the 2.5-point half-width. Expect the race to need more blocks than v11's 5 when
the two peers' makespans differ a lot. Splash does not run the mixed 3+1/2+2/1+3 controls (decision D7).

## 9. ADR-063 amendment

The amendment is applied in `docs/adr/ADR-063-apple-silicon-four-agent-speed-coherence.md`. It adds a new section,
"Splash peer of record for the Qwen families (operator decisions 2026-11-03/11-04)", after "Native-engine peer
arms", and makes these changes:
- the Updated line;
- the Q4 Outcome: both peers for Qwen3.8/Qwen3.6;
- the Q4 win and quality-gate acceptance bullets: the hf2q/Splash ratio meets the same 0.95 bound, and the quality
  gate also runs against Splash;
- the Q4-QWEN38 Outcome and Predecessors: the Splash Qwen3.8 fit and best-configuration record;
- the Q4-QWEN36 workload bullet and Predecessors: Splash runs the declared Qwen3.6 workload.

## 10. Decisions (operator, 2026-11-04)

The operator accepted every recommendation on 2026-11-04 and approved this design (operator, 2026-11-04). The
independent review (ACCEPT-WITH-CONDITIONS) then tightened D6, D9 and D10 as written here. The rejected alternatives
are kept for the record.

- **D1 — Campaign structure: one five-arm Williams campaign per family** (stock, experiment, peer, peer_matched,
  splash), with same-session thermal conditions and one identity freeze. There is no Splash matched-settings arm:
  that diagnostic is sweep-only (§7.2). The rejected alternative was a separate experiment-vs-Splash campaign bound
  to the record campaign's candidate binary.
- **D2 — Pin: source build `031a25a6`** (§2.1 identity). The `1.5.0` bottle is not used.
- **D3 — Repetition penalty: none is sent** (Splash default 1.0), as the TensorFold arm does. The difference from
  hf2q's 1.05 over a 64-token window is recorded in every Splash statement and run audit.
- **D4 — Forced thinking close: hf2q's text**, `--thinking-end-text $'\nI need to answer now.</think>'` (8 tokens,
  §4).
- **D5 — `parallel_tool_calls: false`** is sent (hf2q's default). Splash constrains calls by its grammar.
- **D6 — Memory cap: `--max-memory auto` by default.** If the Q5 floor trips at any sweep point or in the campaign
  (§3.3), an explicit cap is allowed. The cap is recorded as part of that configuration (`max_memory` knob) and
  applied to every later run of it. A cap found during the campaign halts it, and the configuration is re-confirmed
  under the cap before the campaign restarts. It is never silently changed or used to cut context.
- **D7 — Controls: Splash runs in the n4 (four-agent) and n1 (single-agent) stages only.** It does not run in the
  3+1/2+2/1+3 mixed controls.
- **D8 — Symlinks: acceptable.** Splash's assembly symlinks to the GGUF and the draft (no hard link, no write).
- **D9 — Qwen3.6 workload: the declared Qwen3.6 workload.** The Splash Qwen3.6 case runs whatever workload
  Q4-QWEN36 declares for the hf2q and peer Qwen3.6 arms. The operator's direction is the repair-turn workload,
  one repair turn after a failed `run_tests` (`qwen_n4_live_smoke.py --repair-turns 1`, Qwen3.6 only). It is
  recorded in ADR-063 under Q4-QWEN36 (#681, after the floor was cleared: hf2q 21/24 vs peer 20/24, `Pass`).
  The client's repair-turn gate admits `splash` for Qwen3.6 since #715, whose case is
  `q5_cases/splash-qwen36-repair1.json`. Qwen3.8 keeps the canonical six-step protocol.
- **D10 — Lossy engine settings are valid when Splash's own task outcomes pass.** These are int8 KV (the default),
  bf16 state (only if a sweep point selects it; fp32 is the default) and the Neural Engine W8A8 FFN split (the
  Qwen3.8 default). A configuration is valid or invalid on Splash's own runs (§7.1): Q5, the run audits, the Q1
  pass-fraction floor and the task gates. The paired hf2q-vs-Splash quality gate is a Q4 gate on the hf2q
  candidate and never disqualifies a Splash configuration.

## 11. Open risks

- The native weight load of the hf2q GGUFs is unproven until the fit run; screening and `model-check` pass (§4).
- The Q5 floor versus Splash's cache-growth target (§3.3).
- Splash relaunches its engine on failure. A relaunch mid-run is an invalid run, not a recovered one.
- Splash appears much faster than the current peer on both Qwen families **[INFERENCE from its published figures]**.
  If so, the Qwen Win bar becomes Splash's, and Q4-QWEN38's starting gap (−6.4% vs the peer, Q3 v11) widens
  substantially.
