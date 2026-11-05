# Four-agent campaign identities (ADR-063 F1)

Requires Python 3.11 or newer. `identity.py` checks a locally frozen campaign manifest and hashes every referenced file. It does not run inference, build software, contact an upstream service or certify performance. Invoke it only after release-coordination preflight: hashing large models is host work too.

```sh
python3 scripts/four_agent/identity.py /path/to/campaign.json
python3 -m unittest discover -s scripts/four_agent -p 'test_*.py' -v
```

Paths in the manifest are absolute or relative to the manifest directory. Keep campaign JSON and referenced model/binary/receipt files outside Git. Do not put credentials in version output, settings or receipt files. Every file identity is an object with `path`, positive integer `bytes` and lowercase SHA-256 `sha256`. Files must exist and their actual length and digest must match. There is no skip-hashing option.

The manifest uses schema `hf2q.four-agent-identities.v1` and these required fields:

| Object | Required fields |
| --- | --- |
| Root | `schema`, `campaign_id`, timezone-qualified `frozen_at`, `coordination_ref`, `hardware`, `engines`, `models`, explicit `evidence_files` list and `comparators` |
| Hardware | `chip` = `Apple M5 Max`, `physical_bytes` = 137438953472, `os_version`, `os_build`, `metal_version`, `compiler_version` |
| Engines | Exactly `stock`, `experiment`, `latest_peer` |
| Every engine | 40-character `source_commit`, captured executable `version_output`, `binary` file identity, `build_receipt` file identity, explicit `runtime_dependencies` list of file identities |
| Stock and experiment | `native` object and `dependency_provenance_receipt` file identity |
| Native | 40-character `source_commit`, published `version`, registry `source` = `registry+https://github.com/rust-lang/crates.io-index`, locked `checksum`, downloaded `crate` file identity matching that checksum |
| Experiment | `baseline_commit` equal to stock `source_commit`; explicit `identity_mode: stock_control` if its executable bytes equal stock |
| Latest peer | `resolved_upstream_commit` equal to its built `source_commit`, `upstream_resolution_receipt` file identity |
| Models | Exactly `qwen38`, `deepseek4`, `qwen36`, `gemma4` |
| Every model | `hf_repository`, 40-character `hf_revision`, `variant` = `vanilla`, `quantization`; file identities for `gguf`, `conversion_receipt`, `source_integrity_receipt`, `tokenizer`, `template`, `rendered_tokens`, `production_sampling`; `arms` object |
| Every model's arms | Exactly the three engine arm names, each with `model_sha256` equal to the shared GGUF digest, `matched_settings`, `resolved_production_sampling` and `capability_receipt` file identities, explicit `drafts` list of file identities (empty when no separate draft artifact) |
| Every model's latest peer binding | Additional `peer_best_settings` file identity, separate from matched-settings diagnostics |

The downloaded native archive must also contain a unique regular-file Cargo manifest with the declared package/version and VCS metadata matching the declared clean native source commit; metadata is read without extraction. Stock and experiment can pin different published native versions. Their common baseline is explicit; source ancestry and the exact build inputs must be established by review of build/dependency receipts, not inferred from matching text fields. Different draft artifacts are allowed, but each is hashed and capabilities/settings must justify its use. Models with different GGUF bytes belong in separately labeled native-format diagnostics, not this same-artifact manifest.

A successful result states `offline_file_identities_verified: true` and **always** `campaign_qualified: false`. This distinguishes file integrity from semantic proof. The validator does not authenticate HF ownership, infer conversion settings, parse arbitrary build/provenance receipts, establish Git ancestry, prove upstream recency from a self-declared commit, or execute the recorded version command. Independent review must inspect those hashed receipts. F2/Q2 must additionally bind the frozen manifest to live runtime attestations, exact inputs/settings, memory/thermal and coordination admission, real slot/cohort progress and task/tool outcomes. Q3/Q4 require paired independent four-model speed verdicts.

The existing `scripts/grammar_probe/runtime_identity.py` supplies the live runtime-attestation checks. This module adds the campaign-wide cross-arm/file bindings and does not replace that live check or existing thermal/process guards. Synthetic fixtures in `test_identity.py` exercise rejection paths; their passing result is not model or hardware evidence.

See [SETTINGS.md](SETTINGS.md) for source-bound client wire capture, effective sampling controls, repetition-history differences and the separation between production, matched-diagnostic and peer-best arms. `capture_client.py` provides a model-free localhost capture with an isolated OpenCode profile.

The four GGUF digests must be distinct. Each model's tokenizer, template and rendered-input receipt must have distinct digests, and the peer executable must differ from the hf2q executables. An explicit stock-control experiment is an identity diagnostic, not a candidate speed win.

`rendered_tokens` binds the same aggregate `hf2q.offline-rendered-inputs.v1` receipt for all four models. The validator joins its declared model, effective template and tokenizer hashes to the frozen files, checks each prompt/token sidecar hash, token count and context bound, and requires clean source provenance. DeepSeek must identify its native encoding; other families use GGUF metadata templates. The receipt records implementation, configuration and capture-binary hashes. Those provenance fields still require independent source/build review. Successful validation returns both the canonical manifest digest and checked file observations; the CLI also returns the exact raw manifest-file digest.

The ignored Rust offline capture test shares HTTP tokenization without adding BOS/EOS postprocessor tokens and selects the effective serving template. It accepts a restricted text/tool request subset and rejects unsupported controls and context overflow. It does not load model weights or prove API admission, tool execution or generation quality. Build it from a clean committed tree with `GIT_COMMIT_SHA` set to that commit; retain the exact capture binary and configuration. The checked-in canonical coding requests are synthetic renderer fixtures, not completed coding tasks.

Supporting `evidence_files` are rehashed directly, including help output, frozen build inputs and settings registries. Each engine must declare `runtime_dependencies` explicitly; an empty list requires independent build/linkage review establishing no separately distributed non-system runtime files. A build receipt listing dylib hashes does not substitute for this directly checked list. Platform frameworks remain bound through OS/build identity and explicit linkage review.

`comparators` contains exactly `historical`, `basert`, `kapitsa`, `mlx_release`, `mlx_main`, and `mlx_lm`. Every entry declares nonempty `identity`, `status`, `format_route`, `qualification_limit`, and a nonempty list of directly hashed `evidence_files`. These are explicit inventory dispositions, not inherited runtime capability. Settings contents and completeness still require independent source review. The receipt records the validator's own source SHA-256 so a historical result can be tied to the exact checks it ran.

Canonical render evidence must include `cold` and `tool_round_1` through `tool_round_6` for every family. Additional labeled cases are allowed. Each row's request-file digest must match a directly verified supporting evidence file; a token sidecar without its frozen input bytes is insufficient. The DeepSeek loader-source digest is required alongside the renderer/HTTP/capture source identities. This certifies file bindings only; real-client requests rejected by the schema remain an explicit separate capability failure.

The strict physical parser continues to reject negative residuals. The separate
`retryable_correlated_overcount` classifier recognizes only a fully corroborated
CSV/text negative N-page root whose magnitude exactly matches the category subtotal
above physical capacity. It returns no accounting values or admission credit; no
overcount magnitude becomes admissible. The original single-page classifier remains
available for compatibility with its narrower contract.
An operator supervisor may preserve that rejected observation and request a fresh
reading within its existing attempt/deadline limits. It must not retry a valid
low-memory or pressure reading, and an interrupted campaign remains incomplete.

## Q5 per-family calibration (ADR-063 Q5)

`q5_calibration.py` runs one supervised four-agent workload per family and writes a receipt. It follows the measured-admission Q5 text: the host must be running only this workload. The receipt records the host baseline, the host-level and process peaks, the deltas, the effective Free+cached floor with its inputs, and the cleanup. Live stops end the run before memory runs out. The supervisor never retries, never picks a favorable sample and makes no speed claim. Every receipt says `independent_review_required: true`.

Every family run has a `run_kind`:
- **`calibration`** (no `--admission-receipt`). It establishes the family's calibrated peak. Its growth cap is the host-level floor.
- **`admitted`** (`--admission-receipt <CalibrationPass receipt>`). It runs only if the current baseline covers the calibrated charge, and it carries the calibrated footprint cap. A pass is `AdmittedRunPass`.

| File | Role |
| --- | --- |
| `q5_calibration.py` | CLI: dry run, preflight, admission, launch, readiness, client, stops, cleanup, receipt |
| `q5_admission.py` | Pure admission charge, mandatory growth cap and floor seeding |
| `q5_artifacts.py` | Model/projector identity from conversion receipts; text-only locator; input hashes |
| `q5_case.py` | Case-file schema `hf2q.q5.calibration-case.v1`, placeholders, argv rules |
| `q5_policy.py` | Pure stop rules, dynamic Free+cached floor, sample aggregation, verdicts |
| `q5_preflight.py` | Detects competing runtimes; checks the port |
| `q5_host.py` | Parsers and wrappers for `vm_stat`, `sysctl`, `footprint`, `ps`, `lsof` and the thermal state |
| `q5_runtime.py` | Sampler thread (every 500 ms), stop latch, cleanup of the owned process groups |
| `q5_negative_control.py` | Bounded synthetic memory-growth child |
| `q5_cases/{qwen38,qwen36,gemma4,deepseek4}.json` | Example family cases |
| `q5_cases/cancel-{deepseek4,qwen38}.json` | S3 cancel smoke: the family server with the joint-turn trace, client `cancel_smoke.py` (see "S3 cancel smoke") |

### What one run does

1. **Preflight.** It reads `ps -axww` and refuses to start if any other runtime is running.
   - It looks only at each process's executable, or at the program an interpreter or `env` runs. It never searches the rest of the command line, so an agent such as omp whose prompt mentions "hf2q" does not count.
   - Matched as runtimes:
     - `hf2q` and `hf2q-*` (including Cargo test binaries).
     - The peer engine's `llama-*` tools, `ollama` and `lms`.
     - Python-hosted peers: `python -m mlx_lm.server` or `python -m mlx_lm server`, `mlx_lm.*` and `mlx_vlm.*` console scripts, `vllm`, `sglang`.
     - A running `serve_*_opencode.sh` launcher.
   - `cargo` and `rustc` count as competing builds.
   - The supervisor's own process and its ancestors are ignored.
   - The port must have no `lsof` listener, and a connection to it must be refused.
2. **Baseline.** It takes six samples, 500 ms apart. Every sample must be observed, pressure must not be critical, and Free+cached must stay above 2 GiB. An admitted run is also refused unless the baseline Free+cached is at least the calibrated charge plus the run's initial effective floor (see below). It also records:
   - `vm_stat`: free, speculative, file-backed, anonymous, compressor and wired pages.
   - `vm.swapusage`.
   - `kern.memorystatus_vm_pressure_level` and `kern.memorystatus_level`.
   - `hw.memsize` and the thermal state.
3. **Launch.** It starts the case's `serve.argv` directly, without a shell, in a new session (its own process group). The child's environment is `PATH HOME USER LOGNAME TMPDIR LANG LC_ALL LC_CTYPE SHELL TERM` plus the case's explicit `env`; nothing else passes through. A case with `artifact.text_only_locator: true` serves a hard link of the model from a private `{OUT}/text-model/` directory, which is removed after cleanup (see the case-file notes).
4. **Readiness.** It polls the readiness URL, saves the `record` documents and checks each `checks` entry with strict JSON equality. A mismatch, for example a context length other than 262144, stops the run before the client starts.
5. **Client.** It renders the client approval from the template and runs `client.argv` in its own session.
6. **Sampling.** A separate thread samples every 500 ms. Each sample has host memory plus the phys_footprint and RSS of every member of the server's process group. Samples are written to `samples.jsonl`.
7. **Live stops.** Checked on every sample, in this priority:
   - `pressure_critical`: `kern.memorystatus_vm_pressure_level >= 4`. Warning pressure (level 2) is counted in `pressure_counts` but does not stop the run. On 2026-10-30, warning appeared within 17 s while the 107 GB DeepSeek-V4 artifact faulted in, and 6.6 GB was still reclaimable.
   - `swap_growth`: swap used is more than baseline + 512 MiB.
   - `free_cached_floor`: free + speculative + file-backed pages are below the run's *effective floor* (at least 2 GiB; see "Effective Free+cached floor" below).
   - `footprint_cap`: the server group's phys_footprint is above the run's growth cap. Every run has one (see "Growth cap" below).
   - `sampler_blind`: 3 consecutive blind samples.
   The deciding thread sends SIGTERM to the server and client groups immediately. The receipt records the sample→signal→exit latency.
8. **Cleanup.** Every run ends the same way, whether it completed or stopped:
   - SIGTERM to each owned group, a 10 s wait, then SIGKILL and a 5 s wait.
   - Descendants that moved to their own group are signaled only if every member of that group is a known descendant (tracked by PID and start time).
   - The run then verifies that the groups are empty, the children have been reaped and the port has no listener.
   - It takes one more host sample to record memory recovery.

**Footprint source.**
- The primary source is `/usr/bin/footprint -p PID -f bytes --noCategories` (`phys_footprint`).
- If that fails for a member, that member's `ps` RSS is used instead. The sample is then labeled `ps-rss-fallback`, and `samples.footprint_sources` counts both kinds.
- RSS can undercount mapped model weights. So when a footprint cap is enforced, a fallback sample counts as blind.

**Verdicts.**

| Verdict | Meaning | Exit |
| --- | --- | --- |
| `CalibrationPass` | Calibration run. Client succeeded. Pressure was never critical (warning is allowed and recorded), swap growth stayed below 512 MiB, the minimum Free+cached stayed at or above the run's final effective floor, the floor used a measured stop latency, no sample was blind, and cleanup was proven. | 0 |
| `AdmittedRunPass` | The same conditions, for an admitted run. | 0 |
| `Stopped` | A live stop or a supervisor stop ended the run. The stop wins over every other outcome. Supervisor stops: `server_exited`, `readiness_timeout`, `readiness_mismatch`, `client_timeout`, `interrupted`. A stop is a preserved failed case, not a retry trigger. | 1 |
| `ClientFailed` | The client exited nonzero, or `summary.status` is not `success_status`. | 1 |
| `CalibrationFailed` | The client completed, but a pass condition failed. | 1 |
| `PreflightRefused` | Nothing was launched. | 2 |
| `SupervisorError` | The supervisor itself failed. | 1 |

A violation that first appears during the unload after a completed run does not turn the run into `Stopped`. It is recorded as `post_completion_violation`, and it still fails the pass conditions.

### Admission, growth cap and effective floor

**Calibrated host peak.** phys_footprint leaves out file-backed weights. A DeepSeek-V4 receipt shows a 5.3 GiB footprint against a roughly 102 GiB Free+cached drop. So every receipt records `host_peak`:
- `peak_host_used_bytes` = physical − the run's minimum Free+cached.
- `baseline_host_used_bytes` = physical − baseline Free+cached.
- The process `peak_footprint_bytes`.

**Admission** (`--admission-receipt`). The receipt must be a `CalibrationPass` from a calibration run (an admitted-run receipt is refused). It must match this run's family, workload `id`, slots and context, binary sha256, artifact identity (sha256, otherwise size + mtime) and physical memory. Any mismatch means "recalibrate" and refuses the run.
- The charge is `peak_host_used_bytes − baseline_host_used_bytes` from that receipt, which equals its baseline Free+cached minus its minimum Free+cached. Receipts written before this field existed still yield it.
- The run is refused (`PreflightRefused`) unless the current baseline Free+cached ≥ charge + the run's initial effective floor. That floor is computed from the calibration's measured drop rate (below). Receipts that predate the rate record seed nothing, so their initial floor is 2 GiB.
- `receipt.admission` records the charge, the decision and the admission receipt's path and sha256.

**Growth cap.** Every run carries one, recorded as `growth_cap`:
- **Admitted run:** a footprint cap of the calibrated `peak_footprint_bytes × 1.25`. It bounds anonymous growth (caches, scratch, projector) with 25% headroom for run-to-run variation. File-backed weights stay covered by the floor, swap and critical-pressure stops.
- **Calibration run:** the host-level cap. Host used may not exceed physical − the effective floor. This is exactly the `free_cached_floor` stop, and the receipt says so (`growth_cap.bytes` is shown at the 2 GiB minimum).
- **Negative control:** its own footprint cap.

Cases cannot set a footprint cap.

**Effective Free+cached floor** (`receipt.effective_floor`, every run; also `samples.effective_floor`):

```
effective_floor = max(2 GiB, max_post_readiness_drop_rate × (interval + stop latency))
```

- The rate is a running maximum. Each pair of consecutive observed samples whose earlier sample started at or after readiness contributes its Free+cached drop divided by the gap between them. The live stop compares every sample against the current floor, and a sample's own drop raises the floor before that comparison.
- An admitted run starts with its admission receipt's measured rate (`floor_inputs.seed_rate_bytes_per_s`). A calibration run starts from 0, so its floor is 2 GiB until a rate is measured.
- The latency is the `sample_to_exit_ns` of the `--negative-control-receipt`. That argument is required for every family run; the accepted control measured 146 ms. A run that stops also records its own `stop_latency`.
- A pass requires the minimum Free+cached to be at least the final effective floor, and the latency to be measured.
- Replayed on the 2026-10-30 calibrations:
  - DeepSeek-V4: floor 2.00 GiB, minimum 4.09 GiB.
  - Qwen3.8: floor 13.23 GiB (20.5 GiB/s first-request allocation right after readiness), minimum 51.30 GiB.
  - Gemma4: floor 5.33 GiB, minimum 65.87 GiB.
  - All three still pass.
- `max_load_phase_drop_bytes_per_sample` records drops before readiness separately. For example, DeepSeek-V4 page-cached weights becoming wired drop Free+cached by up to about 80 GiB in one sample. No floor can catch that. Load-phase residency is bounded by the artifact's fixed size, which the admission charge already includes.

### Case files and variables

Placeholders are written `{NAME}` and are filled from three sources:
- **Built-ins:** `REPO`, `OUT`, `PYTHON`, `PYTHON_SHA256`, `RUN_ID`, `NONCE32`. `RUN_ID` is derived from the output directory name and matches the client's `[A-Za-z0-9_-]{16,64}` rule. `SERVED_MODEL` is derived: the artifact path, or its hard link in `{OUT}/text-model/` when `artifact.text_only_locator` is true.
- **The case's `variables`.** A `null` value means the variable is required.
- **`HF2Q_Q5_<NAME>` environment overrides.** An override must name a declared variable.

`SERVER_PID`, `CLIENT_APPROVAL` and `CLIENT_APPROVAL_SHA256` exist only after launch and are allowed only in `client`. More rules:
- `argv[0]` must be absolute.
- `env(1)` and shell `-c` strings are refused. A shell is accepted only when its first argument is an absolute script path; the family launchers are run as `/bin/bash <repo>/scripts/serve_<family>_opencode.sh`.
- `client.approval.set` overwrites keys of the operator's approval template after launch. A value that is exactly one placeholder keeps its type, so `server_pid` stays an integer. `{"sha256_of": [paths]}` becomes a `{basename: sha256}` map.

The example cases follow the existing runners:
- They use the canonical launchers with explicit text-only, four-slot settings.
- They use the checked-in four-agent client `qwen_n4_live_smoke.py`, with `--family qwen`, `gemma4` or `deepseek4`.
- Ports are 18238 (Qwen3.8), 18236 (Qwen3.6), 18244 (Gemma4) and 18240 (DeepSeek-V4).
- **Text-only Qwen3.8 and Gemma4.** No launcher variable disables the projector:
  - `QWEN38_VISION=off`, `VISION_MODE=off` and a nonexistent Gemma `MMPROJ` only omit `--mmproj`.
  - `hf2q serve` then loads a sibling `mmproj*`/`*-mmproj*.gguf` automatically (`src/serve/mod.rs`, automatic local-path projector resolution). Both 2026-10-30 calibrations loaded one (884.6 MiB and 1.1 GiB).
  - These cases therefore set `artifact.text_only_locator: true` and `MODEL={SERVED_MODEL}`. They also add a readiness check that `/hf2q/v1/runtime` `measurement_snapshot.active_controls.vision_projector` is `false`, which Qwen3.6 has as well.
  - The hard link needs the receipt directory on the model's volume. It is the same inode, which is verified.
- **DeepSeek-V4** passes `--ctx 262144` and an 8 GiB shared KV budget. The Qwen and Gemma cases use a 48 GiB shared KV budget.
  - `q5_cases/deepseek4.json` is the case of record for the 2026-10-30 calibration, with the same workload id, argv and env.
  - The one change: the blanket `HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE=1` is replaced by the scoped `MAX_COMPRESSOR_USED_GIB=16`. The host's idle compressor holds 9.3–10.0 GiB, above the launcher's 8 GiB default; the launcher's swap and other-process refusals stay active. The justification is in the case notes.
  - It covers the canonical C262144 coding workload only.

Default `HF2Q_BIN` and `MODEL` values are repo-relative placeholders; override them with absolute paths. Required inputs:
- `CARGO`, `TOOLCHAIN`, `SANDBOX_PROOF`, `SANDBOX_PROOF_SHA256`: the client sandbox inputs used by the existing runners.
- `CLIENT_APPROVAL_TEMPLATE`: the family's previous `client-approval.json`, with the same schema, profile, family, `family_controls` and `agent_template`.

The supervisor fills in `server_pid`, `base_url`, `model`, `run_id`, `barrier_token`, `python_sha256` and the fourteen client `sources` digests of the current checkout.

### Peer arm cases (Q3/Q4)

`q5_cases/peer-deepseek4.json`, `peer-gemma4.json`, `peer-qwen38.json` and `peer-qwen36.json` launch the peer server directly as the four-slot benchmark arm. They use the same GGUF and model id as the hf2q case, and run the client with `--engine-kind peer`. `peer-qwen36.json` reproduces the e86 Qwen3.6 peer run's argv exactly, except for the port.
- The server argv is the Qwen3.6 peer run's argv (`peer-qwen36-e86-none-normal-n4-20261027T185853Z/server-launch-command.json`): 4 slots, 262144 tokens per slot, flash attention on, f16 K/V, no context shift, greedy sampling, no speculation, and the reviewed family template.
- `PEER_BIN` defaults to the e86f5fd1 build. Readiness requires `/props` `build_info == PEER_BUILD_INFO`.
- `family_peer.py` holds the DeepSeek-V4 and Gemma 4 adapters. It translates the canonical request into native peer fields and refuses every other field. It also checks `/props` template sha256, sampling and text-only settings. `test_q5_calibration.py` ties each case's flags to the adapter's `CONTROLS`.
- **DeepSeek-V4 omits `--kv-unified`.** The peer's DSV4 cache always keeps one stream per sequence. Unified mode would size each stream to the whole 1048576-token pool, about 4× the KV memory (see the case notes).
- `CLIENT_APPROVAL_TEMPLATE` must carry `engine_kind: "peer"`. For DeepSeek-V4 and Gemma 4 it must also carry `family_controls` equal to `family_peer.CONTROLS[family]` and the peer preflight keys from `family_peer.preflight`.
- Run each peer case as its own calibration (no `--admission-receipt`), because the binary differs from the hf2q calibration.

```sh
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/peer-deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/peer-deepseek4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-peer-deepseek4-$(date -u +%Y%m%dT%H%M%SZ)"
```

`q5_cases/peer-deepseek4-ds4file.json` (ADR-063 Q4-DS4, #661) is the DeepSeek-V4 peer arm on ds4's Flash 0731 file (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, sha256 `ca22ae2f…`) instead of the hf2q artifact, which runs the peer out of device memory. Everything else equals `peer-deepseek4.json`, including the reviewed hf2q template (the file's embedded template has no `reasoning_effort` tiers and the client refuses it), except the port (18242) and three variables for the best-valid-configuration sweep: `LOAD_MODE` (`auto`, i.e. mmap, or `none`), `UBATCH` (512, 1024, 2048) and `PEER_LOG_VERBOSITY` (4, which adds load-time buffer sizes and per-task prompt-cache/checkpoint lines; never per token). Override them with `HF2Q_Q5_<NAME>`. The file is a benchmark comparison artifact, not an hf2q conversion; any comparison against hf2q on its own artifact discloses the artifact difference.

### Gemma 4 peer configuration search and stage profile (ADR-063 Q4-GEMMA)

The method, arm set and commands are in `docs/research/adr063-q4-gemma-profile-method.md`.
- `gemma_peer_spec_arms.py BASE OUT [--drafters drafters.json]` writes one case per speculation type the pinned peer can run on Gemma 4 26B-A4B. Each is `peer-gemma4.json` with only `--spec-type none` replaced. Draft arms bind `DRAFT_MODEL`/`DRAFT_SHA256`, and the runner re-hashes the drafter before launch.
- `gemma_stage_profile.py` reads settled run directories only.
  - `profile` and `compare ROOT --arm NAME=GLOB --base NAME=GLOB` give per-turn and per-run prefill, decode, residual, client and tail times, from the engines' own per-request timing and the client's monotonic clock. `compare` adds deltas with block-paired bootstrap CIs, a warm-prefill cost fit and SSE-gap attribution.
  - `mst` aligns a Metal System Trace export to the request spans.
  - `catsplit` parses hf2q's `[CATSPLIT]` / `[PREFILL_CATSPLIT_JSON]` worker-exit dumps.
- `gemma_mst_capture.py --output DIR -- <q5 command>` runs one calibration under an all-process Metal System Trace. It keeps only credential-scanned GPU-schema XML exports.

### Native-engine registry

`native_peers.py` maps each native-engine kind (`coding_family.NATIVE_KINDS`: `ds4`, `tensorfold`, `omlx`) to its arm module `native_peer_<kind>.py`; no consumer branches on a kind's name. `native_peers.INTERFACE` lists what an arm declares: family and slot scope (`FAMILIES`, the harness families its client runs use, and `Q3_FAMILIES`, its case families; an arm may serve several), request translation and its inverse (`request_controls`, `canonical`), closed reply envelopes (`check_unary`, `check_frame`, `usage`), `readiness`, the optional coordinator `closing` hook (run once every agent finished, outside the makespan), `campaign_identity`, `launch_problems`, `label`/`q3_role`, F2 `BASELINES`, the post-run `AUDIT` module, `PROCESSES` for the competing-runtime inventories (`q5_preflight.py`, `q2_capture_preflight.py`), the verdict reasons `SAME_GGUF_UNAVAILABLE`/`NATIVE_COMPARISON` (`None` where the arm cannot run that way) and the Q3 `PEER_CASE_REFUSAL` text. `quality_campaign.py` accepts `--reference-kind <kind>-native`. A native client run freezes the fourteen client sources plus the arm's `CLIENT_MODULES` (the registry, the arm's client modules and its test); hf2q and same-GGUF peer runs never import the registry.

### Native ds4 arm (DeepSeek-V4, operator decision 2026-10-31)

`q5_cases/native-ds4-deepseek4.json` launches antirez ds4 (`/opt/ds4` at `c97964b8`) as a separately labeled native-engine arm: `ds4 c97964b (native Flash-0731 q2 artifact)`. ds4 serves its own `ds4f-q2` Flash 0731 GGUF (`download_model.sh:16`, about 81 GiB), not the hf2q GGUF; it rejects the hf2q artifact's Q3_K routed experts (`ds4.c:4994`). It is therefore never the same-GGUF peer, and its comparison is task makespan only. The client runs with `--engine-kind ds4` and `native_peer_ds4.py` is the adapter.
- **Request.** The canonical DeepSeek-V4 request becomes `think: true`, `reasoning_effort: "max"`, `temperature: 0`, `top_p: 1.0`, `min_p: 0.0`, `top_k: 0` and `max_completion_tokens: 8192`. Sampling is sent explicitly, because ds4 defaults to temperature 1 and min_p 0.05, and explicit values win with thinking on (`ds4_server.c:13966-13975`; temperature 0 is argmax, `ds4.c:45544`). ds4 has no repetition penalty, which matches hf2q's DeepSeek 1.0. `logprobs` and `parallel_tool_calls` are not ds4 fields (ds4 skips unknown keys), so they are not sent. The reply parser enforces one tool call per response. Every other field refuses.
- **Think Max control.** ds4 renders Think Max only when `--ctx` ≥ 393216; below that it silently uses normal thinking (`ds4.c:424-427, 61096-61100`). `--ctx` is per session. `NATIVE_DS4_CTX` therefore defaults to `393216`, and the adapter refuses lower values. The workload is still 262144 per slot: a turn whose prompt plus 8192 exceeds 262144 refuses on its usage. Four 393216-token sessions cost more KV than 262144 would. The startup line `4 resident sessions request at least X GiB … or Y GiB when the model shares one prefill workspace` is parsed into the client summary (`native_readiness.startup`).
- **Prompt.** ds4 renders DSML and thinking in C, and its prompt differs from the hf2q 0731 template in two ways:
  - it renders the tool schemas before the client system text (`ds4_server.c:3114-3133`), while the template appends them after it (`deepseek-v4-flash-0731.jinja:42-46, 54-60`);
  - its Think Max prefix is the template's `high` text; the template's `max` text has no ds4 counterpart.
  The weaker `high` wording likely shortens ds4's reasoning, which favours ds4 on makespan. Template parity is `Unavailable`.
- **Response.** Unary and SSE envelopes are closed sets: `reasoning_content`, tool-call deltas, `: prefill` keepalives, and finish `stop`/`tool_calls`/`length`. `error` and any unknown field refuse. Usage always carries `prompt_tokens_details.cached_tokens` and the ds4 extension `cache_write_tokens`, recorded as `cache_observation.status: Observed`.
- **Identity and readiness.** q5 readiness is `GET /v1/models` (`object == "list"`); ds4 binds its listener only after the model and all four sessions exist. The client then checks:
  - the `ds4-server` sha256 against `DS4_BIN_SHA256`;
  - the sha256 of all 26 runtime Metal kernel sources (`--chdir {DS4_DIR}`; ds4 reads `metal/*.metal` at startup) against the c97964b pins;
  - the artifact file name, with `MODEL_SHA256` operator-provided and not hashed inline;
  - `/v1/models`: V4 Flash shape, context, `max_completion_tokens` 8192 (`--tokens 8192`);
  - the startup log in `{OUT}/server.stderr`: context, backend `metal`, 4 resident sessions, trace path, listener; and no MTP or disabled DSML replay lines.
- **Trace proof.** `--trace {OUT}/ds4-trace.txt` records each request's raw HTTP body, think mode, sampling, `max_tokens`, token counts and rendered prompt. `native_peer_ds4_audit.py` (called by `q2_run_audit`) refuses a Pass unless:
  - the sha256 multiset of the traced bodies equals the client receipts' `request_sha256`;
  - every request ran `think_mode: max`, greedy sampling and `max_tokens: 8192`, with a rendered prompt starting with BOS plus the ds4 Think Max prefix;
  - each request's `effective_prompt_tokens`/`cached_tokens` equal its replayed usage.
  Trace writes are unbuffered per token piece, a small overhead on this arm.
- **Server-side tool-call recovery.** ds4 can repair an unterminated tool call, or append a model-visible tool error and decode again inside the same request, so the client sees only the second attempt (`ds4_server.c:12627-12639, 14313-14385, 14448-14520`; the hidden tokens count in `completion_tokens`, 14562). The audit counts these trace events per run (`native_trace.recovery_events`). The trace carries no request id, so the count cannot be attributed to a request. The run stays valid, but any non-zero count withholds the Q3 makespan statement.
- **Launch parity.** `--batched-session 4`; no `--mtp`/`--dspark`; no `--kv-disk-dir`, so the disk KV cache is off, like hf2q's DeepSeek `disk_persistence=false` (the `--kv-cache-*` knobs only tune that cache); `--power 100`; `--metal`.
- **Inputs.** `DS4_BIN_SHA256` (the root's c97964b build) and `MODEL_SHA256` are required. `download_model.sh` pins no checksum for this file. `CLIENT_APPROVAL_TEMPLATE` must carry `engine_kind: "ds4"`, `family_controls` equal to `native_peer_ds4.CONTROLS` (`native_ctx_per_session` equal to `NATIVE_DS4_CTX`) and the preflight keys from `native_peer_ds4.preflight`. The supervisor adds `native_ds4` (binary, digests, commit, context, port, log and trace paths) and the seventeen client `sources` digests (the fourteen plus `native_peers.py`, `native_peer_ds4.py` and its test).
- **Q3/Q1.** In `q3_campaign.py` the case is a `--peer-case` together with the same-GGUF `--peer-unavailable-evidence`. The plan labels the arm, and the same-model rule exempts it. Launch refuses a native artifact whose file name is not the ds4f-q2 file, or whose sha256 equals the hf2q artifact's.
  - The family verdict stays `Unavailable` (no same-GGUF peer).
  - `verdict.json` `native_peer` reports "finishes the same coding task X% sooner", with token volumes, output tokens/s, the TTFT ratio and per-run recovery counts. The statement is withheld when ds4 is unavailable, when either arm failed, or when any ds4 run shows tool-call recovery.
  - F2 rows use baseline `ds4_native`. They never count as wins: the scoreboard renders them in their own "Native-engine comparisons" section, with the statement and the companion figures. A campaign with a native peer arm emits only these native rows (plus control failure rows), never an experiment-vs-stock row: that effect is of record only in the same-GGUF campaign.
  - In `quality_campaign.py` it is `--reference-kind ds4-native`.
  - Ambient `HF2Q_Q5_MODEL`/`MODEL_SHA256`/`CLIENT_APPROVAL_TEMPLATE` would also reach this case, so pass its values with `--peer-var`/`--reference-var`.
  - ds4 cohorts are not Q2-certified: `q2_certify.py` refuses every native engine kind and `peer` (only hf2q writes the server phase trace). Q3/Q1 audit them with `q2_run_audit.py`.

```sh
HF2Q_Q5_DS4_BIN_SHA256=<sha256 of /opt/ds4/ds4-server> HF2Q_Q5_MODEL_SHA256=<sha256 of the 0731 q2 GGUF> \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/native-ds4-deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/native-ds4-deepseek4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-native-ds4-deepseek4-$(date -u +%Y%m%dT%H%M%SZ)"
```

### Native TensorFold arm (Qwen3.8, operator decision 2026-10-31)

Two cases launch TensorFold (`/opt/TensorFold` at `1157ea5b`, version 1.2.0) on its own MLX 4-bit checkpoint `TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538` as separately labeled native-engine arms. They are never the same-GGUF peer, and they compare by task makespan only. The client runs with `--engine-kind tensorfold`; `native_peer_tensorfold.py` translates requests and checks replies, and `native_peer_tensorfold_ready.py` checks identity and readiness. Source citations are in both module docstrings and the case notes.

| Case | Label | F2 baseline | ADR-063 D2 arm | Port |
|---|---|---|---|---|
| `q5_cases/native-tensorfold-qwen38.json` | `TensorFold 1157ea5 (native Qwen3.8-27B MLX 4-bit, no drafter)` | `tensorfold_native` | T0 | 18252 |
| `q5_cases/native-tensorfold-qwen38-dflash2.json` | `TensorFold 1157ea5 + DFlash2 (native Qwen3.8-27B MLX 4-bit)` | `tensorfold_dflash2_native` | T2 | 18253 |

- **Launch.** `{TF_BIN} serve {TARGET_DIR} --name qwen3.8-27b --host 127.0.0.1 --port {PORT} --context 262144 --parallel 4 --temperature 0 --max-tokens 8192 --keep-warm 900 --no-update-check`. The DFlash2 case appends `--drafter {DRAFTER_DIR} --drafter-bits 4` and nothing else. The no-drafter case omits `--drafter`; Qwen3.8-27B then "decodes plain" (`README.md:294`). The serve environment is `HF_HUB_OFFLINE=1`, `TENSORFOLD_NO_UPDATE_CHECK=1` and `TENSORFOLD_NO_LIVE=1` (TensorFold's own benchmark child environment, `server/benchmark_server.zig:50-56`), plus `TF_CHECKPOINT_REVISION={TARGET_REVISION}`. Snapshot directories are served directly, so no hub lookup or network access happens.
- **Request.** The canonical Qwen3.8 request (`build_request`) becomes: `temperature: 0`, `top_p: 1.0`, `top_k: 0`, `min_p: 0.0`, `reasoning_effort: "xhigh"`, `chat_template_kwargs: {"enable_thinking": true}`, and `thinking_budget` 2048 on a fresh user turn or 512 on a tool-result continuation (hf2q's Qwen ceilings, reused from `qwen_n4_peer.request_controls`). `logprobs: false` and `max_completion_tokens: 8192` are kept, and `hf2q_overflow_policy` is dropped: TensorFold refuses a prompt that leaves no room for 8192 output tokens (`chat.zig:146-153`). TensorFold ignores unknown fields, so the adapter enforces the closed envelope itself: any other field refuses, and `canonical` refuses native-field drift before the request is sent.
- **Thinking and effort parity.** hf2q renders its Qwen3.8 template with only `preserve_thinking: true` (`handlers.rs:1733-1737`). That template defaults `reasoning_effort` to `xhigh` and injects its xhigh instruction (`rendered-inputs-reviewed/qwen38-*.prompt.txt`). TensorFold's `chat_template.jinja` is the same bytes (sha256 `c3cf9e34…`, git blob `c0c686f9`), so sending `xhigh` explicitly renders the same instruction, and does not depend on a server default. TensorFold never forwards `preserve_thinking` to the template, but the template's undefined default keeps earlier reasoning, the same as hf2q's `true`.
- **`parallel_tool_calls` is not sent.** With `false`, TensorFold keeps the first call and silently drops any others (`openai.zig:333-350`). When the field is omitted, every call is returned, and the client's exactly-one-call rule refuses a second call, so no call is dropped silently.
- **Known differences (recorded, `Unavailable` where they cannot be proven):**
  - TensorFold has no repetition penalty; hf2q's Qwen arm uses 1.05.
  - At the thinking budget, TensorFold forces `"\n</think>\n\n"`; hf2q forces `"\nI need to answer now.</think>"`.
  - TensorFold runs no tool grammar for `strict: false` tools; hf2q charges a lazy grammar.
  - Tool schemas are sent as hf2q renders them: without `strict` and with every object key sorted (`rendered-inputs-reviewed/qwen38-cold.prompt.txt:9-11`); `canonical` restores `strict: false`. Earlier tool calls' arguments cannot be matched: hf2q renders the parsed arguments with sorted keys (`write_file` shows `content` before `path`, `qwen38-tool_round_3.prompt.txt:79-82`), while TensorFold keeps the order the model wrote (`path` first). Rewriting the history's argument strings would break the client transcript lineage, so this difference stays and shows in the token oracle.
  - When an end token arrives inside a tool call, TensorFold closes the call itself and logs nothing (`chat.zig:313-317`). hf2q's grammar prevents that ending.
  - The target's `tokenizer.json` (20 MB) is not the base `Qwen/Qwen3.8-27B` one (12.8 MB) that hf2q's GGUF carries. Per-turn TensorFold token ids are archived for the post-run oracle (below).
- **Replies.** Unary and SSE envelopes are closed sets. They carry `reasoning_content`; the extras `exact_mode`, `tensorfold`, `speculative` and an optional `stop_sequence`; streamed tool-call heads (`call_<24 hex>`) and argument pieces; compact JSON arguments; `: keepalive` comments; and a usage-only `choices: []` chunk (`include_usage`). Usage must carry `cached_tokens` (recorded as `cache_observation.status: Observed`) and `reasoning_tokens`. Finish `cancelled`, `content_filter` and anything unknown refuse. Every reply's runtime extras must say thinking on, effort `xhigh`, engine `qwen27` and sampling `greedy`.
- **Identity and readiness.** q5 readiness is `GET /health` `status == "ok"`; the listener opens only after the engine loaded. The coordinator then checks:
  - `TF_BIN` sha256 against `TF_BIN_SHA256`, with the commit pinned to `1157ea5b` and the version to `1.2.0`. The binary embeds only its version (`build.zig:74-79`), so the commit is operator-declared.
  - the local snapshots: `TARGET_DIR` and `DRAFTER_DIR` must be `<hub>/models--OWNER--NAME/snapshots/<pinned revision>`; `config.json` sha256 must equal `TARGET_CONFIG_SHA256`; `chat_template.jinja` must be the hf2q template bytes; the shards must match the index. Every file's size and cache blob name (the LFS sha256) is recorded; shard bytes are not re-hashed.
  - the server argv (`ps -ww -o args=`), which must equal the approved launch exactly. That rules out prompt-cache, drafter, thinking, loop-guard, learn/recall and KV-precision overrides.
  - `/health`: model `qwen3.8-27b`, `max_batch_size` 4, version 1.2.0, backend `metal`, not warming, with its memory recorded.
  - `/v1/models`: `context_length` and `max_model_len` 262144, thinking on by default, no server effort, and the template's levels `xhigh, medium, low, none`.
  - `/v1/benchmark/manifest`: `model.revision` equal to the pinned target revision, and `config_sha256`/`tokenizer_sha256` equal to the client's recomputation from the local snapshot (`benchmark_model.zig:14,160-189`; this covers `chat_template.jinja`).
  - the startup log (`{OUT}/server.stdout` and `server.stderr`): the serving line (name, listener, `sampling: greedy`, `context: 262144`, and drafts that name `DFlash2 ({DRAFTER_DIR})` exactly when the drafter is on); `Qwen3.8-27B: 4 streams of 262144 tokens`, with `, DFlash2 drafts for one of them` exactly when a drafter loaded (the wording is TensorFold's; every stream drafts); the prompt rows and startup reserve; the loop policy line, `lone loop=plain …` exactly when no drafter loaded (`qwen27_window.zig:86-92`), with its `lane copies` state recorded; and the prompt-cache budget. No idle-unload, Cortheon, over-cap or unsupported-guard line may appear.
- **Rendered-prompt proof.** Once every agent has finished (coordinator `closing`, after the last task-completion clock, so it never overlaps the makespan), the coordinator posts each turn's archived request (messages, tools, `tool_choice`, `reasoning_effort`, `chat_template_kwargs`) to `/tokenize`. It archives `agent-N/turn-N/tensorfold-tokenize.raw` and requires `count == usage.prompt_tokens` (`summary.native_closing.render`). Parity of these ids with hf2q's token ids is `PendingPostOracle`.
- **Post-run audit** (`native_peer_tensorfold_audit.audit_run`, called by `q2_run_audit`). Each client response is bound by its id to TensorFold's own `[tensorfold] done` line (`chat.zig:582-590`). That line's prompt, cached and completion tokens, `thinking=True effort=xhigh`, token sha, finish and `accepted/drafted` counters must equal the replayed reply, and a `done` line no client response claims refuses (a chat request the client did not send). The `/tokenize` archives must equal their proofs. The DFlash2 arm must draft at least once; the no-drafter arm must draft nothing while its policy line says `lane copies=off` (with copies on, copy drafts are recorded). The audit records draft totals, accepted per drafted, tokens per round, and `a slot needs … the request waits` lines (slots that were not resident).
- **Prompt cache and keep-warm.** hf2q reuses prefix KV across turns, so TensorFold's prompt cache stays at its default automatic budget (no `--prompt-cache-gib`), and slot-0 resume stays on. The budget is recorded from the startup log; at 4 × 262144 it is likely 0 GiB. `--keep-warm 900` is the default made explicit.
- **Memory.** Four bf16 262144-token slots take about 64 GiB, on top of about 15 GiB of weights, attention scratch and an 8 GiB margin: about 96–106 GiB against Metal's `maxWorkingSet`. `--parallel 4` is fixed, so a misfit refuses at startup (`--parallel 4 needs X GiB … lower --parallel or --context`) and the q5 run ends with the server exited. Raising `iogpu.wired_limit_mb` is the root's decision, recorded with the run.
- **Inputs.** Pull the pinned snapshots first: `tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`, and for T2 `tensorfold pull z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Build `zig build native -Dcpu=apple_m1` (Zig 0.17.0, full Xcode with the Metal toolchain); `TF_BIN` defaults to `/opt/TensorFold/zig-out/native/bin/tensorfold-native`. `TF_BIN_SHA256` and `TARGET_CONFIG_SHA256` (`shasum -a 256 $TARGET_DIR/config.json`) are required. `CLIENT_APPROVAL_TEMPLATE` must carry `engine_kind: "tensorfold"`, family `qwen`, `family_controls` equal to `native_peer_tensorfold.CONTROLS["none"]` or `["dflash2"]` (matching the case), and the preflight keys from `native_peer_tensorfold.preflight`. The supervisor adds `native_tensorfold` (binary, digests, commit, version, snapshot directories and revisions, drafter state, port, log paths) and eighteen `sources` digests: the fourteen, plus `native_peers.py`, `native_peer_tensorfold.py`, `native_peer_tensorfold_ready.py` and `test_native_peer_tensorfold.py`.
- **Q3.** Qwen3.8's same-GGUF peer is runnable, so a TensorFold `--peer-case` runs only in a separate `--native-comparison` campaign. TensorFold's `SAME_GGUF_UNAVAILABLE` is `None`, so `--peer-unavailable-evidence` refuses. In that campaign:
  - `verdict.json` sets `verdict_of_record: false` and `campaign_label: "native-comparison campaign; not the verdict of record"`, and the family verdict is `Unavailable` with that reason. The verdict of record stays the stock/experiment/same-GGUF-peer campaign.
  - `native_peer` reports the task-makespan statement, token volumes, output tokens/s, the TTFT ratio, and per-run acceptance and slot waits. The statement is withheld when TensorFold is unavailable, when either arm failed, or when a TensorFold run lacks its audit.
  - The F2 group is `qwen38-q3-<baseline>`. The campaign emits no experiment-vs-stock row (that effect belongs to the campaign of record); its rows carry the not-of-record label in variant and note. The TensorFold row uses baseline `tensorfold_native` or `tensorfold_dflash2_native` and renders under "Native-engine comparisons" only, never as a win.
  - The stock arm stays mandatory: an experiment-plus-peer-only campaign would need new plan and verdict logic and is not built.
- **ADR-063 D2** (DFlash2 at four slots vs no drafter, #365). Run T0 and T2 as calibrations with the same inputs, then print each run's D2 record:
  `python3 scripts/four_agent/native_peer_tensorfold_audit.py "$Q5_ROOT/q5-native-tensorfold-t0-…" "$Q5_ROOT/q5-native-tensorfold-t2-…"`.
  Compare them in this order:
  1. four-slot task completion (`tasks_passed`, run audit `Pass`);
  2. task quality (`final_source_sha256` into the Q1/patch review);
  3. memory: the Q5 verdict, free-plus-cached floor and swap growth, then TensorFold's startup reserve, prompt-cache budget, `/health` memory at readiness and after the run, and slot waits;
  4. net makespan, with accepted per drafted and tokens per round (rounds are the verify cost) as the explanation, never the criterion.

  Drafted output must equal the plain decode on this engine (`README.md:203-207`), so final sources may legitimately match.

  For D2's long-context requirement, run the same comparison on the [long-context workload](#long-context-workload-adr-063-d2-365), with T0 and T2 cases derived by `coding_long_context.py derive-case`. The D2 command refuses runs whose `workload.variant` differs.

```sh
TF=/opt/TensorFold/zig-out/native/bin/tensorfold-native
HUB=$HOME/.cache/huggingface/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c
HF2Q_Q5_TF_BIN_SHA256=$(shasum -a 256 "$TF" | cut -d' ' -f1) \
HF2Q_Q5_TARGET_CONFIG_SHA256=$(shasum -a 256 "$HUB/config.json" | cut -d' ' -f1) \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/native-tensorfold-qwen38-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/native-tensorfold-qwen38.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-native-tensorfold-t0-$(date -u +%Y%m%dT%H%M%SZ)"
# T2: the same with q5_cases/native-tensorfold-qwen38-dflash2.json, a dflash2 approval template and a -t2- output.
```

### Native oMLX arm (Qwen3.8, Qwen3.6, Gemma 4; operator decision 2026-10-31)

Three cases launch oMLX (`/opt/omlx` at `6b413c90`, version 0.7.3rc1, an MLX/mlx-lm continuous-batching server) on mlx-community 4-bit checkpoints as separately labeled native-engine arms. They are never the same-GGUF peer, and they compare by task makespan only. The client runs with `--engine-kind omlx`. `native_peer_omlx.py` translates requests and checks replies, `native_peer_omlx_ready.py` generates the launch files and checks identity and readiness, `native_peer_omlx_launch.py` is the q5 serve entry, and `native_peer_omlx_audit.py` is the post-run audit. Source citations are in the module docstrings and the case notes. DeepSeek-V4 is out of scope: no 4-bit conversion fits.

| Case | Harness family | Pinned snapshot | Model id | Label | F2 baseline | Port |
|---|---|---|---|---|---|---|
| `q5_cases/native-omlx-qwen38.json` | `qwen` | `mlx-community/Qwen3.8-27B-4bit@10c35caa` | `Qwen3.8-27B-4bit` | `oMLX 6b413c9 (native Qwen3.8-27B MLX 4-bit)` | `omlx_qwen38_native` | 18254 |
| `q5_cases/native-omlx-qwen36.json` | `qwen` | `mlx-community/Qwen3.6-35B-A3B-4bit@38740b84` | `Qwen3.6-35B-A3B-4bit` | `oMLX 6b413c9 (native Qwen3.6-35B-A3B MLX 4-bit)` | `omlx_qwen36_native` | 18255 |
| `q5_cases/native-omlx-gemma4.json` | `gemma4` | `mlx-community/gemma-4-26b-a4b-it-4bit@0d77464e` | `g4-26b-a4b-it-4bit` | `oMLX 6b413c9 (native Gemma 4 26B-A4B MLX 4-bit)` | `omlx_gemma4_native` | 18256 |

- **Launch.** The serve argv runs `native_peer_omlx_launch.py` with the case variables. It checks the inputs and the identity (below), then:
  - writes `{OUT}/omlx/base/model_settings.json`, generated by `native_peer_omlx_ready.model_settings`;
  - writes `{OUT}/omlx/base/settings.json` with `cache.hot_cache_only: true`;
  - links `{OUT}/omlx/models/<model id>` to `MODEL_DIR`, so the directory name is the served model id (`model_discovery.py:1972-2096`);
  - records `{OUT}/omlx/launch.json` (pid, argv, environment, identity);
  - execs `{OMLX_PY} -I -u -m omlx.cli serve --model-dir {OUT}/omlx/models --base-path {OUT}/omlx/base --host 127.0.0.1 --port {PORT} --headless --no-hf-cache --max-concurrent-requests 4 --paged-ssd-cache-dir {OUT}/omlx/ssd-cache --hot-cache-max-size 16GB --sse-keepalive-mode off --memory-guard aggressive --log-level info` in the same process, so q5's `SERVER_PID` is the server.

  `--no-hf-cache` keeps HF-cache models out of discovery. `--max-concurrent-requests 4` sets `max_num_seqs` and the completion batch (`settings.py:1971-1973`). `--sse-keepalive-mode off` removes the default `{"role":"assistant","content":""}` keepalive chunks. `-I` ignores `PYTHON*` variables, and `-u` keeps the startup prints in `server.stdout`. The serve environment is `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`. Any `OMLX_*`, `PYTHON*` or `MLX_*` variable refuses, because oMLX lets the environment override settings (`settings.py:1270-1340`).
- **Model settings.** oMLX takes sampling, context and thinking per model from `model_settings.json`. It drops unknown keys silently and drops a whole record that fails validation (`model_settings.py:631-648, 853-856`), so the file is generated from known fields only. It sets:
  - `max_context_window` 262144 (oMLX checks it against prompt tokens only, `server.py:2436-2452`) and `max_tokens` 8192;
  - greedy sampling (temperature 0, top_p 1, top_k 0, min_p 0), `repetition_penalty` 1.05, presence 0, `force_sampling: false`;
  - Qwen: `enable_thinking`/`preserve_thinking` true; Gemma: `enable_thinking` false;
  - `model_type_override: "llm"` (text engine);
  - MTP, DFlash, VLM-MTP, SpecPrefill, TurboQuant KV, A8 and ANE prefill, expert offload and fp16 activations all off;
  - pinned and default.

  A model with saved settings gets no initial or Lightning-MTP defaults (`engine_pool.py:1624-1660`). The approval binds the file's sha256 (`native_omlx_settings_sha256`), and it must be unchanged at closing.
- **Request.** The canonical request (`build_request`) gains `temperature: 0`, `top_p: 1.0`, `top_k: 0`, `min_p: 0.0`, `presence_penalty: 0.0`, `frequency_penalty: 0.0`, `repetition_penalty: 1.05` and a per-family `repetition_context_size`: 64 for Qwen (hf2q seeds a 64-token window with the prompt tail, `engine_qwen35.rs:4508-4519`) and 8192 for Gemma (hf2q penalizes every token generated in the request, with no window, `engine.rs:7785-7788`, `sampler_pure.rs:557-574`); mlx-lm's default window is 20 (`server.py:5198-5203`). Whether oMLX seeds that history with prompt tokens is not established from source and is recorded as a parity risk. The request values win over the model settings and over oMLX's global defaults of temperature 1.0 and top_p 0.95.
  - Qwen3.8 and Qwen3.6 add `chat_template_kwargs: {"enable_thinking": true, "preserve_thinking": true}` and `thinking_budget` 2048 on a fresh user turn or 512 on a tool-result continuation (hf2q's ceilings, from `qwen_n4_peer.request_controls`). No `reasoning_effort` is sent: hf2q passes none, so the template's default applies on both engines.
  - Gemma 4 adds `chat_template_kwargs: {"enable_thinking": false}` and no budget.
  - `logprobs: false` and `max_completion_tokens: 8192` are kept, and `hf2q_overflow_policy` is dropped. oMLX has no request field for it; the client refuses a turn whose usage leaves no room for 8192 output tokens.
  - pydantic drops unknown request fields silently, so the adapter enforces the closed envelope itself. Any other field refuses, and `canonical` refuses native-field drift before the request is sent.
- **`parallel_tool_calls` is not sent.** oMLX has no such field. It returns every parsed call (`server.py:5433, 7114-7142`), and the client's exactly-one-call rule refuses a second call.
- **Known differences (recorded, `Unavailable` where they cannot be proven):**
  - These are not the hf2q GGUF bits: mlx-community 4-bit affine MLX quantization versus hf2q's Q4_K_M GGUFs.
  - Tokenizer and template: the snapshot's `chat_template.jinja` sha256 is an operator input (`TEMPLATE_SHA256`). Readiness records whether it equals hf2q's reviewed template (`identical_to_hf2q`). The tokenizer is the mlx-community one, not the GGUF-embedded one.
  - Tool schemas: oMLX drops `strict`, sorts keys and injects `"description": ""` into every property schema (`api/tool_calling.py:48-97, 3741-3801`), so no tool grammar runs; hf2q charges a lazy grammar. The Gemma model id avoids the substring `gemma`, which would otherwise rename `description` parameters and add parameter labels (`server.py:4961-4962`).
  - Tool messages keep only `tool_call_id` and `content` (`api/utils.py:1066-1088`), and unary `content` is stripped (`server.py:5445`).
  - At the thinking budget, oMLX forces its close-think sequence (`api/thinking.py:481-520`); hf2q forces `"\nI need to answer now.</think>"`.
  - Whether mlx-lm's BatchGenerator seeds the repetition window with the prompt tail, as hf2q does, is not established from source.
  - Batched argmax at temperature 0 can depend on batch composition (MLX kernels are not batch-invariant) [INFERENCE].
- **Replies.** Unary and SSE envelopes are closed sets (`exclude_none` dumps), with these checks:
  - ids are `chatcmpl-<8 hex>` and call ids `call_<8 hex>`;
  - `reasoning_content` is a field of its own;
  - each tool call arrives complete in one chunk (`index`, `id`, `type`, `function{name, arguments}`), with arguments that are a JSON object;
  - the role chunk carries only the role, `finish_reason` is absent until the final empty-delta chunk, and a `choices: []` usage chunk follows;
  - usage carries `input_tokens`/`output_tokens`, `cached_tokens` (recorded as `cache_observation.status: Observed`, block-granular) and `reasoning_tokens`. The last is a re-tokenization of the cleaned reasoning, not an engine count (`server.py:5434-5438, 7243-7247`);
  - finish `error`, `abort` and anything unknown refuse;
  - an error frame, or a keepalive-led unary error body under a committed 200 (`server.py:3203-3345`), refuses.
- **Identity and readiness.** q5 readiness is `GET /health` `status == "healthy"`; it answers 503 `loading` until the pinned preload ends. The coordinator then checks:
  - `OMLX_DIR`: a detached checkout of `6b413c90` with no tracked-file changes and `omlx/_version.py` 0.7.3rc1;
  - `OMLX_PY`'s venv, from its dist-info: `mlx` 0.32.4, `mlx-lm` @`0d4d3d04`, `mlx-vlm` @`138d7617`, and the editable `omlx` of `OMLX_DIR`. The pip-freeze-equivalent inventory's sha256 is the build identity (`omlx <version> <commit> venv-freeze-sha256 <sha>`);
  - the snapshot: `MODEL_DIR` is `<hub>/models--mlx-community--<name>/snapshots/<pinned revision>`, `config.json` equals `MODEL_CONFIG_SHA256`, `chat_template.jinja` equals `TEMPLATE_SHA256`, the shards match the index, and the tokenizer files are hashed;
  - `launch.json`: its sha256 is in the approval, its pid is `SERVER_PID`, and its argv and inputs are exact. `omlx serve` retitles the process `omlx-server` (`process_title.py`), so `ps` shows the title, which must match;
  - `/api/status`: version 0.7.3rc1, every custom kernel available (otherwise the Qwen paths fall back to slow generic code), one loaded model, zero requests;
  - `/v1/models/status`: the one served model resolves to `MODEL_DIR`, engine `batched`, type `llm`, the pinned `config_model_type`, `max_context_window` 262144, `max_tokens` 8192;
  - `/health`: healthy, default model, no MCP;
  - the re-saved `settings.json`: listener, keepalive off, 4 concurrent requests, hot-only cache, guard tier and HF-cache discovery off;
  - the startup lines: version, base path, memory guard, hot cache, saved CLI arguments, `Loaded settings for 1 models`, and the `llm` override line. A dropped settings record, initial settings, a missing SSD directory or a dropped tool call refuses.
- **Rendered-prompt proof and closing.** Once every agent has finished (outside the makespan), the coordinator posts each turn's archived `model`, `messages`, `tools` and `chat_template_kwargs` to `/v1/tokenize` (`server.py:8327-8434`, the chat path's own rendering). It archives `omlx-tokenize.raw` and requires `count == usage.prompt_tokens`. `/api/status` session counters must then equal the client receipts (request count, prompt and completion tokens), so no hidden request ran, and both settings files must be unchanged.
- **Post-run audit** (`native_peer_omlx_audit.audit_run`, called by `q2_run_audit`). oMLX logs no response id, so the audit matches the multiset of client turns by (transport, prompt tokens, completion tokens, finish, `max_tokens` 8192) against the server's `Chat completion:` lines (`server.py:5337-5343, 7227-7236`). A unary tool call is logged with the engine's `stop`. A missing or extra line refuses. Tool-parser WARNING and INFO lines (`omlx.api.tool_calling`, e.g. `Native tool parser failed …, recovered via XML fallback`, `Remapped namespaced tool call name`) are counted per run; a dropped call (`Dropping …`, `Dropped …`) refuses. Two server-side repairs log nothing: promoting a call found in the reasoning (`api/tool_calling.py:2509-2556`) and recovering a Qwen call missing its closing markup (`:2347-2464`). They cannot be counted, so every audit, `native_peer` section and F2 note carries `recovery_attribution: Unavailable:…`. The `/tokenize` archives must equal their proofs. `python3 scripts/four_agent/native_peer_omlx_audit.py RUN…` prints each run's audit, task, memory and makespan record.
- **Prefix cache.** hf2q reuses prefix KV across turns, so oMLX's paged prefix cache stays on, in RAM only: `hot_cache_only` with a 16 GB hot tier and a fresh per-run directory that receives no SSD I/O (`cache/paged_ssd_cache.py:1795-1799, 1948-1949`). `cached_tokens` is block-granular (256-token blocks).
- **Memory.** Weights are about 16 GB (Qwen3.8), 20 GB (Qwen3.6) and 15–16 GB (Gemma 4, [INFERENCE]). KV is paged and grows with real tokens; at four full 262144-token slots it is about 64 GiB, 20 GiB and 20 GiB. oMLX calls `mx.set_wired_limit` only when `iogpu.wired_limit_mb` is set (`process_memory_enforcer.py:175-347`); raising it is the root's decision, recorded with the run. `/api/status` model memory is recorded at readiness and closing.
- **Inputs.** `MODEL_CONFIG_SHA256` (`shasum -a 256 $MODEL_DIR/config.json`) and `TEMPLATE_SHA256` (`shasum -a 256 $MODEL_DIR/chat_template.jinja`) are required. `CLIENT_APPROVAL_TEMPLATE` must carry `engine_kind: "omlx"`, the case's harness family, `family_controls` equal to `native_peer_omlx.CONTROLS["<q3 family>"]`, and the preflight keys from `native_peer_omlx.preflight`. The supervisor adds:
  - `native_omlx` (the launch inputs and log paths);
  - `native_omlx_settings_sha256` and `native_omlx_launch_sha256` (`sha256_of` the run's files);
  - eighteen `sources` digests: the fourteen, plus `native_peers.py`, `native_peer_omlx.py`, `native_peer_omlx_ready.py` and `test_native_peer_omlx.py`.
- **Q3.** Every oMLX family has a runnable same-GGUF peer, so an oMLX `--peer-case` runs only in a separate `--native-comparison` campaign; `SAME_GGUF_UNAVAILABLE` is `None`. The semantics are TensorFold's:
  - `verdict_of_record: false`, with the not-of-record label;
  - `native_peer` carries the task-makespan statement with token volumes, output tokens/s and the TTFT ratio. It is withheld when oMLX is unavailable, when either arm failed, when a run lacks its audit, or when any run logged tool-parser WARNING/INFO lines; the silent repairs above are disclosed with the statement;
  - F2 group `<family>-q3-omlx-<family>-native`, rendered only under "Native-engine comparisons", never as a win.
- **Qwen3.6 re-test with a different tokenizer and sampler stack.** `quality_campaign.py --subject-case q5_cases/qwen36.json --reference-kind omlx-native --reference-case q5_cases/native-omlx-qwen36.json` runs oMLX as the separately labeled reference arm. oMLX cannot be the subject: `quality_campaign.py` requires an hf2q subject case.
- **Install (root, once; keeps `/opt/omlx` untouched).** `make install` writes `.so`/`.metallib` files into the source tree, so build in a separate clone:

```sh
xcodebuild -downloadComponent MetalToolchain              # once; full Xcode (omlx README.md:83-100)
D=$HOME/.local/share/hf2q/peers/omlx-6b413c9
git clone /opt/omlx "$D" && git -C "$D" checkout --detach 6b413c900352db1382466cf4b7f0825f61a550a9
uv venv --seed --python 3.12 "$D/.venv"                    # --seed: the Makefile calls "$(PYTHON) -m pip"
make -C "$D" install PYTHON="$D/.venv/bin/python"          # pip install -e . then `make kernels` (fails if a kernel does not load)
"$D/.venv/bin/python" -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
"$D/.venv/bin/python" -m omlx.cli --version                # 0.7.3rc1
"$D/.venv/bin/python" -m pip freeze > "$D.freeze.txt"      # kept with the evidence; readiness recomputes the same inventory
# Snapshots into the HF cache (MODEL_DIR is <hub>/models--mlx-community--<name>/snapshots/<revision>):
hf download mlx-community/Qwen3.8-27B-4bit --revision 10c35caafbb80f7dc6a7a432cdd11af10a6d4818
hf download mlx-community/Qwen3.6-35B-A3B-4bit --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46
hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
```

```sh
HUB=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
HF2Q_Q5_MODEL_CONFIG_SHA256=$(shasum -a 256 "$HUB/config.json" | cut -d' ' -f1) \
HF2Q_Q5_TEMPLATE_SHA256=$(shasum -a 256 "$HUB/chat_template.jinja" | cut -d' ' -f1) \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/native-omlx-qwen36-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/native-omlx-qwen36.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-native-omlx-qwen36-$(date -u +%Y%m%dT%H%M%SZ)"
# Qwen3.8 and Gemma 4: the same with native-omlx-qwen38.json / native-omlx-gemma4.json and their snapshot.
```

### Splash peer of record (Qwen3.8, Qwen3.6; operator decisions 2026-11-03/11-04)

Splash (`incoai/splash`, source build `031a25a6`, engine kind `splash`) is a second **same-artifact** peer of record
for the Qwen families. It serves the hf2q GGUF itself: the same path and sha256 as the family's hf2q case. It
decodes with its family's pinned DFlash2 draft. Q3 runs it as its own measured arm `splash` beside `peer`, and for
`qwen38`/`qwen36` a Win needs ≥5% sooner than both. Design: [docs/research/adr063-splash-peer-design.md](../../docs/research/adr063-splash-peer-design.md).

| File | Role |
| --- | --- |
| `native_peer_splash.py` | Adapter: pinned build, families and drafts, configuration knobs and sweep presets, `family_controls`, closed request/response/usage envelopes |
| `native_peer_splash_ready.py` | Launch inputs, build/venv/toolchain/draft identity, per-run model assembly (`model.json` digests), argv template, readiness, closing render proof (`/apply-template` + `/tokenize`), campaign identity |
| `native_peer_splash_launch.py` | q5 serve entry: verifies, prepares `{OUT}/splash/models`, writes `launch.json`, execs the Splash server in the same process |
| `native_peer_splash_audit.py` | Post-run audit (Done/Started lines bound to client turns, render archive, no cancel/error/restart) and `run_record` |
| `splash_sweep.py` | `identity` (the build variables), `derive-case` (one sweep point, optional D6 `--max-memory` cap), `select` (validity from Splash's own runs and the best configuration), `parity-config` (the input of hf2q's `splash_prompt_parity_post_oracle`) |
| `q5_cases/splash-qwen38.json`, `splash-qwen36.json` | Q5 cases; `SPLASH_CONFIG` is the configuration (default: sweep point `s0`) |

- **Build.** Build a clean detached clone of `031a25a6` with `make -j4 && make install-environment`, only while
  `HOST-LOCK` is absent, and record the toolchain the build used (`xcodebuild -version`, `xcrun metal --version`,
  `clang --version`, `xcrun --show-sdk-version`, `sw_vers`) in `toolchain.txt`. Then
  `python3 scripts/four_agent/splash_sweep.py identity --splash-dir DIR --build-toolchain toolchain.txt` prints
  `SPLASH_BIN_SHA256`, `SPLASH_METALLIB_SHA256`, `SPLASH_HOST_STEP_SHA256`, `SPLASH_TOOLCHAIN_SHA256` and
  `SPLASH_VENV_FREEZE_SHA256`, refusing a host toolchain the build did not record. The case requires all five; it
  also requires `MODEL_SHA256`, the hf2q GGUF's.
- **Drafts.** Qwen3.8 uses `z-lab/Qwen3.8-27B-DFlash2@50307d4c`, which is byte-identical to
  `incoai/Qwen3.8-27B-DFlash2@015e7956`. Qwen3.6 uses `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b69`. Each is an HF
  cache snapshot on the internal disk.
- **Approval template.** `CLIENT_APPROVAL_TEMPLATE` must carry:
  - `engine_kind: "splash"`, family `qwen`;
  - `family_controls` equal to `native_peer_splash.controls("<q3 family>", SPLASH_CONFIG)`;
  - the preflight keys from `native_peer_splash.preflight`.

  The supervisor adds `native_splash` (the launch inputs) and the per-run `native_splash_launch_sha256`.
- **Per run.** Every start digests the GGUF again into a fresh `{OUT}/splash/models`, outside the makespan. The
  campaign identity freezes the build, toolchain, venv, draft, configuration and argv template, never the assembly
  path.
  - Splash's `model.json` must record the hf2q GGUF with `MODEL_SHA256`, the pinned draft weights, and the reviewed
    template as `tokenizer/chat_template.jinja`.
  - `{OUT}` must be on APFS (the internal disk). On the exFAT external drive, Splash's model-directory cleanup stops
    on the `._*` files macOS writes there.
  - The run must add no draft pin to the shared HF cache (`refs/splash`); local sources carry none.
  - `/status` must report the configured KV format and cache dtype and no engine restart, at readiness and closing.
- **Sweep.** After each settled sweep run, write `parity-config RUN --template FILE` (default output
  `RUN/splash-prompt-parity`) and run hf2q's `splash_prompt_parity_post_oracle` on it. `select` counts a
  configuration valid only when every run has a `PassedByteIdentical` result there. A run that stopped before the
  client is an invalid row of its own.
- **Q3.** `--splash-case` defaults to `q5_cases/splash-<family>.json` for qwen38/qwen36. The arm runs only in the
  `n4` and `n1` stages (D7). `--splash-unavailable-evidence` declares Splash unavailable; the family verdict is
  then `Unavailable`, never a Win against the peer alone. A peer-unavailable campaign requires no Splash arm (its
  verdict is Unavailable already).
  - Launch refuses a campaign whose Splash build, toolchain or venv differs from the case's declared hashes, or
    whose output is not on APFS, before any run.
  - A Splash Q5 memory stop halts the campaign, also before the client starts (load and readiness window): record
    a `--max-memory` cap (D6), re-confirm the configuration and start a new campaign.
- **Quality gate.** Run `quality_campaign.py --reference-kind splash` with a Splash reference case. The F2 tier
  requires a `splash` gate for records that require Splash.
- **Qwen3.6 workload.** The arm runs the declared Qwen3.6 workload. The repair-turn workload applies once it is
  recorded on main: add `--repair-turns 1` to the hf2q, peer and Splash Qwen3.6 cases together.

### Root commands

Run the commands from the repository root, one family at a time, with no other runtime loaded. `Q5_ROOT` is a campaign evidence directory outside Git. The supervisor refuses an output directory inside a Git tree or one that already exists.

```sh
export HF2Q_Q5_HF2Q_BIN=/abs/path/release/hf2q          # release build of the source being calibrated
export HF2Q_Q5_CARGO=/abs/cargo HF2Q_Q5_TOOLCHAIN=/abs/toolchain
export HF2Q_Q5_SANDBOX_PROOF=/abs/sandbox-proof.json HF2Q_Q5_SANDBOX_PROOF_SHA256=<sha256>

# Qwen3.8 27B
HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/qwen38.json --dry-run
HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/qwen38.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-qwen38-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 26B-A4B
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-gemma4-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 full-input checkpoint A/B: same case, kill-switch arm (alternate ABAB with the default arm)
HF2Q_Q5_GEMMA_FULL_INPUT_CHECKPOINT=0 \
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-gemma4-ckpt-off-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 joint warm prefill A/B (ADR-063 Stage 1): same case, kill-switch arm (alternate ABAB with the default arm)
HF2Q_Q5_GEMMA_JOINT_WARM_PREFILL=0 \
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-gemma4-joint-warm-off-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 tiled F16 warm attention A/B (ADR-063 Stage 2): same case, tiled opt-in arm (alternate ABAB with the default vec-hybrid arm)
HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION=1 \
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-gemma4-warm-tiled-on-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 prefill MoE route A/B (ADR-063 Q4-GEMMA-PREFILL-MOE): A arm = previous flattened down call
# (plus HF2Q_Q5_MM_ID_SCHEDULE=grid once the compact native schedule is pinned); B arm = defaults. ABBA order.
HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN=flat \
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-gemma4-prefill-moe-A-$(date -u +%Y%m%dT%H%M%SZ)"

# Qwen3.6 35B-A3B
HF2Q_Q5_MODEL=/abs/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen36-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/qwen36.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-qwen36-$(date -u +%Y%m%dT%H%M%SZ)"

# DeepSeek-V4 Flash
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/deepseek4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-deepseek4-$(date -u +%Y%m%dT%H%M%SZ)"

# Any later (admitted) run of a calibrated family: same case, binary, artifact, context and slots
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/deepseek4.json \
  --admission-receipt "$Q5_ROOT/<calibration of record>/receipt.json" \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/q5-deepseek4-admitted-$(date -u +%Y%m%dT%H%M%SZ)"
```

`NC_RECEIPT` is the accepted `NegativeControlPass` receipt (for example `$Q5_ROOT/q5-negative-control-20261030T041437Z/receipt.json`).

Run `--dry-run` before every real run; it launches nothing:
- It prints the expanded plan, `missing_variables` and the file checks (binary, artifact, launcher, client and template).
- It exits 2 unless the plan is launchable.

Optional input:
- `HF2Q_Q5_MODEL_SHA256` adds a provided digest. It must agree with the conversion receipt's digest when there is one; a contradiction refuses the run. Without a receipt, it is recorded as operator-provided and unverified.

**Model and projector identity.**
- `receipt.artifact.sha256` comes from the adjacent `<model>.gguf.receipt.json`, field `output.sha256`. It is accepted only when that receipt's `output.size` equals the file size. `sha256_source` cites the receipt path, the field and the receipt's own sha256.
- Otherwise the identity is size + mtime.
- `projectors_beside_artifact`, and for text-only runs `projectors_beside_served_model`, record the same identity for every projector the server could auto-load.
- Models are never hashed inline.

**Negative control.** One run of the bounded synthetic child:

```sh
python3 scripts/four_agent/q5_calibration.py --negative-control --dry-run
python3 scripts/four_agent/q5_calibration.py --negative-control \
  --output "$Q5_ROOT/q5-negative-control-$(date -u +%Y%m%dT%H%M%SZ)"
```

Defaults:
- The child grows by 256 MiB every 0.25 s toward an 8 GiB ceiling, touching every 16 KiB page.
- The footprint stop is set at half the ceiling (4 GiB).

Limits on the child:
- The supervisor refuses a ceiling above `baseline Free+cached − 2 GiB floor − 4 GiB safety`, above the 32 GiB hard maximum, or at or below the cap.
- The child never allocates past its ceiling, exits when its parent changes, and ends after 180 s.
- If no stop fires, it exits 4 after holding the ceiling for 10 s.

`NegativeControlPass` requires all of the following:
- The `footprint_cap` stop fired.
- The peak footprint stayed below the ceiling.
- The child did not exit 4.
- Cleanup was proven.

The receipt also records `peak_over_cap_bytes`, which is the measured overshoot, and the stop latency.

### Assumptions

- Free+cached means `(free + speculative + file-backed pages) × page size` from `vm_stat`. Pressure comes from `kern.memorystatus_vm_pressure_level`: 1 = normal, 2 = warning, 4 = critical. Only a level of 4 or more stops a run or fails a calibration; levels other than 1, 2 and 4 are recorded as `unknown` and fail the pass. Swap is the `used` field of `vm.swapusage`. The baseline swap is the highest value seen in the baseline samples.
- The hf2q footprint is the sum over the server's process group. The launchers `exec` hf2q, so the group leader becomes hf2q.
- RSS is recorded alongside, but RSS does not count the same memory as phys_footprint: earlier DeepSeek receipts show about 1.2 GB RSS for a roughly 100 GB model. When phys_footprint is unavailable, RSS stands in and is labeled as a fallback.
- `ps` arguments are split on whitespace. An executable path that contains spaces cannot be matched by name; current runtimes and launchers have no spaces in their paths.
- The example cases use the canonical N4 coding workload. Q5 requires the *largest declared* four-agent workload, which is an operator declaration. If a family's largest case is a different client (for example the earlier largest-prompt or largest-tool-result clients), copy the case and replace `client`. Recalibrate whenever the artifact, source, context or slot count changes.
- The Qwen launchers pass no `--ctx`. The readiness check therefore requires `/health` `context_length == 262144` and stops the run before the client if the GGUF default differs.
- The approval template's `agent_template`, preflight keys and `family_controls` must match the current client checkout. The dry run checks only that the template file exists.
- A blind sample, meaning failed host observation or a missing real footprint while a cap is enforced, never counts as a quiet host. Three blind samples in a row stop the run.
- Thermal state is recorded but never stops a run. It is not an out-of-memory risk.

## Long-context workload (ADR-063 D2, #365)

D2's acceptance includes long contexts, and the canonical client tops out near 2.8k prompt tokens per turn. `qwen_n4_live_smoke.py --workload long-context-v1` runs the canonical `parse_port` task with one change: `tests/ports.rs` is the canonical two integration tests followed by [`fixtures/long-context-v1.rs`](fixtures/long-context-v1.rs). That fixture is a 150 KB read-only Rust module, `mod netconf`: the configuration schema of the daemon the port parser feeds. It has 8 sections and 55 settings, each with bounds, a parser, a defaulting reader, a formatter and help text. The module is `#[allow(dead_code)]` and has no `#[test]`. Workflow step (2) reads the file, so every request from turn 2 on carries it.

- **Unchanged.** The prompts and identity line are the canonical ones, so the turn-0 and turn-1 requests are byte-identical to the canonical workload. So are the six-step workflow and its sha256, the tools, the transport matrix, `src/`, the test counts (two per target), grading, the prefix-reuse checks, the 256 KiB wire bound, and the 64 KiB bounds on writes and test output. Only `read_file` gets a 192 KiB bound (`Tools(read_limit=…)`), and only under this variant.
- **Determinism.** `coding_long_context.render()` builds the fixture from its `SCHEMA` table. `FIXTURE_SHA256` pins the committed bytes, and the client refuses drift. To change the fixture, define a new variant id; never re-pin `long-context-v1`.
- **Binding.** The coordinator approval must carry `workload_variant` (equal to `--workload`) and `workload_fixture_sha256`. Its `sources` must also cover `coding_long_context.py` and `test_coding_long_context.py`. A canonical approval that carries either key is refused. The coordinator and every agent summary record `workload_variant: {id, fixture_sha256}`. Aggregation, and therefore `q2_run_audit.py`, refuses an agent from another workload, and the D2 record prints `workload: {id, variant}`.
- **Composition.** `--workload` is N4-only and cannot be combined with `--instance` or `--benchmark-profile`. `--phase-plan` is allowed, so `q3_campaign.py` can derive its controls from a long-context case.
- **Campaigns.** `q3_campaign.py` refuses a plan unless every arm's client `--workload` equals that arm's approval `workload_variant` and all arms select the same variant (or all run the canonical files). `quality_campaign.py` refuses a case that already selects a variant. `native_peer_tensorfold_audit.py` refuses D2 runs with different `workload.variant`.
- **Scope.** The fixture covers D2's *long contexts* only. It is generated from one template per setting, so it is deliberately repetitive code: it is not evidence for D2's *nonrepetitive code* item, which needs its own workload.
- **Prompt growth.** `python3 coding_long_context.py estimate` builds the seven actual requests of one agent offline: the real file contents, a canonical fix and about 1 KiB of reasoning per reply. It reports wire bytes / 4 per turn:

  | Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|---|
  | canonical | 528 | 874 | 1,282 | 1,714 | 2,086 | 2,487 | 2,858 |
  | `long-context-v1` | 528 | 874 | 39,748 | 40,180 | 40,552 | 40,953 | 41,324 |

  The final request is 165,297 bytes, against the 262,144-byte wire bound. Served `prompt_tokens` add the chat template's own text, and Rust tokenizes at about 3–3.5 bytes per token, so served counts from turn 2 on should land around 45–50k (an estimate, not a measurement). That is far inside 262144 tokens per slot.
- **Cases.** `q5_cases/qwen38-long-context.json` and `q5_cases/peer-qwen38-long-context.json` are `derive-case` of `qwen38.json` and `peer-qwen38.json`. They keep the same server, argv and port, and add the client `--workload`, the two approval keys, the two sources and a `-long-context-v1` workload id. Because the workload id differs, Q5 calibrates them separately, and a canonical admission receipt is refused.
- **TensorFold / D2 cases.** Derive T0 and T2 the same way:

  ```sh
  for arm in native-tensorfold-qwen38 native-tensorfold-qwen38-dflash2; do
    python3 scripts/four_agent/coding_long_context.py derive-case \
      scripts/four_agent/q5_cases/$arm.json > "$Q5_ROOT/cases/$arm-long-context.json"
  done
  ```

  Then run them exactly like T0 and T2 above. Keep the derived files outside `q5_cases/`: the campaign tests require ports 18252 and 18253 to appear once each there. Other canonical N4 cases derive the same way; the estimate above is for the Qwen request shape.

## Task-quality campaign (ADR-063 Q1 §3.5)

The canonical four-agent prompts are identical on every run, so repeated cohorts are one sample per agent. The Q1 policy (`hf2q.q1.numeric-policy.v1`) therefore judges task quality over at least 24 distinct, paired task instances.

### Task instances

- `qwen_n4_live_smoke.py --instance N` gives all four agents row `N` of [`fixtures/coding-instances.json`](fixtures/coding-instances.json) (24 rows, ids 0..23). Without `--instance`, nothing changes: the canonical `parse_port` prompts, fixture files and approval are byte-identical to earlier receipts, so existing receipts and prefix oracles stay valid.
- Every row is a different parser of the same difficulty class: another function name, unsigned type, inclusive bounds, reject word and test names. Each row has two explicit bounds (`0 < min < max < T::MAX`), two unit and two integration tests, and the canonical paths, tools, six-step protocol and test counts. `coding_instances.py` renders the prompt, workflow sentence, `src/lib.rs`, `tests/ports.rs` and the initial `src/ports.rs`. It refuses duplicate functions or ranges and rows where a bound would be untested.
- The identity line names the instance (`task instance N`), identically on every arm. The run id is still not part of the prompt.
- The coordinator approval must carry `instance` (an integer equal to `--instance`) and `instance_table_sha256`. Its `sources` must also cover `coding_instances.py` and `test_coding_instances.py`. A canonical approval that carries either key is refused. The coordinator and every agent summary record `instance: {id, table_sha256}`, and aggregation refuses an agent from another instance.
- `--instance` is N4-only and cannot be combined with `--benchmark-profile`.

### Paired campaign driver

`quality_campaign.py` runs instances `0..N-1` (default 24) of one family on a subject arm (an hf2q Q5 case) and a reference arm (`--reference-kind peer` with a peer case, or `stock-hf2q` with a stock hf2q case). Without `--reference-case` the verdict is floor-only.

- **Schedule.** Arms alternate in ABBA blocks: even instances run the subject first, odd instances the reference first. Each run is one `q5_calibration.py` calibration on a derived case (`cases/iNN-<arm>.json`), which is the arm's case plus `--instance N`, the approval `instance`/`instance_table_sha256` and the two instance sources. Runs are sequential, so only one runtime is loaded at a time, and Q5 preflight still refuses competitors.
- **Variables.** Ambient `HF2Q_Q5_<NAME>` overrides go to each arm whose case declares `<NAME>`; one that neither case declares is refused. `--subject-var NAME=VALUE` and `--reference-var NAME=VALUE` override one arm only, for example the two approval templates. The effective overrides are part of the plan.
- **Outcome per run.** An instance passes on an arm when the client summary is `FourAgentProtocolCompleteAwaitingIndependentReview` and the Q2 run audit (`q2_run_audit.py`, without the concurrency-width gate) re-derives that success from the raw client evidence; a summary whose receipts contradict it is `Invalid`. A Q5 memory-envelope failure with a passing client is still a task pass; its Q5 verdict is recorded. After the four-agent release, a failed client, a server exit or a client timeout is a task failure of that arm. A run that never reached the release (Q5 preflight refusal, readiness failure or approval refusal) is `NotStarted`. Any other supervisor stop after the release is `Invalid`.
- **Halting and resume.** The driver halts at the first run without a task outcome. `--resume` with the same `--output` and an identical plan (cases, overrides, instance table, sources, negative control) continues: a `NotStarted` run gets a new attempt, and a run interrupted by the driver itself is scored from what is on disk. An `Invalid` run is terminal: there are no reruns after the task has started, so start a new campaign.
- **Outputs.** `manifest.json` (plan, plan hash, every attempt with its argv, Q5 receipt hash, Q5 verdict, stop, client summary hash and per-agent status), `cases/`, `runs/<index>-iNN-<arm>-aK/` (the Q5 receipt directories) and `logs/`. When complete, it writes `outcomes.json` (`hf2q.q1.paired-task-outcomes.v1`) and adds the verdict to the manifest. The verdict is a port of `q1_decision_policy.py paired`: `Insufficient` if n < 24; `Loss` if the exact one-sided sign test gives p < 0.05; `BelowFloor` if the subject passes fewer than 75% of instances; `Inconclusive` if subject-only failures exceed reference-only failures by more than ⌊n/8⌋; otherwise `Pass`. Every passing patch still needs independent review.
- `--dry-run` prints the plan, the ABBA schedule, instance 0's derived client argv and missing variables per arm, and launches nothing.

Qwen3.8 against the peer (needs `q5_cases/peer-qwen38.json` from the peer adapters change). Use the same exports as the Q5 root commands above, plus `NC_RECEIPT` and `Q5_ROOT`:

```sh
export HF2Q_Q5_HF2Q_BIN=/abs/path/release/hf2q HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf
export HF2Q_Q5_CARGO=/abs/cargo HF2Q_Q5_TOOLCHAIN=/abs/toolchain
export HF2Q_Q5_SANDBOX_PROOF=/abs/sandbox-proof.json HF2Q_Q5_SANDBOX_PROOF_SHA256=<sha256>
python3 -B scripts/four_agent/quality_campaign.py \
  --subject-case scripts/four_agent/q5_cases/qwen38.json \
  --subject-var CLIENT_APPROVAL_TEMPLATE=/abs/qwen38-client-approval.json \
  --reference-case scripts/four_agent/q5_cases/peer-qwen38.json --reference-kind peer \
  --reference-var CLIENT_APPROVAL_TEMPLATE=/abs/peer-qwen38-client-approval.json \
  --instances 24 --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/quality-qwen38-peer-$(date -u +%Y%m%dT%H%M%SZ)"   # --dry-run first; --resume reuses this exact --output
```

A floor-only family (for example DeepSeek-V4) omits the three `--reference-*` options.

## Three-arm speed campaign (ADR-063 Q3)

`q3_campaign.py` runs the per-family Q3 comparison: stock hf2q (`--stock-bin`, the released binary), experiment hf2q (`--experiment-bin`) and the peer's best valid configuration (`q5_cases/peer-<family>.json`); for Qwen3.8 and Qwen3.6 also Splash's best valid configuration (`--splash-case`, default `q5_cases/splash-<family>.json`; see "Splash peer of record"). A labeled matched-settings peer diagnostic (`--peer-matched-case`) is optional. Every run is one `q5_calibration.py` runtime, run one at a time. The design, estimator, verdict rules and root commands are in [docs/research/adr063-q3-campaign.md](../../docs/research/adr063-q3-campaign.md).

- **Identities.** Binaries (with the peer's sibling dylibs), model, peer `build_info`, argv input files, harness sources and commit, Python and host are frozen at start and re-checked before every run and against every receipt. Any change is a terminal `IdentityDrift`.
- **Stages.** `n4` (primary, all four agents together), then the controls `n1`, `3+1`, `2+2` and `1+3`. Each stage runs one discarded warmup per arm per driver session, then Williams-ordered blocks: 5 to 20 blocks, stopping when every required makespan contrast has a 95% half-width ≤ 2.5 points.
- **Controls-only campaigns.** `--controls-of PRIMARY_VERDICT_JSON` runs only the control stages named in `--workloads`, as controls of that verdict of record (#369 amendment). `n4` and any stage already `Complete` in the record are never re-run. The record must be a `verdict.json` with a `Complete` primary; freeze it before binding. The family, host, Python and arm identities must equal the recorded campaign's: binaries, commits, model, cases, override contents, and the peer build. Harness sources and approval templates may differ, and both source maps are recorded. `verdict.json` is `ControlsOnly`, with `primary: null` and a `primary_of_record` field. `--rows` emits `control_of` rows: `Descriptive` effects of completed controls and `Failure` task rows of controls stopped by an experiment failure. The scoreboard renders them in their own section, never as a win or loss.
- **Verdict.** `verdict.json` holds `Win`, `Loss`, `Inconclusive`, `Unavailable`, `Failed` or `Pending`, with effects, CIs and reasons. `--verdict DIR` recomputes it with `--quality-verdict` (a Q1 paired manifest on the experiment binary) and `--patch-review`. `--rows DIR --evidence-root "$E"` prints F2 scoreboard rows (the stock baseline is `v0128_stock`, the v0.1.28 release). `--rows DIR --of-record` with `--quality-gate REFERENCE=MANIFEST` (repeatable) and `--review FILE` prints a reviewed verdict of record for the F2 "Q3 verdicts of record" tier instead.
- **Halting and resume.** A peer that cannot load or runs out of memory becomes `Unavailable`, and the remaining blocks run without it. `--peer-unavailable-evidence` declares this up front. An experiment task failure ends the campaign as `Failed`. A run that did not start halts the campaign, and `--resume` retries it.
- **Client phase plans.** The client takes `--phase-plan {1,3+1,2+2,1+3}`. `1` runs `agent-0` alone on the four-slot server. `K+M` releases K agents, then the remaining M once each of the K has its first decoded token: the first reasoning or semantic SSE frame of its first streamed turn, published as `barrier/<agent>.first-decode.json`. Without the option, the child argv, release record and summaries are byte-identical to earlier runs.

| File | Role |
| --- | --- |
| `q3_campaign.py` | CLI and driver: plan, derived cases, schedule, runs, halts, resume, `--dry-run`, `--verdict`, `--rows` |
| `q3_identity.py` | Frozen identities, launch checks, identity diff |
| `q3_metrics.py` | Run outcome classification and per-run metrics from receipts |
| `q3_stats.py` | Williams sequences, paired log-ratio estimator, stopping rule, family verdict |
| `q3_report.py` | Stage effects, `verdict.json`, quality inputs, F2 rows |

## Q2 adversarial controls and run audit

`q2_run_audit.py` re-derives a settled four-agent run from its raw client evidence. Both campaigns call `check_run` on every claimed Pass and turn a contradicted claim into a terminal `Invalid` with the reason. It does not trust the coordinator's status, makespan or timing fields. It checks:

- that every archived `turn-N/request.json` and `response.bin` matches its receipt hash and size, and that the request is the canonical wire;
- each response, replayed through the actual parser, against the receipt usage and event counts;
- v3 timing, recomputed;
- request lineage, plus unique response ids across all agents and turns;
- assistant output for another agent's identity sentence;
- the scoring, by re-running the coordinator `aggregate`;
- owned cleanup;
- the makespan, recomputed from the release and completion clocks;
- phase-cohort membership and triggers;
- client-observed co-progress width.

The co-progress width is the largest set of agents whose streamed requests stay active together: each request runs from its first to its last progress frame (`progress_event_monotonic_ns`, any SSE frame that grew reasoning or semantic output), and the set needs a common window that either lasts at least half of the shortest member's active interval or holds at least three progress frames of every member. No frame alternation is required, so low-frame bursty streams such as hf2q Gemma qualify. Overlapping HTTP lifetimes alone count for nothing. Q3 requires every active agent to be co-active at some point: 4 for `n4` and for `3+1`, `2+2` and `1+3` (the later cohort only exists after its release, and the earlier cohort must still be running then), and only agent-0 for `n1`. A serialized service, or one capped below four slots, is rejected. The width is a client-side lower bound. Physical batch membership and pure prefill/decode phases come from the server phase trace (`f2_trace.py`).

`q2_fake_service.py` is test support only: a model-free OpenAI-compatible loopback service whose fault table injects:

- role-only and reasoning-only SSE;
- an empty final summary;
- wrong or truncated tool arguments;
- reused tool-call or response ids;
- foreign identity text;
- silent truncation (8192 completion tokens under a non-length finish), an inflated output allowance and a `length` finish;
- cache and usage lies;
- a swapped model;
- known delays, and a bursty four-frame stream like hf2q Gemma's;
- one-request-at-a-time serialization, or a cap of N slots.

`test_q2_adversarial_controls.py` runs the control matrix. The live controls use four real client processes, the receipt and summary forgeries go through the audit, and the false wins go through the actual Q3 and quality drivers. Each test asserts the rejection reason. The matrix, live results and the boundary of what client evidence can prove are in [docs/research/adr063-q2-adversarial-controls.md](../../docs/research/adr063-q2-adversarial-controls.md).

| File | Role |
| --- | --- |
| `q2_run_audit.py` | Settled-run audit used by `q3_campaign.py` and `quality_campaign.py` |
| `q2_fake_service.py` | Model-free fake service, honest-run fixtures and live four-process driver (tests only) |
| `test_q2_adversarial_controls.py` | Wire, audit, live-concurrency, injected-delay and campaign controls |

### Q2 raw-cohort certification (`q2_manifest.py`, `q2_certify.py`)

`q2_manifest.py` hashes the retained files of one cohort into `hf2q.q2.cohort.v1`; `q2_certify.py` replays and validates them against an externally frozen `hf2q.q2.expected-plan.v1`. Without an independent QE attestation the best result is `PendingIndependentQE` (exit 2); a hash-bound `hf2q.q2.qe-verdict.v1` that names the exact manifest and auditor sha256 makes it `Qualified` (exit 0). A refused cohort gets a `Rejected` certificate with the reason (exit 2). Certificates are create-new and never inside a Git tree.

- **Inputs.** `server_receipt` and `readiness` are the files `q2_traced_run.py` (or the F2 operating90 supervisor) writes (`receipt.json` with `protocol_complete`/`final_verified`/`cleanup_proved` and a cleanup list; `readiness.json` with `health` and `runtime`). The q5 calibration receipt and its separate `readiness-N.json` files are not certifiable inputs.
- **Server phase trace.** The JSONL file `hf2q serve` writes under `HF2Q_PHASE_TRACE_CONFIG`, one record per line. Its size bound is its own header's metadata quota (`f2_trace.metadata_total_limit`: 256 MiB, or 2/8 GiB for the declared native capture profiles). Each line is at most one trace event (64 KiB). The file is streamed record by record on every pass, and each pass re-checks its frozen sha256, so a multi-GB trace is never loaded whole.
- **`trace_scenario`** (in the expected plan): the `hf2q.f2.trace-expectation.v1` keys `f2_trace.validate_trace` reads (`schema`, `run_id`, `config_sha256`, and when used `metadata_budget_profile`, `forward_witness_manifest`, `forward_demand_policy`), plus the frozen window: `composition`, `arrival_ns` and `population_policy`. `population_policy` is required, with no default: `ready_now_v1`, `forward_demand_v2` (the Q2 item 4 policy) or `queued_forward_demand_v3`. It is passed to `f2_trace.qualify_window` unchanged.
- **Engines.** Only hf2q cohorts carry the server phase trace, so `engine_kind` `peer` is refused (`peer cohort has no hf2q server phase trace: never Q2-certifiable`) and every native engine kind of the `native_peers` registry (ds4, TensorFold) is refused (`closed engine family`). Peer agents can still be dry-audited with `q2_certify.audit_agent`; native-engine runs are audited by `q2_run_audit.py` with the arm's AUDIT module.

`test_q2_certify.py` runs a synthetic N1 cohort (a fake-service agent, stub compile/test receipts, a synthetic native trace) through `q2_manifest.assemble`, `certify_cohort` and `main`, with the refusal reasons and the JSONL trace bounds.

### Q2 traced cohort (`q2_traced_run.py`, `q2_traced_freeze.py`)

`q2_traced_run.py` runs one supervised attempt of the Qwen3.8 four-slot case (`q5_cases/qwen38.json`) on a given hf2q binary with `HF2Q_PHASE_TRACE_CONFIG` set. `--population` is `n4` (the canonical release of all four agents) or a Q3 control (`n1`, `3+1`, `2+2`, `1+3`, passed to the client as `--phase-plan`). It reuses the Q5 preflight, memory sampler, stop latch and owned-group cleanup. It never retries.

- **One identity.** `--identity` is the eight-key q2_certify identity. Before launch it must match the binary sha256, the build receipt (`source`, `sha256`), the client approval template sha256 (`profile_sha256`), the model's conversion-receipt sha256 and the mlx-native pin in `<server_source>:Cargo.lock`. At readiness, the served tokenizer and template sha256 and the server pid must match too.
- **One run id.** The run id is `q2t-<output directory name>`. The client approval and `native-config.json` both carry it, and the trace header must repeat it.
- **Trace.** `native-config.json` uses `native_command_capture_v3`: server budget 6 GiB, client budget 128 MiB, 8,000,000 native events, 16 MiB native metadata, a fresh control nonce and the `target_input_rms_norm` witness manifest. There is no `forward_demand_policy`. The worker-checked arm ACK is read back before the client starts. After cleanup, the trace must end with `end` (`error` null, `dropped_events` 0).
- **Outputs.** The run writes `receipt.json` (`hf2q.q2.traced-run-receipt.v1`, status `TracedCohortCompleteAwaitingQ2Certification` or `FailedOrHeld` with the reason), `readiness.json` (`{health, runtime, readyz}`), `native-arm-readback.json` and `samples.jsonl`. Every active agent's compile receipt and both test executables are read back through `q2_manifest.test_group` and listed under `retained_test_executables`. The client never deletes them. Exit 0 complete, 1 failed or held, 2 refused before launch.

`q2_traced_freeze.py plan` freezes the expected plan and manifest input of an `n4` run. Its window is `(4,0)` at `barrier/release.json`, and the `--population-policy` is recorded in `trace_scenario`. `q2_traced_freeze.py windows` validates the trace once and qualifies each pre-registered window of any population offline: `n4` `(4,0)` at the release and `(0,4)` at the last agent's first turn-1 decoded delta; `n1` `(1,0)`/`(0,1)` likewise; `3+1` `(1,3)`, `2+2` `(2,2)` and `1+3` `(3,1)` at `barrier/release-2.json`. Arrivals are client CLOCK_MONOTONIC instants minus the trace header origin, so they are fixed before any trace body is read. n1 and phase-plan runs are not q2_certify cohorts. `test_q2_traced_run.py` covers the argument, identity, readiness, readback, cleanup and run-id checks.

## F2 scoreboard rendering

`f2_scoreboard_render.py` renders [docs/research/adr063-f2-scoreboard.md](../../docs/research/adr063-f2-scoreboard.md) from the checked-in rows manifest `docs/research/adr063-f2-scoreboard-rows.json` (schema `hf2q.f2.scoreboard-rows.v1`) plus any `f2_scoreboard_cli.py` outputs. Only those fixed-six outputs can produce Qualified F2 primary results; manifest rows are Descriptive, NegativeResult or Failure receipts and cite `$E/...` receipt paths and extraction keys, never absolute local paths. `--verify` re-reads every Verified row from its receipt and refuses drift:

```sh
python3 scripts/four_agent/f2_scoreboard_render.py --rows docs/research/adr063-f2-scoreboard-rows.json \
  --evidence-root "$E" --verify [--scoreboard qwen38="$E/<run>/scoreboard.json"] \
  --output docs/research/adr063-f2-scoreboard.md
```

**Q3 verdicts of record** (`f2_q3_tier.py` records, `f2_q3_view.py` markdown; F2 closure option B) render above the Descriptive wins, one line per family, from a group `q3_record` and its `q3_of_record` rows (`q3_campaign.py --rows --of-record`, see [adr063-q3-campaign.md](../../docs/research/adr063-q3-campaign.md) "Outputs").
- **Each line shows:** the Q3 verdict; the makespan effect of `hf2q <commit8>` vs v0.1.28 stock; the task-makespan sentence vs the peer (X = 1 − faster/slower on the slower arm's time; a native-engine statement for an Unavailable family); worst-slot TTFT p95 and SSE-gap max ratios; the quality gate per reference and any pending patch reviews; the campaign status with its halt, precision shortfall and measured/unmeasured control stages; and the verdict and review references. A table below gives token volumes and output rate per arm.
- **SSE gaps:** against the peer they are shown as not comparable (different event granularity). One the verdict could not estimate is shown as not measurable and does not gate.
- **Win:** appears only vs stock, and only when the stock contrast, stock fairness, stop, thermal, memory and patch-review conditions hold and the required quality gates (stock plus the peer, or the native stand-in when the peer is unavailable) passed. Otherwise the line lists the gating reasons.
- **Placement:** these rows never enter the comparison, win or gap sections. The record's completed controls render in the Q3 controls section. Validation refuses a row outside the record that cites its receipts, such as the same campaign's default fragment.
- **Extraction:** lives in `f2_rows_extract.py` (`median_of` takes the median of one extract per item, for example per run receipt). `--verify` also re-derives every `q3_record` from its receipts.

## C5 warm frontend timing

`c5_frontend_summary.py` reads `server.stderr` files from runs whose server had `HF2Q_FRONTEND_TIMING=1` and prints a `hf2q.c5.frontend-summary.v1` JSON per family: handler frontend work (render, tokenize, stable-prefix render/tokenize/match, probe render, budget markers, constraint compile, overflow rewrite) against warm semantic TTFT, with the ADR-063 C5 10% gate verdict (`material`, `not_material` or `unmeasured`). The measurement plan and root commands are in [docs/research/adr063-c5-warm-frontend.md](../../docs/research/adr063-c5-warm-frontend.md).

```sh
python3 scripts/four_agent/c5_frontend_summary.py "$RUN/server.stderr"
```

## S3 cancel smoke and joint row widths (ADR-063 S3: S-C1, S-C2)

`cancel_smoke.py run` is a client for an already running hf2q server. It replays four recorded requests, `agent-N/turn-K/request.json` (default turn 2) from a four-agent run's `client/` directory, as four concurrent streams. Each request is sent verbatim except that `stream` is forced to `true` with `stream_options.include_usage`, and its `model` must equal `--model`.

**Cancel pass.** The client closes the `--cancel-agent` stream (default `agent-0`) once it has received `--cancel-after-events` semantic events (default 16: deltas carrying content, reasoning or tool-call text). It closes only while each of the other three streams has at least one event and no `finish_reason`. If that window never opens, the run is `CancelSmokeInvalid`: no claim either way; pick a longer victim or a smaller K.

**Control pass** (`--no-control` skips it). The same four requests run again without a cancel. All four completing shows that the cancelled slot was released.

**`CancelSmokePass` requires all of the following:**
- The three survivors end with valid SSE: HTTP 200 `text/event-stream`; one choice at index 0; a `stop`, `tool_calls` or `length` finish; usage with a consistent total; `[DONE]` last; no error frame or `finish_reason: error`. Tool calls need dense indices, an id, a name and JSON-object arguments, and `tool_calls` is the finish reason exactly when calls exist.
- `/metrics` `hf2q_sse_cancellations` rises by exactly 1 in the cancel pass and by 0 in the control pass.
- The last joint decode turn the server logged before the close has width ≥ 2.
- The run window of the server log has no `ERROR` or `poison` lines, and `/readyz` is ready afterwards.
- Every control-pass stream is valid.

**Reported, not gated.** `control_comparison` records whether each survivor's output (content, reasoning, finish reason, tool-call names and arguments; ids excluded) equals its control output. It also records whether the victim's partial output is a prefix of its control output. Exact equality is expected on bitwise or cohort-invariant routes. The control runs on caches warmed by the cancel pass, so a difference needs diagnosis before it counts against the route.

**Row widths (S-C2).** Width is `lanes` for DeepSeek-V4 and `rows` for Qwen, taken from the per-turn `DeepSeek-V4 joint decode turn` and `Qwen joint decode turn` debug lines. The receipt histograms them for the whole run, the cancel pass, the part after the close and the control pass. A DeepSeek-V4 turn runs up to `budget` lockstep steps, so `budget_steps_by_width_upper_bound` is an upper bound: lanes that finish mid-turn narrow later steps. `cancel_smoke.py widths --family F --server-log LOG` prints the same histogram for any log, such as a canonical four-agent run.

**Debug lines need `LOG_FILTER`.** hf2q ignores `RUST_LOG` when `-v` is given, and the family launchers pass `-v`. So no retained run has these lines. `serve_deepseek4_opencode.sh` and `serve_qwen36_opencode.sh` (which `serve_qwen38_opencode.sh` execs) take `LOG_FILTER=<filter>`: it replaces `-v` with `RUST_LOG=<filter>`. Empty or unset keeps `-v`. `deepseek4.json` and `qwen38.json` declare `LOG_FILTER` with an empty default; the cancel cases default it to the family's debug filter.

**Receipt.** `<output>/summary.json`, schema `hf2q.s3.cancel-smoke.v1`, with `status`, `reasons`, per-pass per-agent results, the close point (server-log offset and survivor event counts), counter deltas, joint turns around the close, histograms and the comparison. Per-agent request, raw SSE and assembled output are written under `<output>/{cancel,control}/agent-N/`. The exit code is 0 only on `CancelSmokePass`.

**Q5 contract.** The cancel cases reuse the canonical case's launcher, environment, readiness and memory supervision. The client:
- runs with `cwd` at the run directory;
- reads `{OUT}/server.stderr`, which the supervisor writes;
- writes `{OUT}/client/summary.json`, whose `success_status` is `CancelSmokePass`.

There is no client approval or barrier: the client runs no tools, no cargo and no sandbox, and releases its four streams together itself. A cancel run is a calibration-kind Q5 run, so it needs `--negative-control-receipt` and no `--admission-receipt`.

Root commands, one family at a time, from the repository root of the binary under test (`E`, `Q5_ROOT` and `NC_RECEIPT` as in the Q5 commands):

```sh
# DeepSeek-V4: S-C1 cancel smoke plus S-C2 widths
HF2Q_Q5_HF2Q_BIN=/abs/target/release/hf2q \
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_RECORDED_CLIENT="$E/q5-deepseek4-final-20261030T051035Z/client" \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/cancel-deepseek4.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/s3-cancel-deepseek4-$(date -u +%Y%m%dT%H%M%SZ)"

# Qwen3.8
HF2Q_Q5_HF2Q_BIN=/abs/target/release/hf2q HF2Q_Q5_MODEL=/abs/Qwen3.8-27B-vanilla.gguf \
HF2Q_Q5_RECORDED_CLIENT="$E/q5-qwen38-final-20261030T051303Z/client" \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/cancel-qwen38.json \
  --negative-control-receipt "$NC_RECEIPT" \
  --output "$Q5_ROOT/s3-cancel-qwen38-$(date -u +%Y%m%dT%H%M%SZ)"

# S-C2 on the canonical four-agent workload: the family case with the trace arm, then the histogram
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
HF2Q_Q5_LOG_FILTER='hf2q=info,mlx_native=info,hf2q::serve::api::engine_deepseek4::progress=debug' \
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/deepseek4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/s3-trace-deepseek4-$STAMP"
python3 scripts/four_agent/cancel_smoke.py widths --family deepseek4 \
  --server-log "$Q5_ROOT/s3-trace-deepseek4-$STAMP/server.stderr"
# Qwen3.8: q5_cases/qwen38.json with HF2Q_Q5_LOG_FILTER='hf2q=info,mlx_native=info,hf2q::serve::api::engine_qwen35::progress=debug', then --family qwen38
```

Standalone, with a server the root starts itself, the client needs only the base URL, the model id and the server's stderr file:

```sh
LOG_FILTER='hf2q=info,mlx_native=info,hf2q::serve::api::engine_deepseek4::progress=debug' \
MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf PORT=18240 MAX_SLOTS=4 \
KV_CACHE_BUDGET_BYTES=8589934592 MAX_COMPRESSOR_USED_GIB=16 HF2Q_BIN=/abs/target/release/hf2q \
  scripts/serve_deepseek4_opencode.sh --ctx 262144 2> "$RUN/server.stderr" &
# after http://127.0.0.1:18240/readyz reports ready:
python3 scripts/four_agent/cancel_smoke.py run --base-url http://127.0.0.1:18240/v1 \
  --model 'DeepSeek V4 Flash 0731' --family deepseek4 \
  --requests "$E/q5-deepseek4-final-20261030T051035Z/client" \
  --server-log "$RUN/server.stderr" --output "$RUN/client"
```

Model-free controls are in `test_cancel_smoke.py`. A loopback fake server streams SSE, counts write failures as cancellations and logs joint turns. The tests check:
- the pass path;
- survivor error finishes and malformed tool arguments;
- a missed window;
- an uncounted cancel;
- a missing trace;
- a divergent control (reported, not gated);
- that the case argv matches the CLI.
