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
- **Outcome per run.** An instance passes on an arm when the client summary is `FourAgentProtocolCompleteAwaitingIndependentReview`. A Q5 memory-envelope failure with a passing client is still a task pass; its Q5 verdict is recorded. After the four-agent release, a failed client, a server exit or a client timeout is a task failure of that arm. A run that never reached the release (Q5 preflight refusal, readiness failure or approval refusal) is `NotStarted`. Any other supervisor stop after the release is `Invalid`.
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

`q3_campaign.py` runs the per-family Q3 comparison: stock hf2q (`--stock-bin`, the released binary), experiment hf2q (`--experiment-bin`) and the peer's best valid configuration (`q5_cases/peer-<family>.json`). A labeled matched-settings peer diagnostic (`--peer-matched-case`) is optional. Every run is one `q5_calibration.py` runtime, run one at a time. The design, estimator, verdict rules and root commands are in [docs/research/adr063-q3-campaign.md](../../docs/research/adr063-q3-campaign.md).

- **Identities.** Binaries (with the peer's sibling dylibs), model, peer `build_info`, argv input files, harness sources and commit, Python and host are frozen at start and re-checked before every run and against every receipt. Any change is a terminal `IdentityDrift`.
- **Stages.** `n4` (primary, all four agents together), then the controls `n1`, `3+1`, `2+2` and `1+3`. Each stage runs one discarded warmup per arm per driver session, then Williams-ordered blocks: 5 to 20 blocks, stopping when every required makespan contrast has a 95% half-width ≤ 2.5 points.
- **Verdict.** `verdict.json` holds `Win`, `Loss`, `Inconclusive`, `Unavailable`, `Failed` or `Pending`, with effects, CIs and reasons. `--verdict DIR` recomputes it with `--quality-verdict` (a Q1 paired manifest on the experiment binary) and `--patch-review`. `--rows DIR --evidence-root "$E"` prints F2 scoreboard rows.
- **Halting and resume.** A peer that cannot load or runs out of memory becomes `Unavailable`, and the remaining blocks run without it. `--peer-unavailable-evidence` declares this up front. An experiment task failure ends the campaign as `Failed`. A run that did not start halts the campaign, and `--resume` retries it.
- **Client phase plans.** The client takes `--phase-plan {1,3+1,2+2,1+3}`. `1` runs `agent-0` alone on the four-slot server. `K+M` releases K agents, then the remaining M once each of the K has its first decoded token: the first reasoning or semantic SSE frame of its first streamed turn, published as `barrier/<agent>.first-decode.json`. Without the option, the child argv, release record and summaries are byte-identical to earlier runs.

| File | Role |
| --- | --- |
| `q3_campaign.py` | CLI and driver: plan, derived cases, schedule, runs, halts, resume, `--dry-run`, `--verdict`, `--rows` |
| `q3_identity.py` | Frozen identities, launch checks, identity diff |
| `q3_metrics.py` | Run outcome classification and per-run metrics from receipts |
| `q3_stats.py` | Williams sequences, paired log-ratio estimator, stopping rule, family verdict |
| `q3_report.py` | Stage effects, `verdict.json`, quality inputs, F2 rows |

## F2 scoreboard rendering

`f2_scoreboard_render.py` renders [docs/research/adr063-f2-scoreboard.md](../../docs/research/adr063-f2-scoreboard.md) from the checked-in rows manifest `docs/research/adr063-f2-scoreboard-rows.json` (schema `hf2q.f2.scoreboard-rows.v1`) plus any `f2_scoreboard_cli.py` outputs. Only those fixed-six outputs can produce Qualified F2 primary results; manifest rows are Descriptive, NegativeResult or Failure receipts and cite `$E/...` receipt paths and extraction keys, never absolute local paths. `--verify` re-reads every Verified row from its receipt and refuses drift:

```sh
python3 scripts/four_agent/f2_scoreboard_render.py --rows docs/research/adr063-f2-scoreboard-rows.json \
  --evidence-root "$E" --verify [--scoreboard qwen38="$E/<run>/scoreboard.json"] \
  --output docs/research/adr063-f2-scoreboard.md
```

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
