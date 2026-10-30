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

`q5_calibration.py` runs one supervised four-agent workload per family and writes a receipt. It follows the measured-admission Q5 text: the host must be running only this workload. The receipt records the host baseline, the hf2q peak, the deltas and the cleanup. Live stops end the run before memory runs out. The supervisor never retries, never picks a favorable sample and makes no speed claim. Every receipt says `independent_review_required: true`.

| File | Role |
| --- | --- |
| `q5_calibration.py` | CLI: dry run, preflight, launch, readiness, client, stops, cleanup, receipt |
| `q5_case.py` | Case-file schema `hf2q.q5.calibration-case.v1`, placeholders, argv rules |
| `q5_policy.py` | Pure stop rules, sample aggregation, verdicts |
| `q5_preflight.py` | Detects competing runtimes; checks the port |
| `q5_host.py` | Parsers and wrappers for `vm_stat`, `sysctl`, `footprint`, `ps`, `lsof` and the thermal state |
| `q5_runtime.py` | Sampler thread (every 500 ms), stop latch, cleanup of the owned process groups |
| `q5_negative_control.py` | Bounded synthetic memory-growth child |
| `q5_cases/{qwen38,qwen36,gemma4,deepseek4}.json` | Example family cases |

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
2. **Baseline.** It takes six samples, 500 ms apart. Every sample must be observed, pressure must not be critical, and Free+cached must stay above 2 GiB. It also records:
   - `vm_stat`: free, speculative, file-backed, anonymous, compressor and wired pages.
   - `vm.swapusage`.
   - `kern.memorystatus_vm_pressure_level` and `kern.memorystatus_level`.
   - `hw.memsize` and the thermal state.
3. **Launch.** It starts the case's `serve.argv` directly, without a shell, in a new session (its own process group). The child's environment is `PATH HOME USER LOGNAME TMPDIR LANG LC_ALL LC_CTYPE SHELL TERM` plus the case's explicit `env`; nothing else passes through.
4. **Readiness.** It polls the readiness URL, saves the `record` documents and checks each `checks` entry with strict JSON equality. A mismatch, for example a context length other than 262144, stops the run before the client starts.
5. **Client.** It renders the client approval from the template and runs `client.argv` in its own session.
6. **Sampling.** A separate thread samples every 500 ms. Each sample has host memory plus the phys_footprint and RSS of every member of the server's process group. Samples are written to `samples.jsonl`.
7. **Live stops.** Checked on every sample, in this priority:
   - `pressure_critical`: `kern.memorystatus_vm_pressure_level >= 4`. Warning pressure (level 2) is counted in `pressure_counts` but does not stop the run. On 2026-10-30, warning appeared within 17 s while the 107 GB DeepSeek-V4 artifact faulted in, and 6.6 GB was still reclaimable.
   - `swap_growth`: swap used is more than baseline + 512 MiB.
   - `free_cached_floor`: free + speculative + file-backed pages are below 2 GiB.
   - `footprint_cap`: only when a cap is set, with `--footprint-cap-bytes` or `policy.footprint_cap_bytes`.
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
| `CalibrationPass` | Client succeeded. Pressure was never critical (warning is allowed and recorded), swap growth stayed below 512 MiB, Free+cached stayed at or above 2 GiB in every sample, no sample was blind, and cleanup was proven. | 0 |
| `Stopped` | A live stop or a supervisor stop ended the run. The stop wins over every other outcome. Supervisor stops: `server_exited`, `readiness_timeout`, `readiness_mismatch`, `client_timeout`, `interrupted`. A stop is a preserved failed case, not a retry trigger. | 1 |
| `ClientFailed` | The client exited nonzero, or `summary.status` is not `success_status`. | 1 |
| `CalibrationFailed` | The client completed, but a pass condition failed. | 1 |
| `PreflightRefused` | Nothing was launched. | 2 |
| `SupervisorError` | The supervisor itself failed. | 1 |

A violation that first appears during the unload after a completed run does not turn the run into `Stopped`. It is recorded as `post_completion_violation`, and it still fails the pass conditions.

### Case files and variables

Placeholders are written `{NAME}` and are filled from three sources:
- **Built-ins:** `REPO`, `OUT`, `PYTHON`, `PYTHON_SHA256`, `RUN_ID`, `NONCE32`. `RUN_ID` is derived from the output directory name and matches the client's `[A-Za-z0-9_-]{16,64}` rule.
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
- Gemma4 is made text-only by pointing `MMPROJ` at a nonexistent file inside the receipt directory, which selects the launcher's no-projector branch.
- DeepSeek-V4 passes `--ctx 262144` and an 8 GiB shared KV budget. The Qwen and Gemma cases use a 48 GiB shared KV budget.

Default `HF2Q_BIN` and `MODEL` values are repo-relative placeholders; override them with absolute paths. Required inputs:
- `CARGO`, `TOOLCHAIN`, `SANDBOX_PROOF`, `SANDBOX_PROOF_SHA256`: the client sandbox inputs used by the existing runners.
- `CLIENT_APPROVAL_TEMPLATE`: the family's previous `client-approval.json`, with the same schema, profile, family, `family_controls` and `agent_template`.

The supervisor fills in `server_pid`, `base_url`, `model`, `run_id`, `barrier_token`, `python_sha256` and the twelve client `sources` digests of the current checkout.

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
  --output "$Q5_ROOT/q5-qwen38-$(date -u +%Y%m%dT%H%M%SZ)"

# Gemma4 26B-A4B
HF2Q_Q5_MODEL=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/gemma4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --output "$Q5_ROOT/q5-gemma4-$(date -u +%Y%m%dT%H%M%SZ)"

# Qwen3.6 35B-A3B
HF2Q_Q5_MODEL=/abs/Qwen3.6-35B-A3B-vanilla-Q4_K_M.gguf HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/qwen36-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/qwen36.json \
  --output "$Q5_ROOT/q5-qwen36-$(date -u +%Y%m%dT%H%M%SZ)"

# DeepSeek-V4 Flash
HF2Q_Q5_MODEL=/abs/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf \
HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE=/abs/deepseek4-client-approval.json \
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/deepseek4.json \
  --output "$Q5_ROOT/q5-deepseek4-$(date -u +%Y%m%dT%H%M%SZ)"
```

Run `--dry-run` before every real run; it launches nothing:
- It prints the expanded plan, `missing_variables` and the file checks (binary, artifact, launcher, client and template).
- It exits 2 unless the plan is launchable.

Optional inputs:
- `HF2Q_Q5_MODEL_SHA256` adds a provided digest. The receipt labels it as not verified by the supervisor; artifact identity is size plus mtime.
- After a family's first pass, later runs of the same artifact, source, context and slot count can add `--footprint-cap-bytes <calibrated peak + margin>`. Derive the margin from the negative control's measured `peak_over_cap_bytes` and `stop_latency`.

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
## F2 scoreboard rendering

`f2_scoreboard_render.py` renders [docs/research/adr063-f2-scoreboard.md](../../docs/research/adr063-f2-scoreboard.md) from the checked-in rows manifest `docs/research/adr063-f2-scoreboard-rows.json` (schema `hf2q.f2.scoreboard-rows.v1`) plus any `f2_scoreboard_cli.py` outputs. Only those fixed-six outputs can produce Qualified F2 primary results; manifest rows are Descriptive, NegativeResult or Failure receipts and cite `$E/...` receipt paths and extraction keys, never absolute local paths. `--verify` re-reads every Verified row from its receipt and refuses drift:

```sh
python3 scripts/four_agent/f2_scoreboard_render.py --rows docs/research/adr063-f2-scoreboard-rows.json \
  --evidence-root "$E" --verify [--scoreboard qwen38="$E/<run>/scoreboard.json"] \
  --output docs/research/adr063-f2-scoreboard.md
```
