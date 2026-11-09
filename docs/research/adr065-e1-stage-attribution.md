# ADR-065 E1: complete-round stage attribution

Status: complete (2026-11-09). Issue: #504.

## Diagnostic contract

`HF2Q_ADR065_STAGE_PROFILE` is an exact-value opt-in:

- unset or any value other than `wall` or `1`: no round guard, timestamp, native boundary or record;
- `wall`: complete-round wall timing only, buffered until the HTTP server has drained and then emitted during graceful shutdown;
- `1`: complete-round wall timing, native command-buffer attribution and CPU spans; this mode also requires `MLX_PROFILE_CB=1`.

Each record uses schema `hf2q.adr065.stage-profile-round.v1` and identifies family, route and active width. GPU command buffers are reduced into projections, DeltaNet conv, DeltaNet GDN, KV write, attention, FFN/MoE, head, MTP draft and MTP catch-up. CPU spans cover logit readback, host selection, rollback and per-row arena setup. Host gap is the non-negative round remainder after known GPU and CPU time. DeepSeek decode-scratch release and reallocation events are pending counters consumed by the next complete-round record, so events between rounds are not lost.

Unrecognized native labels remain `unknown`, are retained by raw label, and are excluded from every decision rule. `scripts/four_agent/adr065_stage_profile.py` validates Q5 verdicts, cleanup, artifact and binary identities, profile controls, the closed record schema, and canonical transcript equality before reducing any data.

Qwen solo K3 performs multiple target forwards in one round. While the E1 guard is active it owns the native profile table across those forwards; the legacy per-forward diagnostic therefore does not reset the table. This preserves MTP draft and catch-up in one complete-round split.

## Frozen identities

| Item | Identity |
| --- | --- |
| Previous-default source | `c4f4701c18bbe0e78da18d3d6f69a814be3bb237` |
| Previous-default binary | sha256 `1555a72187a2abf2e8502a904c1ff6abe0010242d93276e2df48be34670241b1` |
| Qwen joint observer source | `cd4c74bb9da3df7cbaffa41eb2f66e6f3e81c656` |
| Qwen joint observer binary | sha256 `e87d517468789e20dc6660d9e26bed87deace1dc5b3b9cbc08ca0cf31c40f34b` |
| DeepSeek/Gemma observer source | `caeca1b21664066a8365b999d58d37c3eb63d9b8` |
| DeepSeek/Gemma observer binary | sha256 `0a2981ff1a62d3a1df89d149337c38535fcd6b8d15bf77779bff05aa6c755ddf` |
| Qwen solo K3 source | `8ed0ddc743e27953aa42d1982f61893965fb52b2` |
| Qwen3.8 GGUF | sha256 `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| DeepSeek-V4 GGUF | sha256 `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` |
| Gemma 4 GGUF | sha256 `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53` |
| Qwen / DeepSeek / Gemma Q5 cases | `37ab3e832f06e15b6465757feaec6d7d337bb4d477cb819bf50dad1f20caee6f` / `b78360154fe140275a91de1d5b2928862ebe6b3d1a88f031e3951c17e2783c82` / `6f2bea3c84338e3652c0aa74ba7353949104937f0d6bbe06ee1df1676929c245` |
| Sandbox proof | sha256 `4b2bea7715c6244e7636d6d07b4e33bd41b7c525313488dcca0f107cc79d486e` |
| Toolchain | `stable-aarch64-apple-darwin`, rustc/cargo 1.99.1, locked dependencies |

Each Q5 arm used the official artifact, four slots, context 262144 per slot, the canonical 28-turn coding workload, the retained negative control and family approval template. Candidate control and profile used the same binary. The reference arm used the previous-default binary. Gemma candidate control/profile both set `HF2Q_FUSE_LMHEAD=0` so head work remained independently attributable; its previous-default reference retained the default setting.

## Retained Q5 result and observer distortion

All nine Q5 arms returned `CalibrationPass` and proved owned-process and listener cleanup. For each family, the wall-control transcript and previous-default transcript were `ExactTranscripts`: 28 same-request groups, 28 turn positions and zero different outputs.

| Family | Wall makespan | Profile makespan | Profile / wall | Width-4 wall median | Width-4 profile median | Width-4 profile / wall | Width-4 Unknown / round wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 139.462278 s | 261.518457 s | 1.8752× | 73.333 ms | 135.001 ms | 1.8409× | 0.00773% |
| DeepSeek-V4 | 100.761924 s | 231.759982 s | 2.3001× | 77.112 ms | 185.078 ms | 2.4001× | 0% |
| Gemma 4 | 12.586108 s | 61.458296 s | 4.8830× | 18.516 ms | 117.303 ms | 6.3351× | 0% |

The native observer is intentionally diagnostic, not a production timing path. Its complete-round distortion is large, especially for Gemma; stage totals MUST NOT be used as unprofiled latency. Wall controls provide the undistorted complete-round denominator.

### Width-four stage totals

These are aggregate totals over each profiled width-four cohort. Unknown is shown but excluded from known-stage interpretation.

| Route | Rounds | Projections | Delta conv | Delta GDN | KV write | Attention | FFN/MoE | Head | Unknown | Host gap | Readback | Selection | Arena |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen joint | 1,483 | 32.422 s | 0.801 s | 3.512 s | 0.396 s | 5.245 s | 71.066 s | 5.163 s | 0.0168 s | 98.110 s | 0.106 s | 0 | 0.643 s |
| DeepSeek joint | 664 | 22.762 s | 0 | 0 | 0 | 26.719 s | 22.159 s | 1.824 s | 0 | 55.034 s | 0.022 s | 0.048 s | 0.002 s |
| Gemma joint | 396 | 2.232 s | 0 | 0 | 0.196 s | 1.207 s | 4.609 s | 0.911 s | 0 | 39.244 s | 0.032 s | 0.172 s | 0.001 s |

DeepSeek recorded 18 decode-scratch releases and 18 reallocations: width two `5/4`, width three `12/13`, width four `1/1`, and width one `0/0` for release/reallocation respectively. The one-event width shift is expected from pending counters: an event between completions is charged to the next completion rather than discarded.

## Qwen solo K3 result

The official-artifact ignored hardware gate ran the `off` and slot-aware `auto` arms, exercised solo K3, a cohort and cancellation, and passed: 9/9 responses identical, 151 MTP proposals, 440 accepted draft tokens and 8 rejected draft tokens. The retained profile contains 151 complete `solo_k3` rounds.

| Rounds | Median round | MTP draft | MTP catch-up | Projections | Delta GDN | KV write | Attention | FFN/MoE | Head | Host selection | Rollback | Host gap | Unknown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 151 | 118.590 ms | 0.974 s | 0.026 s | 0.417 s | 0.888 s | 0.030 s | 0.312 s | 5.069 s | 0.337 s | 0.0145 s | 0.000043 s | 8.583 s | 1.298 s |

Known stages cover 92.77% of solo-round wall time. Unknown is 7.23%, dominated by native `layer.gdn.stage_a`, which combines projection/conv work and cannot be split truthfully from the available command-buffer label. It is retained and excluded rather than assigned heuristically.

## Evidence

`$E1` is `/Users/robert/.local/state/hf2q/v0130/e1-504`.

| Evidence | SHA-256 |
| --- | --- |
| Combined stage report, `$E1/stage-profile-report.json` | `10d5836c853b9bb87f6f7b342869005d4a152980a16e99806ad2d8bcc4f75827` |
| Qwen K3 profile log | `60c45b00109819543ed6a0787bd5e1c1ed2501ff9e038d15123cc9ef9d9214e6` |
| Qwen control / profile / reference receipts | `73f18dee2ecfe5a98d9e1d2448bc4c4918840a405cbc65f8f38ba6f992bbe432` / `945dfaa7a569ee4b3b0ea3e41aa174a1087ad1c87b9affad231cacc4aef1158f` / `9b2220f80e1e2b665e09480d9baaf01d9e0eeaa72e120cda0061450266738fca` |
| DeepSeek control / profile / reference receipts | `5e7a0ef9450c589a696bf274582f3e28b982b9c853d0d8979a2ef48f3e5f8f0d` / `81f84da8c9eee16b00976ebe040c6ed5c757b24b126e9b30aba57ed23fb9dd8b` / `8996061766778456cdebbf7f2f968020ad12f65d9fb0f9599a07ebc48a5594a5` |
| Gemma control / profile / reference receipts | `60087865b6dd52d0ec62cbc97e0872149839d456dd85757ee7766923b9f47d1e` / `852773bdd71aeb59e772151ac2ebaeed62671a4d175c70c25105a3594b907544` / `8112a48c5a491b93264a0ab8bb9780e741c80d1e71c2e77cf063dd6b94810679` |

Excluded attempts are not pooled: initial exploratory Qwen arms; the first DeepSeek reference stopped before load because another process exceeded the host gate; the first Gemma reference was `PreflightRefused` because the operator-supplied digest contradicted the artifact receipt; and two Qwen K3 commands selected zero tests because they targeted the library instead of the binary test harness. None loaded a model or contributed measurements.

## Reproduction

Run one model at a time. Each Q5 case binds the paths and hashes above through `HF2Q_Q5_*` variables.

```sh
python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/qwen38.json \
  --output "$E1/qwen38-final-wall" --negative-control-receipt "$NC/receipt.json"
# Repeat with ADR065_STAGE_PROFILE=1, NATIVE_PROFILE_CB=1 for the profile arm,
# and with the previous-default binary and profiling unset for the reference.
# Repeat the same three-arm protocol for deepseek4.json and gemma4.json.

HF2Q_QWEN38_GGUF="$QWEN_GGUF" \
HF2Q_ADR065_STAGE_PROFILE=1 MLX_PROFILE_CB=1 \
  cargo test --release --locked --bin hf2q \
  qwen38_slot_aware_speculation_auto_matches_off -- \
  --ignored --nocapture --test-threads=1

python3 scripts/four_agent/adr065_stage_profile.py \
  --pair qwen38 "$E1/qwen38-final-wall" "$E1/qwen38-final-profile" "$E1/qwen38-reference" \
  --pair deepseek4 "$E1/deepseek4-caeca1b2-wall" "$E1/deepseek4-caeca1b2-profile" "$E1/deepseek4-reference-02" \
  --pair gemma4 "$E1/gemma4-caeca1b2-wall" "$E1/gemma4-caeca1b2-profile" "$E1/gemma4-reference-02" \
  --solo-profile qwen38 "$E1/qwen38-k3-profile.log" \
  --output "$E1/stage-profile-report.json"
```
