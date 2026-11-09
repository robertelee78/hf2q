# ADR-065 E7: bounded joint-verify prototype

Story: [#510](https://github.com/robertelee78/hf2q/issues/510). Governing decision: [ADR-065 E7](../adr/ADR-065-four-agent-joint-speculative-verify.md#e7-time-a-bounded-joint-verify-prototype).

## Diagnostic boundary

The retained prototype is commit `11a7e0b433e538d8ca39529fb843156a7799745a` on published branch `experiment/adr065-e7-joint-verify`. It is not merged to `main` and has no default runtime path. Its checked-in driver, `scripts/run_adr065_joint_verify_diagnostic.sh`, invokes one ignored Rust diagnostic test directly; it does not start the HTTP server or route production traffic.

The graph executes:

- batched MTP drafting by depth;
- grouped target projections over one packed verify window;
- per-slot DeltaNet and attention cores;
- per-slot recurrent-state capture and shared prefix recovery;
- host accept selection and logit readback;
- MTP catch-up, settlement and synchronization.

Serving remains capped at eight rows. The diagnostic-only caller has a separate sixteen-row bound for four slots × four rows. Passing `None` for the diagnostic caps preserves the prototype branch's serving assignment; only the ignored test can request the sixteen-row grid.

The driver binds the frozen Qwen3.8 Q5 case and the official Q5 artifact before loading. Context timing uses one real target/MTP seed row followed by logical cache-cursor advance in a preallocated four-slot TQ cache. Prior cache values are not used as quality evidence; only dispatch shape and complete-round cost are measured.

## Bound identities

| Item | Identity |
| --- | --- |
| Prototype source | `11a7e0b433e538d8ca39529fb843156a7799745a` |
| Published branch | `origin/experiment/adr065-e7-joint-verify` |
| Native dependency | published `mlx-native =0.18.0` |
| Model | `Qwen3.8-27B-vanilla.gguf`, sha256 `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| Frozen Q5 case | `scripts/four_agent/q5_cases/qwen38-long-context.json`, sha256 `8b9e135d729af204fd9b330bd929c719565ae4d1a489442ad954671bee23d30b` |
| Smoke report | `$E7/smoke/report.json`, sha256 `d769babc7e4f6972061afade2675fdd69986229644d4148a2aa89fc598058efd` |
| Toolchain | stable Rust 1.99.1, Apple Metal |

`$E7` is `/Users/robert/.local/state/hf2q/v0130/e7-510`.

## Bounded smoke result

One warm-up and one retained round ran for each required endpoint at near-2k context.

| Shape | Rows | Draft | Verify + readback | Host selection | Capture / recovery | Catch-up / settle | Complete round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4×1 | 4 | 0.001 ms | 54.611 ms | 0.012 ms | 0.002 ms | 1.434 ms | 56.061 ms |
| 4×4 | 16 | 12.702 ms | 201.261 ms | 0.009 ms | 2.988 ms | 2.093 ms | 219.055 ms |

Both rounds completed drafting where applicable, target verify, readback, selection, capture/recovery and MTP catch-up. All four lanes settled; the deliberately untrained diagnostic drafts were rejected, committing one canonical token per lane. Rejection is part of the measured complete-round path, not a failed run.

## Reproduction

```sh
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_ADR065_E_Q5_CASE=/opt/hf2q/scripts/four_agent/q5_cases/qwen38-long-context.json \
HF2Q_ADR065_E_OUTPUT="$E7/smoke/report.json" \
HF2Q_ADR065_E_GRID=smoke \
CARGO_TARGET_DIR=/Users/robert/.cache/hf2q/target-adr065-e7-rust199 \
scripts/run_adr065_joint_verify_diagnostic.sh
```

Focused gates also passed:

```text
serve::api::engine_qwen35::joint_mtp::tests::diagnostic_depths_cover_four_by_four
inference::models::qwen35::joint_decode_routes::tests::verify_width_covers_the_diagnostic_grid
serve::api::engine_qwen35::joint_mtp::tests::qwen38_joint_verify_diagnostic_grid
```
