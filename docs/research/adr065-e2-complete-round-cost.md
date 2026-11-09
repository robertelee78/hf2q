# ADR-065 E2: complete-round cost grid

Story: [#505](https://github.com/robertelee78/hf2q/issues/505). Governing decision: [ADR-065 E2](../adr/ADR-065-four-agent-joint-speculative-verify.md#e2-measure-complete-round-cost-for-valid-window-configurations).

## Method and identities

E7's diagnostic-only prototype at `11a7e0b433e538d8ca39529fb843156a7799745a` ran the official Qwen3.8 27B artifact and frozen long-context Q5 case. The full report is `$E2/full/report.json`, sha256 `82e7f4bbe2bc4558b8128cc7d2beb6f811e7624688db4ce75f708bad313e6021`; `$E2` is `/Users/robert/.local/state/hf2q/v0130/e2-505`.

The report contains all 27 preregistered configurations: nine shapes at logical context positions 2,048, 16,384 and 65,536. Every configuration has one excluded warm-up followed by repetitions 1–7, for 216 retained records: 27 warm-ups and 189 timed rounds. The driver rejects a missing configuration or repetition before E8 consumes the report.

The target is the real model. Every layer dispatch reads that layer's own Qwen3.8 matrices; no synthetic repeated matrix stands in for the layer chain. Model weights stay resident as in serving. A one-token target/MTP prefill seeds each slot, then the diagnostic advances the logical cursor in a preallocated TQ cache. This preserves the attention lengths and dispatch shapes being timed without claiming semantic quality for fabricated prior cache contents.

The stage split is wall time around the complete graph:

- `draft`: batched MTP drafting and draft-cursor restore;
- `verify_readback`: grouped projections, per-slot cores, head synchronization and logit/argmax readback;
- `selection`: host accept walk;
- `capture_recovery`: shared recurrent-state recovery and attention-cursor commit;
- `settle_catch_up`: per-lane target settlement, MTP catch-up and output drain.

E1's GPU observer establishes the finer command-buffer classes inside `verify_readback`; E2 does not relabel unattributable time. `prepare` and host gaps remain in complete-round total and are fully retained in the JSON.

## Seven-run medians

Times are milliseconds except host selection, shown in microseconds.

| Context | Shape | Complete | Draft | Verify + readback | Selection µs | Recovery | Catch-up / settle |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4×1 | 56.153 | 0.001 | 54.781 | 7.542 | 0.001 | 1.273 |
| 2,048 | 4×2 | 202.141 | 4.235 | 192.979 | 8.334 | 2.948 | 1.816 |
| 2,048 | 4×3 | 211.226 | 8.581 | 197.415 | 8.333 | 2.975 | 2.175 |
| 2,048 | 4×4 | 219.317 | 13.063 | 201.370 | 6.917 | 2.953 | 2.157 |
| 2,048 | 4+4+1+1 | 209.342 | 11.611 | 194.125 | 5.333 | 1.598 | 1.855 |
| 2,048 | 4+1+1+1 | 201.465 | 8.458 | 190.343 | 7.458 | 0.904 | 1.652 |
| 2,048 | 1×4 | 66.854 | 8.613 | 56.591 | 3.917 | 0.953 | 0.803 |
| 2,048 | 2×4 | 210.419 | 11.784 | 195.616 | 6.209 | 1.602 | 1.308 |
| 2,048 | 3×4 | 218.472 | 15.486 | 198.993 | 7.292 | 2.295 | 1.860 |
| 16,384 | 4×1 | 65.262 | 0.001 | 64.008 | 5.166 | 0.001 | 1.433 |
| 16,384 | 4×2 | 234.207 | 4.933 | 224.340 | 6.042 | 2.982 | 2.038 |
| 16,384 | 4×3 | 244.056 | 9.623 | 229.338 | 6.250 | 2.957 | 2.060 |
| 16,384 | 4×4 | 261.576 | 14.438 | 242.059 | 6.541 | 2.961 | 2.198 |
| 16,384 | 4+4+1+1 | 239.287 | 12.359 | 223.720 | 8.375 | 1.614 | 1.823 |
| 16,384 | 4+1+1+1 | 225.020 | 8.912 | 213.399 | 7.625 | 0.934 | 1.609 |
| 16,384 | 1×4 | 77.956 | 8.898 | 67.660 | 4.334 | 0.925 | 0.620 |
| 16,384 | 2×4 | 234.065 | 12.420 | 218.406 | 6.500 | 1.640 | 1.190 |
| 16,384 | 3×4 | 251.104 | 16.917 | 229.707 | 8.041 | 2.277 | 1.560 |
| 65,536 | 4×1 | 91.662 | 0.001 | 90.402 | 6.250 | 0.001 | 1.381 |
| 65,536 | 4×2 | 276.316 | 6.361 | 264.993 | 8.250 | 3.008 | 1.924 |
| 65,536 | 4×3 | 315.840 | 12.678 | 298.159 | 8.875 | 2.992 | 2.118 |
| 65,536 | 4×4 | 354.661 | 18.931 | 330.690 | 5.750 | 3.012 | 2.123 |
| 65,536 | 4+4+1+1 | 299.721 | 14.905 | 280.829 | 8.125 | 1.686 | 1.860 |
| 65,536 | 4+1+1+1 | 270.370 | 10.231 | 257.768 | 8.167 | 1.034 | 1.615 |
| 65,536 | 1×4 | 103.962 | 10.203 | 92.475 | 4.666 | 0.957 | 0.431 |
| 65,536 | 2×4 | 288.753 | 15.416 | 270.342 | 6.792 | 1.686 | 0.861 |
| 65,536 | 3×4 | 329.681 | 20.763 | 304.776 | 7.625 | 2.457 | 1.728 |

The dominant cost is target verify and readback. At four active slots, increasing each window from one row to four rows raises the median complete round from 56.153 ms to 219.317 ms near 2k, 65.262 ms to 261.576 ms near 16k, and 91.662 ms to 354.661 ms near 64k. Host selection remains 3.9–8.9 µs across the grid and is not the limiting stage.

## Reproduction

From the published experiment branch:

```sh
HF2Q_QWEN38_GGUF=/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf \
HF2Q_ADR065_E_Q5_CASE=/opt/hf2q/scripts/four_agent/q5_cases/qwen38-long-context.json \
HF2Q_ADR065_E_OUTPUT="$E2/full/report.json" \
HF2Q_ADR065_E_GRID=full \
CARGO_TARGET_DIR=/Users/robert/.cache/hf2q/target-adr065-e7-rust199 \
scripts/run_adr065_joint_verify_diagnostic.sh
```

The run completed the ignored real-model driver in 75.38 seconds. The report's embedded source commit, Q5 case hash, grid mode and repetition identities are validated again by `scripts/four_agent/adr065_g0.py` before the G0 computation.
