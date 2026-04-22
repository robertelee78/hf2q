# TQ C0b Localize — Dequant Diff Summary

Date: 2026-04-21  |  CFA session: cfa-20260421-C0b-localize  |  Worker 2

## Verdict: E1

All nrmse <= 0.15 and max_abs_diff <= 1.0: packed cache dequantizes within kernel bound. Bug NOT in H3 (encode/cache). Downstream: H1 kernel / H2 FWHT / H4 dispatch.

## Per-layer x op worst-case

| layer | op | max_nrmse | worst_head | worst_pos | max_abs_diff | nrmse_violations |
|------:|---:|----------:|-----------:|----------:|-------------:|-----------------:|
| 0 | k | 0.138983 | 1 | 20 | 0.048441 | 0 |
| 0 | v | 0.125411 | 0 | 21 | 0.408404 | 0 |
| 5 | k | 0.120948 | 1 | 4 | 0.030501 | 0 |
| 5 | v | 0.116922 | 1 | 4 | 0.444982 | 0 |

## Worst-case cell (nrmse)

layer=0 op=k head=1 pos=20 nrmse=0.138983 max_abs_diff=0.048014

## Worst-case cell (max_abs_diff)

layer=5 op=v head=0 pos=12 nrmse=0.099221 max_abs_diff=0.444982

## Total cells with nrmse > 0.15: 0

## Kernel bound status

nrmse bound (< 0.15) holds: YES
max_abs_diff bound (< 1.0) holds: YES
