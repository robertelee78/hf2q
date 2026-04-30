# ADR-017 Phase A0 — falsification-harness results

**Status:** Phase A0.1 substrate. Per-cell numbers below are NaN where the matrix is not yet executed; A0.2/A0.3 land the measurement passes.

## Reproducer

```bash
# 1. Pre-bench process audit (fail if mcp-brain-server / llama-server / ollama running)
ps -Ao comm,pid,%cpu | grep -E 'mcp-brain-server|llama-server|ollama' || echo OK

# 2. Run matrix on M5 Max (clean SoC required)
HF2Q_KV_PERSIST_E2E=1 \
  cargo test --release --test kv_persist_harness \
  -- --test-threads=1 --nocapture kv_persist_matrix_e2e
```

## Thermal-state log

Phase A0.2 fills this section with `pmset -g thermlog` excerpt + skin-temperature notation per `feedback_perf_gate_thermal_methodology` (cold SoC for perf, parity second).

## mcp-brain-server STOP/RESUME log

Per `feedback_bench_process_audit`: the operator records the `kill -STOP $(pgrep mcp-brain-server)` PID and the `kill -CONT $PID` time here. Phase A0.1 substrate emits the placeholder; A0.2 fills.

## Per-cell results

| Cell label | Ran | no_cache TTFT (ms) | cache_hit TTFT (ms) | pre_evict (ms) | insert (ms) | load (ms) | kv_sha256 pre/post match | note |
|---|---|---|---|---|---|---|---|---|
| Gemma4_26b/Q4_0/Dense/L0/Miss/ColdResume | yes | NaN | NaN | 0.000 | NaN | NaN | match | subprocess_driver error: transport: error sending request for url (http://127.0.0.1:52338/v1/chat/completions) |
| Gemma4_26b/Q4_0/Dense/L0/Miss/SharedPrefix4Agents | yes | 236.536 | 236.544 | 0.000 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L512/Miss/ColdResume | yes | NaN | NaN | 8.018 | NaN | NaN | match | subprocess_driver error: transport: error sending request for url (http://127.0.0.1:52338/v1/chat/completions) |
| Gemma4_26b/Q4_0/Dense/L512/Miss/SharedPrefix4Agents | yes | 235.659 | 125.430 | 7.953 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L2048/Miss/ColdResume | yes | 242.675 | 60.470 | 29.442 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L2048/Miss/SharedPrefix4Agents | yes | NaN | NaN | 28.036 | NaN | NaN | match | subprocess_driver error: transport: request or response body error |
| Gemma4_26b/Q4_0/Dense/L8192/Miss/ColdResume | yes | 236.446 | 126.280 | 221.238 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L8192/Miss/SharedPrefix4Agents | yes | NaN | NaN | 118.628 | NaN | NaN | match | subprocess_driver error: transport: request or response body error |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/ColdResume | yes | 255.331 | 477.907 | 627.999 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/SharedPrefix4Agents | yes | 236.094 | 474.677 | 514.852 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/SwapBackInSameCtx | yes | NaN | NaN | 636.712 | NaN | NaN | match | subprocess_driver error: http 500: {"error":{"message":"Generation failed: model load failed: Failed to parse config.json","type":"server_error","param":null,"code":"generation_error"}} |
| Gemma4_26b/Q4_0/Dense/L32768/SsdColdPostRestart/ColdResume | yes | 234.801 | 489.006 | 811.911 | NaN | NaN | match | subprocess driven; tokens_total=16, log_lines=29 |

## Summary

- **yes:** 12 cell(s)
- **Total cells:** 12
- **KV cache format version:** 1
- **Block size:** 256 tokens
