# HF2Q environment surface inventory

This is the source-bound lexical inventory accepted by ADR-050 on 2026-08-23.
It includes names in production code, embedded test modules, comments, and
regression guards. A name appearing here is not automatically a supported
operator setting.

Regenerate the table with:

```bash
bash scripts/audit_hf2q_env_surface.sh
```

The snapshot contains **356 distinct names** after the ADR-050 removals.
The pre-change audit contained 363. Dispositions mean:

- `appropriate-env`: environment is an intentional boundary for a secret,
  XDG-like location, or shell/package integration.
- `promote-or-internalize`: normal or qualified production behavior must move
  to typed CLI/config or an internal family policy; it must not remain hidden
  shell UX.
- `documented-escape`: a safe, technical algorithm escape hatch that is not
  part of the ordinary setup path.
- `development-only`: benchmark, profile, dump, test, fixture, fault-injection,
  unsafe experiment, or unfinished feature control. These do not warrant CLI
  expansion while they remain development surfaces.
- `removed-guard`: only a regression assertion names the removed production
  environment reader.

## Completed in ADR-050

The production readers for scheduler, max slots, shared KV budget, persistent
KV disk budget, DeepSeek context, and the three server behavior defaults were
removed. `HF2Q_KV_PERSIST=0` also no longer defeats an explicit typed serve
path. Their supported replacements are typed CLI/config fields. The late
process mutation behind `generate --kv-bits` was replaced with a typed
in-process override; its legacy environment input remains development-only.

## Remaining promotion queue

The inventory deliberately identifies follow-up work without smuggling it into
this release:

1. Finish unifying development-generation persistent-KV activation, cache
   management root selection, LCP memory budget, and writer policy into typed
   persistence config. Serve activation/path and the on-disk budget are typed,
   complete, and fail-loud in this release.
2. Promote the multi-model pool byte ceiling to typed config.
3. Replace hidden batching ceilings and cross-slot collection controls with
   qualified scheduler/admission policy. Ordinary operators should not need
   `HF2Q_MAX_BATCHED_SLOTS`, its legacy spec-decode alias, or launcher-only
   admission variables.
4. Make required family policies—TQ KV, hybrid KV, encoder sessions, terminal-K
   batching, qualified LM-head routing, and proven batched serving—qualified
   typed defaults. Keep only explicitly unsafe diagnostic opt-outs.
5. Move Qwen speculation and process-global decode routing into typed backend
   policy. The current load-time environment mutation is order-dependent under
   multi-model serving and cannot be the final architecture.

## Complete snapshot

| Name | Disposition | Source occurrences |
|---|---|---:|
| `HF2Q_A4_INFLECTION_BENCH` | `development-only` | 2 |
| `HF2Q_A4_MOE_AB_VALIDATION_E2E` | `development-only` | 3 |
| `HF2Q_ACTIVATION_SCHEMA` | `development-only` | 14 |
| `HF2Q_ADMIT_COALESCE_US` | `promote-or-internalize` | 2 |
| `HF2Q_AUTH_TOKEN` | `appropriate-env` | 7 |
| `HF2Q_B9_FORCE_SEQUENTIAL` | `development-only` | 2 |
| `HF2Q_BARRIER_NS` | `development-only` | 2 |
| `HF2Q_BATCHED_ATTNPRE` | `development-only` | 4 |
| `HF2Q_BATCHED_BENCH` | `development-only` | 3 |
| `HF2Q_BATCHED_BODY` | `development-only` | 11 |
| `HF2Q_BATCHED_DUMP` | `development-only` | 7 |
| `HF2Q_BATCHED_FLASH` | `development-only` | 6 |
| `HF2Q_BATCHED_KVENC` | `development-only` | 4 |
| `HF2Q_BATCHED_LAYER_SCAN` | `development-only` | 4 |
| `HF2Q_BATCHED_PREFILL` | `documented-escape` | 9 |
| `HF2Q_BATCHED_WSUM` | `development-only` | 2 |
| `HF2Q_BENCH_CONC` | `development-only` | 2 |
| `HF2Q_BENCH_N` | `development-only` | 3 |
| `HF2Q_BENCH_PROMPT_LEN` | `development-only` | 2 |
| `HF2Q_BENCH_REPEAT` | `development-only` | 2 |
| `HF2Q_BENCH_SETTLE_MS` | `development-only` | 2 |
| `HF2Q_BENCH_TOKENS` | `development-only` | 2 |
| `HF2Q_BENCH_TOKIO_THREADS` | `development-only` | 2 |
| `HF2Q_BINARY_NOT_RELEASE` | `development-only` | 6 |
| `HF2Q_BISECT_ALEN` | `development-only` | 2 |
| `HF2Q_BISECT_ASEED` | `development-only` | 2 |
| `HF2Q_BISECT_BLEN` | `development-only` | 2 |
| `HF2Q_BUILD_GIT_SHA` | `development-only` | 2 |
| `HF2Q_BYTE_EQUIV_E2E` | `development-only` | 40 |
| `HF2Q_BYTE_EQUIV_E2E_GGUF` | `development-only` | 30 |
| `HF2Q_C2C_E2E` | `development-only` | 14 |
| `HF2Q_C2C_E2E_GGUF` | `development-only` | 7 |
| `HF2Q_CACHE_DIR` | `appropriate-env` | 15 |
| `HF2Q_CACHE_DIR_ENV` | `development-only` | 2 |
| `HF2Q_CB_THROUGHPUT_MODEL` | `development-only` | 1 |
| `HF2Q_CHUNK_SCAN_PREFILL` | `development-only` | 20 |
| `HF2Q_CKSUM_PERSEQ` | `development-only` | 2 |
| `HF2Q_COMPLETE` | `development-only` | 11 |
| `HF2Q_COMPLETER_PATH_PLACEHOLDER__` | `development-only` | 3 |
| `HF2Q_COMPLETION_STARTUP_FILE` | `appropriate-env` | 1 |
| `HF2Q_COMPLETION_ZDOTDIR_PROBE` | `development-only` | 3 |
| `HF2Q_CROSS_SLOT_ADMIT` | `promote-or-internalize` | 9 |
| `HF2Q_DEBUG_TOKENIZE_ONLY` | `development-only` | 5 |
| `HF2Q_DEBUG_TQ_RMS` | `development-only` | 20 |
| `HF2Q_DECODE_CATSPLIT` | `development-only` | 6 |
| `HF2Q_DECODE_CB_CHUNKS` | `development-only` | 3 |
| `HF2Q_DECODE_EMIT` | `development-only` | 9 |
| `HF2Q_DECODE_EMIT_TOKENS` | `development-only` | 3 |
| `HF2Q_DECODE_INPUT_TOKENS` | `development-only` | 5 |
| `HF2Q_DECODE_MVN` | `promote-or-internalize` | 5 |
| `HF2Q_DECODE_MV_EXT` | `promote-or-internalize` | 5 |
| `HF2Q_DECODE_PROFILE` | `development-only` | 8 |
| `HF2Q_DECODE_SPLIT_CB_AT_LAYER` | `development-only` | 2 |
| `HF2Q_DECODE_TRACE` | `development-only` | 3 |
| `HF2Q_DEEPSEEK4_AGENTIC_CONTRACT_RECEIPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_AGENTIC_PROMPT_CONTRACT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_AGENTIC_REQUEST_JSON` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_COHORT_BENCH_PAIRS` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_COHORT_RECEIPT` | `development-only` | 2 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_PHASE_DIR` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_RECEIPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_DECODE_COHORT_RUN_UUID` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_GGUF` | `development-only` | 7 |
| `HF2Q_DEEPSEEK4_PHASE` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_RENDERED_PROMPT` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_REQUEST_JSON` | `development-only` | 1 |
| `HF2Q_DEEPSEEK4_TOKENIZER` | `development-only` | 8 |
| `HF2Q_DEEPSEEK4_TOKENIZER_JSON` | `development-only` | 2 |
| `HF2Q_DEEPSEEK_COMPRESSED_STAGE_PROFILE` | `development-only` | 4 |
| `HF2Q_DEEPSEEK_DUMP_ATTENTION_DIR` | `development-only` | 6 |
| `HF2Q_DEEPSEEK_DUMP_LAYER_DIR` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_ENCODER_STAGES` | `development-only` | 4 |
| `HF2Q_DEEPSEEK_GRAPH_DIAG` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_GRAPH_LAYERS_PER_CB` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_GRAPH_REORDER` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_LAYERS_PER_CB` | `development-only` | 5 |
| `HF2Q_DEEPSEEK_MMAP_WEIGHTS` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_PREFILL_TIMING` | `development-only` | 3 |
| `HF2Q_DEEPSEEK_PREFILL_WINDOWS` | `development-only` | 6 |
| `HF2Q_DEEPSEEK_SEQUENTIAL_PREFILL` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_SKIP_WARMUP` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_SLOT_DECODE_QUANTUM` | `development-only` | 1 |
| `HF2Q_DEEPSEEK_STAGE_PROFILE` | `development-only` | 5 |
| `HF2Q_DENSE_Q_ARENA_RESET` | `development-only` | 6 |
| `HF2Q_DENSE_Q_LEGACY` | `development-only` | 5 |
| `HF2Q_DFLASH_ACCEPT` | `development-only` | 2 |
| `HF2Q_DFLASH_BATCH_ARGMAX` | `development-only` | 2 |
| `HF2Q_DFLASH_BLOCK_SIZE` | `development-only` | 8 |
| `HF2Q_DFLASH_DRAFTER_DUMP` | `development-only` | 1 |
| `HF2Q_DFLASH_DRAFTER_PATH` | `development-only` | 9 |
| `HF2Q_DFLASH_HIDDEN_DEBUG` | `development-only` | 1 |
| `HF2Q_DFLASH_KV_CPU` | `development-only` | 1 |
| `HF2Q_DFLASH_PROFILE` | `development-only` | 6 |
| `HF2Q_DFLASH_SDPA_CPU` | `development-only` | 2 |
| `HF2Q_DFLASH_XLEN_BF16` | `development-only` | 1 |
| `HF2Q_DFLASH_XLEN_D512_OFF` | `development-only` | 2 |
| `HF2Q_DFLASH_XLEN_DEBUG` | `development-only` | 4 |
| `HF2Q_DFLASH_XLEN_SDPA` | `development-only` | 68 |
| `HF2Q_DISP_PROFILE` | `development-only` | 3 |
| `HF2Q_DUAL_BUFFER` | `development-only` | 4 |
| `HF2Q_DUMP_ALL_CACHE` | `development-only` | 10 |
| `HF2Q_DUMP_BOUNDARY` | `development-only` | 3 |
| `HF2Q_DUMP_CB_COUNT` | `development-only` | 2 |
| `HF2Q_DUMP_COUNTERS` | `development-only` | 7 |
| `HF2Q_DUMP_DIR` | `development-only` | 6 |
| `HF2Q_DUMP_FA_BF16` | `development-only` | 2 |
| `HF2Q_DUMP_FPRINT` | `development-only` | 2 |
| `HF2Q_DUMP_LAYER` | `development-only` | 32 |
| `HF2Q_DUMP_LAYERS` | `development-only` | 7 |
| `HF2Q_DUMP_LAYERS_LIST` | `development-only` | 3 |
| `HF2Q_DUMP_LAYER_ACTIVATIONS` | `development-only` | 5 |
| `HF2Q_DUMP_LAYER_ALL` | `development-only` | 2 |
| `HF2Q_DUMP_LAYER_DETAIL` | `development-only` | 3 |
| `HF2Q_DUMP_LAYER_N` | `development-only` | 5 |
| `HF2Q_DUMP_LOGITS` | `development-only` | 6 |
| `HF2Q_DUMP_NORM_WEIGHT` | `development-only` | 4 |
| `HF2Q_DUMP_PRE_QUANT` | `development-only` | 9 |
| `HF2Q_DUMP_PRE_QUANT_LAYERS` | `development-only` | 3 |
| `HF2Q_DUMP_PRE_QUANT_POSITIONS` | `development-only` | 3 |
| `HF2Q_DUMP_PROMPT_TOKENS` | `development-only` | 6 |
| `HF2Q_DUMP_RENDERED_PROMPT` | `development-only` | 10 |
| `HF2Q_DUMP_RUN_ID` | `development-only` | 2 |
| `HF2Q_DUMP_RUN_NAME` | `development-only` | 9 |
| `HF2Q_DUMP_SDPA_MAX_POS` | `development-only` | 7 |
| `HF2Q_DUMP_SLIDING_LAYER_0` | `development-only` | 9 |
| `HF2Q_DUMP_SLIDING_MASK` | `development-only` | 1 |
| `HF2Q_DUMP_TQ_STATE` | `development-only` | 4 |
| `HF2Q_EAGLE3_DRAFTER_PATH` | `development-only` | 5 |
| `HF2Q_EAGLE3_EQUIVALENCE_REPETITIONS` | `development-only` | 3 |
| `HF2Q_EAGLE3_TOP_K` | `development-only` | 2 |
| `HF2Q_EAGLE3_TREE_BUDGET` | `development-only` | 2 |
| `HF2Q_EAGLE3_TREE_MAX_DEPTH` | `development-only` | 2 |
| `HF2Q_EMIT_NLL` | `development-only` | 3 |
| `HF2Q_ENCODER_SESSION` | `promote-or-internalize` | 18 |
| `HF2Q_F16_KV` | `development-only` | 17 |
| `HF2Q_F16_SHADOW` | `development-only` | 8 |
| `HF2Q_F32_MATVEC` | `development-only` | 2 |
| `HF2Q_FALSIFIER_EVICT` | `development-only` | 2 |
| `HF2Q_FALSIFIER_STAGGER_MS` | `development-only` | 3 |
| `HF2Q_FA_F16` | `development-only` | 13 |
| `HF2Q_FA_LAYER_CKSUM` | `development-only` | 1 |
| `HF2Q_FA_PEER_PORT` | `development-only` | 7 |
| `HF2Q_FA_PEER_PORT_NWG32` | `development-only` | 3 |
| `HF2Q_FA_TRACE` | `development-only` | 2 |
| `HF2Q_FFN_POOLED_LEGACY` | `development-only` | 2 |
| `HF2Q_FFN_SPLIT` | `development-only` | 6 |
| `HF2Q_FFN_TERMINAL_K_BATCH` | `promote-or-internalize` | 2 |
| `HF2Q_FISH_COMPLETIONS_DIR` | `appropriate-env` | 2 |
| `HF2Q_FORCE_DENSE_SDPA_ON_TQ_KV` | `development-only` | 1 |
| `HF2Q_FULL_F16_KV` | `development-only` | 55 |
| `HF2Q_FUSED_END_OF_LAYER` | `development-only` | 7 |
| `HF2Q_FUSED_GATE_UP_SILU` | `development-only` | 4 |
| `HF2Q_FUSED_MOE_GATE_UP_MM_ID` | `development-only` | 3 |
| `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2` | `development-only` | 5 |
| `HF2Q_FUSED_QKVG` | `development-only` | 5 |
| `HF2Q_FUSED_TRIPLE_NORM` | `development-only` | 3 |
| `HF2Q_FUSE_LMHEAD` | `development-only` | 5 |
| `HF2Q_G4_TREE_VERIFY_NAN_DEBUG` | `development-only` | 2 |
| `HF2Q_GEMMA4_31B_DRAFTER` | `development-only` | 5 |
| `HF2Q_GEMMA4_31B_GGUF` | `development-only` | 8 |
| `HF2Q_GEMMA4_EAGLE3_MIN_ACCEPT` | `development-only` | 2 |
| `HF2Q_GEMMA_N8_EXPECTED_KV_REGIME` | `development-only` | 3 |
| `HF2Q_GEMMA_N8_PARITY_MAX_TOKENS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_PARITY_ROUNDS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_PREFILL_REPEATS` | `development-only` | 1 |
| `HF2Q_GEMMA_N8_RESUME_REPEATS` | `development-only` | 1 |
| `HF2Q_GEMMA_TILED_LIVE` | `development-only` | 1 |
| `HF2Q_GLOBAL_FA` | `development-only` | 7 |
| `HF2Q_GPU_BUSY` | `development-only` | 5 |
| `HF2Q_GPU_SAMPLE` | `development-only` | 4 |
| `HF2Q_GQA_EXPAND_LEGACY` | `development-only` | 5 |
| `HF2Q_GRAPH_OPT` | `development-only` | 6 |
| `HF2Q_GRAPH_OPT_PREFILL` | `development-only` | 3 |
| `HF2Q_GROUP_STATS` | `development-only` | 1 |
| `HF2Q_HB_DUAL_LEGACY` | `development-only` | 3 |
| `HF2Q_HOST_PHASES` | `development-only` | 6 |
| `HF2Q_HYBRID_KV` | `promote-or-internalize` | 122 |
| `HF2Q_HYBRID_NWG` | `development-only` | 2 |
| `HF2Q_IMATRIX_CPU_REF` | `development-only` | 4 |
| `HF2Q_IMATRIX_METAL_REF` | `development-only` | 4 |
| `HF2Q_IMATRIX_REAL_REF` | `development-only` | 4 |
| `HF2Q_ITERGA_KRUNS` | `development-only` | 2 |
| `HF2Q_ITERGA_N8` | `development-only` | 2 |
| `HF2Q_KV_DUAL_LEGACY` | `development-only` | 3 |
| `HF2Q_KV_LCP_CAPACITY` | `promote-or-internalize` | 1 |
| `HF2Q_KV_LCP_CHUNKED_PREFILL` | `promote-or-internalize` | 11 |
| `HF2Q_KV_LCP_DELTANET_CHECKPOINT_STRIDE` | `development-only` | 2 |
| `HF2Q_KV_LCP_DISABLE_MID_STORE` | `development-only` | 4 |
| `HF2Q_KV_LCP_LONG_RESUME` | `promote-or-internalize` | 5 |
| `HF2Q_KV_LCP_RESUME` | `promote-or-internalize` | 62 |
| `HF2Q_KV_LCP_RESUME_CAPACITY` | `promote-or-internalize` | 11 |
| `HF2Q_KV_PERSIST` | `promote-or-internalize` | 19 |
| `HF2Q_KV_PERSIST_PATH` | `promote-or-internalize` | 15 |
| `HF2Q_KV_PERSIST_PATH_ENV` | `development-only` | 9 |
| `HF2Q_KV_WRITER_CAPACITY` | `development-only` | 1 |
| `HF2Q_LAYER_POLICY` | `development-only` | 20 |
| `HF2Q_LEGACY_PER_LAYER_CB` | `development-only` | 5 |
| `HF2Q_LEGACY_TQ_SDPA` | `development-only` | 1 |
| `HF2Q_LMHEAD_COMPARE` | `development-only` | 3 |
| `HF2Q_LMHEAD_Q6K` | `promote-or-internalize` | 9 |
| `HF2Q_LMHEAD_Q8` | `documented-escape` | 9 |
| `HF2Q_LMHEAD_RERANK` | `development-only` | 8 |
| `HF2Q_LOAD_TIMING` | `development-only` | 5 |
| `HF2Q_MAX_BATCHED_SLOTS` | `promote-or-internalize` | 12 |
| `HF2Q_MAX_SLOTS` | `removed-guard` | 1 |
| `HF2Q_METAL_CAPTURE` | `development-only` | 5 |
| `HF2Q_METAL_CAPTURE_LAYERS` | `development-only` | 3 |
| `HF2Q_MLX_KERNEL_PROFILE` | `development-only` | 4 |
| `HF2Q_MLX_PROFILE` | `development-only` | 4 |
| `HF2Q_MLX_TIMING` | `development-only` | 3 |
| `HF2Q_MM_ID_ROUTING_THRESHOLD` | `development-only` | 3 |
| `HF2Q_MTP_PHASE_PROFILE` | `development-only` | 1 |
| `HF2Q_MTP_PROFILE` | `development-only` | 5 |
| `HF2Q_MVN_BARRIER_TRACE` | `development-only` | 1 |
| `HF2Q_MVN_ENCODE_TRACE` | `development-only` | 3 |
| `HF2Q_NETWORK_TESTS` | `development-only` | 7 |
| `HF2Q_NLL` | `development-only` | 8 |
| `HF2Q_NO_COMPLETION_INSTALL` | `appropriate-env` | 5 |
| `HF2Q_NO_FA` | `development-only` | 16 |
| `HF2Q_NO_FUSED_STAGE_AB_VEC` | `development-only` | 3 |
| `HF2Q_NO_RESIDENCY` | `development-only` | 7 |
| `HF2Q_NO_VEC_SMALL_PATH` | `development-only` | 2 |
| `HF2Q_PARALLEL_ENCODE` | `development-only` | 24 |
| `HF2Q_PARALLEL_ENCODE_KV_THRESHOLD` | `development-only` | 2 |
| `HF2Q_PARALLEL_PROFILE` | `development-only` | 2 |
| `HF2Q_PARTIAL_CHAIN_LEGACY` | `development-only` | 3 |
| `HF2Q_PARTIAL_CHAIN_N` | `development-only` | 6 |
| `HF2Q_PER_LAYER_DISP` | `development-only` | 15 |
| `HF2Q_PER_LAYER_GPU_TIME` | `development-only` | 2 |
| `HF2Q_PER_LAYER_PHASE_GPU_TIME` | `development-only` | 4 |
| `HF2Q_PIPELINE_PREWARM` | `development-only` | 3 |
| `HF2Q_PIPELINE_PREWARM_LOG` | `development-only` | 1 |
| `HF2Q_POOL_BUDGET_BYTES` | `promote-or-internalize` | 3 |
| `HF2Q_PREFILL_CROSS_SLOT` | `promote-or-internalize` | 1 |
| `HF2Q_PREFILL_DUMP` | `development-only` | 4 |
| `HF2Q_PREFILL_SLOT_BATCHED` | `promote-or-internalize` | 4 |
| `HF2Q_PREFILL_TIMING` | `development-only` | 2 |
| `HF2Q_PROFILE_BUCKETS` | `development-only` | 10 |
| `HF2Q_PROFILE_DENSE_Q_SPLIT_COMMITS` | `development-only` | 5 |
| `HF2Q_PROFILE_FA` | `development-only` | 5 |
| `HF2Q_PROFILE_GPU_TS` | `development-only` | 4 |
| `HF2Q_PROFILE_LAYERS` | `development-only` | 5 |
| `HF2Q_PROFILE_MM` | `development-only` | 9 |
| `HF2Q_PROFILE_MOE` | `development-only` | 7 |
| `HF2Q_PROFILE_MOE_POST` | `development-only` | 2 |
| `HF2Q_PROFILE_RESIDENCY_ABORT` | `development-only` | 1 |
| `HF2Q_PROFILE_SYNC` | `development-only` | 2 |
| `HF2Q_PROFILE_W5B17` | `development-only` | 5 |
| `HF2Q_PROFILE_W5B22` | `development-only` | 5 |
| `HF2Q_PROFILE_W5B8` | `development-only` | 19 |
| `HF2Q_Q6K_ID_MV_NR2` | `development-only` | 2 |
| `HF2Q_Q6K_MV_NR2` | `development-only` | 3 |
| `HF2Q_Q8_0_ID_MV_NR2` | `development-only` | 1 |
| `HF2Q_QKV_SPLIT_LEGACY` | `development-only` | 2 |
| `HF2Q_QWEN35_DROP_MTP` | `development-only` | 8 |
| `HF2Q_QWEN35_E2E_GGUF` | `development-only` | 7 |
| `HF2Q_QWEN35_FA_LEGACY` | `development-only` | 2 |
| `HF2Q_QWEN35_PREFILL_SWEEP` | `development-only` | 9 |
| `HF2Q_QWEN35_PREFILL_SWEEP_COMPARE_FULL_LAST` | `development-only` | 1 |
| `HF2Q_QWEN35_PREFILL_SWEEP_FULL_LOGITS` | `development-only` | 1 |
| `HF2Q_QWEN35_PREFILL_SWEEP_TRIALS` | `development-only` | 2 |
| `HF2Q_QWEN35_PREFILL_SWEEP_WARMUPS` | `development-only` | 2 |
| `HF2Q_QWEN35_TOKENIZER` | `development-only` | 3 |
| `HF2Q_QWEN36_AUTOREG` | `development-only` | 2 |
| `HF2Q_QWEN36_WATCHDOG_FIXTURE_MODEL` | `development-only` | 2 |
| `HF2Q_QWEN36_WATCHDOG_FIXTURE_OUTPUT` | `development-only` | 1 |
| `HF2Q_QWEN36_WATCHDOG_SHORT_FIXTURE_OUTPUT` | `development-only` | 1 |
| `HF2Q_QWEN3VL_E2E` | `development-only` | 1 |
| `HF2Q_QWEN3VL_LM_LOAD` | `development-only` | 9 |
| `HF2Q_QWEN_GQA_Q2` | `documented-escape` | 2 |
| `HF2Q_QWEN_SPECULATION` | `promote-or-internalize` | 6 |
| `HF2Q_QWEN_VISION_MMPROJ` | `development-only` | 3 |
| `HF2Q_RERANK_PROFILE` | `development-only` | 5 |
| `HF2Q_RMS_NORM_V2` | `development-only` | 3 |
| `HF2Q_RUNTIME_SCHEMA` | `development-only` | 2 |
| `HF2Q_S019_CKSUM` | `development-only` | 11 |
| `HF2Q_S019_MAXTOK` | `development-only` | 2 |
| `HF2Q_S019_NO_REMASK` | `development-only` | 4 |
| `HF2Q_S019_NSEQ` | `development-only` | 2 |
| `HF2Q_S019_PROMPT_LEN` | `development-only` | 2 |
| `HF2Q_S019_REPEATS` | `development-only` | 2 |
| `HF2Q_S2C` | `development-only` | 1 |
| `HF2Q_SCALE_FORMULA` | `development-only` | 9 |
| `HF2Q_SCHEDULER` | `removed-guard` | 1 |
| `HF2Q_SERVE_BATCHED` | `promote-or-internalize` | 13 |
| `HF2Q_SERVE_BATCHED_` | `development-only` | 12 |
| `HF2Q_SERVE_BATCHED_PREFILL` | `promote-or-internalize` | 11 |
| `HF2Q_SETUP_ABORT_AT` | `development-only` | 2 |
| `HF2Q_SETUP_CRASH_CHILD` | `development-only` | 1 |
| `HF2Q_SETUP_CRASH_ROOT` | `development-only` | 2 |
| `HF2Q_SETUP_UMASK_CHILD` | `development-only` | 1 |
| `HF2Q_SETUP_UMASK_ROOT` | `development-only` | 2 |
| `HF2Q_SKIP_ATTN_QKV` | `development-only` | 2 |
| `HF2Q_SKIP_DENSE_MLP` | `development-only` | 2 |
| `HF2Q_SKIP_EMBED` | `development-only` | 1 |
| `HF2Q_SKIP_END_OF_LAYER` | `development-only` | 4 |
| `HF2Q_SKIP_END_OF_LAYER_FINAL` | `development-only` | 2 |
| `HF2Q_SKIP_HEAD_NORM_ROPE` | `development-only` | 2 |
| `HF2Q_SKIP_MMPROJ_LOAD` | `development-only` | 4 |
| `HF2Q_SKIP_MOE_EXPERTS` | `development-only` | 2 |
| `HF2Q_SKIP_MOE_SWIGLU` | `development-only` | 2 |
| `HF2Q_SKIP_O_PROJ` | `development-only` | 2 |
| `HF2Q_SKIP_POST_ATTN_NORM` | `development-only` | 2 |
| `HF2Q_SKIP_ROUTING` | `development-only` | 2 |
| `HF2Q_SKIP_TQ_ENCODE` | `development-only` | 5 |
| `HF2Q_SKIP_TQ_SDPA` | `development-only` | 4 |
| `HF2Q_SKIP_VIT_WARMUP` | `development-only` | 4 |
| `HF2Q_SKIP_V_NORM` | `development-only` | 2 |
| `HF2Q_SKIP_WEIGHTED_SUM` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE` | `development-only` | 54 |
| `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED` | `development-only` | 14 |
| `HF2Q_SPEC_DECODE_K` | `development-only` | 16 |
| `HF2Q_SPEC_DECODE_K1` | `development-only` | 10 |
| `HF2Q_SPEC_DECODE_K1_NO_AMORT` | `development-only` | 3 |
| `HF2Q_SPEC_DECODE_K1_TRACE` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE_K1_TWO_CALLS` | `development-only` | 2 |
| `HF2Q_SPEC_DECODE_KN_HIDDEN_ROW_CAP` | `development-only` | 3 |
| `HF2Q_SPEC_DECODE_MAX_BATCHED_SLOTS` | `promote-or-internalize` | 18 |
| `HF2Q_SPEC_DFLASH` | `development-only` | 30 |
| `HF2Q_SPEC_DFLASH_BLOCK_SIZE` | `development-only` | 1 |
| `HF2Q_SPEC_DFLASH_PHASE` | `development-only` | 1 |
| `HF2Q_SPEC_EAGLE3` | `development-only` | 25 |
| `HF2Q_SPEC_NGRAM` | `development-only` | 22 |
| `HF2Q_SPEC_NGRAM_K` | `development-only` | 4 |
| `HF2Q_SPEC_NGRAM_MAX` | `development-only` | 3 |
| `HF2Q_SPEC_NGRAM_MIN` | `development-only` | 3 |
| `HF2Q_SPEC_NGRAM_PROFILE` | `development-only` | 4 |
| `HF2Q_SPLIT_POSTATTN_NORM` | `development-only` | 1 |
| `HF2Q_SPLIT_POSTFF_NORMADD` | `development-only` | 2 |
| `HF2Q_SPLIT_POSTFF_NORMADDSCALAR` | `development-only` | 1 |
| `HF2Q_SPLIT_TIMING` | `development-only` | 4 |
| `HF2Q_STEP_PROFILE` | `development-only` | 1 |
| `HF2Q_SYNC_PER_LAYER` | `development-only` | 5 |
| `HF2Q_TEST_DESCENDANT_PID_FILE` | `development-only` | 2 |
| `HF2Q_TEST_HELPER_PID_FILE` | `development-only` | 4 |
| `HF2Q_TEST_PROMPT` | `development-only` | 1 |
| `HF2Q_TEST_QWEN38_EXPECT_MV_EXT` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_EXPECT_Q4K_MVN` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_EXPECT_Q5_K_M` | `development-only` | 2 |
| `HF2Q_TEST_QWEN38_GGUF` | `development-only` | 3 |
| `HF2Q_TOKENIZER_GGUF_EMBEDDED` | `development-only` | 4 |
| `HF2Q_TQ_CODEBOOK_BITS` | `development-only` | 44 |
| `HF2Q_TQ_FAST_FUSED_KV` | `development-only` | 3 |
| `HF2Q_TQ_FUSE_FWHT_PRE` | `development-only` | 2 |
| `HF2Q_TQ_HB_OUT_FUSED` | `development-only` | 1 |
| `HF2Q_TQ_KV` | `promote-or-internalize` | 26 |
| `HF2Q_UNSAFE_EXPERIMENTS` | `development-only` | 37 |
| `HF2Q_USE_DENSE` | `development-only` | 79 |
| `HF2Q_VERIFIER_NBENCH` | `development-only` | 2 |
| `HF2Q_VIT_DUMP` | `development-only` | 23 |
| `HF2Q_VIT_DUMP_DTYPE_AUDIT` | `development-only` | 6 |
| `HF2Q_VIT_F32_ATTENTION` | `development-only` | 10 |
| `HF2Q_W5B4_DIVERGENCE` | `development-only` | 5 |
| `HF2Q_ZDOTDIR_V1` | `development-only` | 2 |
| `HF2Q_ZSH_COMPLETIONS_DIR` | `appropriate-env` | 2 |
| `HF2Q_ZSH_STARTUP_DIR` | `appropriate-env` | 1 |
