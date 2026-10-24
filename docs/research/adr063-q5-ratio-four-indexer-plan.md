# Q5 ratio-four indexer constructor plan

This slice exposes pure, checked shape/payload arithmetic from the existing
DeepSeek `RatioFourIndexerArena::new`. The allocator consumes the same ten
ordered descriptors through the existing phase-sensitive `forward_support::alloc`
helper. Labels, dtypes, shape axes, minimum-one output/workspace semantics,
allocation order, native dispatch and legacy pool selection remain unchanged
for valid inputs. No native dependency/API or model/context/chunk policy changes.

The plan computes requested bytes and separately projected legacy power-of-two
bucket bytes with checked shape products, byte products, bucket rounding and
aggregate sums. The optimized top-k workspace formula comes from published
mlx-native0.15.2's existing pure helper. Zero row/head/dimension and arithmetic
overflow now reject before any arena payload allocation; previously some invalid
inputs could allocate earlier roots before failing or reach division by zero.
The plan is not an enforced memory cap, physical allocation bound or full B.
Inactive phases allocate exact requested bytes rather than pooled buckets.

The source-bound F1 dimension fixture is index_heads64/index_dim128. Selected
context262144 gives at most65536 compressed positions at ratio4. For one
2048-row arena and512 completed groups, the ten requested roots total
1,179,910,144bytes: query67,108,864+33,554,432; two weight roots524,288each;
two projected compressor roots2,097,152each; two BF16 output roots131,072each;
and score/top-k workspaces536,870,912each. All endpoint sizes are powers of two.
The same rows/groups with32768 valid compressed positions is a distinct131072
context diagnostic, totaling643,039,232bytes; it does not change F1 context.

2048 is an existing aggregate cooperative transaction-row limit, not full model
context or four independent2048-row allocations. The generic constructor plan
adds no such new ceiling; callers select their existing routes and arguments.
Per-lane pool rounding, retained historical buckets, concurrent arenas, outputs,
submission-input ownership, graphs, workers and opaque driver/host costs remain
outside this constructor subtotal. No model fit, runtime admission, speed,
warmup-policy adoption or full Q5 qualification is claimed.

Seven pure tests cover all ordered F1 descriptors and exact totals, the separate
131072 diagnostic, zero completed groups, the native1024-block top-k boundary,
irregular bucket rounding, malformed dimensions and independent shape/byte/
bucket/aggregate overflow. Fixtures create config/shape metadata only; no file,
model, Metal device or payload is allocated. The constructor and descriptor
allocation path share the plan rather than retaining a duplicate shape table.

Integration-owner validation: all seven pure controls passed in the release
profile (no ignored tests, device or model); locked all-target/all-feature check
also passed, as did the locked production release build. The author did not
run builds or tests. Exact focused command:

```sh
cargo test --release --locked --bin hf2q indexer_plan_ -- --nocapture --test-threads=1
```

ADR063 Q5's complete prelaunch ledger and bounded real stress requirements remain
open. This is one constructor arithmetic prerequisite, preserving the selected
inference path while making its payload projection concrete and reviewable.
