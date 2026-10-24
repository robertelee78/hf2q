# Q5 FA prefill constructor plan

This HF-only slice extracts the existing seven BF16 FaPrefillArena requests into
a pure checked FaPrefillPlan consumed by the actual constructor. Original zero
dimension then GQA validation, allocation order, dtypes, seq/head-major shapes,
allocation-error labels and arena capacity fields remain unchanged. validate_fits
and all existing hardware tests remain untouched. No native API/dependency,
allocator, kernel, pool, policy or resource-lifetime change is selected.

The plan checks shape products, BF16 bytes and aggregate requested-byte sum before
the first arena allocation. Malformed overflow now returns an error instead of
unchecked wrap/panic or partial allocation. All descriptor shapes are finite host
metadata. Actual allocation still calls device.alloc_buffer with the exact bytes;
no power-of-two bucket projection is introduced for this direct allocator.

The ordered roots are q_bf16_seq, q_bf16_hm, k_bf16_seq, k_bf16_hm,
v_bf16_seq, v_bf16_hm and out_bf16_hm. Three Q/output roots contain seq*Q*D BF16
elements; four K/V roots contain seq*KV*D. Total requested bytes are
2*seq*D*(3Q+4KV).

| Cached F1 geometry / transaction rows | Each Q/output root | Each K/V root | Seven-root bytes |
| --- | ---: | ---: | ---: |
| Qwen3.8 Q24/KV4/D256 /2048 |25,165,824|4,194,304|92,274,688|
| Qwen3.6 Q16/KV2/D256 /2048 |16,777,216|2,097,152|58,720,256|

These shapes come from the selected model metadata. The2048 rows describe the
existing four-slot prefill transaction quantum; they are not the262144-token
per-slot context. The generic constructor adds no such row/context/head-width or
alignment limit. Seq1 and irregular101-row plans remain valid. The unchanged
forward caller selects the arena only for seq_len>1 with a full-attention layer
and shares it across FA layers for that forward's established lifetime.

This subtotal excludes the F32 out_seq return, FA projection/prefix-dequant
buffers, DN/FFN/native chunk arenas, logits, MTP, retained view parents, other
in-flight work and host/Metal/framework/driver overhead. It is not physical peak,
an enforced memory cap, whole B, a preload admission result or full Q5 proof.
No model quality, performance or warmup policy is qualified here.

Seven pure tests assert both models' seven ordered roles/dtypes/shapes/totals,
seq1/101 exact direct bytes, original zero/GQA errors and ordering, lack of a new
head-width restriction, shape/byte overflow and aggregate overflow when individual
roots fit. Tests construct only small descriptor metadata; no file/model/device,
GPU lock or payload allocation. Existing hardware tests are separate and their
skip behavior cannot substitute for these pure controls.

Integration-owner validation passed: seven pure release-profile controls, the
locked all-target/all-feature check and locked production release build. The
author did not run builds or tests. Exact focused command:

```sh
cargo test --release --locked --bin hf2q fa_prefill_plan_ -- --nocapture --test-threads=1
```

All complete-ledger, selected-path high-water, observer, reserve and bounded real
stress requirements of ADR063 Q5 remain open. This is one shared-constructor
arithmetic prerequisite only.
