# ADR-030 Phase 4 — Target integration plan

**Date**: 2026-05-13 (iter-36)
**Status**: planning artifact, no code change yet
**Audience**: this is a planning doc to lock in the integration shape
before modifying `forward_mlx.rs` / `forward_prefill_batched.rs`.

## The remaining Phase 4 work

To get end-to-end DFlash spec-decode running, we need:

1. **Target hidden-state capture** at `target_layer_ids = [1, 6, 11, 17, 22, 27]`
   during the target's verify forward pass. The captured hidden states are
   the input to `dispatch_dflash_fc` (the drafter's `target_hidden_concat`
   argument).

2. **Per-position argmax emission** from the target's verify forward.
   Currently `forward_prefill_batched` returns only the LAST-row argmax;
   we need all `seq_len` positions for `accept_prefix_argmax`.

3. **`forward_decode_verify_batched` body replacement** at
   `forward_prefill_batched.rs:2665`. Currently a temporary delegation
   to `forward_decode_verify_serial`. The replacement calls the modified
   `forward_prefill_batched` (with capture + per-position argmax) +
   returns both outputs.

4. **Target KV rollback** after spec-decode round: existing
   `MlxModelWeights::rollback_kv` (`forward_mlx.rs:5733`) handles this —
   just needs to be called by the orchestrator with `K - accept_count`.

5. **Embed lookup + lm_head matmul** wrappers callable from the orchestrator
   without re-entering the full `forward_prefill_batched` body. The drafter
   needs embedded tokens as input (`h` in `dispatch_dflash_model_forward`)
   and target's lm_head applied to its output.

## Proposed API shape

### 1. Modify `forward_prefill_batched` signature

```rust
// Before (iter-138, current production):
pub fn forward_prefill_batched(
    &mut self,
    prompt_tokens: &[u32],
    max_decode_tokens: usize,
    start_pos: usize,
    gpu: &mut GpuContext,
) -> Result<u32>;

// After (Phase 4 iter-X):
pub fn forward_prefill_batched(
    &mut self,
    prompt_tokens: &[u32],
    max_decode_tokens: usize,
    start_pos: usize,
    gpu: &mut GpuContext,
    capture: Option<&mut PrefillCapture>,
) -> Result<PrefillResult>;

pub struct PrefillCapture<'a> {
    /// Layer indices at which to capture pf_hidden. Indices outside
    /// this set are not captured.
    pub target_layer_ids: &'a [usize],
    /// Output buffer, layout [num_capture_layers, seq_len, hidden_size]
    /// F32 row-major. Pre-allocated by caller.
    pub hidden_output: &'a mut [f32],
    /// Per-position argmax output. Length seq_len. Pre-allocated.
    /// When None, function preserves legacy single-argmax-return behavior.
    pub per_position_argmaxes: Option<&'a mut [u32]>,
}

pub struct PrefillResult {
    /// Last-row argmax (= legacy return value). Always emitted.
    pub last_argmax: u32,
    /// True if capture was performed and per_position_argmaxes was
    /// populated.
    pub did_capture: bool,
}
```

**Compatibility**: production callers pass `None` and read
`PrefillResult.last_argmax`. Byte-identical to legacy behavior.

### 2. Hook point for hidden capture

End of layer loop iteration, after `pf_hidden` has been written with
the layer's output. Approximate line: TBD after closer reading of the
layer-loop body (line 765-end-of-iter). Pseudocode:

```rust
for (layer_idx, layer) in self.layers.iter().enumerate() {
    // ... existing layer body (norms, QKV, attn, MLP, residuals) ...
    // ... after pf_hidden is written with layer output ...

    if let Some(cap) = capture.as_deref_mut() {
        if let Some(target_idx) = cap.target_layer_ids.iter().position(|&i| i == layer_idx) {
            let pf_data: &[f32] = pf_hidden.as_slice()
                .map_err(|e| anyhow!("capture pf_hidden L{layer_idx}: {e}"))?;
            let dst_start = target_idx * seq_len * hs;
            let dst_end = dst_start + seq_len * hs;
            cap.hidden_output[dst_start..dst_end].copy_from_slice(pf_data);
        }
    }
}
```

### 3. Per-position argmax emission — REFINED iter-45

**Current tail** (forward_prefill_batched.rs:2245-2382):
1. Copy last row of pf_hidden → activations.hidden
2. final_norm: activations.hidden → activations.norm_out
3. lm_head matmul (Q6K/Q8/F16 path): norm_out × lm_head → activations.logits
4. softcap on activations.logits in-place
5. argmax: activations.logits → activations.argmax_index, argmax_value
6. s.finish() commits the Metal session
7. first_token = argmax_index[0]

**Phase 4 modification** (~80 LOC):

After the legacy `first_token` computation (existing path preserved
byte-identical), check if `self.dflash_capture` requires per-position
argmaxes. If yes, loop pos = 0..(seq_len-1):

```rust
if let Some(cap) = self.dflash_capture.as_ref() {
    if cap.per_position_argmaxes.is_some() {
        // Pre-compute all argmaxes into a local Vec to avoid
        // borrow-checker conflicts with self.activations + self.dflash_capture
        let mut local_argmaxes: Vec<u32> = vec![0; seq_len];
        local_argmaxes[seq_len - 1] = first_token; // already done

        for pos in 0..(seq_len - 1) {
            let mut s = exec.begin()?;
            // 1. copy row `pos` of pf_hidden to activations.hidden
            mlx_native::ops::copy::dispatch_copy_f32(
                s.encoder_mut(), reg, metal_dev,
                &pf_hidden, &self.activations.hidden,
                pos * hs, 0, hs,
            )?;
            // 2-5. final_norm + lm_head + softcap + argmax (same as
            //      the legacy block above; refactor into helper to
            //      avoid duplication)
            s.finish()?;
            let argmax_val: u32 = self.activations.argmax_index.as_slice()?[0];
            local_argmaxes[pos] = argmax_val;
        }

        // Now write back to capture (single mut borrow on dflash_capture)
        if let Some(cap) = self.dflash_capture.as_mut() {
            if let Some(pa) = cap.per_position_argmaxes.as_mut() {
                pa.copy_from_slice(&local_argmaxes);
            }
        }
    }
}
```

**Cost**: seq_len × (1 copy + 1 norm + 1 matmul + 1 softcap + 1
argmax) Metal dispatches. For seq_len=8, ~40 dispatches in 8 sessions
= ~5× the existing tail cost. Total forward is unchanged.

**Borrow-checker pattern**: use a local Vec to buffer argmaxes,
write back after the loop with a single mut borrow on dflash_capture.
Inside the loop we only need an immutable .is_some() check.

**Helper extraction recommended** (~30 LOC refactor of existing
tail): factor lines 2245-2375 into `fn argmax_for_row(&mut self,
exec, reg, metal_dev, pf_hidden, pos, hs) -> Result<u32>`. Call it
once at pos=seq_len-1 for legacy, then in the loop for capture.

### 4. Embed + lm_head wrappers

These already exist as internal helpers. We just need to expose them
as `pub fn` for orchestrator use:
- `pub fn dispatch_embed_tokens(...)` — looks up embed_tokens[token_id]
- `pub fn dispatch_target_lm_head(...)` — applies lm_head + softcap

Both can be thin wrappers around the existing internal dispatches in
forward_prefill_batched's tail.

### 5. `forward_decode_verify_batched` replacement

```rust
pub fn forward_decode_verify_batched(
    &mut self,
    tokens: &[u32],
    start_seq_pos: usize,
    gpu: &mut GpuContext,
    capture: Option<&mut PrefillCapture>,
) -> Result<Vec<u32>> {
    // No longer a delegation to serial. Real batched body:
    let mut argmaxes = vec![0u32; tokens.len()];
    let mut cap = capture.unwrap_or_else(...).extend_or_set_per_position(&mut argmaxes);
    let _ = self.forward_prefill_batched(
        tokens, 0 /* no further decode */, start_seq_pos, gpu, Some(&mut cap),
    )?;
    Ok(argmaxes)
}
```

## Risk + mitigation

| Concern | Risk | Mitigation |
|---|---|---|
| 4 production call sites need update | Low | Pass `None` for `capture`, read `PrefillResult.last_argmax`. Mechanical |
| Per-position argmax breaks legacy callers | Low | Only runs when `capture.per_position_argmaxes` is Some |
| Capture buffer layout mismatch with drafter | Medium | Strict shape check at orchestrator entry; assert pf_input_dim = num_capture_layers × hidden_size |
| Captured hidden differs from Python `model._hidden_states` | High — coherence-blocking | Phase 4 parity test: capture from hf2q + Python, compare element-wise within bf16 tolerance |
| `forward_decode_verify_batched` body change breaks ADR-028 iter-139 callers | Low (none in production yet) | Existing function is dead-code-flagged (no callers); behavior was "serial via fallback" anyway |

## Phase 4 milestone gate

After this integration, run:

```bash
HF2Q_SPEC_DFLASH=1 hf2q generate --temp 0 --model gemma-4-26b-a4b-it-ara-abliterated.gguf \
  --prompt "How many positive whole-number divisors does 196 have?"
```

Compare output byte-for-byte against `HF2Q_SPEC_DFLASH=0` at temp=0.
**Byte-identity required** — any divergence fails the coherence gate.

Then run `scripts/coherence-harness/coherence_bench.sh` on all 18 golden
fixtures + 100 random prompts.

## Order of operations (proposed)

1. (1 iter) Add `pub fn` wrappers for embed_tokens + lm_head in
   `forward_mlx.rs`. No layer-body changes.
2. (1-2 iters) Add `Option<PrefillCapture>` parameter + capture hook at
   end-of-layer-iter in `forward_prefill_batched.rs`. Update 4 production
   call sites to pass None. Test that build + existing test suite still
   pass byte-for-byte.
3. (1 iter) Add per-position argmax emission in the tail. Smoke test
   verifies argmaxes match `forward_decode_verify_serial` byte-for-byte
   at K=0.
4. (1-2 iters) Replace `forward_decode_verify_batched` body.
5. (1 iter) Wire orchestrator end-to-end: drafter forward + target
   verify + accept_prefix + KV rollback. Smoke test on one prompt.
6. (1-2 iters) Run coherence gate + golden harness. Debug any
   divergences.
7. (1 iter) Run perf gate vs hf2q baseline.

Total: ~7-10 iters to complete Phase 4. Per the per-iter cadence
established (~5 min wall, ~150-300 LOC per iter for non-monolith work,
slower for monolith touches) this is 1-3 hours of remaining work for
Phase 4.
