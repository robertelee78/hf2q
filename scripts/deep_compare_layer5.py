"""Deep comparison of layer 5 (first Global) intermediates between mlx-lm and hf2q.

Hooks into mlx-lm's DecoderLayer to capture every intermediate value,
then prints them for comparison against hf2q's DIAG output.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import inspect
from mlx_lm import load

MODEL_PATH = "/opt/hf2q/models/gemma4-mlx-auto"
PROMPT = "What is 2+2? Explain your reasoning step by step in great detail."
TARGET_LAYER = 5  # First Global layer


def fmt(arr, n=5):
    """Format first n elements of last token."""
    vals = arr[0, -1, :n].tolist() if len(arr.shape) == 3 else arr[-1, :n].tolist()
    return "[" + ", ".join(f"{v:.8f}" for v in vals) + "]"


def l2(arr):
    """L2 norm of last token."""
    last = arr[0, -1, :] if len(arr.shape) == 3 else arr[-1, :]
    return float(mx.sqrt(mx.sum(last ** 2)).item())


def run():
    model, tokenizer = load(MODEL_PATH)
    lm = model.language_model
    inner = lm.model

    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(prompt_str)
    input_ids = mx.array([tokens])
    print(f"Tokens ({len(tokens)}): {tokens[:10]}...")

    # Run forward through layers 0-4 to get input to layer 5
    h = inner.embed_tokens(input_ids)
    h = h * inner.embed_scale
    mx.eval(h)

    # Set up cache and masks
    cache = model.make_cache()
    cache = cache + [None] * (len(inner.layers) - len(cache))
    masks = inner._make_masks(h, cache)

    per_layer_inputs = [None] * len(inner.layers)

    intermediates_list = [(None, None)] * len(inner.layers)

    # Run layers 0-4
    for idx in range(TARGET_LAYER):
        kvs, offset = intermediates_list[inner.previous_kvs[idx]]
        h, kvs_out, offset_out = inner.layers[idx](
            h, masks[idx], cache[idx],
            per_layer_input=per_layer_inputs[idx],
            shared_kv=kvs, offset=offset,
        )
        mx.eval(h)
        intermediates_list[idx] = (kvs_out, offset_out)

    print(f"\n=== Input to Layer {TARGET_LAYER} (Global) ===")
    print(f"  L2={l2(h):.4f}  first5={fmt(h)}")

    # Now manually step through layer 5
    layer = inner.layers[TARGET_LAYER]
    attn = layer.self_attn
    mask = masks[TARGET_LAYER]
    c = cache[TARGET_LAYER]
    kvs, offset = intermediates_list[inner.previous_kvs[TARGET_LAYER]]

    B, L, D = h.shape
    residual = h

    # Step 1: Input layernorm
    h_normed = layer.input_layernorm(h)
    mx.eval(h_normed)
    print(f"\n=== Layer {TARGET_LAYER} Intermediates ===")
    print(f"1. input_layernorm:  L2={l2(h_normed):.4f}  {fmt(h_normed)}")

    # Step 2: Q projection
    q = attn.q_proj(h_normed)
    mx.eval(q)
    q = q.reshape(B, L, attn.n_heads, attn.head_dim)
    mx.eval(q)
    print(f"2. Q proj (reshaped): L2={l2(q.reshape(B,L,-1)):.4f}  {fmt(q.reshape(B,L,-1))}")

    # Step 3: Q norm
    q = attn.q_norm(q)
    mx.eval(q)
    print(f"3. Q norm:           L2={l2(q.reshape(B,L,-1)):.4f}  {fmt(q.reshape(B,L,-1))}")

    # Step 4: K projection (and V = K for k_eq_v)
    k = attn.k_proj(h_normed)
    mx.eval(k)
    k = k.reshape(B, L, attn.n_kv_heads, attn.head_dim)
    mx.eval(k)
    print(f"4. K proj (reshaped): L2={l2(k.reshape(B,L,-1)):.4f}  {fmt(k.reshape(B,L,-1))}")

    # V = K before norms (k_eq_v)
    values = k  # same tensor

    # Step 5: K norm
    k_normed = attn.k_norm(k)
    mx.eval(k_normed)
    print(f"5. K norm:           L2={l2(k_normed.reshape(B,L,-1)):.4f}  {fmt(k_normed.reshape(B,L,-1))}")

    # Step 6: V norm (on original k, before k_norm)
    v_normed = attn.v_norm(values)
    mx.eval(v_normed)
    print(f"6. V norm (from K):  L2={l2(v_normed.reshape(B,L,-1)):.4f}  {fmt(v_normed.reshape(B,L,-1))}")

    # Step 7: Transpose for RoPE: [B, n_heads, L, head_dim]
    k_t = k_normed.transpose(0, 2, 1, 3)
    v_t = v_normed.transpose(0, 2, 1, 3)
    q_t = q.transpose(0, 2, 1, 3)
    mx.eval(k_t); mx.eval(v_t); mx.eval(q_t)

    # Step 8: Get offset
    offset = mx.array(c.offset) if c is not None else 0
    mx.eval(offset)
    print(f"7. RoPE offset:      {offset}")

    # Step 9: RoPE on K
    k_roped = attn.rope(k_t, offset=offset)
    mx.eval(k_roped)
    # Print head 0, last position
    k_h0_last = k_roped[0, 0, -1, :5]
    mx.eval(k_h0_last)
    print(f"8. K after RoPE (head0,last): {k_h0_last.tolist()}")

    # Step 10: RoPE on Q
    q_roped = attn.rope(q_t, offset=offset)
    mx.eval(q_roped)
    q_h0_last = q_roped[0, 0, -1, :5]
    mx.eval(q_h0_last)
    print(f"9. Q after RoPE (head0,last): {q_h0_last.tolist()}")

    # Step 11: Also show the RoPE frequencies
    print(f"10. RoPE freqs (first 10): {attn.rope._freqs[:10].tolist()}")
    print(f"    RoPE dims: {attn.rope.dims}")
    print(f"    RoPE total freqs: {attn.rope._freqs.shape}")

    # Step 12: KV cache update
    keys_cached, values_cached = c.update_and_fetch(k_roped, v_t)
    mx.eval(keys_cached); mx.eval(values_cached)
    print(f"11. K cached shape: {keys_cached.shape}, V cached shape: {values_cached.shape}")

    # Step 13: SDPA
    from mlx.nn.layers.transformer import scaled_dot_product_attention
    attn_out = scaled_dot_product_attention(
        q_roped, keys_cached, values_cached,
        cache=c, scale=attn.scale, mask=mask
    )
    mx.eval(attn_out)
    # attn_out: [B, n_heads, L, head_dim]
    a_last = attn_out[0, 0, -1, :5]
    mx.eval(a_last)
    print(f"12. SDPA out (head0,last): {a_last.tolist()}")
    print(f"    SDPA scale: {attn.scale}")

    # Step 14: Transpose back + O proj
    attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    mx.eval(attn_flat)
    print(f"13. Attn flat:       L2={l2(attn_flat):.4f}  {fmt(attn_flat)}")

    o_out = attn.o_proj(attn_flat)
    mx.eval(o_out)
    print(f"14. O proj:          L2={l2(o_out):.4f}  {fmt(o_out)}")

    # Step 15: Post-attention norm + residual
    h2 = layer.post_attention_layernorm(o_out)
    mx.eval(h2)
    print(f"15. post_attn_norm:  L2={l2(h2):.4f}  {fmt(h2)}")

    h2 = residual + h2
    mx.eval(h2)
    print(f"16. residual1:       L2={l2(h2):.4f}  {fmt(h2)}")

    # Step 16: MoE FFN
    residual2 = h2

    # Dense MLP branch
    h1 = layer.pre_feedforward_layernorm(h2)
    mx.eval(h1)
    print(f"17. pre_ff_norm:     L2={l2(h1):.4f}  {fmt(h1)}")

    h1 = layer.mlp(h1)
    mx.eval(h1)
    print(f"18. dense_mlp:       L2={l2(h1):.4f}  {fmt(h1)}")

    h1 = layer.post_feedforward_layernorm_1(h1)
    mx.eval(h1)
    print(f"19. post_ff_norm1:   L2={l2(h1):.4f}  {fmt(h1)}")

    # MoE branch
    top_k_indices, top_k_weights = layer.router(h2)
    mx.eval(top_k_indices); mx.eval(top_k_weights)
    print(f"20. MoE router (last tok): experts={top_k_indices[0,-1,:].tolist()}")
    print(f"    weights={top_k_weights[0,-1,:].tolist()}")

    h2_normed = layer.pre_feedforward_layernorm_2(h2)
    mx.eval(h2_normed)
    h2_moe = layer.experts(h2_normed, top_k_indices, top_k_weights)
    mx.eval(h2_moe)
    print(f"21. MoE experts:     L2={l2(h2_moe):.4f}  {fmt(h2_moe)}")

    h2_moe = layer.post_feedforward_layernorm_2(h2_moe)
    mx.eval(h2_moe)
    print(f"22. post_ff_norm2:   L2={l2(h2_moe):.4f}  {fmt(h2_moe)}")

    # Combine
    h_combined = h1 + h2_moe
    mx.eval(h_combined)
    print(f"23. dense+moe:       L2={l2(h_combined):.4f}  {fmt(h_combined)}")

    h_combined = layer.post_feedforward_layernorm(h_combined)
    mx.eval(h_combined)
    print(f"24. post_ff_norm:    L2={l2(h_combined):.4f}  {fmt(h_combined)}")

    h_out = residual2 + h_combined
    mx.eval(h_out)
    print(f"25. FINAL output:    L2={l2(h_out):.4f}  {fmt(h_out)}")

    # Layer scalar
    if hasattr(layer, 'layer_scalar') and layer.layer_scalar is not None:
        scalar = layer.layer_scalar
        mx.eval(scalar)
        print(f"26. layer_scalar:    {scalar.tolist()}")
        h_out = h_out * scalar
        mx.eval(h_out)
        print(f"27. After scalar:    L2={l2(h_out):.4f}  {fmt(h_out)}")

    print(f"\nDone. Compare against HF2Q_DIAG=1 output for layer {TARGET_LAYER}.")


if __name__ == "__main__":
    run()
