"""Layer-by-layer comparison: extract hidden states from mlx-lm prefill."""
import mlx.core as mx
import numpy as np
import json
from mlx_lm import load

MODEL_PATH = "/opt/hf2q/models/gemma4-mlx-auto"
PROMPT = "What is 2+2? Explain your reasoning step by step in great detail."


def run():
    model, tokenizer = load(MODEL_PATH)
    lm = model.language_model
    inner = lm.model  # Gemma4TextModel

    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(prompt_str)
    print(f"Tokens ({len(tokens)}): {tokens}")
    input_ids = mx.array([tokens])

    # Replicate TextModel.__call__ step by step, capturing intermediates
    h = inner.embed_tokens(input_ids)
    h = h * inner.embed_scale
    mx.eval(h)
    print(f"After embed+scale (last tok, first 5): {h[0, -1, :5].tolist()}")

    # Per-layer inputs (PLE)
    if inner.hidden_size_per_layer_input:
        per_layer_inputs = inner._get_per_layer_inputs(input_ids, h)
        per_layer_inputs = inner._project_per_layer_inputs(h, per_layer_inputs)
        per_layer_inputs = [per_layer_inputs[:, :, i, :] for i in range(len(inner.layers))]
    else:
        per_layer_inputs = [None] * len(inner.layers)

    # Cache
    cache = model.make_cache()
    cache = cache + [None] * (len(inner.layers) - len(cache))

    # Masks
    masks = inner._make_masks(h, cache)

    # Process each layer
    intermediates = [(None, None)] * len(inner.layers)
    print(f"\nprevious_kvs: {inner.previous_kvs}")
    print(f"\n{'Layer':>5} {'L2':>10} {'First 5 (last token)'}")
    print("=" * 70)

    for idx, (layer, c, mask, prev_idx, pli) in enumerate(
        zip(inner.layers, cache, masks, inner.previous_kvs, per_layer_inputs)
    ):
        kvs, offset = intermediates[prev_idx]
        h, kvs_out, offset_out = layer(
            h, mask, c,
            per_layer_input=pli,
            shared_kv=kvs,
            offset=offset,
        )
        mx.eval(h)
        intermediates[idx] = (kvs_out, offset_out)

        last = h[0, -1, :]
        l2 = float(mx.sqrt(mx.sum(last ** 2)).item())
        f5 = last[:5].tolist()
        print(f"  {idx:>3d}   {l2:>10.4f}   [{', '.join(f'{v:.8f}' for v in f5)}]")

    # Final norm
    h = inner.norm(h)
    mx.eval(h)
    f5 = h[0, -1, :5].tolist()
    print(f"\nAfter final norm: [{', '.join(f'{v:.6f}' for v in f5)}]")

    # lm_head + softcap
    if lm.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    mx.eval(logits)

    vals = logits[0, -1, :].tolist()
    top = sorted(range(len(vals)), key=lambda i: -vals[i])[:10]
    print(f"\nPre-softcap top-10:")
    for r, i in enumerate(top):
        print(f"  #{r}: id={i:>6d} logit={vals[i]:>10.4f} '{tokenizer.decode([i])}'")

    cap = lm.final_logit_softcapping
    if cap:
        capped = mx.tanh(logits / cap) * cap
        mx.eval(capped)
        vals2 = capped[0, -1, :].tolist()
        top2 = sorted(range(len(vals2)), key=lambda i: -vals2[i])[:10]
        print(f"\nPost-softcap (cap={cap}) top-10:")
        for r, i in enumerate(top2):
            print(f"  #{r}: id={i:>6d} logit={vals2[i]:>10.4f}")

    # Save
    out = {}
    for idx in range(len(inner.layers)):
        # Re-eval not needed, but let's just save what we printed
        pass
    print("\nDone.")


if __name__ == "__main__":
    run()
