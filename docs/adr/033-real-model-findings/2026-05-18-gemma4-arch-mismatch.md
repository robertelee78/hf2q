# Real-Model Finding: Gemma 4 is a NEW architecture, not Gemma 3-compatible

**Date:** 2026-05-18
**Reproduced:** `hf2q convert-v2 /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m -o out.gguf`
**Output:** `convert-v2: tensor 'model.language_model.layers.4.post_feedforward_layernorm_2.weight' not recognized by 'gemma4' mapper`

## Summary

The `gemma4` per-arch mapper at `src/convert/arch/gemma4.rs` was written
against ADR-033's spec, which described Gemma 4 as "Gemma 3-architecture
compatible." Real Gemma 4 release (google/gemma-4-26b-a4b-it, May 2026)
has a **substantially different architecture** with multimodal wrapping
and MoE experts. The mapper needs expansion before convert-v2 works on
real Gemma 4 checkpoints.

## Actual tensor inventory (from `model.safetensors.index.json`)

Top-level prefixes:
- `model.embed_vision.embedding_projection.*` (multimodal projector)
- `model.language_model.embed_tokens` (← text decoder, nested under `language_model.`)
- `model.language_model.layers.<N>.*`
- `model.language_model.norm`
- `model.vision_tower.encoder.*` (SigLIP vision encoder)
- `model.vision_tower.patch_embedder.*`
- `model.vision_tower.std_bias` / `std_scale` (normalization)

Per-block (layer 0 sample):
```
model.language_model.layers.0.experts.down_proj          ← MoE expert (fused 3-D)
model.language_model.layers.0.experts.gate_up_proj       ← MoE expert (fused gate+up)
model.language_model.layers.0.input_layernorm.weight
model.language_model.layers.0.layer_scalar               ← per-layer scalar (Gemma 4 new)
model.language_model.layers.0.mlp.down_proj.weight       ← parallel DENSE FFN
model.language_model.layers.0.mlp.gate_proj.weight       ← (alongside MoE — Gemma 4 has BOTH)
model.language_model.layers.0.mlp.up_proj.weight
model.language_model.layers.0.post_attention_layernorm.weight
model.language_model.layers.0.post_feedforward_layernorm.weight
model.language_model.layers.0.post_feedforward_layernorm_1.weight   ← second pair
model.language_model.layers.0.post_feedforward_layernorm_2.weight   ← third pair
model.language_model.layers.0.pre_feedforward_layernorm.weight
model.language_model.layers.0.pre_feedforward_layernorm_2.weight    ← second pre-pair
model.language_model.layers.0.router.per_expert_scale    ← router scaling
model.language_model.layers.0.router.proj.weight         ← router projection
model.language_model.layers.0.router.scale               ← scalar gating
model.language_model.layers.0.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.0.self_attn.{q,k}_norm.weight
```

## Required mapper expansions

1. **Prefix strip `model.language_model.`** — multimodal-wrapper convention. The mapper's HF tensor-name match table needs to accept either `model.<...>` or `model.language_model.<...>`. Recommend a `strip_text_decoder_prefix(name) -> Option<&str>` helper.

2. **MoE expert fusion** — `experts.gate_up_proj` is FUSED gate+up (3-D tensor), unlike Qwen3MoE's separate `experts.<E>.gate_proj` / `experts.<E>.up_proj` per-expert tensors. This requires a `MappedTensor::FusedGateUp` variant or per-arch splitting before orchestrator emission.

3. **Multiple norms (_1, _2 suffixes)** — Gemma 4 has 3 post_feedforward_layernorm variants and 2 pre_feedforward_layernorm variants. The GGUF target names need disambiguation (e.g., `blk.<N>.post_ffw_norm_1.weight`).

4. **Router tensors** — `router.proj` → `blk.<N>.ffn_gate_inp.weight`. `router.scale` and `router.per_expert_scale` need llama.cpp counterparts (may be Gemma 4-specific KV-pair metadata instead of tensors).

5. **Parallel dense + MoE** — `mlp.{gate,up,down}_proj` exist alongside `experts.*`. Gemma 4 appears to be a sparse-MoE that also keeps a dense shortcut. llama.cpp's loader needs the right GGUF tensor names for this hybrid pattern (`ffn_gate.weight` for dense vs `ffn_gate_exps.weight` for routed).

6. **layer_scalar** — a per-layer scalar tensor unique to Gemma 4. Maps to `blk.<N>.layer_scale.weight` per llama.cpp's recent Gemma 3 additions, OR may need a new KV.

7. **`build_metadata` expansion** — Gemma 4 config has `num_experts`, `experts_per_token`, plus the MoE-specific keys that need to surface as GGUF metadata.

8. **Vision/embed_vision/vision_tower filter** — `is_vision_tensor_pattern` (canonical at `src/quantize/ggml_quants/vision.rs`) catches `vision_tower.` and `embed_vision`. Verify these short-circuit BEFORE the language_model mapper sees them.

## ADR alignment

ADR-033 §P0 amendment H stated `src/calibrate/apex.rs` is implicit-delete; nothing about Gemma 4 specifically. The gemma4 mapper at commit 6f264204 was written against the gemma3 spec because that's what ADR-033 said. **Recommend an ADR amendment N adding Gemma 4 as a distinct arch (`gemma4` GGUF architecture string, MoE-with-dense-shortcut model, 4 norm pairs per block, fused gate_up_proj experts) rather than an extension of gemma3.**

## Workaround for iter-43+ work

Until the mapper is expanded:
- Convert-v2 works for Llama3 dense models (verified by llama3_end_to_end_tiny test).
- Convert-v2 works for Qwen3MoE Mixtral-style models (verified by convert_v2_apex_balanced_tiny_qwen35moe test).
- Convert-v2 BLOCKED for real Gemma 4 release until mapper expansion.
- Operator's existing workflow (mudler/apex-quant on a separate machine, consuming pre-converted GGUFs) is unaffected — it was never reaching `hf2q convert-v2` to begin with.

## Test plan for the mapper expansion

When iter-43+ tackles this:
1. Add a synthetic Gemma 4-shaped safetensors fixture (`model.language_model.*` prefix, `experts.gate_up_proj` fused tensors, 4 norm pairs, router subblock, parallel dense mlp).
2. Add `convert_v2_gemma4_real_arch_round_trip` integration test that invokes convert-v2 on the fixture and asserts the GGUF tensor names match what llama.cpp's loader for gemma4 expects.
3. Once green, retry `hf2q convert-v2 /opt/hf2q/models/google-gemma-4-26b-a4b-it --quant q5_k_m`.
4. Byte-cmp against `(convert_hf_to_gguf.py | llama-quantize)` if llama.cpp supports Gemma 4 at this point (verify against `vendor/apex-quant`'s `apex_pipeline.sh` which is operator's current ground truth).
