# Parity harness — ADR-034 P1

Reference-vs-implementation numerical-parity test harness for the
speculative-decode end-to-end mission (ADR-034).

## Purpose

The convert side is validated by **byte-identity** against canonical GGUF
output (ADR-033 §P1). The runtime side (MTP forward, DFlash forward,
rejection sampler, spec-decode orchestration) has no analogous byte-cmp gate
because Metal kernel output has inherent FP non-determinism in the absence
of strict ordering controls.

This harness provides the **numerical-parity gate**: dump intermediate
activations from a Python reference, load them in a Rust test, run the
hf2q forward path with byte-identical inputs, compare max-abs-diff.

## Scripts

- `mtp_parity.py` — runs Qwen 3.5/3.6 MTP forward via HF transformers,
  dumps per-step intermediates (eh_proj output, attention block output,
  FFN output, final logits) as `.npy` files keyed by tensor name.
- `dflash_parity.py` — runs `/opt/dflash/dflash/model_mlx.py` DFlash
  draft forward, dumps per-layer activations.

## Tests

- `tests/parity/mtp_python_ref.rs` — loads the `.npy` dumps + runs
  `MtpWeights::forward_draft` on the SAME (token, hidden_state) inputs;
  asserts max-abs-diff < threshold for each intermediate.
- `tests/parity/dflash_python_ref.rs` — same shape for DFlash.

## Threshold

Per ADR-034 §6 G2 (numerical-parity committed-ε ladder):
- F32 forward: max-abs-diff < 1e-5
- BF16 forward: max-abs-diff < 1e-3
- Q4_K_M forward: max-abs-diff < 5e-2 (quant noise floor)

## Status (2026-05-21)

- 🟡 Scaffold created
- ⚪ `mtp_parity.py` — not yet implemented (blocked on Qwen 3.6 MTP-bearing safetensors download)
- ⚪ `dflash_parity.py` — not yet implemented (blocked on /opt/dflash z-lab drafter checkpoint download)
- ⚪ `tests/parity/mtp_python_ref.rs` — not yet implemented
- ⚪ `tests/parity/dflash_python_ref.rs` — not yet implemented

## Notes from prep deep-research

ADR-034 §3.5 originally claimed MTP inner FFN is dense even for MoE-A3B targets.
This is **wrong** — verified by:
- HF safetensors: 773 mtp.layers.0.mlp.experts.* tensors in Qwen 3.5 35B-A3B
- Canonical GGUF: 16 MoE-style tensors at blk.40.* (ffn_gate_exps etc.)

The MTP inner FFN matches the main-stack topology: MoE for MoE targets,
dense for dense targets. The parity harness MUST exercise both paths.

See [[project_adr034_mtp_loader_moe_bug_2026_05_21]] memory entry.
