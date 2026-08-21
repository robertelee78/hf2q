#!/usr/bin/env python3
"""dflash_parity.py — Reference DFlash draft forward for ADR-034 numerical-parity gate.

Loads `/opt/dflash/dflash/model_mlx.py` (582-LOC reference), bound to a target
model + drafter checkpoint, runs a single draft step, and dumps per-layer
intermediates as .npy files. The Rust parity test loads these dumps and
compares against hf2q's `src/inference/spec_decode/dflash/forward.rs`
(7011-LOC scaffold, never validated against this Python reference).

Usage:
    dflash_parity.py <target_model_dir> <drafter_dir> <output_dir>

Outputs (all .npy in <output_dir>/):
    target_hidden_states.npy   # [B, L, H] hidden states captured from target_layer_ids
    drafter_input_concat.npy   # [B, L, K*H] post-concat input to drafter.fc
    drafter_fc_out.npy         # [B, L, H] post-fc projection
    drafter_layer_{i}_out.npy  # per-layer outputs
    drafter_final_norm_out.npy
    drafter_logits.npy         # final draft logits

Status: scaffold (2026-05-21). Implementation blocked on:
  1. z-lab DFlash drafter checkpoint download (Qwen 3.6 or Gemma 4 variant)
  2. Target model (Qwen 3.6 dense or 35B-A3B) safetensors
  3. mlx installation (the reference is mlx-based, not torch)

When unblocked, fill in:
  1. Import /opt/dflash/dflash/model_mlx.py:DFlashDraftModel
  2. Call load() for the target model_id, load_draft() for the drafter
  3. Run draft.bind(target_model) to wire embed_tokens + lm_head
  4. Prefill the target on a fixed prompt; capture hidden_states from
     drafter.config.target_layer_ids via _LayerHook (already in model_mlx.py)
  5. Manually run drafter forward, dumping per-step intermediates

CRITICAL: the reference is mlx-based. The Rust scaffold is on the Metal
side via mlx-native. Output values may differ by 1-2 ULP due to floating-
point ordering, BUT the math should be identical. Threshold per ADR-034 §G2.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="DFlash numerical-parity reference dumper")
    p.add_argument("target_model_dir", type=Path)
    p.add_argument("drafter_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()

    if not args.target_model_dir.is_dir():
        print(f"FAIL: target_model_dir {args.target_model_dir} not found", file=sys.stderr)
        return 1
    if not args.drafter_dir.is_dir():
        print(f"FAIL: drafter_dir {args.drafter_dir} not found", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("dflash_parity.py — ADR-034 P1 scaffold (NOT YET IMPLEMENTED)")
    print("=" * 60)
    print(f"Target:    {args.target_model_dir}")
    print(f"Drafter:   {args.drafter_dir}")
    print(f"Output:    {args.output_dir}")
    print()
    print("STATUS: This is a scaffold. Implementation requires:")
    print("  1. z-lab/Qwen3.6-27B-DFlash (or similar) drafter checkpoint downloaded")
    print("  2. Target model safetensors (Qwen 3.6 27B or 35B-A3B)")
    print("  3. mlx installation: pip install mlx")
    print("  4. ~150-250 LOC to wrap /opt/dflash/dflash/model_mlx.py + dump intermediates")
    print()
    print("Reference: /opt/dflash/dflash/model_mlx.py (582 LOC, pinned @ 94e4abc5)")
    print("Target Rust path: src/inference/spec_decode/dflash/forward.rs (2158 LOC)")
    print("ADR: docs/adr/ADR-034-speculative-decode-end-to-end.md §P4")
    return 2


if __name__ == "__main__":
    sys.exit(main())
