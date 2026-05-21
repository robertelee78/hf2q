#!/usr/bin/env python3
"""mtp_parity.py — Reference MTP forward for ADR-034 numerical-parity gate.

Loads a Qwen 3.5/3.6 MTP-bearing checkpoint via HF transformers, runs a
single MTP draft step, and dumps every intermediate activation as .npy
files keyed by tensor name. The Rust parity test loads these dumps and
compares against hf2q's MtpWeights::forward_draft output.

Usage:
    mtp_parity.py <model_path> <output_dir> [--token-id N] [--hidden-init FILE]

Outputs (all .npy in <output_dir>/):
    input_hidden.npy        # [1, H] verifier hidden state for token t
    input_embed.npy         # [1, H] embedding of accepted token t+1
    input_position.npy      # [4] IMROPE position ids
    enorm_out.npy           # post-enorm output (embed branch)
    hnorm_out.npy           # post-hnorm output (hidden branch)
    eh_proj_out.npy         # [1, H] projected sum (the input to MTP inner block)
    attn_norm_out.npy       # post-attn_norm output
    attn_out.npy            # post-self-attention output
    post_attn_norm_out.npy  # post-post_attention_norm output
    ffn_out.npy             # post-FFN output (the inner block output)
    shared_head_norm_out.npy
    logits.npy              # [1, V] final draft logits

Status: scaffold (2026-05-21). Implementation blocked on Qwen 3.6 MTP-bearing
safetensors download. When unblocked, fill in:
  1. Model load via transformers AutoModelForCausalLM (trust_remote_code=True
     for Qwen3.5/3.6 MoE).
  2. Hidden-state capture from a forward pass on a fixed prompt
     ("The capital of France is", greedy, max_new_tokens=2).
  3. Manual replay of the MTP block on the captured hidden state +
     next-token embedding, with per-step .npy dumps.

The implementation MUST use deterministic seeding (torch.manual_seed(42))
and CPU-only execution to eliminate Metal non-determinism from the
reference dumps.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="MTP numerical-parity reference dumper")
    p.add_argument("model_path", type=Path, help="Path to Qwen 3.5/3.6 MTP-bearing checkpoint dir")
    p.add_argument("output_dir", type=Path, help="Where to write .npy dumps")
    p.add_argument("--token-id", type=int, default=None,
                   help="Override token id for embedding (default: ' Paris' token)")
    p.add_argument("--hidden-init", type=Path, default=None,
                   help="Override hidden init from .npy file (default: capture from forward)")
    args = p.parse_args()

    if not args.model_path.is_dir():
        print(f"FAIL: model_path {args.model_path} not a directory", file=sys.stderr)
        return 1

    config_json = args.model_path / "config.json"
    if not config_json.exists():
        print(f"FAIL: {config_json} missing", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("mtp_parity.py — ADR-034 P1 scaffold (NOT YET IMPLEMENTED)")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output dir:   {args.output_dir}")
    print()
    print("STATUS: This is a scaffold. Implementation requires:")
    print("  1. Qwen 3.6 MTP-bearing safetensors on disk")
    print("  2. transformers + torch installation with Qwen 3.5/3.6 trust_remote_code support")
    print("  3. ~200-400 LOC to capture hidden states, run MTP block, dump intermediates")
    print()
    print("Implementation gates landed in ADR-034 P1 + project_adr034_mtp_loader_moe_bug_2026_05_21.")
    print("See scripts/parity/README.md for full design.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
