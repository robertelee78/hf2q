#!/usr/bin/env python3
"""
Analyze whether Q8_0 quantization systematically lifts the <pad> (id 0)
logit relative to F16 quantization, by measuring:

  1. Per-row reconstruction error for pad vs median vs common tokens.
  2. Per-block amax distribution in pad row vs other rows (Q8_0 uses
     per-block amax/127 scale; near-zero blocks have compressed dynamic
     range).
  3. Empirical logit gap `Q8_logit - F16_logit` for pad on a distribution
     of realistic hidden states.

Inputs:
  - Gemma-4 GGUF model (embed_weight, F16 stored as F16, dequantized to F32).
  - Dumped hidden state from an actual decode run (pre-lm_head).

Output: numerical evidence for whether pad is a row-specific pathology
or just a near-tie manifestation.
"""

import gguf
import numpy as np
from pathlib import Path

GGUF_PATH = '/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf'
HS = 2816
VOCAB = 262144

def quantize_q8_0_row(row_f32: np.ndarray) -> np.ndarray:
    """Q8_0 quantize a row, then dequantize back — returns the effective
    weight as the hf2q kernel would see it (F16 scale, int8 quants,
    scale = amax/127 per 32-block)."""
    assert row_f32.size % 32 == 0
    n_blocks = row_f32.size // 32
    out = np.zeros_like(row_f32)
    for b in range(n_blocks):
        block = row_f32[b*32:(b+1)*32]
        amax = float(np.abs(block).max())
        d_f32 = amax / 127.0
        # hf2q stores d as F16
        d_f16 = np.float32(np.float16(d_f32))
        if d_f16 == 0.0:
            inv_d = 0.0
        else:
            inv_d = 1.0 / d_f16
        q = np.clip(np.round(block * inv_d), -127, 127).astype(np.int8)
        out[b*32:(b+1)*32] = q.astype(np.float32) * d_f16
    return out

def quantize_f16_row(row_f32: np.ndarray) -> np.ndarray:
    """F16 roundtrip."""
    return np.float16(row_f32).astype(np.float32)

def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a-b)**2)) / (np.sqrt(np.mean(a*a)) + 1e-12))

# --- Load embed ---
print("Loading GGUF embed_weight ...")
r = gguf.GGUFReader(GGUF_PATH)
embed = None
for t in r.tensors:
    if t.name == 'token_embd.weight':
        embed = np.array(t.data).astype(np.float32).reshape(VOCAB, HS)
        print(f"  shape={embed.shape} dtype={t.tensor_type.name}")
        break
assert embed is not None

# --- Per-row reconstruction error ---
print("\n=== Per-row reconstruction error (F16 vs Q8_0 vs F32 reference) ===")

# Sample tokens: pad, eos, bos, turn tokens, and a random sample
sample_ids = [0, 1, 2, 105, 106] + [int(i) for i in np.random.default_rng(42).integers(1000, VOCAB, 20)]

def row_stats(rid):
    row = embed[rid]
    f16 = quantize_f16_row(row)
    q8 = quantize_q8_0_row(row)
    return {
        'std': row.std(),
        'amax': np.abs(row).max(),
        'f16_err': rel_err(row, f16),
        'q8_err':  rel_err(row, q8),
        'f16_abs_err': float(np.abs(row - f16).mean()),
        'q8_abs_err':  float(np.abs(row - q8).mean()),
        'f16_max_abs': float(np.abs(row - f16).max()),
        'q8_max_abs':  float(np.abs(row - q8).max()),
    }

stats_by_id = {rid: row_stats(rid) for rid in sample_ids}
print(f"{'id':>8} {'std':>8} {'amax':>8} {'f16_rrms':>10} {'q8_rrms':>10} "
      f"{'f16_mae':>10} {'q8_mae':>10} {'q8/f16':>8}")
for rid in sample_ids:
    s = stats_by_id[rid]
    ratio = s['q8_err'] / s['f16_err'] if s['f16_err'] > 0 else float('inf')
    marker = "  ← pad" if rid == 0 else ""
    print(f"{rid:>8} {s['std']:>8.4f} {s['amax']:>8.4f} "
          f"{s['f16_err']:>10.3e} {s['q8_err']:>10.3e} "
          f"{s['f16_abs_err']:>10.3e} {s['q8_abs_err']:>10.3e} "
          f"{ratio:>8.2f}{marker}")

# --- Per-block amax distribution for pad row ---
print("\n=== Per-block amax distribution (88 blocks per row) ===")
pad_row = embed[0]
tok1k_row = embed[1000]  # random "typical" token

def per_block_amax(row):
    return np.array([np.abs(row[b*32:(b+1)*32]).max() for b in range(HS // 32)])

pad_amax = per_block_amax(pad_row)
tok_amax = per_block_amax(tok1k_row)
print(f"  pad block amax:  min={pad_amax.min():.4e}  max={pad_amax.max():.4e}  "
      f"mean={pad_amax.mean():.4e}  median={np.median(pad_amax):.4e}")
print(f"  tok1k amax:      min={tok_amax.min():.4e}  max={tok_amax.max():.4e}  "
      f"mean={tok_amax.mean():.4e}  median={np.median(tok_amax):.4e}")

# Number of "tiny" blocks (amax < 0.01) — these get amplified Q8 error
print(f"  pad blocks with amax < 0.01: {(pad_amax < 0.01).sum()}/{len(pad_amax)}")
print(f"  tok1k blocks with amax < 0.01: {(tok_amax < 0.01).sum()}/{len(tok_amax)}")

# --- Empirical F16 vs Q8 logit gap for pad on realistic hidden states ---
print("\n=== F16 vs Q8 logit for pad, across many hidden states ===")
# Use a distribution that mimics post-final-norm hidden state.
# From measurement: pre-lm_head has ~std=0.34 in this model (seen in earlier
# bisection). Generate N random hidden states.
rng = np.random.default_rng(0)
N = 512
H = rng.standard_normal((N, HS)).astype(np.float32) * 0.34

# Compute logits for rows: pad, top-10 typical
test_ids = [0, 1, 2, 105, 106, 100, 1000, 10000, 50000, 100000, 200000]

# Store full F16 and Q8 rows
rows_f16 = {rid: quantize_f16_row(embed[rid]) for rid in test_ids}
rows_q8 = {rid: quantize_q8_0_row(embed[rid]) for rid in test_ids}

def logits(h, row):
    return h @ row  # scalar: dot product

# For each hidden state, compute pad's F16 and Q8 logit, plus a reference logit (say tok 1000)
stats = {rid: {'gaps': []} for rid in test_ids}
for i in range(N):
    h = H[i]
    for rid in test_ids:
        l_f16 = float(h @ rows_f16[rid])
        l_q8 = float(h @ rows_q8[rid])
        stats[rid]['gaps'].append(l_q8 - l_f16)

print(f"  per-token (logit_Q8 - logit_F16) over {N} random hidden states (std=0.34):")
print(f"  {'id':>8} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'max_abs':>10}")
for rid in test_ids:
    g = np.array(stats[rid]['gaps'])
    marker = "  ← pad" if rid == 0 else ""
    print(f"  {rid:>8} {g.mean():>10.4e} {g.std():>10.4e} {g.min():>10.4e} "
          f"{g.max():>10.4e} {np.abs(g).max():>10.4e}{marker}")

# --- What could make pad win? Check variance of pad's Q8 error projection on h ---
# logit_Q8[pad] - logit_F16[pad] = h @ (q8_pad - f16_pad)
# magnitude ~ ||h|| * ||q8_pad - f16_pad|| * cos(theta)
pad_f16 = rows_f16[0]
pad_q8 = rows_q8[0]
pad_err_vec = pad_q8 - pad_f16  # the "error direction" of Q8 vs F16 for pad
print(f"\n  pad Q8-F16 error vector: std={pad_err_vec.std():.4e}  norm={np.linalg.norm(pad_err_vec):.4e}")
# For tok1k
t1k_err = rows_q8[1000] - rows_f16[1000]
print(f"  tok1k Q8-F16 error vector: std={t1k_err.std():.4e}  norm={np.linalg.norm(t1k_err):.4e}")
