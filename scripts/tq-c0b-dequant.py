#!/usr/bin/env python3
"""
tq-c0b-dequant.py — CFA-20260421-C0b-localize Worker 2: dequantizer + diff

Mirrors the GPU dequantization path in flash_attn_vec_tq.metal exactly:
  1. Unpack nibbles  (low nibble = even coord, high nibble = odd coord)
     Source: hadamard_quantize_kv.metal lines 130-138 (tid%2==0 → low, tid%2==1 → high)
  2. Centroid lookup from CODEBOOK_4BIT
     Source: mlx-native/src/turboquant.rs lines 27-32
  3. Scale by norm * rsqrt(head_dim)
     Source: flash_attn_vec_tq.metal line 304: k_sn = K_norms[...] * inv_sqrt_dk
  4. Inverse FWHT (self-inverse; same function as forward)
     Source: mlx-native/src/turboquant.rs lines 94-97 (H*H=I with 1/sqrt(N) normalization)

NRMSE formula: from test_flash_attn_vec_tq.rs lines 384-387
  nrmse = sqrt(sum_sq_diff / sum_sq_ref)
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants — copied verbatim from mlx-native/src/turboquant.rs lines 27-32
# ---------------------------------------------------------------------------
CODEBOOK_4BIT = np.array([
    -2.7325896, -2.0690172, -1.6180464, -1.2562312,
    -0.9423405, -0.6567591, -0.3880483, -0.1283950,
     0.1283950,  0.3880483,  0.6567591,  0.9423405,
     1.2562312,  1.6180464,  2.0690172,  2.7325896,
], dtype=np.float32)

# ---------------------------------------------------------------------------
# FWHT — mirrors turboquant.rs fwht_inplace (lines 79-97)
# Self-inverse: H*H = I via 1/sqrt(N) normalization.
# ---------------------------------------------------------------------------
def fwht_inplace(x: np.ndarray) -> None:
    """In-place normalized Walsh-Hadamard transform (self-inverse).
    x must have last dimension a power of two.
    Source: turboquant.rs lines 79-97."""
    n = x.shape[-1]
    h = 1
    while h < n:
        # Slice view: pair (x[..., i], x[..., i+h]) for each block
        x_view = x.reshape(x.shape[:-1] + (n // (h * 2), h * 2))
        a = x_view[..., :h].copy()
        b = x_view[..., h:].copy()
        x_view[..., :h] = a + b
        x_view[..., h:] = a - b
        h *= 2
    # Normalize: 1/sqrt(N) — turboquant.rs line 95
    x *= 1.0 / math.sqrt(n)


# ---------------------------------------------------------------------------
# NRMSE formula — copied from test_flash_attn_vec_tq.rs lines 384-387
# nrmse = sqrt(sum_sq_diff / sum_sq_ref)
# Note: the test uses sum_sq_ref > 0 guard; we use max(..., 1e-30) per spec.
# ---------------------------------------------------------------------------
def nrmse(reference: np.ndarray, actual: np.ndarray) -> float:
    """Normalized RMSE.
    Source: test_flash_attn_vec_tq.rs lines 384-387.
    nrmse = sqrt(sum(diff^2) / max(sum(ref^2), 1e-30))"""
    diff = actual - reference
    sum_sq_diff = float(np.sum(diff.astype(np.float64) ** 2))
    sum_sq_ref  = float(np.sum(reference.astype(np.float64) ** 2))
    return math.sqrt(sum_sq_diff / max(sum_sq_ref, 1e-30))


# ---------------------------------------------------------------------------
# Dequantize packed buffer
# packed shape: [nkv, P, hd/2]  (P = number of positions in the dump)
# norms shape:  [nkv, P]
# Returns: float32 [nkv, P, hd]
# ---------------------------------------------------------------------------
def dequant_packed(packed: np.ndarray, norms: np.ndarray, hd: int) -> np.ndarray:
    """Dequantize TQ packed buffer to float32 vectors.

    Steps mirror flash_attn_vec_tq.metal and hadamard_quantize_kv.metal:
    1. Unpack nibbles:
       coord 2*c   (even) = low  nibble of packed[h, p, c]
       coord 2*c+1 (odd)  = high nibble of packed[h, p, c]
       Source: hadamard_quantize_kv.metal lines 130-138
    2. CODEBOOK_4BIT lookup
       Source: turboquant.rs lines 27-32
    3. Multiply by norm * rsqrt(hd)
       Source: flash_attn_vec_tq.metal line 304: k_sn = K_norms[...] * inv_sqrt_dk
    4. Inverse FWHT (self-inverse — same transform)
       Source: turboquant.rs lines 94-97
    """
    nkv, P, _ = packed.shape
    assert packed.shape[2] == hd // 2, f"packed hd/2 mismatch: {packed.shape[2]} != {hd//2}"
    assert norms.shape == (nkv, P), f"norms shape mismatch: {norms.shape} != ({nkv},{P})"

    # Step 1: Unpack nibbles → centroid indices [nkv, P, hd]
    # low nibble  → even coord (2*c)
    # high nibble → odd  coord (2*c+1)
    # Source: hadamard_quantize_kv.metal lines 130-138
    p_u8 = packed.astype(np.uint8)  # [nkv, P, hd/2]
    low  = (p_u8 & 0x0F).astype(np.int32)   # even coords: [nkv, P, hd/2]
    high = ((p_u8 >> 4) & 0x0F).astype(np.int32)  # odd coords: [nkv, P, hd/2]

    # Interleave: indices[..., 2*c] = low[..., c], indices[..., 2*c+1] = high[..., c]
    indices = np.empty((nkv, P, hd), dtype=np.int32)
    indices[..., 0::2] = low   # even coordinates
    indices[..., 1::2] = high  # odd coordinates

    # Step 2: Centroid lookup
    # Source: turboquant.rs lines 27-32
    rotated = CODEBOOK_4BIT[indices]  # [nkv, P, hd] float32

    # Step 3: Scale by norm * rsqrt(hd)
    # Source: flash_attn_vec_tq.metal line 304: k_sn = K_norms[...] * inv_sqrt_dk
    inv_sqrt_hd = 1.0 / math.sqrt(hd)
    scale = norms * inv_sqrt_hd  # [nkv, P]
    rotated *= scale[:, :, np.newaxis]  # broadcast over hd

    # Step 4: Inverse FWHT (self-inverse, same function)
    # Source: turboquant.rs lines 94-97
    fwht_inplace(rotated)  # operates on last dim

    return rotated  # [nkv, P, hd] float32


# ---------------------------------------------------------------------------
# Round-trip self-test
# ---------------------------------------------------------------------------
def forward_quantize(x: np.ndarray, hd: int):
    """Python implementation of hadamard_quantize_kv.metal for self-test.

    1. Apply forward FWHT
    2. Scale by 1/sqrt(hd)  (turboquant.rs line 81 analog)
    3. Compute L2 norm of scaled vector
    4. Unit-normalize and scale by sqrt(hd) → approx N(0,1) domain
    5. Nearest CODEBOOK_4BIT centroid
    6. Pack nibbles (even coord → low nibble, odd coord → high nibble)

    Source: hadamard_quantize_kv.metal lines 80-138,
            test_flash_attn_vec_tq.rs nibble_quantize lines 98-123.
    """
    nkv, P, _ = x.shape

    # Step 1+2: FWHT + 1/sqrt(hd) normalization
    val = x.copy()
    fwht_inplace(val)  # now val is H(x) / sqrt(hd), shape [nkv, P, hd]

    # Step 3: L2 norm of the scaled vector (per head, per position)
    norms = np.sqrt(np.sum(val.astype(np.float64) ** 2, axis=-1)).astype(np.float32)  # [nkv, P]

    # Step 4: unit-normalize + scale to N(0,1) domain
    # unit_val = val / norm;  scaled = unit_val * sqrt(hd)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    scaled = val / safe_norms[:, :, np.newaxis] * math.sqrt(hd)  # [nkv, P, hd]

    # Step 5: nearest centroid — midpoints between adjacent codebook entries
    # Source: test_flash_attn_vec_tq.rs boundaries_4bit + nearest_centroid_4bit (lines 76-93)
    boundaries = 0.5 * (CODEBOOK_4BIT[:-1] + CODEBOOK_4BIT[1:])  # 15 boundaries
    # indices: count how many boundaries scaled > b
    # shape: [nkv, P, hd, 15] — use broadcasting
    cmp = (scaled[:, :, :, np.newaxis] > boundaries[np.newaxis, np.newaxis, np.newaxis, :])
    indices = cmp.sum(axis=-1).astype(np.uint8)  # [nkv, P, hd]

    # Step 6: pack nibbles
    # even coord (indices[..., 2*c])   → low nibble  of packed[..., c]
    # odd  coord (indices[..., 2*c+1]) → high nibble of packed[..., c]
    # Source: hadamard_quantize_kv.metal lines 130-138
    even = indices[..., 0::2] & 0xF  # [nkv, P, hd/2]
    odd  = indices[..., 1::2] & 0xF  # [nkv, P, hd/2]
    packed = (even | (odd << 4)).astype(np.uint8)  # [nkv, P, hd/2]

    return packed, norms


def run_self_test(hd: int = 256, n_vecs: int = 1000, seed: int = 42) -> float:
    """Round-trip self-test: encode then decode, compute nrmse.

    Gate: nrmse < 0.12.

    NOTE on the gate value: The spec references 0.06 as "the kernel's own replay test",
    but test_flash_attn_vec_tq.rs uses nrmse < 0.15 for SDPA *output* (post attention
    averaging, not per-vector). The per-vector encode-decode NRMSE for 4-bit Lloyd-Max
    on N(0,1) hd=256 vectors is ~0.097 by construction (same as the Rust nibble_quantize
    + nibble_dequantize reference). Using 0.12 catches implementation bugs (wrong nibble
    order, wrong centroid table, wrong FWHT normalization) while admitting the correct
    4-bit quantization noise floor of ~0.097."""
    rng = np.random.default_rng(seed)
    nkv, P = 4, n_vecs // 4

    # Random N(0,1) input vectors
    x = rng.standard_normal((nkv, P, hd)).astype(np.float32)

    # Encode
    packed, enc_norms = forward_quantize(x, hd)

    # Decode
    x_hat = dequant_packed(packed, enc_norms, hd)

    # NRMSE — formula from test_flash_attn_vec_tq.rs lines 384-387
    err = nrmse(x, x_hat)
    return err


# ---------------------------------------------------------------------------
# Load binary dump utilities
# ---------------------------------------------------------------------------
def load_f32_bin(path: str, shape) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def load_u8_bin(path: str, shape) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    return data.reshape(shape)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    tq_root: str,
    dense_root: str,
    csv_path: str,
    summary_path: str,
    layers: list,
):
    rows = []

    for layer_cfg in layers:
        ll = layer_cfg["layer"]
        nkv = layer_cfg["nkv"]
        hd  = layer_cfg["hd"]
        kv_seq_len = layer_cfg["kv_seq_len"]
        P = kv_seq_len  # number of positions to compare
        tag = f"L{ll:02d}"

        # Load TQ dumps
        # Packed dump shape: [nkv, P, hd/2]
        # (dump contains only kv_seq_len positions, not full kv_capacity)
        k_packed = load_u8_bin(
            os.path.join(tq_root, f"hf2q_k_packed_layer{ll:02d}_pos22.u8.bin"),
            (nkv, P, hd // 2),
        )
        k_norms = load_f32_bin(
            os.path.join(tq_root, f"hf2q_k_norms_layer{ll:02d}_pos22.f32.bin"),
            (nkv, P),
        )
        v_packed = load_u8_bin(
            os.path.join(tq_root, f"hf2q_v_packed_layer{ll:02d}_pos22.u8.bin"),
            (nkv, P, hd // 2),
        )
        v_norms = load_f32_bin(
            os.path.join(tq_root, f"hf2q_v_norms_layer{ll:02d}_pos22.f32.bin"),
            (nkv, P),
        )

        # Load dense dumps — shape [nkv, 23, hd]; slice [:22] for the 22 prefilled tokens
        # Dense row 22 is the first decode K/V — NOT present on TQ side; skip it.
        k_dense_full = load_f32_bin(
            os.path.join(dense_root, f"hf2q_cache_k_layer{ll:02d}_pos22.bin"),
            (nkv, 23, hd),
        )
        k_dense = k_dense_full[:, :P, :]  # [nkv, 22, hd]

        v_dense_full = load_f32_bin(
            os.path.join(dense_root, f"hf2q_cache_v_layer{ll:02d}_pos22.bin"),
            (nkv, 23, hd),
        )
        v_dense = v_dense_full[:, :P, :]  # [nkv, 22, hd]

        # Dequantize TQ
        k_tq = dequant_packed(k_packed, k_norms, hd)   # [nkv, 22, hd]
        v_tq = dequant_packed(v_packed, v_norms, hd)   # [nkv, 22, hd]

        # Compute per-head, per-position metrics
        for (op_label, tq_data, dense_data) in [
            ("k", k_tq, k_dense),
            ("v", v_tq, v_dense),
        ]:
            for h in range(nkv):
                for p in range(P):
                    ref = dense_data[h, p, :]   # [hd]
                    act = tq_data[h, p, :]       # [hd]
                    diff = act - ref
                    mad = float(np.max(np.abs(diff)))
                    err = nrmse(ref, act)
                    rows.append({
                        "layer": ll,
                        "op": op_label,
                        "head": h,
                        "pos": p,
                        "max_abs_diff": mad,
                        "nrmse": err,
                    })

    # Write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("layer,op,head,pos,max_abs_diff,nrmse\n")
        for r in rows:
            f.write(f"{r['layer']},{r['op']},{r['head']},{r['pos']},"
                    f"{r['max_abs_diff']:.6f},{r['nrmse']:.6f}\n")

    print(f"CSV written: {csv_path} ({len(rows)} rows)")

    # Summarize per (layer, op)
    from collections import defaultdict
    summary = defaultdict(lambda: {"max_nrmse": 0.0, "worst_head": -1, "worst_pos": -1,
                                    "max_abs": 0.0, "violations": 0})
    for r in rows:
        key = (r["layer"], r["op"])
        s = summary[key]
        if r["nrmse"] > s["max_nrmse"]:
            s["max_nrmse"] = r["nrmse"]
            s["worst_head"] = r["head"]
            s["worst_pos"] = r["pos"]
        if r["max_abs_diff"] > s["max_abs"]:
            s["max_abs"] = r["max_abs_diff"]
        if r["nrmse"] > 0.15:
            s["violations"] += 1

    # Print summary
    print("\n--- Summary (per layer x op) ---")
    print(f"{'layer':>5} {'op':>2} {'max_nrmse':>10} {'worst_head':>11} {'worst_pos':>10} "
          f"{'max_abs_diff':>13} {'violations>0.15':>16}")
    for (ll, op) in sorted(summary.keys()):
        s = summary[(ll, op)]
        print(f"{ll:>5} {op:>2} {s['max_nrmse']:>10.6f} {s['worst_head']:>11} {s['worst_pos']:>10} "
              f"{s['max_abs']:>13.6f} {s['violations']:>16}")

    # Verdict
    all_nrmse_ok = all(s["max_nrmse"] <= 0.15 for s in summary.values())
    all_abs_ok   = all(s["max_abs"] <= 1.0    for s in summary.values())
    any_violation = any(s["violations"] > 0    for s in summary.values())

    if all_nrmse_ok and all_abs_ok:
        verdict = "E1"
        verdict_reason = "All nrmse <= 0.15 and max_abs_diff <= 1.0: packed cache dequantizes within kernel bound. Bug NOT in H3 (encode/cache). Downstream: H1 kernel / H2 FWHT / H4 dispatch."
    elif any_violation:
        verdict = "E2"
        worst_r = max(rows, key=lambda r: r["nrmse"])
        verdict_reason = (f"E2: nrmse violation at layer={worst_r['layer']} op={worst_r['op']} "
                          f"head={worst_r['head']} pos={worst_r['pos']} nrmse={worst_r['nrmse']:.6f}. "
                          f"Encode/cache is broken.")
    else:
        verdict = "E3"
        verdict_reason = "E3: mixed result — partial encode errors, check per-layer breakdown."

    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_reason}")

    # Write summary markdown
    # (explicitly requested by spec as output artifact — not a generic doc)
    worst_overall = max(rows, key=lambda r: r["nrmse"])
    worst_abs_overall = max(rows, key=lambda r: r["max_abs_diff"])
    total_violations = sum(1 for r in rows if r["nrmse"] > 0.15)

    with open(summary_path, "w") as f:
        f.write("# TQ C0b Localize — Dequant Diff Summary\n\n")
        f.write(f"Date: 2026-04-21  |  CFA session: cfa-20260421-C0b-localize  |  Worker 2\n\n")
        f.write(f"## Verdict: {verdict}\n\n")
        f.write(f"{verdict_reason}\n\n")
        f.write("## Per-layer x op worst-case\n\n")
        f.write("| layer | op | max_nrmse | worst_head | worst_pos | max_abs_diff | nrmse_violations |\n")
        f.write("|------:|---:|----------:|-----------:|----------:|-------------:|-----------------:|\n")
        for (ll, op) in sorted(summary.keys()):
            s = summary[(ll, op)]
            f.write(f"| {ll} | {op} | {s['max_nrmse']:.6f} | {s['worst_head']} | {s['worst_pos']} "
                    f"| {s['max_abs']:.6f} | {s['violations']} |\n")
        f.write("\n## Worst-case cell (nrmse)\n\n")
        f.write(f"layer={worst_overall['layer']} op={worst_overall['op']} "
                f"head={worst_overall['head']} pos={worst_overall['pos']} "
                f"nrmse={worst_overall['nrmse']:.6f} max_abs_diff={worst_overall['max_abs_diff']:.6f}\n\n")
        f.write("## Worst-case cell (max_abs_diff)\n\n")
        f.write(f"layer={worst_abs_overall['layer']} op={worst_abs_overall['op']} "
                f"head={worst_abs_overall['head']} pos={worst_abs_overall['pos']} "
                f"nrmse={worst_abs_overall['nrmse']:.6f} max_abs_diff={worst_abs_overall['max_abs_diff']:.6f}\n\n")
        f.write(f"## Total cells with nrmse > 0.15: {total_violations}\n\n")
        f.write("## Kernel bound status\n\n")
        f.write(f"nrmse bound (< 0.15) holds: {'YES' if all_nrmse_ok else 'NO'}\n")
        f.write(f"max_abs_diff bound (< 1.0) holds: {'YES' if all_abs_ok else 'NO'}\n")

    print(f"Summary written: {summary_path}")
    return verdict, summary, rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TQ dequantizer + diff — CFA C0b Worker 2")
    parser.add_argument("--self-test", action="store_true", help="Run round-trip self-test only")
    parser.add_argument("--hd", type=int, default=256, help="head_dim for self-test")
    parser.add_argument(
        "--tq-root",
        default="/tmp/cfa-20260421-C0b-localize/dumps/tq",
        help="Root dir for TQ dumps",
    )
    parser.add_argument(
        "--dense-root",
        default="/tmp/cfa-20260421-C0b-localize/dumps/dense",
        help="Root dir for dense F32 dumps",
    )
    parser.add_argument(
        "--csv",
        default="/opt/hf2q/docs/tq-c0b-localize-2026-04-21-raw.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--summary",
        default="/opt/hf2q/docs/tq-c0b-localize-2026-04-21-summary.md",
        help="Output summary markdown path",
    )
    args = parser.parse_args()

    if args.self_test:
        print("Running round-trip self-test (1000 vectors, hd=256)...")
        err = run_self_test(hd=256, n_vecs=1000, seed=42)
        print(f"  hd=256: nrmse={err:.6f}")
        err512 = run_self_test(hd=512, n_vecs=1000, seed=99)
        print(f"  hd=512: nrmse={err512:.6f}")
        worst = max(err, err512)
        # Gate is 0.12: must be above 4-bit noise floor (~0.097) but catches bugs.
        # Spec said 0.06 but that is below the mathematical 4-bit quantization noise
        # floor for this formula; 0.12 catches all implementation bugs while passing
        # the expected ~0.097 from correct 4-bit Lloyd-Max encode+decode.
        if worst < 0.12:
            print(f"PASS nrmse={worst:.6f} (< 0.12 gate; expected ~0.097 for correct 4-bit)")
            sys.exit(0)
        else:
            print(f"FAIL nrmse={worst:.6f} (>= 0.12 gate) -- dequantizer is wrong")
            sys.exit(1)

    # Layer configs from meta files
    layers = [
        {"layer": 0,  "nkv": 8, "hd": 256, "kv_seq_len": 22},
        {"layer": 5,  "nkv": 2, "hd": 512, "kv_seq_len": 22},
    ]

    verdict, summary, rows = run_analysis(
        tq_root=args.tq_root,
        dense_root=args.dense_root,
        csv_path=args.csv,
        summary_path=args.summary,
        layers=layers,
    )

    # Machine-readable final line for automation
    worst_r = max(rows, key=lambda r: r["nrmse"])
    worst_abs = max(rows, key=lambda r: r["max_abs_diff"])
    print(f"\nFINAL: verdict={verdict} worst_nrmse={worst_r['nrmse']:.6f} "
          f"(L{worst_r['layer']:02d} {worst_r['op']} h{worst_r['head']} p{worst_r['pos']}) "
          f"worst_abs={worst_abs['max_abs_diff']:.6f}")


if __name__ == "__main__":
    main()
