#!/usr/bin/env python3
"""
tq-c0-diff.py — Per-layer numerical diff of TQ vs dense SDPA dumps.
Session: cfa-20260421-C0-audit, Worker 2 (analyst-differ).

File naming: hf2q_<op>_layer<LL>_pos<PP>.bin  (raw F32 little-endian)
seq_pos = 22 + (decode_step - 1)
decode_steps:  1,  5, 10, 15, 20, 25, 30
seq_pos:      22, 26, 31, 36, 41, 46, 51

Gemma 4 has two attention-layer types (interleaved every 6):
  Local SWA layers (default): nh=16, nkv=8, hd=256
    q_normed  : [16, 256] → 4096 floats → 16384 bytes
    k_normed  : [ 8, 256] → 2048 floats →  8192 bytes
    v_normed  : [ 8, 256] → 2048 floats →  8192 bytes
    sdpa_out  : [16, 256] → 4096 floats → 16384 bytes

  Global attention layers (5, 11, 17, 23, 29): nh=32, nkv=4, hd=256
    q_normed  : [32, 256] → 8192 floats → 32768 bytes
    k_normed  : [ 4, 256] → 1024 floats →  4096 bytes
    v_normed  : [ 4, 256] → 1024 floats →  4096 bytes
    sdpa_out  : [32, 256] → 8192 floats → 32768 bytes
"""

import numpy as np
import csv
import os
import sys

# ── config ────────────────────────────────────────────────────────────────────
DENSE_ROOT = "/tmp/cfa-20260421-C0-audit/dumps/dense"
TQ_ROOT    = "/tmp/cfa-20260421-C0-audit/dumps/tq"
CSV_OUT    = "/opt/hf2q/docs/tq-c0-audit-2026-04-21-raw.csv"
SUMMARY_OUT = "/opt/hf2q/docs/tq-c0-audit-2026-04-21-summary.md"

NUM_LAYERS   = 30
DECODE_STEPS = [1, 5, 10, 15, 20, 25, 30]
SEQ_POS      = [22 + (d - 1) for d in DECODE_STEPS]  # [22,26,31,36,41,46,51]
OPS = ["q_normed", "k_normed", "v_normed", "sdpa_out"]

# Gemma 4: every 6th layer starting at layer 5 is global attention (nh=32, nkv=4)
GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29}

def get_op_shape(layer, op):
    """Return (n_floats, shape) for a given layer + op."""
    if layer in GLOBAL_ATTN_LAYERS:
        # Global attention: nh=32, nkv=4, hd=256
        if op in ("q_normed", "sdpa_out"):
            return 8192, (32, 256)
        else:  # k_normed, v_normed
            return 1024, (4, 256)
    else:
        # Local SWA attention: nh=16, nkv=8, hd=256
        if op in ("q_normed", "sdpa_out"):
            return 4096, (16, 256)
        else:  # k_normed, v_normed
            return 2048, (8, 256)

# Thresholds.
# THRESHOLD (analyst's internal breach trigger, held at 1e-3 for continuity with
# earlier sessions). This is NOT the kernel test's declared max_abs gate.
# MAX_AD_KERNEL_BOUND is the kernel test's own declared ceiling (test_flash_attn_vec_tq.rs:403).
# NRMSE_KERNEL_BOUND is the kernel test's nrmse ceiling (test_flash_attn_vec_tq.rs:399).
THRESHOLD            = 1e-3  # analyst internal breach trigger
MAX_AD_KERNEL_BOUND  = 1.0   # kernel test: assert!(max_abs_diff < 1.0)
NRMSE_KERNEL_BOUND   = 0.15  # kernel test: assert!(nrmse < 0.15)
NRMSE_DRILLDOWN      = 0.10  # 2/3 of 0.15 — triggers drilldown warning

# ── helpers ───────────────────────────────────────────────────────────────────
def load_tensor(root, layer, seq_pos, op):
    """Load raw F32 tensor. Returns (array, error_str|None)."""
    fname = f"hf2q_{op}_layer{layer:02d}_pos{seq_pos:02d}.bin"
    path  = os.path.join(root, fname)
    n_floats, shape = get_op_shape(layer, op)
    expected_bytes = n_floats * 4
    if not os.path.exists(path):
        return None, f"MISSING:{path}"
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        return None, f"SIZE_MISMATCH:{path}:expected={expected_bytes},got={actual_bytes}"
    arr = np.frombuffer(open(path, "rb").read(), dtype="<f4").copy()
    return arr.reshape(shape), None


def diff_metrics(d, t):
    """Compute diff metrics between two arrays of the same shape.

    nrmse formula matches mlx-native/tests/test_flash_attn_vec_tq.rs (lines 384-388):
        nrmse = sqrt(sum_sq_diff / sum_sq_ref)
    (CORRECTED 2026-04-21 — prior formula used |dense|.mean() as denominator, which
    under-measured against the kernel's own declared bound. See queen-directive
    in cfa-20260421-C0-audit for the review trail.)
    """
    diff      = d - t
    abs_diff  = np.abs(diff)
    max_ad    = float(abs_diff.max())
    mean_ad   = float(abs_diff.mean())
    mse       = float(np.mean(diff ** 2))
    # kernel-test nrmse: sqrt(sum_sq_diff / sum_sq_ref)
    sum_sq_diff = float((diff.astype(np.float64) * diff.astype(np.float64)).sum())
    sum_sq_ref  = float((d.astype(np.float64) * d.astype(np.float64)).sum())
    nrmse     = float(np.sqrt(sum_sq_diff / sum_sq_ref)) if sum_sq_ref > 0.0 else 0.0
    # rel_err_pct kept on the old denom purely as an auxiliary human-scale metric
    denom_legacy = float(np.abs(d).mean()) + 1e-12
    rel_err   = 100.0 * mean_ad / denom_legacy
    dense_mag = float(np.abs(d).mean())
    tq_mag    = float(np.abs(t).mean())
    return max_ad, mean_ad, mse, nrmse, rel_err, dense_mag, tq_mag


# ── main pass ─────────────────────────────────────────────────────────────────
rows = []

for layer in range(NUM_LAYERS):
    for decode_step, seq_pos in zip(DECODE_STEPS, SEQ_POS):
        for op in OPS:
            d_arr, d_err = load_tensor(DENSE_ROOT, layer, seq_pos, op)
            t_arr, t_err = load_tensor(TQ_ROOT,    layer, seq_pos, op)

            if d_err or t_err:
                reason = d_err or ""
                if t_err:
                    reason += (";" if reason else "") + t_err
                rows.append({
                    "layer": layer, "pos": decode_step, "op": op,
                    "max_abs_diff": "MISSING", "mean_abs_diff": "MISSING",
                    "mse": "MISSING", "nrmse": "MISSING",
                    "rel_err_pct": "MISSING",
                    "dense_mean_abs": "MISSING", "tq_mean_abs": "MISSING",
                    "note": reason,
                })
                continue

            max_ad, mean_ad, mse, nrmse, rel_err, dense_mag, tq_mag = diff_metrics(d_arr, t_arr)
            rows.append({
                "layer": layer, "pos": decode_step, "op": op,
                "max_abs_diff": max_ad, "mean_abs_diff": mean_ad,
                "mse": mse, "nrmse": nrmse, "rel_err_pct": rel_err,
                "dense_mean_abs": dense_mag, "tq_mean_abs": tq_mag,
                "note": "",
            })

print(f"Computed {len(rows)} rows ({sum(1 for r in rows if r['note'])} skipped/missing).")

# ── write CSV ─────────────────────────────────────────────────────────────────
csv_fields = ["layer","pos","op","max_abs_diff","mean_abs_diff","mse",
              "nrmse","rel_err_pct","dense_mean_abs","tq_mean_abs","note"]
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    w.writerows(rows)
print(f"CSV written: {CSV_OUT}")

# ── build lookup tables ────────────────────────────────────────────────────────
# data[layer][decode_step][op] = row dict (numeric only, skipped rows excluded)
data = {}
for r in rows:
    if r["note"]:
        continue
    L, P, op = r["layer"], r["pos"], r["op"]
    data.setdefault(L, {}).setdefault(P, {})[op] = r

# ── analysis ──────────────────────────────────────────────────────────────────
# (A) First threshold breach per op (across all layer x pos)
first_breach = {}  # op -> (layer, pos, max_abs_diff) or None
for op in OPS:
    best = None
    for layer in range(NUM_LAYERS):
        for pos in DECODE_STEPS:
            row = data.get(layer, {}).get(pos, {}).get(op)
            if row is None:
                continue
            if row["max_abs_diff"] > THRESHOLD:
                if best is None or (layer, pos) < (best[0], best[1]):
                    best = (layer, pos, row["max_abs_diff"])
    first_breach[op] = best  # None if no breach

# (B) Per-layer progression at pos=5 (decode_step=5)
pos5_data = []
for layer in range(NUM_LAYERS):
    entry = {"layer": layer}
    for op in OPS:
        row = data.get(layer, {}).get(5, {}).get(op)
        if row:
            entry[f"{op}_max"] = row["max_abs_diff"]
            entry[f"{op}_nrmse"] = row["nrmse"]
        else:
            entry[f"{op}_max"] = None
            entry[f"{op}_nrmse"] = None
    pos5_data.append(entry)

# (C) Per-layer max nrmse for sdpa_out across all positions
sdpa_nrmse_by_layer = {}
for layer in range(NUM_LAYERS):
    vals = []
    for pos in DECODE_STEPS:
        row = data.get(layer, {}).get(pos, {}).get("sdpa_out")
        if row:
            vals.append(row["nrmse"])
    sdpa_nrmse_by_layer[layer] = max(vals) if vals else None

kernel_bound_holds = all(
    v is not None and v < NRMSE_KERNEL_BOUND
    for v in sdpa_nrmse_by_layer.values()
)

# ── find overall first breach (any op) ────────────────────────────────────────
global_first_breach = None
for op in OPS:
    b = first_breach[op]
    if b:
        if global_first_breach is None or (b[0], b[1]) < (global_first_breach[0], global_first_breach[1]):
            global_first_breach = (b[0], b[1], op, b[2])

# ── write summary ─────────────────────────────────────────────────────────────
def fmt(v, decimals=6):
    if v is None:
        return "MISSING"
    return f"{v:.{decimals}e}"

lines = []
lines.append("# TQ vs Dense C-0 Layer-by-Layer Audit — 2026-04-21")
lines.append("")
lines.append(f"Session: `cfa-20260421-C0-audit` | Worker 2 (analyst-differ)")
lines.append(f"Threshold: `max_abs_diff > {THRESHOLD}` = BUG signal | `nrmse < {NRMSE_KERNEL_BOUND}` = kernel bound")
lines.append("")

# ── Table A: gating verdict ────────────────────────────────────────────────────
lines.append("## Table A — Gating Verdict (First Threshold Breach per Op)")
lines.append("")
lines.append("| Op | First Breach Layer | First Breach Pos | max_abs_diff |")
lines.append("|----|--------------------|------------------|--------------|")
for op in OPS:
    b = first_breach[op]
    if b:
        lines.append(f"| {op} | {b[0]} | {b[1]} | {b[2]:.6e} |")
    else:
        lines.append(f"| {op} | NONE | NONE | NONE — within threshold at all 210 measurements |")
lines.append("")

# ── Table B: per-layer at pos=5 ────────────────────────────────────────────────
lines.append("## Table B — Per-Layer Progression at Decode Pos 5 (seq_pos=26)")
lines.append("")
lines.append("| Layer | q_max | q_nrmse | k_max | k_nrmse | v_max | v_nrmse | sdpa_max | sdpa_nrmse |")
lines.append("|-------|-------|---------|-------|---------|-------|---------|----------|------------|")
for e in pos5_data:
    L = e["layer"]
    row_str = f"| {L:2d} "
    for op in OPS:
        key_max  = f"{op}_max"
        key_nrmse = f"{op}_nrmse"
        mx  = e[key_max]
        nr  = e[key_nrmse]
        mark_max  = "**" if (mx is not None and mx > THRESHOLD) else ""
        mark_nrmse = "**" if (nr is not None and nr > NRMSE_KERNEL_BOUND) else ""
        row_str += f"| {mark_max}{fmt(mx)}{mark_max} | {mark_nrmse}{fmt(nr)}{mark_nrmse} "
    row_str += "|"
    lines.append(row_str)
lines.append("")
lines.append("Values exceeding threshold/bound are **bold**.")
lines.append("")

# ── Table C: kernel-bound check for sdpa_out ─────────────────────────────────
lines.append("## Table C — Kernel-Bound Check: sdpa_out max nrmse across positions per layer")
lines.append("")
lines.append(f"Kernel bound: nrmse < {NRMSE_KERNEL_BOUND}  |  Overall: {'HOLDS' if kernel_bound_holds else 'VIOLATED'}")
lines.append("")
lines.append("| Layer | sdpa_out max_nrmse (over 7 positions) | Within Bound? |")
lines.append("|-------|---------------------------------------|---------------|")
for layer in range(NUM_LAYERS):
    v = sdpa_nrmse_by_layer[layer]
    within = "YES" if (v is not None and v < NRMSE_KERNEL_BOUND) else ("MISSING" if v is None else "**NO**")
    mark = "**" if (v is not None and v >= NRMSE_KERNEL_BOUND) else ""
    lines.append(f"| {layer:2d} | {mark}{fmt(v)}{mark} | {within} |")
lines.append("")

# ── narrative ──────────────────────────────────────────────────────────────────
lines.append("## Narrative Summary")
lines.append("")
if global_first_breach:
    L, P, op, mag = global_first_breach
    lines.append(f"CLASSIFICATION: **bug-candidate**")
    lines.append(f"First threshold breach: layer={L}, pos={P}, op={op}, max_abs_diff={mag:.6e}")
elif kernel_bound_holds:
    lines.append("CLASSIFICATION: **floor-candidate**")
    lines.append("No op exceeds max_abs_diff > 1e-3 at any (layer, pos). nrmse < 0.15 holds for sdpa_out per-layer.")
else:
    lines.append("CLASSIFICATION: **ambiguous**")
    lines.append("No clean threshold breach but nrmse >= 0.15 at some layer. Drilldown required.")
lines.append("")
lines.append(f"Kernel bound (nrmse<{NRMSE_KERNEL_BOUND} for sdpa_out per-layer): {'HOLDS' if kernel_bound_holds else 'VIOLATED'}")
lines.append("")

# pos_5 snippet for the money table (layers 0..9)
lines.append("### pos=5 excerpt (layers 0-9, sdpa_out columns)")
lines.append("")
lines.append("| Layer | sdpa_max | sdpa_nrmse |")
lines.append("|-------|----------|------------|")
for e in pos5_data[:10]:
    L = e["layer"]
    mx = e["sdpa_out_max"]
    nr = e["sdpa_out_nrmse"]
    lines.append(f"| {L:2d} | {fmt(mx)} | {fmt(nr)} |")
lines.append("")

with open(SUMMARY_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Summary written: {SUMMARY_OUT}")

# ── print key numbers to stdout ────────────────────────────────────────────────
print("\n=== KEY NUMBERS ===")
print(f"Global first breach: {global_first_breach}")
print(f"Kernel bound holds: {kernel_bound_holds}")
print("\nFirst breach per op:")
for op in OPS:
    b = first_breach[op]
    print(f"  {op}: {b}")

print("\nsdpa_out max nrmse by layer (first 10):")
for L in range(10):
    print(f"  layer {L:2d}: {sdpa_nrmse_by_layer[L]}")

# ── check drilldown trigger ────────────────────────────────────────────────────
drilldown_needed = False
drilldown_reason = ""
if global_first_breach:
    # Unambiguous bug-candidate — check if clean (single op, clear early breach)
    breach_layer, breach_pos, breach_op, breach_mag = global_first_breach
    classification = "bug-candidate"
    # Check if only one op breaches, and it breaches early
    num_ops_breaching = sum(1 for op in OPS if first_breach[op] is not None)
    if num_ops_breaching > 1 or breach_layer > 15:
        drilldown_needed = True
        drilldown_reason = f"Multiple ops breach ({num_ops_breaching}) or late breach (layer={breach_layer})"
    else:
        drilldown_needed = False
        drilldown_reason = f"Single-op clean breach at layer={breach_layer} pos={breach_pos} op={breach_op}"
else:
    # No breach — check nrmse accumulation
    max_sdpa_nrmse_at_pos30 = max(
        (data.get(L, {}).get(30, {}).get("sdpa_out", {}).get("nrmse", 0.0) or 0.0)
        for L in range(NUM_LAYERS)
    )
    if max_sdpa_nrmse_at_pos30 > NRMSE_DRILLDOWN:
        drilldown_needed = True
        drilldown_reason = f"sdpa_out nrmse at pos30 reaches {max_sdpa_nrmse_at_pos30:.4f} > {NRMSE_DRILLDOWN}"
        classification = "floor-candidate" if kernel_bound_holds else "ambiguous"
    else:
        classification = "floor-candidate" if kernel_bound_holds else "ambiguous"
        drilldown_reason = "no breach, nrmse well within bounds"

print(f"\nClassification: {classification}")
print(f"Drilldown needed: {drilldown_needed}  ({drilldown_reason})")

# Export results for caller
print("\n=== EXPORT ===")
print(f"CLASSIFICATION={classification}")
print(f"DRILLDOWN_NEEDED={drilldown_needed}")
if global_first_breach:
    L, P, op, mag = global_first_breach
    print(f"BREACH_LAYER={L}")
    print(f"BREACH_POS={P}")
    print(f"BREACH_OP={op}")
    print(f"BREACH_MAG={mag:.6e}")
else:
    print("BREACH_LAYER=None")
    print("BREACH_POS=None")
    print("BREACH_OP=None")
    print("BREACH_MAG=None")
print(f"KERNEL_BOUND_HOLDS={kernel_bound_holds}")
print(f"DRILLDOWN_REASON={drilldown_reason}")

# Summary of max_abs_diff for largest value seen (for "no breach" reporting)
if not global_first_breach:
    all_max = [
        (r["layer"], r["pos"], r["op"], r["max_abs_diff"])
        for r in rows if r["note"] == ""
    ]
    worst = max(all_max, key=lambda x: x[3])
    print(f"LARGEST_MAX_ABS_DIFF: layer={worst[0]} pos={worst[1]} op={worst[2]} val={worst[3]:.6e}")
