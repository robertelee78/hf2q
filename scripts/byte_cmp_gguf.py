#!/usr/bin/env python3
"""ADR-033 §P1 per-tensor byte-equivalence check.

Compares two GGUF files tensor-by-tensor and reports any divergence.
Usage: byte_cmp_gguf.py <canonical.gguf> <candidate.gguf>
Exit 0 iff every tensor matches byte-for-byte.
"""
import sys
import gguf
import numpy as np

if len(sys.argv) != 3:
    print(__doc__, file=sys.stderr)
    sys.exit(2)

canonical_path, candidate_path = sys.argv[1], sys.argv[2]

a = gguf.GGUFReader(canonical_path, "r")
b = gguf.GGUFReader(candidate_path, "r")

by_name_a = {t.name: t for t in a.tensors}
by_name_b = {t.name: t for t in b.tensors}

missing_in_b = set(by_name_a) - set(by_name_b)
missing_in_a = set(by_name_b) - set(by_name_a)
if missing_in_b:
    print(f"[ERR] {len(missing_in_b)} tensors missing in candidate: {sorted(missing_in_b)[:5]}")
if missing_in_a:
    print(f"[ERR] {len(missing_in_a)} extra tensors in candidate: {sorted(missing_in_a)[:5]}")

per_type_total = {}
per_type_mismatch = {}
mismatches = []

for name in sorted(by_name_a):
    if name not in by_name_b:
        continue
    ta, tb = by_name_a[name], by_name_b[name]
    if ta.tensor_type != tb.tensor_type:
        print(f"[TYPE-MISMATCH] {name}: canonical={ta.tensor_type.name} vs candidate={tb.tensor_type.name}")
        continue
    tname = ta.tensor_type.name
    per_type_total[tname] = per_type_total.get(tname, 0) + 1
    ba = bytes(ta.data.tobytes())
    bb = bytes(tb.data.tobytes())
    if ba == bb:
        continue
    per_type_mismatch[tname] = per_type_mismatch.get(tname, 0) + 1
    arr_a = np.frombuffer(ba, dtype=np.uint8)
    arr_b = np.frombuffer(bb, dtype=np.uint8)
    n = min(len(arr_a), len(arr_b))
    diff_mask = arr_a[:n] != arr_b[:n]
    n_diff = int(diff_mask.sum())
    first_diff = int(np.argmax(diff_mask)) if n_diff > 0 else -1
    pct = 100.0 * n_diff / n if n > 0 else 0.0
    mismatches.append((name, tname, len(ba), n_diff, first_diff, pct))

print("\n=== PER-TYPE SUMMARY ===")
for tname in sorted(per_type_total):
    mm = per_type_mismatch.get(tname, 0)
    tot = per_type_total[tname]
    status = "OK" if mm == 0 else "DIFF"
    print(f"  {tname:8s} {mm:>4}/{tot:<4} mismatches  [{status}]")

if mismatches:
    print(f"\n=== {len(mismatches)} MISMATCHED TENSORS (first 20) ===")
    for name, tname, nb, nd, fd, pct in mismatches[:20]:
        print(f"  {name:50s} {tname:8s} bytes={nb:>10}  diffs={nd:>8}  first_off={fd:>10}  pct={pct:.4f}%")
    sys.exit(1)

total = sum(per_type_total.values())
print(f"\n[OK] All {total} tensors byte-identical.")
sys.exit(0)
