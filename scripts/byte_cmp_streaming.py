#!/usr/bin/env python3
"""Memory-efficient streaming byte-cmp of two GGUF files.

The default in-line `tobytes()` + zip approach buffers entire tensor blobs
(plus the diff iterator state) in Python RAM, which OOMs for 16GB GGUFs.
This script uses GGUFReader's per-tensor numpy arrays (memmapped) and
streams 64KB chunks via numpy.frombuffer to keep peak RAM under ~256MB.

Usage: byte_cmp_streaming.py <canonical.gguf> <hf2q.gguf>
"""
import sys
import numpy as np
import gguf

CHUNK = 64 * 1024 * 1024  # 64MB chunks


def diff_bytes(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing bytes between two memmapped ndarrays of equal shape/dtype."""
    av = a.view(np.uint8).ravel()
    bv = b.view(np.uint8).ravel()
    if av.shape != bv.shape:
        return -1
    total_diff = 0
    n = av.shape[0]
    for off in range(0, n, CHUNK):
        end = min(off + CHUNK, n)
        total_diff += int(np.count_nonzero(av[off:end] != bv[off:end]))
    return total_diff


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <canonical.gguf> <hf2q.gguf>", file=sys.stderr)
        sys.exit(2)
    canon_path, hf2q_path = sys.argv[1], sys.argv[2]

    canon = gguf.GGUFReader(canon_path, "r")
    hf2q = gguf.GGUFReader(hf2q_path, "r")

    c_dict = {t.name: t for t in canon.tensors}
    h_dict = {t.name: t for t in hf2q.tensors}

    totals: dict[str, list[int]] = {}
    missing = []
    shape_skip = []

    for name in sorted(c_dict):
        if name not in h_dict:
            missing.append(name)
            continue
        ct, ht = c_dict[name], h_dict[name]
        c_bytes = ct.data.nbytes
        h_bytes = ht.data.nbytes
        if c_bytes != h_bytes:
            shape_skip.append((name, c_bytes, h_bytes))
            continue
        t_type = ct.tensor_type.name
        d = diff_bytes(ct.data, ht.data)
        totals.setdefault(t_type, [0, 0])
        totals[t_type][0] += d
        totals[t_type][1] += c_bytes

    print("Per-tensor-type byte residuals:")
    overall_d, overall_t = 0, 0
    for k, (d, t) in sorted(totals.items()):
        if t > 0:
            pct = 100 * d / t
            print(f"  {k:>8}: {d:>14}/{t:<14} = {pct:.6f}%")
            overall_d += d
            overall_t += t
    print()
    if overall_t > 0:
        print(f"  OVERALL: {overall_d}/{overall_t} = {100*overall_d/overall_t:.6f}%")
    if missing:
        print(f"\nMissing in hf2q ({len(missing)}):")
        for m in missing[:10]:
            print(f"  {m}")
    if shape_skip:
        print(f"\nShape-mismatched (skipped, {len(shape_skip)}):")
        for n, cb, hb in shape_skip[:10]:
            print(f"  {n}: canonical={cb}B hf2q={hb}B")


if __name__ == "__main__":
    main()
