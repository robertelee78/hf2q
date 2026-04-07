#!/usr/bin/env python3
"""
Verify that our Rust mlx_affine_quantize packing matches MLX's mx.quantize
for all bit widths (2, 3, 4, 6, 8).

This script:
1. Creates a known weight tensor
2. Quantizes with mx.quantize for each bit width
3. Prints the raw byte layout of the packed weights
4. Shows how many values per pack and the bit arrangement
"""

import mlx.core as mx
import numpy as np

def print_bytes_as_hex(data_bytes, label, max_bytes=48):
    """Print raw bytes as hex."""
    hex_str = ' '.join(f'{b:02x}' for b in data_bytes[:max_bytes])
    print(f"  {label} (first {min(len(data_bytes), max_bytes)} bytes): {hex_str}")

def print_u32_words(data_bytes, label, max_words=12):
    """Print as uint32 LE words."""
    words = []
    for i in range(0, min(len(data_bytes), max_words * 4), 4):
        w = int.from_bytes(data_bytes[i:i+4], 'little')
        words.append(f'{w:08x}')
    print(f"  {label} (first {len(words)} uint32 LE): {' '.join(words)}")

def analyze_packing(bits, group_size=64):
    """Analyze MLX packing for a given bit width."""
    print(f"\n{'='*60}")
    print(f"  {bits}-bit quantization (group_size={group_size})")
    print(f"{'='*60}")

    # Create a small reproducible weight tensor
    mx.random.seed(42)
    w = mx.random.normal((1, group_size))  # single row, single group

    qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(qw, qs, qb)

    print(f"  Weight shape: {qw.shape}, dtype: {qw.dtype}")
    print(f"  Scales shape: {qs.shape}, dtype: {qs.dtype}")
    print(f"  Biases shape: {qb.shape}, dtype: {qb.dtype}")

    # Get raw bytes
    qw_np = np.array(qw)
    raw_bytes = qw_np.tobytes()

    print(f"  Total bytes: {len(raw_bytes)}")
    print_bytes_as_hex(raw_bytes, "Raw bytes")
    print_u32_words(raw_bytes, "As uint32")

    # Decode the packed values to verify
    n_bins = (1 << bits) - 1
    scale_val = float(qs[0, 0])
    bias_val = float(qb[0, 0])
    print(f"  Scale: {scale_val:.6f}, Bias: {bias_val:.6f}")

    power_of_2 = (bits & (bits - 1)) == 0
    if power_of_2:
        el_per_int = 32 // bits
        print(f"  Power-of-2: {el_per_int} values per uint32")
        # Decode first uint32
        first_word = int.from_bytes(raw_bytes[0:4], 'little')
        vals = []
        for k in range(el_per_int):
            q = (first_word >> (k * bits)) & ((1 << bits) - 1)
            vals.append(q)
        print(f"  First uint32 decoded values: {vals}")
    else:
        el_per_int = 8 if bits == 3 else 4
        print(f"  Non-power-of-2: {el_per_int} values per 3 bytes (24 bits)")
        # Decode first 3 bytes
        first_pack = raw_bytes[0] | (raw_bytes[1] << 8) | (raw_bytes[2] << 16)
        vals = []
        for k in range(el_per_int):
            q = (first_pack >> (k * bits)) & ((1 << bits) - 1)
            vals.append(q)
        print(f"  First 3-byte pack decoded values: {vals}")

    # Verify round-trip: dequantize and compare
    dw = mx.dequantize(qw, qs, qb, group_size=group_size, bits=bits)
    mx.eval(dw)
    max_err = float(mx.max(mx.abs(w - dw)))
    print(f"  Max dequantization error: {max_err:.6f}")

    # Verify total value count
    if power_of_2:
        total_packed = (len(raw_bytes) // 4) * (32 // bits)
    else:
        total_packed = (len(raw_bytes) // 3) * el_per_int
    print(f"  Total values packed: {total_packed} (expected: {group_size})")

    return raw_bytes, scale_val, bias_val


def verify_rust_packing(bits, group_size=64):
    """
    Reproduce the Rust mlx_affine_quantize logic in Python to verify
    it produces the same output as mx.quantize.
    """
    print(f"\n  --- Rust-equivalent packing for {bits}-bit ---")

    mx.random.seed(42)
    w = mx.random.normal((1, group_size))
    mx.eval(w)
    w_np = np.array(w).flatten().astype(np.float32)

    # MLX quantize for reference
    qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(qw, qs, qb)
    mlx_bytes = np.array(qw).tobytes()

    # Reproduce Rust logic
    n_bins = float((1 << bits) - 1)
    eps = 1e-7

    w_min = float(np.min(w_np))
    w_max = float(np.max(w_np))
    mask = abs(w_min) > abs(w_max)
    scale = max((w_max - w_min) / n_bins, eps)
    if not mask:
        scale = -scale
    edge = w_min if mask else w_max
    q0 = round(edge / scale)
    bias = 0.0
    if q0 != 0:
        scale = edge / q0
        bias = edge

    # Convert to bf16 and back (to match precision loss)
    import struct
    def to_bf16(f):
        b = struct.pack('f', f)
        return struct.unpack('f', bytes([0, 0, b[2], b[3]]))[0]

    scale_bf16 = to_bf16(scale)
    bias_bf16 = to_bf16(bias)

    power_of_2 = (bits & (bits - 1)) == 0
    if bits == 3:
        el_per_int = 8
    elif bits == 6:
        el_per_int = 4
    else:
        el_per_int = 32 // bits

    bytes_per_pack = 1 if power_of_2 else 3
    int_per_group = group_size * bytes_per_pack // el_per_int
    packs = int_per_group // bytes_per_pack

    rust_bytes = bytearray()
    val_idx = 0
    for j in range(packs):
        out_el = 0
        for k in range(el_per_int):
            w_el = float(w_np[val_idx]) if val_idx < len(w_np) else 0.0
            q = round((w_el - bias) / scale)
            q = max(0.0, min(q, n_bins))
            out_el |= int(q) << (k * bits)
            val_idx += 1
        if power_of_2:
            rust_bytes.extend(out_el.to_bytes(4, 'little'))
        else:
            rust_bytes.append(out_el & 0xff)
            rust_bytes.append((out_el >> 8) & 0xff)
            rust_bytes.append((out_el >> 16) & 0xff)

    rust_bytes = bytes(rust_bytes)
    match = mlx_bytes == rust_bytes
    print(f"  Rust-equiv bytes match MLX: {match}")
    if not match:
        print(f"  MLX  bytes: {' '.join(f'{b:02x}' for b in mlx_bytes[:48])}")
        print(f"  Rust bytes: {' '.join(f'{b:02x}' for b in rust_bytes[:48])}")
        # Find first difference
        for i in range(min(len(mlx_bytes), len(rust_bytes))):
            if mlx_bytes[i] != rust_bytes[i]:
                print(f"  First diff at byte {i}: MLX={mlx_bytes[i]:02x} Rust={rust_bytes[i]:02x}")
                break
    print(f"  Values packed: {val_idx}")
    return match


if __name__ == '__main__':
    all_pass = True
    for bits in [2, 3, 4, 6, 8]:
        analyze_packing(bits)
        ok = verify_rust_packing(bits)
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("  ALL BIT WIDTHS MATCH")
    else:
        print("  SOME BIT WIDTHS MISMATCH -- see above")
    print(f"{'='*60}")
