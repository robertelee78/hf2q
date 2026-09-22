#!/usr/bin/env python3
"""Build ADR-059 canary graft artifacts for a qwen35/qwen35moe model.

Two artifacts, both GGUF v3 with `graft.*` metadata (ADR-059 container):

* zero-slot canary — `graft.n_slots = 0`, no K/V tensors. The engine
  treats it as a no-op (splice returns 0, nothing touched): grafted
  output must be byte-identical to the ungrafted baseline.
* live graft — deterministic, distinct K/V rows over every full-attention
  layer (complete site coverage). Grafted output must differ from the
  baseline (the splice participates in attention).

Pure stdlib: the GGUF writer mirrors the hf2q reader's format exactly
(magic, v3, tensor count, metadata kv list, tensor infos, 32-byte
alignment, F32 tensor data).

Usage:
  python3 build_canary_graft.py --model /path/to/model.gguf \
      --out-zero zero.graft.gguf --out-live live.graft.gguf \
      [--n-slots 8] [--scale 0.5]
"""

from __future__ import annotations

import argparse
import math
import struct


def read_gguf_meta(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, = struct.unpack("<I", f.read(4))
        assert version == 3, version
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))
        out = {}
        for _ in range(n_kv):
            klen, = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode()
            vtype, = struct.unpack("<I", f.read(4))
            if vtype == 8:
                slen, = struct.unpack("<Q", f.read(8))
                out[key] = f.read(slen).decode()
            elif vtype == 4:
                out[key], = struct.unpack("<I", f.read(4))
            elif vtype == 6:
                out[key], = struct.unpack("<f", f.read(4))
            elif vtype == 10:
                out[key], = struct.unpack("<Q", f.read(8))
            elif vtype == 5:
                out[key], = struct.unpack("<i", f.read(4))
            elif vtype == 7:
                out[key] = f.read(1) != b"\0"
            elif vtype == 9:
                etype, = struct.unpack("<I", f.read(4))
                count, = struct.unpack("<Q", f.read(8))
                skip_array(f, etype, count)
            else:
                raise ValueError(f"metadata type {vtype} for {key}")
        return out


def skip_array(f, etype, count):
    sizes = {0: 4, 1: 4, 2: 8, 3: 1, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 2, 13: 2}
    for _ in range(count):
        if etype == 8:
            slen, = struct.unpack("<Q", f.read(8))
            f.read(slen)
        else:
            f.read(sizes[etype])


def full_attention_layers(block_count, interval):
    if interval == 0:
        return []
    return [i for i in range(block_count) if (i + 1) % interval == 0]


def kv_string(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) + \
        struct.pack("<Q", len(value)) + value.encode()


def kv_u32(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + \
        struct.pack("<I", value)


def kv_f32(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 6) + \
        struct.pack("<f", value)


def kv_bool(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 7) + \
        bytes([1 if value else 0])


def write_graft(path, meta, n_slots, scale, quant_lane):
    arch = meta["general.architecture"]
    block_count = meta[f"{arch}.block_count"]
    interval = meta[f"{arch}.full_attention_interval"]
    heads = meta[f"{arch}.attention.head_count_kv"]
    head_dim = meta[f"{arch}.attention.key_length"]
    rope_theta = meta[f"{arch}.rope.freq_base"]
    rotary_dim = meta[f"{arch}.rope.dimension_count"]
    layers = full_attention_layers(block_count, interval)
    assert layers, "model exposes no full-attention layers"

    metadata = [
        kv_string("graft.mode", "splice_prefix"),
        kv_u32("graft.spec_version", 1),
        kv_string("graft.hook_point", "full_attn_kv"),
        kv_string("graft.kind", "direct_kv"),
        kv_u32("graft.n_slots", n_slots),
        kv_f32("graft.rope_theta", rope_theta),
        kv_u32("graft.rotary_dim", rotary_dim),
        kv_u32("graft.position_base", 0),
        kv_bool("graft.mrope_interleaved", True),
        kv_string("graft.quant_lane", quant_lane),
    ]

    tensors = []
    if n_slots > 0:
        for layer in layers:
            for side in ("k", "v"):
                n = n_slots * heads * head_dim
                rows = []
                for i in range(n):
                    # Deterministic, distinct, zero-mean pattern: a smooth
                    # wave scaled by `scale` (and sign-flipped for V) so
                    # the graft perturbs attention measurably at any
                    # reasonable scale.
                    phase = (i % 97) / 97.0 * 2.0 * math.pi
                    value = scale * math.sin(phase + (0.3 if side == "v" else 0.0))
                    rows.append(struct.pack("<f", value))
                tensors.append((f"graft.{side}.{layer}", [n_slots, heads, head_dim], b"".join(rows)))

    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))
    for kv in metadata:
        out += kv
    offset = 0
    for name, dims, _payload in tensors:
        out += struct.pack("<Q", len(name)) + name.encode()
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", 0)  # F32
        out += struct.pack("<Q", offset)
        offset += 4 * dims[0] * dims[1] * dims[2]
    while len(out) % 32 != 0:
        out += b"\0"
    for _name, _dims, payload in tensors:
        out += payload
    with open(path, "wb") as f:
        f.write(bytes(out))
    print(
        f"wrote {path}: n_slots={n_slots} layers={layers} "
        f"heads={heads} head_dim={head_dim} bytes={len(out)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model GGUF (metadata source)")
    parser.add_argument("--out-zero", required=True, help="zero-slot canary output path")
    parser.add_argument("--out-live", required=True, help="live graft output path")
    parser.add_argument("--n-slots", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--quant-lane", default="canary-synthetic")
    args = parser.parse_args()

    meta = read_gguf_meta(args.model)
    arch = meta["general.architecture"]
    assert arch in ("qwen35", "qwen35moe"), f"unsupported canary arch {arch!r}"

    write_graft(args.out_zero, meta, 0, args.scale, args.quant_lane)
    write_graft(args.out_live, meta, args.n_slots, args.scale, args.quant_lane)


if __name__ == "__main__":
    main()
