#!/usr/bin/env python3
"""Build ADR-059 canary graft artifacts for a qwen35/qwen35moe/gemma4/deepseek4 model.

Two artifacts, both GGUF v3 with `graft.*` metadata (ADR-059 container):

* zero-slot canary — `graft.n_slots = 0`, no K/V tensors. The engine
  treats it as a no-op (splice returns 0, nothing touched): grafted
  output must be byte-identical to the ungrafted baseline.
* live graft — deterministic, distinct K/V rows over every covered
  layer (complete site coverage). Grafted output must differ from the
  baseline (the splice participates in attention).

deepseek4 is the `compressed_kv` site (ADR-059 #308): the bank is
FABRICATED COMPRESSED ROWS for every `compress_ratios[layer] != 0`
layer, exported in the #309 producer's exact `to_gguf` shape
(`graft.compress_ratios` i32 array parallel to the model's, shared-KV
MQA tensors `[n_slots, 1, head_dim]` with K == V bytes, BF16-rounded
F32 values — the splice encodes through the cache's BF16 rows —
`graft.content_sha256` over raw F32 K-then-V bytes ascending-layer,
RoPE keys for checkpoint identity only, no `position_base` / family
rope flag at this site).

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
import hashlib
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
                # Arrays the graft bind reads (gemma4 site geometry):
                # INT32 (per-layer head_count_kv), BOOL
                # (sliding_window_pattern), and U32 (deepseek4
                # attention.compress_ratios). Other element types are
                # skipped (the bind does not consume them).
                etype, = struct.unpack("<I", f.read(4))
                count, = struct.unpack("<Q", f.read(8))
                if etype == 5:
                    out[key] = list(struct.unpack(f"<{count}i", f.read(4 * count)))
                elif etype == 4:
                    out[key] = list(struct.unpack(f"<{count}I", f.read(4 * count)))
                elif etype == 7:
                    out[key] = [f.read(1) != b"\0" for _ in range(count)]
                else:
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


def site_geometry(meta):
    """The full_attn_kv site geometry for a model GGUF's architecture,
    mirroring GraftModelShape::from_gguf (the bind the canary must pass).

    gemma4: full-attn layers from the per-layer
    `gemma4.attention.sliding_window_pattern` bool array (True=sliding,
    False=full; every-6th fallback when absent), per-layer
    `gemma4.attention.head_count_kv` i32 array (the FULL layers' count
    is the site geometry), standard RoPE (mrope_interleaved=False).

    deepseek4: the `compressed_kv` site (CompressedKvShape::from_gguf) —
    covered layers are `compress_ratios[layer] != 0`, shared-KV MQA
    (`head_count_kv` must be 1), head dim `key_length`, RoPE identity
    = the COMPRESSOR rope (`compress_rope_freq_base` +
    `rope.dimension_count`), and the model's full per-layer
    `compress_ratios` schedule travels with the geometry.
    """
    arch = meta["general.architecture"]
    block_count = meta[f"{arch}.block_count"]
    if arch in ("qwen35", "qwen35moe"):
        interval = meta[f"{arch}.full_attention_interval"]
        return {
            "arch": arch,
            "layers": full_attention_layers(block_count, interval),
            "heads": meta[f"{arch}.attention.head_count_kv"],
            "head_dim": meta[f"{arch}.attention.key_length"],
            "rope_theta": meta[f"{arch}.rope.freq_base"],
            "rotary_dim": meta[f"{arch}.rope.dimension_count"],
            "mrope_interleaved": True,
        }
    if arch == "gemma4":
        pattern = meta.get("gemma4.attention.sliding_window_pattern")
        if isinstance(pattern, list) and len(pattern) == block_count:
            layers = [i for i in range(block_count) if not pattern[i]]
        else:
            # Fallback mirroring the bind: every 6th layer is full.
            layers = [i for i in range(block_count) if (i + 1) % 6 == 0]
        assert layers, "model exposes no full-attention layers"
        head_count_kv = meta["gemma4.attention.head_count_kv"]
        assert isinstance(head_count_kv, list) and len(head_count_kv) == block_count, \
            "gemma4.attention.head_count_kv must be a per-layer array"
        heads = head_count_kv[layers[0]]
        assert heads > 0, "full-attention layer reports 0 KV heads"
        return {
            "arch": arch,
            "layers": layers,
            "heads": heads,
            # Global (full-attention) geometry — NOT the _swa variants.
            "head_dim": meta["gemma4.attention.key_length"],
            "rope_theta": meta["gemma4.rope.freq_base"],
            "rotary_dim": meta["gemma4.rope.dimension_count"],
            # Gemma-4 uses standard RoPE; IMROPE interleaving is the
            # Qwen3.5-family convention (bind refuses the mismatch).
            "mrope_interleaved": False,
        }
    if arch == "deepseek4":
        compress_ratios = meta["deepseek4.attention.compress_ratios"]
        assert isinstance(compress_ratios, list) \
            and len(compress_ratios) == block_count, \
            "deepseek4.attention.compress_ratios must be a per-layer array"
        layers = [i for i in range(block_count) if compress_ratios[i] != 0]
        assert layers, "model exposes no compressed (ratio != 0) layers"
        heads = meta["deepseek4.attention.head_count_kv"]
        assert heads == 1, \
            "the compressed_kv site is shared-KV MQA (head_count_kv must be 1)"
        return {
            "arch": arch,
            "layers": layers,
            "heads": heads,
            "head_dim": meta["deepseek4.attention.key_length"],
            # Checkpoint identity ONLY: the compressor's rope space
            # (compress_rope_freq_base + the rope head dim). Compressed
            # rows are merged, positionless state.
            "rope_theta": meta["deepseek4.attention.compress_rope_freq_base"],
            "rotary_dim": meta["deepseek4.rope.dimension_count"],
            "compress_ratios": compress_ratios,
        }
    raise AssertionError(f"unsupported canary arch {arch!r}")


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


def kv_i32_array(key, values):
    """GGUF v3 metadata array: type 9, element type i32 (4), count, items —
    byte-identical to hf2q's writer (backends/gguf/types.rs write_array)
    and the #309 producer's `_gguf_kv_i32_array`."""
    body = b"".join(struct.pack("<i", v) for v in values)
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 9) + \
        struct.pack("<I", 4) + struct.pack("<Q", len(values)) + body


def bf16_round(value):
    """Round an f32 through bf16 exactly as Rust's `f32_to_bf16_bits`
    (round-to-nearest-even on the low 16 bits). The compressed splice
    encodes through the cache's BF16 rows, so every exported F32 value
    must be exactly bf16-representable (the #309 to-gguf contract)."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def write_graft(path, meta, n_slots, scale, quant_lane):
    arch = meta["general.architecture"]
    geometry = site_geometry(meta)
    layers = geometry["layers"]
    assert layers, "model exposes no full-attention layers"

    metadata = [
        kv_string("graft.mode", "splice_prefix"),
        kv_u32("graft.spec_version", 1),
        kv_string("graft.hook_point", "full_attn_kv"),
        kv_string("graft.kind", "direct_kv"),
        kv_u32("graft.n_slots", n_slots),
        kv_f32("graft.rope_theta", geometry["rope_theta"]),
        kv_u32("graft.rotary_dim", geometry["rotary_dim"]),
        kv_u32("graft.position_base", 0),
        kv_bool("graft.mrope_interleaved", geometry["mrope_interleaved"]),
        kv_string("graft.quant_lane", quant_lane),
    ]

    tensors = []
    if n_slots > 0:
        heads = geometry["heads"]
        head_dim = geometry["head_dim"]
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
        f"heads={geometry['heads']} head_dim={geometry['head_dim']} "
        f"mrope_interleaved={geometry['mrope_interleaved']} bytes={len(out)}"
    )


def write_graft_compressed(path, meta, n_slots, scale, quant_lane):
    """Write the graft.* GGUF for the deepseek4 `compressed_kv` site —
    the #309 producer's exact export shape (phantom_compressed.py
    `write_compressed_gguf`), on a synthetic bank.

    Contract keys (ADR-059 compressed_kv section): hook_point=
    compressed_kv; graft.compress_ratios i32 array parallel to the
    model's per-layer array; RoPE keys for CHECKPOINT IDENTITY ONLY
    (graft.position_base does not apply at this site, and no family
    rope flag applies — neither is emitted); content_sha256 over raw
    f32 K-then-V bytes in ascending layer order (the reader's rule);
    tensors graft.k.<layer>/graft.v.<layer> in [n_slots, kv_heads,
    head_dim] with K == V bytes (shared-KV MQA) and bf16-rounded values.
    """
    geometry = site_geometry(meta)
    layers = geometry["layers"]
    assert layers, "model exposes no compressed (ratio != 0) layers"

    metadata = [
        kv_string("graft.mode", "splice_prefix"),
        kv_u32("graft.spec_version", 1),
        kv_string("graft.hook_point", "compressed_kv"),
        kv_string("graft.kind", "direct_kv"),
        kv_u32("graft.n_slots", n_slots),
        kv_i32_array("graft.compress_ratios", geometry["compress_ratios"]),
        kv_f32("graft.rope_theta", geometry["rope_theta"]),
        kv_u32("graft.rotary_dim", geometry["rotary_dim"]),
        kv_string("graft.quant_lane", quant_lane),
    ]

    tensors = []
    if n_slots > 0:
        heads = geometry["heads"]
        head_dim = geometry["head_dim"]
        for layer in layers:
            # Deterministic, distinct, zero-mean wave (the full_attn_kv
            # canary pattern), bf16-rounded so the splice through the
            # cache's BF16 rows is lossless. Shared-KV MQA: K and V are
            # the SAME rows — identical payloads for both tensor names.
            n = n_slots * heads * head_dim
            rows = []
            for i in range(n):
                phase = (i % 97) / 97.0 * 2.0 * math.pi
                rows.append(struct.pack("<f", bf16_round(scale * math.sin(phase))))
            payload = b"".join(rows)
            for side in ("k", "v"):
                tensors.append(
                    (f"graft.{side}.{layer}", [n_slots, heads, head_dim], payload)
                )
        digest = hashlib.sha256()
        for _name, _dims, payload in tensors:
            digest.update(payload)
        metadata.append(kv_string("graft.content_sha256", digest.hexdigest()))
    metadata.append(
        kv_string("general.base_model.0.name", "DeepSeek-V4-Flash-0731")
    )

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
        f"heads={geometry['heads']} head_dim={geometry['head_dim']} "
        f"compress_ratios={geometry['compress_ratios']} bytes={len(out)}"
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
    assert arch in ("qwen35", "qwen35moe", "gemma4", "deepseek4"), \
        f"unsupported canary arch {arch!r}"

    if arch == "deepseek4":
        # The compressed_kv site: the #309 producer's export shape.
        write_graft_compressed(args.out_zero, meta, 0, args.scale, args.quant_lane)
        write_graft_compressed(
            args.out_live, meta, args.n_slots, args.scale, args.quant_lane
        )
        return

    write_graft(args.out_zero, meta, 0, args.scale, args.quant_lane)
    write_graft(args.out_live, meta, args.n_slots, args.scale, args.quant_lane)


if __name__ == "__main__":
    main()
