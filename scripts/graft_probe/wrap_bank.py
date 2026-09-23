#!/usr/bin/env python3
"""Wrap an hf2q self-donor derivation dump into a graft.* GGUF (ADR-059).

Input: the sidecar + raw rows written by `HF2Q_GRAFT_DERIVE=<prefix>:<n>`
during a donor prefill on the F32 control path (hf2q generate):

  <prefix>.meta.json            — identity + geometry (see derive_and_dump)
  <prefix>.layer<L>.k.f32 / .v.f32 — [n_slots, n_kv_heads, head_dim] F32

Output: a `graft.*` GGUF whose checkpoint identity
(general.base_model.0.*) is exactly what the bind's CheckpointIdentity
trust boundary resolved for the source artifact — the bank is
checkpoint-bound to the model it was derived from.

The container layout mirrors scripts/graft_probe/build_canary_graft.py
(GGUF v3, graft.* metadata, F32 tensors graft.k.<layer>/graft.v.<layer>).

Usage:
  python3 wrap_bank.py --prefix /tmp/apex-bank --out /tmp/apex.graft.gguf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32


def kv_string(key: str, value: str) -> bytes:
    # type 8: string
    return (
        struct.pack("<Q", len(key))
        + key.encode()
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(value))
        + value.encode()
    )


def kv_u32(key: str, value: int) -> bytes:
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_f32(key: str, value: float) -> bytes:
    # GGUF metadata type 6 = F32
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 6) + struct.pack("<f", value)


def kv_bool(key: str, value: bool) -> bytes:
    # GGUF metadata type 7 = BOOL
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 7) + struct.pack("<B", 1 if value else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="derivation dump prefix")
    parser.add_argument("--out", required=True, help="output graft GGUF path")
    parser.add_argument(
        "--scale-k", type=float, default=1.0,
        help="multiply K rows by this factor (attention-pull axis)")
    parser.add_argument(
        "--scale-v", type=float, default=1.0,
        help="multiply V rows by this factor (injected-content axis)")
    args = parser.parse_args()

    prefix = Path(args.prefix)
    meta = json.loads((prefix.parent / (prefix.name + ".meta.json")).read_text())
    n_slots = int(meta["n_slots"])
    layers = [int(layer) for layer in meta["layers"]]
    heads = int(meta["n_kv_heads"])
    head_dim = int(meta["head_dim"])
    if n_slots <= 0:
        print("refusing to wrap a zero-slot derivation (use the canary builder for that)", file=sys.stderr)
        return 2

    def scale_rows(payload: bytes, factor: float) -> bytes:
        if factor == 1.0:
            return payload
        values = struct.unpack("<%df" % (len(payload) // 4), payload)
        return struct.pack("<%df" % len(values), *(v * factor for v in values))

    metadata = [
        kv_string("graft.mode", "splice_prefix"),
        kv_u32("graft.spec_version", 1),
        kv_string("graft.hook_point", "full_attn_kv"),
        kv_string("graft.kind", "direct_kv"),
        kv_u32("graft.n_slots", n_slots),
        kv_f32("graft.rope_theta", float(meta["rope_theta"])),
        kv_u32("graft.rotary_dim", int(meta["rotary_dim"])),
        kv_u32("graft.position_base", 0),
        kv_bool("graft.mrope_interleaved", True),
        kv_string("graft.quant_lane", "f32"),
    ]
    # Checkpoint identity: exactly the fields the bind resolves for the
    # source artifact. Undeclared fields are omitted (never invented).
    identity = meta.get("identity", {})
    if identity.get("name"):
        metadata.append(kv_string("general.base_model.0.name", identity["name"]))
    if identity.get("organization"):
        metadata.append(kv_string("general.base_model.0.organization", identity["organization"]))
    if identity.get("repository"):
        metadata.append(kv_string("general.base_model.0.repository", identity["repository"]))
    if identity.get("revision"):
        metadata.append(kv_string("general.base_model.0.version", identity["revision"]))

    tensors = []
    expected_elems = n_slots * heads * head_dim
    for layer in layers:
        for side in ("k", "v"):
            payload = (prefix.parent / (prefix.name + f".layer{layer}.{side}.f32")).read_bytes()
            if len(payload) != 4 * expected_elems:
                print(
                    f"layer {layer} {side}: {len(payload)} bytes != {4 * expected_elems}",
                    file=sys.stderr,
                )
                return 2
            factor = args.scale_k if side == "k" else args.scale_v
            payload = scale_rows(payload, factor)
            tensors.append(
                (f"graft.{side}.{layer}", [n_slots, heads, head_dim], payload)
            )

    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack("<I", GGUF_VERSION)
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
    while len(out) % GGUF_DEFAULT_ALIGNMENT != 0:
        out += b"\0"
    for _name, _dims, payload in tensors:
        out += payload
    Path(args.out).write_bytes(bytes(out))
    print(
        f"wrote {args.out}: n_slots={n_slots} layers={layers} "
        f"heads={heads} head_dim={head_dim} identity={identity} bytes={len(out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
