#!/usr/bin/env python3
"""Synthesize a GLP test vector for the hardware gate.

Two artifacts:
- zero.glp.gguf: all-zero directions at the steered layers. The no-op canary:
  serving with this must produce logits identical to unsteered.
- probe.glp.gguf: a fixed pseudo-random direction at a few layers. The live
  canary: logits must shift by >1e-3 on a fixed probe prompt.

Writes spec-conformant GGUF v3: glp.mode=project, glp.spec_version=1,
glp.hook_point=residual_stream_post_layer, direction.<N> fp32 width=4096
(DeepSeek-V4 hidden). Layers chosen inside the GLP-29 range (L10-38).
"""

from __future__ import annotations

import hashlib
import random
import struct
import sys

HIDDEN = 4096  # DeepSeek-V4 hidden width
LAYERS = [10, 20, 30]  # a few steered layers inside the published GLP-29 range


def write_gguf(path: str, vectors: dict[int, list[float]]) -> None:
    meta = [
        ("glp.mode", "project"),
        ("glp.spec_version", 1),
        ("glp.hook_point", "residual_stream_post_layer"),
        ("glp.alpha_default", 1.0),
        ("glp.rank", 1),
        ("general.name", "hf2q-glp-canary"),
        ("glp.method", "synthetic-canary"),
    ]
    tensors = [(f"direction.{layer}", vec) for layer, vec in sorted(vectors.items())]

    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(meta))

    def wstr(s: str) -> None:
        b = s.encode()
        out.extend(struct.pack("<Q", len(b)))
        out.extend(b)

    for key, value in meta:
        wstr(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8)  # string
            wstr(value)
        elif isinstance(value, int):
            out += struct.pack("<I", 4)  # u32
            out += struct.pack("<I", value)
        elif isinstance(value, float):
            out += struct.pack("<I", 6)  # f32
            out += struct.pack("<f", value)

    offset = 0
    for name, vec in tensors:
        wstr(name)
        out += struct.pack("<I", 1)  # n_dims
        out += struct.pack("<Q", len(vec))
        out += struct.pack("<I", 0)  # F32
        out += struct.pack("<Q", offset)
        offset += len(vec) * 4

    # align tensor data to 32
    pad = (32 - len(out) % 32) % 32
    out += b"\x00" * pad
    for _, vec in tensors:
        out += struct.pack(f"<{len(vec)}f", *vec)

    sys.stdout.write(f"wrote {path} ({len(out)} bytes, sha256 {hashlib.sha256(bytes(out)).hexdigest()[:12]})\n")
    with open(path, "wb") as fh:
        fh.write(out)


def main() -> None:
    zero = {layer: [0.0] * HIDDEN for layer in LAYERS}
    write_gguf("/opt/hf2q/scripts/grammar_probe/zero.glp.gguf", zero)

    rng = random.Random(20260904)
    probe = {}
    for layer in LAYERS:
        vec = [rng.gauss(0, 1) for _ in range(HIDDEN)]
        norm = sum(x * x for x in vec) ** 0.5
        probe[layer] = [x / norm for x in vec]
    write_gguf("/opt/hf2q/scripts/grammar_probe/probe.glp.gguf", probe)


if __name__ == "__main__":
    main()
