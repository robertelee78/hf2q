#!/usr/bin/env python3
"""ADR-054 follow-up item 3 (#298): re-point a calibrated GLP export at its
capture site, the DeepSeek post-layer residual.

Modes:
  hook <in> <out>
      Copy the calibrated export, rewriting ONLY the `glp.hook_point` string
      KV to `residual_stream_post_layer`. `glp.derived_at` (already
      `residual_stream_post_layer`) is untouched; every other KV keeps its
      exact wire bytes; the tensor-data blob is copied verbatim. Tensor
      offsets are relative to the (re-padded) data-section start, so they stay
      valid. `glp.content_sha256` covers the raw F32 direction tensors only
      (src/inference/glp/reader.rs, producer ordering by numeric layer) —
      the direction bytes are unchanged, so the hash stays valid.

  tile <in> <out> --hc N [--hidden H]
      The hook rewrite PLUS tiling every direction.<N> tensor across the
      model's HC streams: dims [hidden] -> [hc*hidden], payload = the
      hidden-width F32 vector repeated hc times. This is the per-stream
      width `apply_layer_gpu_mhc_in_session` validates
      (src/inference/glp/apply_gpu.rs: dir slice = [stream*hidden,
      (stream+1)*hidden)). The tiled direction bytes differ, so
      `glp.content_sha256` is recomputed over the tiled bytes — a stale hash
      is refused at load (reader.rs content_sha256 mismatch check).

GGUF v3 seek-back layout (matches src/backends/gguf/writer.rs):
  magic u32 | version u32 | tensor_count u64 | kv_count u64 |
  KVs (key: u64-len string; value_type u32; value) |
  tensor infos (name string; n_dims u32; dims u64[]; ggml_type u32; offset u64)
  | zero pad to alignment (general.alignment, default 32) | tensor data.
"""
import argparse
import hashlib
import struct
import sys

MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT_DEFAULT = 32
GGML_F32 = 0
NEW_HOOK = "residual_stream_post_layer"

VALUE_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                     10: 8, 11: 8, 12: 8}


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        if self.pos + n > len(self.data):
            raise ValueError(f"read past EOF at {self.pos}+{n}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8")


def read_value_raw(reader, value_type):
    """Return the raw wire bytes of one metadata value."""
    if value_type == 8:
        start = reader.pos
        n = reader.u64()
        reader.take(n)
        return reader.data[start:reader.pos]
    if value_type == 9:
        start = reader.pos
        elem_type = reader.u32()
        count = reader.u64()
        for _ in range(count):
            read_value_raw(reader, elem_type)
        return reader.data[start:reader.pos]
    size = VALUE_FIXED_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"unsupported metadata value type {value_type}")
    return reader.take(size)


def parse(path):
    data = open(path, "rb").read()
    r = Reader(data)
    if r.take(4) != MAGIC:
        raise ValueError("bad GGUF magic")
    version = r.u32()
    if version != VERSION:
        raise ValueError(f"expected GGUF v3, got v{version}")
    tensor_count = r.u64()
    kv_count = r.u64()
    kvs = []  # (key, value_type, raw_value_bytes)
    for _ in range(kv_count):
        key = r.string()
        value_type = r.u32()
        raw = read_value_raw(r, value_type)
        kvs.append([key, value_type, raw])
    tensors = []  # dict(name, n_dims, dims, ggml_type, offset)
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        ggml_type = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "n_dims": n_dims, "dims": dims,
                        "ggml_type": ggml_type, "offset": offset})
    alignment = ALIGNMENT_DEFAULT
    for key, vt, raw in kvs:
        if key == "general.alignment" and vt == 4:
            alignment = struct.unpack("<I", raw)[0]
    data_offset = -(-r.pos // alignment) * alignment  # div_ceil -> pad
    return data, kvs, tensors, alignment, data_offset


def encode_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def meta_bytes(kvs):
    out = bytearray()
    for key, vt, raw in kvs:
        out += encode_string(key) + struct.pack("<I", vt) + raw
    return bytes(out)


def align_up(x, a):
    return -(-x // a) * a


def layer_of(name):
    if not name.startswith("direction."):
        return None
    suffix = name[len("direction."):]
    if not suffix.isdigit() or int(suffix) == 0:
        return None
    return int(suffix)


def check_source(kvs, tensors, hidden):
    meta = {k: (vt, raw) for k, vt, raw in kvs}
    vt, raw = meta["glp.hook_point"]
    if vt != 8:
        raise ValueError("glp.hook_point is not a string")
    hook = raw[8:].decode()
    if hook != "ffn_out_pre_residual":
        raise ValueError(
            f"source glp.hook_point is {hook!r}, expected ffn_out_pre_residual")
    vt, raw = meta["glp.derived_at"]
    derived = raw[8:].decode()
    if derived != "residual_stream_post_layer":
        raise ValueError(
            f"glp.derived_at is {derived!r}, expected residual_stream_post_layer")
    for t in tensors:
        if layer_of(t["name"]) is None:
            raise ValueError(f"non-direction tensor {t['name']!r}")
        if t["ggml_type"] != GGML_F32:
            raise ValueError(f"{t['name']} is not F32")
        if t["n_dims"] != 1:
            raise ValueError(f"{t['name']} is not 1-D")
        if t["dims"][0] != hidden:
            raise ValueError(
                f"{t['name']} width {t['dims'][0]} != hidden {hidden}")


def rewrite(in_path, out_path, tile, hc, hidden_arg):
    data, kvs, tensors, alignment, data_offset = parse(in_path)
    hidden = tensors[0]["dims"][0]
    if hidden_arg and hidden_arg != hidden:
        raise ValueError(
            f"--hidden {hidden_arg} does not match tensor width {hidden}")
    check_source(kvs, tensors, hidden)

    # 1. the hook rewrite: glp.hook_point -> residual_stream_post_layer
    #    (string raw value = u64 length prefix + bytes, same wire shape the
    #    reader's read_value_raw returns)
    for kv in kvs:
        if kv[0] == "glp.hook_point":
            kv[2] = encode_string(NEW_HOOK)

    # 2. tile mode: widen direction tensors, recompute content_sha256
    new_payloads = None
    if tile:
        new_payloads = {}
        for t in tensors:
            w = t["dims"][0]
            start = data_offset + t["offset"]
            payload = data[start:start + w * 4]
            if len(payload) != w * 4:
                raise ValueError(f"{t['name']} payload exceeds file")
            tiled = payload * hc
            t["dims"] = [w * hc]
            new_payloads[t["name"]] = tiled
        hasher = hashlib.sha256()
        for name in sorted(new_payloads, key=layer_of):
            hasher.update(new_payloads[name])
        content_sha = hasher.hexdigest()
        for kv in kvs:
            if kv[0] == "glp.content_sha256":
                kv[2] = encode_string(content_sha)

    # 3. re-serialize: header + KVs + tensor infos + pad + data
    #    (tensor offsets are recomputed relative to the new data-section
    #    start, mirroring the seek-back writer's sequential aligned layout)
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(kvs))
    out += meta_bytes(kvs)
    infos = bytearray()
    offsets = []
    rel = 0
    for t in tensors:
        if new_payloads:
            payload = new_payloads[t["name"]]
        else:
            start = data_offset + t["offset"]
            payload = data[start:start + t["dims"][0] * 4]
        infos += encode_string(t["name"])
        infos += struct.pack("<I", t["n_dims"])
        for d in t["dims"]:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", t["ggml_type"])
        offsets.append(len(infos))
        infos += struct.pack("<Q", rel)
        if not new_payloads and rel != t["offset"]:
            # hook mode copies the data blob verbatim, so the recomputed
            # offsets MUST equal the originals — a mismatch means the
            # layout model is wrong; refuse rather than ship bad offsets.
            raise ValueError(
                f"{t['name']}: recomputed offset {rel} != original "
                f"{t['offset']}")
        rel = align_up(rel + len(payload), alignment)
    out += infos
    pad = align_up(len(out), alignment) - len(out)
    out += b"\x00" * pad
    data_start = len(out)
    if new_payloads:
        for i, t in enumerate(tensors):
            rel_off = struct.unpack_from("<Q", infos, offsets[i])[0]
            out[data_start + rel_off:data_start + rel_off] = \
                new_payloads[t["name"]]
    else:
        out += data[data_offset:]
    with open(out_path, "wb") as fh:
        fh.write(out)

    for label, path in (("in ", in_path), ("out", out_path)):
        print(f"sha256 {label} {hashlib.sha256(open(path, 'rb').read()).hexdigest()}  {path}")
    print(f"hook_point: ffn_out_pre_residual -> {NEW_HOOK}")
    print(f"derived_at: unchanged (residual_stream_post_layer)")
    if tile:
        print(f"tiled: direction tensors {hidden} -> {hc}*{hidden} "
              f"({hc} copies of the stream-0 vector); glp.content_sha256 "
              f"recomputed over the tiled bytes: {content_sha}")
    else:
        print("direction bytes: unchanged (glp.content_sha256 stays valid)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["hook", "tile"])
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--hc", type=int, default=None,
                    help="HC stream count (tile mode; required)")
    ap.add_argument("--hidden", type=int, default=None,
                    help="expected hidden width (verified, not assumed)")
    args = ap.parse_args()
    if args.mode == "tile" and not args.hc:
        ap.error("tile mode requires --hc")
    rewrite(args.in_path, args.out_path,
            tile=(args.mode == "tile"), hc=args.hc, hidden_arg=args.hidden)


if __name__ == "__main__":
    main()
