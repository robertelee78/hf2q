#!/usr/bin/env python3
"""Independent per-tensor and full-file sha256 of a GGUF file (developer harness; stdlib only).

ADR-066 / ADR-063 Q4-DS4-RACE section 6.1 T2 (docs/research/adr066-iq2xxs-subset-design.md): the hf2q loader test
(tests/deepseek4_ds4_file_parity.rs) compares the sha256 of the bytes it exposes for every tensor with this reader's,
and this reader's full-file digest verifies the MODEL_SHA256 the case files declare. It never loads a model, maps
the file, or depends on hf2q, the peer or any converter: the header is parsed with struct and the data is streamed
once, sequentially, in 64 MiB chunks into the file digest and the digest of whichever tensor the bytes belong to.

Usage: ds4_file_digest.py FILE --output OUT.json [--expect-sha256 HEX]

OUT.json (schema hf2q.adr066.gguf-file-digest.v1):
  file, bytes, file_sha256, version, alignment, data_offset, n_tensors, and
  tensors: [{name, ggml_type, offset, nbytes, sha256}] in header order, where
    ggml_type is the GGML type id (16 = IQ2_XXS, 10 = Q2_K, 8 = Q8_0, 1 = F16, 0 = F32, ...),
    offset is the absolute byte offset of the tensor's data in the file (data_offset + the header's relative offset),
    nbytes is the block-geometry size (elements / block size * type size), never the padding up to the next tensor,
    sha256 is the digest of exactly those nbytes bytes.
  With --expect-sha256: expected_file_sha256 and file_sha256_matches; a mismatch exits 3 after writing the receipt.
An unknown GGML type, a misaligned, overlapping or out-of-file tensor, or a truncated file refuses (exit 2).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

SCHEMA = "hf2q.adr066.gguf-file-digest.v1"
CHUNK = 64 << 20
MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
# GGUF metadata value type id -> fixed byte size (8 string and 9 array are variable).
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY, UINT32 = 8, 9, 4
# GGML type id -> (name, elements per block, bytes per block), as ggml-common.h defines the block structs.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2)}


class DigestError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise DigestError(reason)


class Reader:
    """Little-endian GGUF header reader over a buffered file; refuses a short read."""

    def __init__(self, stream):
        self.stream = stream

    def take(self, n):
        data = self.stream.read(n)
        require(len(data) == n, "truncated GGUF header")
        return data

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8")

    def skip_value(self, kind):
        """Skip one metadata value without materializing it (tokenizer arrays stay unread)."""
        if kind in SCALAR_BYTES:
            self.take(SCALAR_BYTES[kind])
        elif kind == STRING:
            self.take(self.unpack("<Q"))
        elif kind == ARRAY:
            inner, count = self.unpack("<I"), self.unpack("<Q")
            if inner in SCALAR_BYTES:
                self.take(SCALAR_BYTES[inner] * count)
            else:
                for _ in range(count):
                    self.skip_value(inner)
        else:
            raise DigestError(f"unknown GGUF metadata value type {kind}")


def tensor_nbytes(name, dims, ggml_type):
    require(ggml_type in GGML_TYPES, f"{name}: unknown GGML type id {ggml_type}")
    _, block, size = GGML_TYPES[ggml_type]
    require(dims and all(d > 0 for d in dims), f"{name}: empty tensor shape {dims}")
    require(dims[0] % block == 0, f"{name}: first dimension {dims[0]} is not a multiple of the {block}-element block")
    elements = 1
    for d in dims:
        elements *= d
    return elements // block * size


def read_header(path):
    """(version, alignment, data_offset, tensors in header order with absolute offsets and block-geometry nbytes)."""
    with open(path, "rb") as stream:
        r = Reader(stream)
        require(r.take(4) == MAGIC, f"{path} is not a GGUF file")
        version = r.unpack("<I")
        require(version in (2, 3), f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        alignment = DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            key, kind = r.string(), r.unpack("<I")
            if key == "general.alignment":
                require(kind == UINT32, "general.alignment must be uint32")
                alignment = r.unpack("<I")
                require(alignment > 0 and alignment % 8 == 0, f"invalid general.alignment {alignment}")
            else:
                r.skip_value(kind)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.unpack("<Q") for _ in range(r.unpack("<I"))]
            ggml_type, relative = r.unpack("<I"), r.unpack("<Q")
            tensors.append({"name": name, "ggml_type": ggml_type, "relative": relative,
                            "nbytes": tensor_nbytes(name, dims, ggml_type)})
        data_offset = (stream.tell() + alignment - 1) // alignment * alignment
    require(len({t["name"] for t in tensors}) == len(tensors), "duplicate tensor names")
    for t in tensors:
        require(t["relative"] % alignment == 0, f"{t['name']}: offset {t['relative']} is not {alignment}-aligned")
        t["offset"] = data_offset + t.pop("relative")
    return version, alignment, data_offset, tensors


def digest(path, chunk=CHUNK, expect=None):
    """The receipt document: one sequential pass feeds the file digest and each tensor's digest."""
    path = Path(path).resolve()
    size = os.path.getsize(path)
    version, alignment, data_offset, tensors = read_header(path)
    spans = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    for previous, current in zip(spans, spans[1:]):
        a, b = tensors[previous], tensors[current]
        require(a["offset"] + a["nbytes"] <= b["offset"], f"{a['name']} overlaps {b['name']}")
    if spans:
        last = tensors[spans[-1]]
        require(last["offset"] + last["nbytes"] <= size, f"{last['name']} extends past the end of the file")
    file_hash, hashes = hashlib.sha256(), [hashlib.sha256() for _ in tensors]
    buffer = bytearray(chunk)
    view = memoryview(buffer)
    position, first = 0, 0  # spans[first:] are the tensors not yet fully read
    with open(path, "rb", buffering=0) as stream:
        while True:
            n = stream.readinto(buffer)
            if not n:
                break
            block, end = view[:n], position + n
            file_hash.update(block)
            j = first
            while j < len(spans):
                t = tensors[spans[j]]
                start, stop = t["offset"], t["offset"] + t["nbytes"]
                if start >= end:
                    break
                low, high = max(start, position), min(stop, end)
                if high > low:
                    hashes[spans[j]].update(block[low - position:high - position])
                if stop > end:
                    break
                j += 1
            first, position = j, end
    require(position == size, f"{path} changed size while it was read ({position} of {size} bytes)")
    out = {"schema": SCHEMA, "file": str(path), "bytes": size, "file_sha256": file_hash.hexdigest(), "version": version,
           "alignment": alignment, "data_offset": data_offset, "n_tensors": len(tensors),
           "tensors": [{"name": t["name"], "ggml_type": t["ggml_type"], "offset": t["offset"], "nbytes": t["nbytes"],
                        "sha256": h.hexdigest()} for t, h in zip(tensors, hashes)]}
    if expect is not None:
        out.update(expected_file_sha256=expect, file_sha256_matches=out["file_sha256"] == expect)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="the GGUF file to digest (read only)")
    parser.add_argument("--output", type=Path, required=True, help="new receipt JSON (never overwritten)")
    parser.add_argument("--expect-sha256", help="64-hex full-file sha256 the file must have (exit 3 when it differs)")
    args = parser.parse_args(argv)
    if args.expect_sha256 is not None and not (len(args.expect_sha256) == 64
                                               and all(c in "0123456789abcdef" for c in args.expect_sha256)):
        parser.error("--expect-sha256 must be 64 lowercase hex")
    try:
        require(not args.output.exists(), f"{args.output} exists; a receipt is never overwritten")
        document = digest(args.file, expect=args.expect_sha256)
    except (OSError, DigestError, UnicodeDecodeError) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 2
    with open(args.output, "x") as stream:
        json.dump(document, stream, indent=1)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "file_sha256": document["file_sha256"],
                      "n_tensors": document["n_tensors"],
                      **({"file_sha256_matches": document["file_sha256_matches"]} if args.expect_sha256 else {})}))
    return 3 if args.expect_sha256 is not None and not document["file_sha256_matches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
