#!/usr/bin/env python3
"""Extract `tokenizer.chat_template` string from a GGUF file.

2026-05-03 — built per user directive for the coherence harness. Walks the
GGUF KV header in pure Python (no llama.cpp dep required) and writes the
chat_template value to stdout."""
import sys
import struct

# GGML value types (gguf-py spec).
TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

SCALAR_SIZE = {
    TYPE_UINT8: 1, TYPE_INT8: 1,
    TYPE_UINT16: 2, TYPE_INT16: 2,
    TYPE_UINT32: 4, TYPE_INT32: 4, TYPE_FLOAT32: 4,
    TYPE_BOOL: 1,
    TYPE_UINT64: 8, TYPE_INT64: 8, TYPE_FLOAT64: 8,
}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def skip_value(f, vtype):
    if vtype in SCALAR_SIZE:
        f.read(SCALAR_SIZE[vtype])
    elif vtype == TYPE_STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(n)
    elif vtype == TYPE_ARRAY:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if etype in SCALAR_SIZE:
            f.read(SCALAR_SIZE[etype] * n)
        elif etype == TYPE_STRING:
            for _ in range(n):
                m = struct.unpack("<Q", f.read(8))[0]
                f.read(m)
        else:
            raise ValueError(f"unsupported nested array etype={etype}")
    else:
        raise ValueError(f"unknown value type {vtype}")


def main(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"not a GGUF file: magic={magic!r}", file=sys.stderr)
            sys.exit(2)
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            print(f"unsupported GGUF version {version}", file=sys.stderr)
            sys.exit(2)
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if key == "tokenizer.chat_template" and vtype == TYPE_STRING:
                value = read_string(f)
                sys.stdout.write(value)
                return
            skip_value(f, vtype)
    print("tokenizer.chat_template not found", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: extract_template.py <gguf>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
