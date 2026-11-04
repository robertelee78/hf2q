"""Controls of the section 6.1 T2 GGUF digest harness on tiny synthetic GGUF files (no model file is read)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import ds4_file_digest as dd


def gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def synthetic(tensors, alignment=32, version=3, extra_kv=True, tail=b""):
    """(GGUF bytes, {name: (absolute offset, data bytes)}). tensors: [(name, dims, ggml type, data bytes)]; data is laid
    out in order with alignment padding, like a writer would; tail is appended after the last tensor."""
    kv = [gguf_string("general.alignment") + struct.pack("<II", dd.UINT32, alignment)]
    if extra_kv:
        kv += [gguf_string("general.name") + struct.pack("<I", dd.STRING) + gguf_string("tiny"),
               gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", dd.ARRAY, dd.STRING, 3)
               + b"".join(gguf_string(t) for t in ("a", "bb", "ccc")),
               gguf_string("tokenizer.ggml.scores") + struct.pack("<IIQ", dd.ARRAY, 6, 2) + struct.pack("<ff", 0.5, 1.5),
               gguf_string("deepseek4.block_count") + struct.pack("<IQ", 10, 43)]
    infos, relative, layout = [], 0, []
    for name, dims, kind, data in tensors:
        relative = (relative + alignment - 1) // alignment * alignment
        infos.append(gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                     + struct.pack("<IQ", kind, relative))
        layout.append((name, relative, data))
        relative += len(data)
    head = gguf_header(version, len(tensors), len(kv)) + b"".join(kv) + b"".join(infos)
    data_offset = (len(head) + alignment - 1) // alignment * alignment
    blob = bytearray(head + b"\0" * (data_offset - len(head)))
    where = {}
    for name, offset, data in layout:
        blob += b"\0" * (data_offset + offset - len(blob))
        where[name] = (data_offset + offset, data)
        blob += data
    return bytes(blob) + tail, where


def gguf_header(version, n_tensors, n_kv):
    return dd.MAGIC + struct.pack("<IQQ", version, n_tensors, n_kv)


def payload(n, seed):
    return bytes((seed * 31 + i * 7) % 251 for i in range(n))


TENSORS = [("token_embd.weight", [64, 3], 1, payload(64 * 3 * 2, 1)),          # F16
           ("blk.0.ffn_gate_exps.weight", [512, 2], 16, payload(2 * 2 * 66, 2)),  # IQ2_XXS: 4 blocks of 66 bytes
           ("blk.0.attn_q_a.weight", [32, 5], 8, payload(5 * 34, 3)),          # Q8_0
           ("output_norm.weight", [7], 0, payload(28, 4))]                     # F32


class Digest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, blob, name="tiny.gguf"):
        path = self.root / name
        path.write_bytes(blob)
        return path

    def test_every_tensor_and_the_file_match_an_independent_hash_at_any_chunk_size(self):
        blob, where = synthetic(TENSORS, tail=b"\0" * 13)
        path = self.write(blob)
        for chunk in (1, 7, 64, 65, 4096, dd.CHUNK):
            with self.subTest(chunk=chunk):
                document = dd.digest(path, chunk)
                self.assertEqual(document["file_sha256"], hashlib.sha256(blob).hexdigest())
                self.assertEqual((document["schema"], document["bytes"], document["version"], document["alignment"],
                                  document["n_tensors"]), (dd.SCHEMA, len(blob), 3, 32, 4))
                self.assertEqual([t["name"] for t in document["tensors"]], [t[0] for t in TENSORS])  # header order
                for entry, (name, _, kind, data) in zip(document["tensors"], TENSORS):
                    offset, _ = where[name]
                    self.assertEqual((entry["ggml_type"], entry["offset"], entry["nbytes"]), (kind, offset, len(data)))
                    self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
                    self.assertEqual(blob[offset:offset + entry["nbytes"]], data)
                self.assertEqual(document["tensors"][0]["offset"], document["data_offset"])

    def test_block_geometry_sizes(self):
        self.assertEqual(dd.tensor_nbytes("x", [4096, 2048], 16), 4096 * 2048 // 256 * 66)  # IQ2_XXS
        self.assertEqual(dd.tensor_nbytes("x", [4096, 2048, 256], 10), 4096 * 2048 * 256 // 256 * 84)  # Q2_K
        self.assertEqual(dd.tensor_nbytes("x", [4096, 1024], 8), 4096 * 1024 // 32 * 34)  # Q8_0
        self.assertEqual(dd.tensor_nbytes("x", [10, 3], 1), 60)  # F16
        with self.assertRaisesRegex(dd.DigestError, "unknown GGML type id 43"):
            dd.tensor_nbytes("x", [256], 43)
        with self.assertRaisesRegex(dd.DigestError, "not a multiple of the 256-element block"):
            dd.tensor_nbytes("x", [100, 2], 16)

    def test_malformed_files_are_refused(self):
        good, _ = synthetic(TENSORS)
        cases = {"not a GGUF file": b"GGML" + good[4:],
                 "unsupported GGUF version 1": dd.MAGIC + struct.pack("<I", 1) + good[8:],
                 "truncated GGUF header": good[:40],
                 "extends past the end of the file": good[:-1]}
        for reason, blob in cases.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(dd.DigestError, reason):
                dd.digest(self.write(blob, "bad.gguf"))
        overlap, _ = synthetic([("a", [8], 0, payload(32, 1)), ("b", [8], 0, payload(32, 2))])
        # point tensor b's relative offset at a's data
        info = gguf_string("b") + struct.pack("<IQIQ", 1, 8, 0, 32)
        overlap = overlap.replace(info, gguf_string("b") + struct.pack("<IQIQ", 1, 8, 0, 0))
        with self.assertRaisesRegex(dd.DigestError, "a overlaps b"):
            dd.digest(self.write(overlap, "overlap.gguf"))
        misaligned = overlap.replace(gguf_string("b") + struct.pack("<IQIQ", 1, 8, 0, 0),
                                     gguf_string("b") + struct.pack("<IQIQ", 1, 8, 0, 40))
        with self.assertRaisesRegex(dd.DigestError, "offset 40 is not 32-aligned"):
            dd.digest(self.write(misaligned, "misaligned.gguf"))

    def test_cli_writes_a_new_receipt_and_reports_an_unexpected_file_digest(self):
        blob, _ = synthetic(TENSORS)
        path, out = self.write(blob), self.root / "t2-digest.json"
        expect = hashlib.sha256(blob).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dd.main([str(path), "--output", str(out), "--expect-sha256", expect]), 0)
        document = json.loads(out.read_text())
        self.assertEqual((document["file_sha256"], document["file_sha256_matches"]), (expect, True))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(dd.main([str(path), "--output", str(out)]), 2)  # never overwritten
        other = self.root / "t2-other.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dd.main([str(path), "--output", str(other), "--expect-sha256", "0" * 64]), 3)
        self.assertFalse(json.loads(other.read_text())["file_sha256_matches"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dd.main([str(path), "--output", str(self.root / "x.json"), "--expect-sha256", "ABC"])


if __name__ == "__main__":
    unittest.main()
