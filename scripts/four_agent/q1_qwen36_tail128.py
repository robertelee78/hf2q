"""ADR-063 Q1-QWEN36-SPEC (#653) Tail128 workload: mechanical derivation and check.

  python3 -B q1_qwen36_tail128.py derive SUFFIX144_WORKLOAD > tail128.json
  python3 -B q1_qwen36_tail128.py record SUFFIX144_WORKLOAD TAIL128 > derivation.json
  python3 -B q1_qwen36_tail128.py check SUFFIX144_WORKLOAD TAIL128 DERIVATION

The T5 input `Qwen36ToolRoundTail128` is derived from the D1-accepted
`Qwen36ToolRoundSuffix144` workload (sha256 INPUT_SHA256) and nothing else:
for each lane, Tail128.prompts[l] = Suffix144.lanes[l][16..144], the last 128
tokens of each reviewed prompt. No lane is reselected and no other prompt
source is read. The output is the T5 runner's preparation `Prefixes` document
{family, vocab_size, prompts[4]} (closed keys, serde form of `Family`).

`record` re-derives TAIL128, requires its bytes to equal the derived bytes and
writes the derivation record binding the input sha256 and the output sha256.
`check` re-derives both the Tail128 bytes and the derivation record and
byte-compares them with the given files before preparation; any difference
refuses (non-zero exit).
"""
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "hf2q.q1.qwen36-tail128-derivation.v1"
CHECK_SCHEMA = "hf2q.q1.qwen36-tail128-check.v1"
INPUT_SHA256 = "db413896a984f8a7559df70739738c85a8dcc059fecf307c00e3051846ae9880"
FAMILY = "Qwen36"
LANES = 4
FULL = 144
START = 16
RULE = "Tail128.prompts[l] = Suffix144.lanes[l][16..144]"


def require(condition, message):
    if not condition:
        raise SystemExit(f"q1_qwen36_tail128: {message}")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def derive(source, expected_sha256=INPUT_SHA256):
    """Suffix144 workload bytes -> Tail128 `Prefixes` bytes."""
    require(sha256(source) == expected_sha256, f"input sha256 is not the accepted Suffix144 workload {expected_sha256}")
    work = json.loads(source)
    require(isinstance(work, dict) and set(work) == {"family", "lanes", "vocab_size"},
            "Suffix144 workload keys must be exactly family, lanes, vocab_size")
    vocab = work["vocab_size"]
    require(work["family"] == FAMILY, "Suffix144 family must be Qwen36")
    require(type(vocab) is int and vocab > 0, "vocab_size must be a positive integer")
    lanes = work["lanes"]
    require(isinstance(lanes, list) and len(lanes) == LANES, "exactly four lanes")
    for lane, tokens in enumerate(lanes):
        require(isinstance(tokens, list) and len(tokens) == FULL, f"lane {lane}: exactly 144 tokens")
        require(all(type(t) is int and 0 <= t < vocab for t in tokens), f"lane {lane}: token outside vocabulary")
    prompts = [tokens[START:FULL] for tokens in lanes]
    require(len({tuple(p) for p in prompts}) == LANES, "Tail128 lane prompts must stay distinct")
    return encode({"family": FAMILY, "vocab_size": vocab, "prompts": prompts})


def record(source_path, source, tail, expected_sha256=INPUT_SHA256):
    """Derivation record for TAIL128; refuses unless TAIL128 is the derived bytes."""
    derived = derive(source, expected_sha256)
    require(tail == derived, "TAIL128 bytes differ from the mechanical derivation")
    return encode({
        "schema": SCHEMA,
        "label": "Qwen36ToolRoundTail128",
        "source_label": "Qwen36ToolRoundSuffix144",
        "rule": RULE,
        "window": {"start": START, "end": FULL, "tokens_per_lane": FULL - START},
        "lanes": LANES,
        "family": FAMILY,
        "input": {"path": str(source_path), "sha256": sha256(source), "bytes": len(source)},
        "output": {"sha256": sha256(derived), "bytes": len(derived)},
        "lanes_reselected": False,
        "other_prompt_sources": False,
        "whole_q1_qualified": False,
    })


def check(source_path, source, tail, derivation, expected_sha256=INPUT_SHA256):
    expected = record(source_path, source, tail, expected_sha256)
    require(derivation == expected, "DERIVATION bytes differ from the re-derived record")
    return encode({"schema": CHECK_SCHEMA, "verdict": "Match", "input_sha256": sha256(source),
                   "output_sha256": sha256(tail), "derivation_sha256": sha256(derivation)})


def main(argv):
    if len(argv) == 3 and argv[1] == "derive":
        sys.stdout.buffer.write(derive(Path(argv[2]).read_bytes()))
    elif len(argv) == 4 and argv[1] == "record":
        path = Path(argv[2]).resolve()
        sys.stdout.buffer.write(record(path, path.read_bytes(), Path(argv[3]).read_bytes()))
    elif len(argv) == 5 and argv[1] == "check":
        path = Path(argv[2]).resolve()
        sys.stdout.buffer.write(check(path, path.read_bytes(), Path(argv[3]).read_bytes(), Path(argv[4]).read_bytes()))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
