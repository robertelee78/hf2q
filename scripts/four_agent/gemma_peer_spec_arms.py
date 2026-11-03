#!/usr/bin/env python3
"""ADR-063 Q4-GEMMA (#660): the peer's Gemma 4 speculation arms for the best-valid-configuration search.

Every arm is q5_cases/peer-gemma4.json (the Q3 v9 peer arm) with only its trailing `--spec-type none`
replaced, so model, slots, context, KV, sampling, template and readiness stay byte-identical. The arm set
is every speculation type the pinned peer build (b128-e86f5fd) can run on a Gemma 4 26B-A4B target:

  P0  none            the Q3 v9 peer arm (baseline and transcript reference)
  NM  ngram-mod       shared-pool rolling-hash n-gram drafts (the peer's --spec-default type), defaults
  NS  ngram-simple    last-match n-gram drafts, defaults (n 12, m 48)
  NK  ngram-map-k     keyed n-gram map, defaults (n 12, m 48, min hits 1)
  NV  ngram-map-k4v   keyed map with four values, the peer documentation's repetition setting (8/8/2, 64)
  NC  ngram-cache     n-gram statistics cache (no static/dynamic lookup files), defaults
  MT  draft-mtp       Google's Gemma 4 assistant drafter (gemma4-assistant, shares the target KV)
  E3  draft-eagle3    Red Hat's EAGLE-3 speculator for this target
  DF  draft-dflash    z-lab's DFlash block drafter for this target (block 16, n-max 15)

Draft-model arms need a reference-only drafter GGUF (converted by the peer's own converter, never shipped
or served by hf2q); its path and sha256 are case variables DRAFT_MODEL / DRAFT_SHA256, and the runner must
re-hash before launch. DSpark is absent: the pinned build supports only Qwen3 DSpark backbones and no
26B-A4B DSpark checkpoint is published. Greedy drafting throughout, as the target samples at temperature 0.

  gemma_peer_spec_arms.py BASE_CASE OUT_DIR [--drafters drafters.json]

drafters.json: {"MT": {"path": ..., "sha256": ..., "source": ...}, "E3": {...}, "DF": {...}}; draft arms
without an entry are not written. Refuses to change an existing file whose bytes differ; writes
OUT_DIR/cases-receipt.json with every case's sha256.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import q5_case

ARMS = {
    "P0": ("none", []),
    "NM": ("ngram-mod", ["--spec-type", "ngram-mod"]),
    "NS": ("ngram-simple", ["--spec-type", "ngram-simple"]),
    "NK": ("ngram-map-k", ["--spec-type", "ngram-map-k"]),
    "NV": ("ngram-map-k4v", ["--spec-type", "ngram-map-k4v", "--spec-ngram-map-k4v-size-n", "8",
                             "--spec-ngram-map-k4v-size-m", "8", "--spec-ngram-map-k4v-min-hits", "2",
                             "--spec-draft-n-max", "64"]),
    "NC": ("ngram-cache", ["--spec-type", "ngram-cache"]),
    "MT": ("mtp-assistant", ["--spec-type", "draft-mtp", "--spec-draft-model", "{DRAFT_MODEL}",
                             "--spec-draft-n-max", "3", "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
                             "--spec-draft-sampling", "greedy", "--spec-draft-backend-sampling"]),
    "E3": ("eagle3", ["--spec-type", "draft-eagle3", "--spec-draft-model", "{DRAFT_MODEL}",
                      "--spec-draft-n-max", "3", "--spec-draft-sampling", "greedy"]),
    "DF": ("dflash", ["--spec-type", "draft-dflash", "--spec-draft-model", "{DRAFT_MODEL}",
                      "--spec-draft-n-max", "15", "--spec-draft-sampling", "greedy"]),
}
DRAFT_ARMS = ("MT", "E3", "DF")
# The `--spec-type` each arm selects: the name the peer prints in its `adding speculative implementation`
# and `statistics <type>:` lines (visible at --log-verbosity 4) for qualification evidence.
EXPECTED_TYPE = {arm: (flags[1] if flags else None) for arm, (_, flags) in ARMS.items()}


class ArmError(ValueError):
    pass


def case_name(arm):
    return f"G-{arm}-{ARMS[arm][0]}"


def derive(base, arm, drafter=None):
    """The arm's case: base with its final `--spec-type none` pair replaced by the arm's flags."""
    if arm not in ARMS:
        raise ArmError(f"unknown arm {arm}")
    if (arm in DRAFT_ARMS) != (drafter is not None):
        raise ArmError(f"arm {arm}: a drafter is {'required' if arm in DRAFT_ARMS else 'not allowed'}")
    case = copy.deepcopy(base)
    argv = case["serve"]["argv"]
    if argv[-2:] != ["--spec-type", "none"] or argv.count("--spec-type") != 1:
        raise ArmError("base serve.argv must end with its only --spec-type none")
    label, flags = ARMS[arm]
    if arm != "P0":
        case["serve"]["argv"] = argv[:-2] + flags
    case["workload"]["id"] += f"-spec-{label}"
    note = (f"ADR-063 Q4-GEMMA peer configuration search arm {arm} ({label}): q5_cases/peer-gemma4.json with "
            f"`--spec-type none` replaced by {' '.join(flags) if flags else '(unchanged)'}; nothing else changed.")
    case["notes"] = case["notes"] + [note]
    if drafter is not None:
        for key in ("path", "sha256"):
            if not isinstance(drafter.get(key), str) or not drafter[key]:
                raise ArmError(f"arm {arm}: drafter {key} required")
        if len(drafter["sha256"]) != 64:
            raise ArmError(f"arm {arm}: drafter sha256 must be 64 hex digits")
        case["variables"]["DRAFT_MODEL"] = drafter["path"]
        case["variables"]["DRAFT_SHA256"] = drafter["sha256"]
        case["notes"].append(
            f"Drafter: reference-only benchmark input ({drafter.get('source', 'unrecorded source')}), never "
            "shipped or served by hf2q. Q5 binds one artifact sha256 (the target); the runner must verify "
            "shasum -a 256 {DRAFT_MODEL} == DRAFT_SHA256 before launch.")
    return q5_case.validate(case)


def save(path, case):
    data = (json.dumps(case, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise ArmError(f"refusing to change existing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--drafters")
    args = ap.parse_args(argv)
    base_path = Path(args.base)
    base = q5_case.validate(q5_case.loads(base_path.read_text()))
    drafters = json.loads(Path(args.drafters).read_text()) if args.drafters else {}
    receipt = {"base": str(base_path), "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
               "cases": {}, "skipped": {}}
    for arm in ARMS:
        if arm in DRAFT_ARMS and arm not in drafters:
            receipt["skipped"][arm] = "no drafter entry"
            continue
        case = derive(base, arm, drafters.get(arm))
        path = Path(args.out) / f"{case_name(arm)}.json"
        receipt["cases"][arm] = {"path": str(path), "sha256": save(path, case),
                                 "expected_type": EXPECTED_TYPE[arm],
                                 **({"drafter_sha256": drafters[arm]["sha256"]} if arm in drafters else {})}
    (Path(args.out) / "cases-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
