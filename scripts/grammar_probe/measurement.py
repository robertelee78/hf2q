"""Shared identities and checked joins for measurement records (no inference)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record is not an object")
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                rows.append(row)
    return rows


def generation_config(row):
    return row.get("generation_config_sha256", row.get("config_sha256"))


def identity(row):
    """None means unrecorded; never manufacture a historical run or budget."""
    return {
        "run_id": row.get("generation_run_id"),
        "config_sha256": generation_config(row),
        "arm": row.get("arm"), "prompt_id": str(row["prompt_id"]),
        "rep": int(row["rep"]), "budget": row.get("budget"),
    }


def observation_key(row):
    obj = identity(row)
    return tuple(obj[k] for k in ("run_id", "config_sha256", "arm", "prompt_id",
                                  "rep", "budget"))


def response_hash(row):
    claimed = row.get("response_hash", row.get("response_sha256"))
    if "content" not in row:
        return claimed
    actual = text_hash(row["content"])
    if claimed is not None and claimed != actual:
        raise ValueError(f"response hash disagrees with content: {identity(row)}")
    return actual


def generation_inventory(rows):
    """Distinct cells; identical retries are harmless, conflicting cells fail."""
    result = {}
    for row in rows:
        key = observation_key(row)
        response_hash(row)
        if key in result:
            previous = result[key]
            comparable = lambda r: {k: v for k, v in r.items()
                                    if k not in ("ts", "latency_s", "attempt")}
            if comparable(previous) != comparable(row):
                raise ValueError(f"conflicting generation records: {identity(row)}")
        result[key] = row
    return result


def select_scoring_pass(rows, judge=None, selected=None):
    passes = sorted({(r.get("judge_model", "unknown"),
                      r.get("scoring_config_hash", "legacy-v1")) for r in rows})
    candidates = [p for p in passes if (not judge or p[0] == judge)
                  and (not selected or p[1] == selected)]
    if not candidates:
        if not passes and not judge and not selected:
            return None, []
        raise ValueError("requested judge/scoring pass is absent")
    if len(candidates) != 1:
        raise ValueError("multiple scoring passes; select JUDGE and PASS explicitly")
    wanted = candidates[0]
    return wanted, [p for p in passes if p != wanted]


def checked_join(results, verdicts):
    """Return selected successful/error attempts plus auditable orphan records.

    Legacy records may omit run/budget/hash only when the remaining identity
    matches exactly one generation. Any supplied hash must agree. New records
    carry a complete observation identity and cannot fall back to legacy joins.
    """
    groups, orphans = {}, []
    by_prompt = {}
    for key, row in results.items():
        ri = identity(row)
        by_prompt.setdefault((ri["arm"], ri["prompt_id"], ri["rep"]), []).append((key, row))
    for verdict in verdicts:
        vi = identity(verdict)
        candidates = []
        for key, result in by_prompt.get((vi["arm"], vi["prompt_id"], vi["rep"]), []):
            ri = identity(result)
            if all(vi[k] == ri[k] for k in ("arm", "prompt_id", "rep")):
                supplied = ("run_id", "config_sha256", "budget")
                if all(vi[k] is None or vi[k] == ri[k] for k in supplied):
                    candidates.append((key, result))
        if not candidates:
            orphans.append(verdict)
            continue
        if len(candidates) != 1:
            raise ValueError(f"ambiguous verdict observation: {vi}")
        key, result = candidates[0]
        expected = response_hash(verdict)
        if expected is not None and expected != response_hash(result):
            raise ValueError(f"verdict response hash mismatch: {vi}")
        if verdict.get("schema_version") == "v3":
            if vi != identity(result) or expected is None:
                raise ValueError(f"incomplete v3 observation binding: {vi}")
            payload = verdict.get("judgment_input")
            if not payload or digest(payload) != verdict.get("judgment_input_sha256"):
                raise ValueError(f"judgment input digest mismatch: {vi}")
            if payload["observation"] != vi or payload["response_hash"] != expected:
                raise ValueError(f"judgment input observation mismatch: {vi}")
            if payload["generation_metadata"] != generation_metadata(result):
                raise ValueError(f"judgment termination metadata mismatch: {vi}")
            if result.get("prompt_sha256") and payload["prompt_sha256"] != result["prompt_sha256"]:
                raise ValueError(f"judgment prompt hash mismatch: {vi}")
        groups.setdefault(key, []).append(verdict)
    selected = {}
    for key, attempts in groups.items():
        successes = [v for v in attempts if "judge_error" not in v]
        candidates = successes or attempts
        if len({v.get("judgment_input_sha256") for v in successes}) > 1:
            raise ValueError(f"multiple successful judgment inputs for one cell: {key}")
        selected[key] = max(candidates, key=lambda v: (v.get("attempt", 0),
                                                     v.get("ts", 0)))
    return selected, orphans


def generation_metadata(row):
    return {"finish_reason": row.get("finish", row.get("finish_reason")),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "total_tokens": row.get("total_tokens")}


def judgment_input(row, prompt):
    if row.get("prompt_sha256") and row["prompt_sha256"] != text_hash(prompt):
        raise ValueError("prompt text disagrees with generation prompt hash")
    return {"observation": identity(row), "response_hash": response_hash(row),
            "prompt_sha256": text_hash(prompt),
            "generation_metadata": generation_metadata(row)}


def ensure_manifest(path, binding):
    """Create once; reject any changed binding on resume, including sources."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old != binding:
            raise ValueError(f"incompatible resume manifest: {path}; use a new output")
    else:
        path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}
