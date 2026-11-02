"""Receipt extraction for the ADR-063 F2 rows manifest: strict JSON, `$E/` and `repo:` references and the
extract specs `f2_scoreboard_render.py --verify` re-reads. No launch, measurement or retries."""
import json
import math
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def require(ok, why):
    if not ok:
        raise ValueError(why)


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError(f"nonfinite JSON {token}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def number(value, why):
    require(type(value) in (int, float) and math.isfinite(value), why)


def resolve(ref, evidence_root):
    if ref.startswith("$E/"):
        require(evidence_root is not None, "--evidence-root required to verify $E references")
        base, rel = Path(evidence_root), ref[3:]
    else:
        require(ref.startswith("repo:"), f"bad reference {ref}")
        base, rel = REPO, ref[5:]
    path = (base / rel).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), f"missing receipt {ref}")
    return path


def _walk(value, path):
    values = [value]
    for part in path.split(".") if path else []:
        nxt = []
        for v in values:
            if part == "*" and isinstance(v, dict):
                nxt.extend(v[k] for k in sorted(v))
            elif part == "*" and isinstance(v, list):
                nxt.extend(v)
            elif isinstance(v, dict) and part in v:
                nxt.append(v[part])
            elif isinstance(v, list) and part.isdigit() and int(part) < len(v):
                nxt.append(v[int(part)])
        values = nxt
    return values


def _reduce(values, spec):
    if spec.get("skip_null"):
        values = [v for v in values if v is not None]
    how = spec["reduce"]
    if how == "count_equal":
        return float(sum(1 for v in values if v == spec["equal"]))
    if how == "one":
        require(len(values) == 1, f"expected exactly one value, found {len(values)}")
    require(values, "extract matched nothing")
    if how == "list":
        return [float(v) for v in values]
    numbers = [float(v) for v in values]
    for x in numbers:
        number(x, "extracted value")
    return {"one": lambda: numbers[0], "median": lambda: statistics.median(numbers), "sum": lambda: math.fsum(numbers),
            "min": lambda: min(numbers), "max": lambda: max(numbers)}[how]()


def extract(spec, evidence_root, cache):
    kind = spec["kind"]
    if kind == "ratio":
        num, den = extract(spec["numerator"], evidence_root, cache), extract(spec["denominator"], evidence_root, cache)
        require(den != 0, "zero denominator")
        value = num / den * spec.get("scale", 1)
    elif kind == "median_of":  # one value per item (for example one per run receipt), then their median
        require(isinstance(spec.get("items"), list) and spec["items"], "median_of needs items")
        value = statistics.median(extract(item, evidence_root, cache) for item in spec["items"])
    else:
        path = resolve(spec["file"], evidence_root)
        if path not in cache:
            cache[path] = path.read_text(errors="replace")
        if kind == "json":
            value = _reduce(_walk(strict_json(cache[path]), spec["path"]), spec)
        else:
            require(kind == "regex", f"unknown extract kind {kind}")
            value = _reduce(re.findall(spec["regex"], cache[path]), spec)
    if spec.get("transform") == "improvement_percent_from_ratio":
        value = 100 * (1 - value)
    return value
