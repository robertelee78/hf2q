#!/usr/bin/env python3
"""Render the ADR-063 F2 markdown scoreboard. No launch, measurement or retries.

Inputs are zero or more `hf2q.f2.fixed-six-scoreboard.v1` results from
`f2_scoreboard_cli.py` (the only source of Qualified F2 primary results) and a
checked-in rows manifest of retained-receipt measurements. `--verify` re-reads
every Verified row from its receipt under `--evidence-root` and refuses drift.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

from f2_scoreboard import SCHEMA as SCOREBOARD_SCHEMA

ROWS_SCHEMA = "hf2q.f2.scoreboard-rows.v1"
REPO = Path(__file__).resolve().parents[2]
ARMS = {"hf2q_current": "hf2q current", "candidate": "candidate",
        "v0127_stock": "v0.1.27 stock", "peer": "peer"}
STATUSES = {"Qualified": "Q", "Descriptive": "D", "NegativeResult": "NEG", "Failure": "FAIL"}
# key: (title, unit, direction, gap key); direction None = not a ratio metric.
METRICS = {
    "decode_tps_per_agent": ("Decode tok/s per agent (median per-request server rate)", "tok/s", "higher", "decode_tps_per_agent"),
    "decode_tps_aggregate": ("Aggregate output tok/s (completion tokens / client elapsed)", "tok/s", "higher", "decode_tps_aggregate"),
    "decode_tps_single_stream": ("Single-stream decode tok/s", "tok/s", "higher", "decode_tps_single_stream"),
    "prefill_tps": ("Prefill tok/s", "tok/s", "higher", "prefill_tps"),
    "semantic_ttft_s": ("Semantic TTFT (median, streamed turns)", "s", "lower", "semantic_ttft_s"),
    "makespan_s": ("Own-output four-task makespan", "s", "lower", "makespan"),
    "client_elapsed_s": ("Client coordinator elapsed", "s", "lower", "makespan"),
    "cached_token_share_pct": ("Cached-token share of prompt tokens", "%", "higher", "cached_token_share_pct"),
    "task_pass": ("Successful agent tasks", "tasks", None, "task_pass"),
}
GAP_METRICS = ("decode_tps_per_agent", "decode_tps_aggregate", "decode_tps_single_stream", "prefill_tps",
               "semantic_ttft_s", "makespan", "cached_token_share_pct", "task_pass")
EFFECT_UNIT = "% improvement"
SCOREBOARD_STATUSES = {"Qualified", "InconclusivePrecision", "QualityIneligible", "InvalidCampaign"}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


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


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _number(value, why):
    require(type(value) in (int, float) and math.isfinite(value), why)


def validate_manifest(m):
    require(m.get("schema") == ROWS_SCHEMA, f"rows schema must be {ROWS_SCHEMA}")
    require(m.get("evidence_root_symbol") == "$E", "evidence root symbol must be $E")
    band = m.get("neutral_band_ratio")
    _number(band, "neutral band ratio")
    require(1 < band < 1.5, "neutral band ratio in (1, 1.5)")
    for text in _strings(m):
        require(not text.startswith(("/", "~")) and "/Users/" not in text and "/home/" not in text,
                f"absolute local path in manifest: {text[:60]!r}")
    families, groups = m.get("families"), m.get("groups")
    require(isinstance(families, dict) and families and isinstance(groups, dict), "families and groups")
    for f in families.values():
        require(isinstance(f.get("title"), str) and f["title"] and isinstance(f.get("note"), str) and
                isinstance(f.get("artifact_sha256"), str) and SHA64.match(f["artifact_sha256"]), "family fields")
    for g in groups.values():
        require(g.get("family") in families and g.get("title") and g.get("workload"), "group binding")
    ids = set()
    for row in m.get("rows", []):
        rid = row.get("id")
        require(isinstance(rid, str) and rid and rid not in ids, f"unique row id {rid!r}")
        ids.add(rid)
        require(row.get("family") in families and groups.get(row.get("group"), {}).get("family") == row["family"],
                f"{rid}: family/group binding")
        require(row.get("metric") in METRICS and row.get("arm") in ARMS and row.get("status") in STATUSES,
                f"{rid}: metric/arm/status vocabulary")
        require(row.get("variant") is None or (isinstance(row["variant"], str) and row["variant"]), f"{rid}: variant")
        _number(row.get("value"), f"{rid}: finite value")
        require(type(row.get("decimals")) is int and 0 <= row["decimals"] <= 6, f"{rid}: decimals")
        kind = row.get("kind")
        if kind == "effect":
            require(row.get("unit") == EFFECT_UNIT, f"{rid}: effect unit")
            require(row.get("baseline_arm") in ARMS and row["baseline_arm"] != row["arm"], f"{rid}: effect baseline")
            ci = row.get("ci95")
            if ci is not None:
                require(isinstance(ci, list) and len(ci) == 2 and ci[0] <= row["value"] <= ci[1], f"{rid}: ci95")
                for x in ci:
                    _number(x, f"{rid}: ci95")
        else:
            require(kind == "measurement", f"{rid}: kind")
            require(row.get("unit") == METRICS[row["metric"]][1], f"{rid}: unit must be {METRICS[row['metric']][1]}")
            require("ci95" not in row and "baseline_arm" not in row, f"{rid}: measurement has no effect fields")
        require(type(row.get("n_slots")) is int and row["n_slots"] >= 1, f"{rid}: n_slots")
        if row["metric"] == "task_pass":
            require(float(row["value"]).is_integer() and 0 <= row["value"] <= row["n_slots"], f"{rid}: task pass count")
        require(isinstance(row.get("context"), str) and row["context"], f"{rid}: context")
        require(isinstance(row.get("date"), str) and DATE.match(row["date"]), f"{rid}: date")
        source = row.get("source")
        require(isinstance(source, dict) and set(source) == {"commit", "binary_sha256"}, f"{rid}: source")
        require(source["commit"] is None or SHA40.match(source["commit"]), f"{rid}: source commit")
        require(source["binary_sha256"] is None or SHA64.match(source["binary_sha256"]), f"{rid}: binary")
        require(source["commit"] or source["binary_sha256"], f"{rid}: source identity required")
        require(isinstance(row.get("artifact_sha256"), str) and SHA64.match(row["artifact_sha256"]), f"{rid}: artifact")
        require(isinstance(row.get("receipt"), str) and row["receipt"].startswith(("$E/", "repo:")), f"{rid}: receipt ref")
        v = row.get("verification")
        require(isinstance(v, dict) and v.get("state") in ("Verified", "Unverified"), f"{rid}: verification")
        if v["state"] == "Verified":
            require(isinstance(v.get("key"), str) and v["key"] and isinstance(v.get("extract"), dict) and
                    "value" in v["extract"], f"{rid}: verified rows need key and value extract")
            require(("ci95" in v["extract"]) == (row.get("ci95") is not None), f"{rid}: ci95 extract")
        else:
            require(isinstance(v.get("reason"), str) and v["reason"], f"{rid}: unverified reason")
        require(isinstance(row.get("note"), str), f"{rid}: note")
    return m


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
        _number(x, "extracted value")
    return {"one": lambda: numbers[0], "median": lambda: statistics.median(numbers), "sum": lambda: math.fsum(numbers),
            "min": lambda: min(numbers), "max": lambda: max(numbers)}[how]()


def extract(spec, evidence_root, cache):
    kind = spec["kind"]
    if kind == "ratio":
        num, den = extract(spec["numerator"], evidence_root, cache), extract(spec["denominator"], evidence_root, cache)
        require(den != 0, "zero denominator")
        value = num / den * spec.get("scale", 1)
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


def verify_rows(manifest, evidence_root):
    cache, checked = {}, []
    for row in manifest["rows"]:
        v = row["verification"]
        if v["state"] != "Verified":
            continue
        got = extract(v["extract"]["value"], evidence_root, cache)
        require(round(got, row["decimals"]) == round(row["value"], row["decimals"]),
                f"{row['id']}: receipt gives {got!r}, manifest records {row['value']!r}")
        if "ci95" in v["extract"]:
            ci = extract(v["extract"]["ci95"], evidence_root, cache)
            require([round(x, row["decimals"]) for x in ci] == [round(x, row["decimals"]) for x in row["ci95"]],
                    f"{row['id']}: ci95 drift")
        checked.append(row["id"])
    return checked


def load_scoreboard(family, ref, data, families):
    require(family in families, f"scoreboard family {family!r} not declared")
    require(ref.startswith("$E/"), "scoreboard reference must be symbolic $E/...")
    require(data.get("schema") == SCOREBOARD_SCHEMA and data.get("status") in SCOREBOARD_STATUSES,
            "fixed-six scoreboard schema/status")
    if data["status"] == "Qualified":
        require(data.get("primary_precision_met") is True and data.get("peer_precision_met") is True and
                isinstance(data.get("primary"), dict) and isinstance(data.get("secondary"), dict),
                "Qualified scoreboard must carry met precision and effects")
    return {"family": family, "ref": ref, "data": data}


def eligible(row):
    return (row["verification"]["state"] == "Verified" and row["status"] in ("Qualified", "Descriptive", "NegativeResult")
            and row["kind"] == "measurement" and METRICS[row["metric"]][2] is not None)


def comparisons(manifest):
    band = manifest["neutral_band_ratio"]
    buckets = {}
    for row in manifest["rows"]:
        if eligible(row):
            buckets.setdefault((row["group"], row["metric"]), {}).setdefault((row["arm"], row["variant"]), []).append(row)
    result = []
    for (group, metric), arms in buckets.items():
        pairs = [(k, ("hf2q_current", None)) for k in arms if k[0] == "candidate"]
        pairs += [(("hf2q_current", None), k) for k in arms if k[0] in ("peer", "v0127_stock")]
        for num_key, den_key in pairs:
            if num_key not in arms or den_key not in arms:
                continue
            num_rows, den_rows = arms[num_key], arms[den_key]
            num = statistics.median(r["value"] for r in num_rows)
            den = statistics.median(r["value"] for r in den_rows)
            if num <= 0 or den <= 0:
                continue
            higher = METRICS[metric][2] == "higher"
            ratio = num / den if higher else den / num
            rows = num_rows + den_rows
            label = "Qualified" if all(r["status"] == "Qualified" for r in rows) else "Descriptive"
            # Orient so larger score is better; a win/regression needs >= 2 runs per arm
            # and non-overlapping ranges, otherwise a single noisy run could read as a win.
            num_s = [r["value"] if higher else -r["value"] for r in num_rows]
            den_s = [r["value"] if higher else -r["value"] for r in den_rows]
            repeated = len(num_rows) >= 2 and len(den_rows) >= 2
            if any(r["status"] == "NegativeResult" for r in rows):
                verdict = "NegativeResult"
            elif 1 / band < ratio < band:
                verdict = "Neutral"
            elif not (repeated and (min(num_s) > max(den_s) if ratio > 1 else max(num_s) < min(den_s))):
                verdict = "Unseparated"
            else:
                verdict = "Win" if ratio > 1 else "Regression"
            result.append({"group": group, "metric": metric, "arm": num_key, "baseline": den_key, "value": num,
                           "baseline_value": den, "ratio": ratio, "label": label, "verdict": verdict,
                           "decimals": max(r["decimals"] for r in rows), "rows": sorted(r["id"] for r in rows)})
    order = list(manifest["groups"])
    return sorted(result, key=lambda c: (order.index(c["group"]), list(METRICS).index(c["metric"]), c["arm"][0],
                                         c["arm"][1] or "", c["baseline"][0]))


def effect_is_win(row):
    return (row["kind"] == "effect" and row["verification"]["state"] == "Verified" and
            row["status"] in ("Qualified", "Descriptive") and row.get("ci95") is not None and row["ci95"][0] > 0)


def fmt(value, decimals):
    return f"{value:.{decimals}f}"


def arm_label(key):
    return ARMS[key[0]] + (f" ({key[1]})" if key[1] else "")


def row_value(row):
    if row["kind"] == "effect":
        ci = f" [{fmt(row['ci95'][0], row['decimals'])}, {fmt(row['ci95'][1], row['decimals'])}]" if row.get("ci95") else " (no CI)"
        return f"{fmt(row['value'], row['decimals'])}{EFFECT_UNIT}{ci} vs {ARMS[row['baseline_arm']]}"
    if row["metric"] == "task_pass":
        return f"{int(row['value'])}/{row['n_slots']} tasks"
    return f"{fmt(row['value'], row['decimals'])} {row['unit']}"


def tag(row):
    return f"[{STATUSES[row['status']]}]" + ("" if row["verification"]["state"] == "Verified" else " (unverified)")


def cell(rows):
    parts = []
    for key in sorted({(r["variant"] or "", r["status"], r["kind"]) for r in rows}):
        subset = sorted((r for r in rows if (r["variant"] or "", r["status"], r["kind"]) == key), key=lambda r: r["id"])
        prefix = f"{key[0]}: " if key[0] else ""
        if len(subset) == 1 or key[2] == "effect" or key[1] == "Failure":
            text = "; ".join(row_value(r) for r in subset)
        elif subset[0]["metric"] == "task_pass":
            values = sorted({row_value(r) for r in subset})
            text = f"{values[0]} in each of {len(subset)} runs" if len(values) == 1 else "; ".join(row_value(r) for r in subset)
        else:
            d, values = subset[0]["decimals"], [r["value"] for r in subset]
            text = (f"median {fmt(statistics.median(values), d)} {subset[0]['unit']} "
                    f"(n={len(values)}, {fmt(min(values), d)}–{fmt(max(values), d)})")
        parts.append(f"{prefix}{text} {tag(subset[0])}")
    return "<br>".join(parts) if parts else "—"


def short(sha):
    return f"`{sha[:12]}`" if sha else "—"


def render(manifest, scoreboards, manifest_sha256):
    fams, groups, rows = manifest["families"], manifest["groups"], manifest["rows"]
    band = manifest["neutral_band_ratio"]
    comps = comparisons(manifest)
    out = ["# ADR-063 F2 four-agent scoreboard", "",
           "<!-- Generated by scripts/four_agent/f2_scoreboard_render.py; do not edit by hand. -->", "",
           f"Rows manifest: `docs/research/adr063-f2-scoreboard-rows.json` (SHA-256 `{manifest_sha256}`). "
           "`$E` is the ADR-063 campaign evidence root (local, not committed); every number cites its receipt and key.", "",
           "Labels:", "",
           "- **[Q] Qualified**: an F2 primary fixed-six result from `f2_scoreboard_cli.py` with status `Qualified`. "
           "Only these are F2 primary qualified results.",
           "- **[D] Descriptive**: measured from a retained receipt, but **not** an F2 primary qualified result "
           "(no fixed-six blocks, no Q2 certification, usually pending independent review). Never presented as a Qualified win.",
           "- **[NEG] NegativeResult**: a recorded verdict of no win or a failed acceptance criterion.",
           "- **[FAIL] Failure**: the run failed its task; its numbers are never used as a ratio baseline.",
           "- `(unverified)`: the value could not be re-read from its receipt; excluded from wins.", "",
           f"Ratios are oriented so **> 1 means the first arm is better** (tok/s: arm/baseline; seconds: baseline/arm). "
           f"Repeated runs compare by median. Ratios within ×{band} either way are Neutral. Outside that band, a Win or "
           "Regression also needs at least two runs per arm with non-overlapping ranges; otherwise it is Unseparated, never a win.", ""]
    out += ["## F2 primary results (fixed-six, Qualified only)", ""]
    if not scoreboards:
        out += ["No `hf2q.f2.fixed-six-scoreboard.v1` result has been rendered: no family has an F2 primary result yet.", ""]
    else:
        out += ["| Family | Status | B vs A makespan reduction % [95% CI] | B vs C | A vs C | Result |", "|---|---|---|---|---|---|"]
        for sb in sorted(scoreboards, key=lambda s: (list(fams).index(s["family"]), s["ref"])):
            d = sb["data"]
            def eff(e):
                return f"{e['reduction_percent']:.2f} [{e['nominal_pointwise_ci95_percent'][0]:.2f}, {e['nominal_pointwise_ci95_percent'][1]:.2f}]"
            if d["status"] == "Qualified":
                cols = [eff(d["primary"]), eff(d["secondary"]["B_over_C"]), eff(d["secondary"]["A_over_C"])]
            else:
                cols = ["—", "—", "—"]
            out.append(f"| {fams[sb['family']]['title']} | {d['status']} | {' | '.join(cols)} | `{sb['ref']}` |")
        out.append("")
    qualified_wins = []
    for sb in scoreboards:
        d = sb["data"]
        if d["status"] != "Qualified":
            continue
        for name, e, who in (("B vs A", d["primary"], "candidate vs hf2q current"),
                             ("B vs C", d["secondary"]["B_over_C"], "candidate vs peer")):
            if e["nominal_pointwise_ci95_percent"][0] > 0:
                qualified_wins.append(f"| {fams[sb['family']]['title']} | F2 primary makespan | {who} ({name}) | "
                                      f"{1 / e['geometric_ratio']:.3f}× | {e['reduction_percent']:.2f}% "
                                      f"[{e['nominal_pointwise_ci95_percent'][0]:.2f}, {e['nominal_pointwise_ci95_percent'][1]:.2f}] | `{sb['ref']}` |")
    qualified_wins += [_comp_line(c, groups, fams) for c in comps if c["verdict"] == "Win" and c["label"] == "Qualified"]
    out += ["## Wins vs baseline", "", "### Qualified wins", ""]
    out += (["| Family | Metric | Arm vs baseline | Ratio | Detail | Evidence |", "|---|---|---|---|---|---|"] + qualified_wins
            if qualified_wins else ["None yet: no F2 primary Qualified win exists."])
    out += ["", "### Descriptive wins (not F2 primary qualified results)", ""]
    dwins = [_comp_line(c, groups, fams) for c in comps if c["verdict"] == "Win" and c["label"] == "Descriptive"]
    dwins += [f"| {fams[r['family']]['title']} | {groups[r['group']]['title']}: {METRICS[r['metric']][0]} | "
              f"{arm_label((r['arm'], r['variant']))} vs {ARMS[r['baseline_arm']]} | — | {row_value(r)} {tag(r)} | `{r['id']}` |"
              for r in rows if effect_is_win(r)]
    out += (["| Family | Metric | Arm vs baseline | Ratio | Detail | Evidence |", "|---|---|---|---|---|---|"] + dwins
            if dwins else ["None."])
    out += ["", "## Negative results and non-wins", "",
            "| Family | Metric | Arm vs baseline | Ratio | Detail | Evidence |", "|---|---|---|---|---|---|"]
    out += [_comp_line(c, groups, fams) for c in comps if c["verdict"] != "Win"]
    out += [f"| {fams[r['family']]['title']} | {groups[r['group']]['title']}: {METRICS[r['metric']][0]} | "
            f"{arm_label((r['arm'], r['variant']))} vs {ARMS[r['baseline_arm']]} | — | {row_value(r)} {tag(r)} | `{r['id']}` |"
            for r in rows if r["kind"] == "effect" and not effect_is_win(r)]
    out += [f"| {fams[r['family']]['title']} | {groups[r['group']]['title']}: {METRICS[r['metric']][0]} | "
            f"{arm_label((r['arm'], r['variant']))} | — | {row_value(r)} {tag(r)}: {r['note']} | `{r['id']}` |"
            for r in rows if r["status"] == "Failure"]
    out += ["", "## Open gaps", "", "| Family | Metric | Missing arms | Arms with only failed runs |", "|---|---|---|---|"]
    for fam in fams:
        frows = [r for r in rows if r["family"] == fam]
        for gap in GAP_METRICS:
            missing, failed = [], []
            for arm in ARMS:
                arm_rows = [r for r in frows if r["arm"] == arm and METRICS[r["metric"]][3] == gap]
                if any(r["verification"]["state"] == "Verified" and r["status"] != "Failure" for r in arm_rows):
                    continue
                failed_arm = any(r["status"] == "Failure" and r["arm"] == arm for r in frows)
                (failed if failed_arm else missing).append(ARMS[arm])
            if missing or failed:
                out.append(f"| {fams[fam]['title']} | {gap} | {', '.join(missing) or '—'} | {', '.join(failed) or '—'} |")
    out.append("")
    covered = {sb["family"] for sb in scoreboards if sb["data"]["status"] == "Qualified"}
    out += [f"- {fams[f]['title']}: no F2 primary Qualified fixed-six result." for f in fams if f not in covered]
    out += [f"- Unverified row `{r['id']}`: {r['verification']['reason']}" for r in rows if r["verification"]["state"] != "Verified"]
    out.append("")
    for fam in fams:
        out += [f"## {fams[fam]['title']}", "", f"Artifact SHA-256 `{fams[fam]['artifact_sha256']}`. {fams[fam]['note']}".rstrip(), "",
                "| Workload | Metric | " + " | ".join(ARMS.values()) + " |", "|---|---|" + "---|" * len(ARMS)]
        for gid, g in groups.items():
            if g["family"] != fam:
                continue
            grows = [r for r in rows if r["group"] == gid]
            for metric in METRICS:
                mrows = [r for r in grows if r["metric"] == metric]
                if mrows:
                    out.append(f"| {g['title']} | {METRICS[metric][0]} | " +
                               " | ".join(cell([r for r in mrows if r["arm"] == a]) for a in ARMS) + " |")
        out += ["", "Workloads:", ""] + [f"- **{g['title']}**: {g['workload']}" for g in groups.values() if g["family"] == fam]
        out += ["", "| Row | Metric | Arm | Value | N | Context | Date | Source | Label | Receipt · key | Note |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["family"] == fam:
                v = r["verification"]
                src = short(r["source"]["commit"]) if r["source"]["commit"] else f"bin {short(r['source']['binary_sha256'])}"
                key = f"`{v['key']}`" if v["state"] == "Verified" else f"unverified: {v['reason']}"
                out.append(f"| `{r['id']}` | {r['metric']} | {arm_label((r['arm'], r['variant']))} | {row_value(r)} | "
                           f"{r['n_slots']} | {r['context']} | {r['date']} | {src} | {r['status']} | `{r['receipt']}` · {key} | {r['note']} |")
        out.append("")
    out += ["## Dated history", ""]
    runs = {}
    for r in rows:
        run = r["receipt"].split("/")[1] if r["receipt"].startswith("$E/") else r["receipt"]
        runs.setdefault((r["date"], list(fams).index(r["family"]), r["group"], r["arm"], r["variant"] or "", run), []).append(r)
    for key in sorted(runs):
        rr = runs[key]
        parts = "; ".join(f"{r['metric']} {row_value(r)} {tag(r)}" for r in sorted(rr, key=lambda r: list(METRICS).index(r["metric"])))
        out.append(f"- {key[0]} · {fams[rr[0]['family']]['title']} · {groups[key[2]]['title']} · "
                   f"{arm_label((key[3], key[4] or None))}: {parts}")
    out.append("")
    return "\n".join(out)


def _comp_line(c, groups, fams):
    g = groups[c["group"]]
    unit = METRICS[c["metric"]][1]
    detail = (f"{fmt(c['value'], c['decimals'])} vs {fmt(c['baseline_value'], c['decimals'])} {unit}; "
              f"{c['verdict']} [{STATUSES[c['label']]}]")
    return (f"| {fams[g['family']]['title']} | {g['title']}: {METRICS[c['metric']][0]} | "
            f"{arm_label(c['arm'])} vs {arm_label(c['baseline'])} | {c['ratio']:.3f}× | {detail} | "
            + ", ".join(f"`{i}`" for i in c["rows"]) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="rows manifest JSON")
    parser.add_argument("--scoreboard", action="append", default=[], metavar="FAMILY=PATH",
                        help="fixed-six scoreboard JSON under --evidence-root; repeatable")
    parser.add_argument("--evidence-root", help="local directory that $E denotes")
    parser.add_argument("--verify", action="store_true", help="re-read every Verified row from its receipt")
    parser.add_argument("--output", help="markdown path; stdout when omitted")
    args = parser.parse_args(argv)
    raw = Path(args.rows).read_bytes()
    manifest = validate_manifest(strict_json(raw))
    if args.verify:
        verify_rows(manifest, args.evidence_root)
    scoreboards = []
    for item in args.scoreboard:
        family, sep, path = item.partition("=")
        require(sep and args.evidence_root, "--scoreboard FAMILY=PATH requires --evidence-root")
        p, root = Path(path).resolve(), Path(args.evidence_root).resolve()
        require(p.is_relative_to(root), "scoreboard must live under the evidence root")
        scoreboards.append(load_scoreboard(family, "$E/" + p.relative_to(root).as_posix(),
                                           strict_json(p.read_bytes()), manifest["families"]))
    text = render(manifest, scoreboards, hashlib.sha256(raw).hexdigest())
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
