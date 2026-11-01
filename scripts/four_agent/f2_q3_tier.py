"""ADR-063 F2 tier 'Q3 verdicts of record' (F2 closure option B, operator decision 2026-10-31/11-01).

One record per family, carried by the rows group of a Q3 `verdict.json` of record (`q3_campaign.py --rows
CAMPAIGN --of-record`): the Q3 family verdict, campaign status and halt, the quality gate against both
references and the reviewed verdict. Rows flagged `q3_of_record` carry its numbers: makespan and worst-slot
fairness effects against stock and peer, token volumes and per-output-token rate. Validation and `--verify`
re-derive the record from the verdict, campaign and quality manifests; nothing here launches or measures.
"""
import json
from pathlib import Path

from f2_rows_extract import require
import native_peers

# q3_report rows of a native-comparison campaign carry this label; they are never a verdict of record.
NATIVE_COMPARISON = "native-comparison campaign; not the verdict of record"
VERDICTS = ("Win", "Loss", "Unavailable", "Failed", "Inconclusive", "Pending")
GATE_STATUSES = ("Pass", "Pending", "Fail")
STOCK_REFERENCE = "stock-hf2q"
# quality_campaign.py --reference-kind vocabulary (test_f2_q3_tier binds it to quality_campaign.REFERENCE_KINDS).
REFERENCES = {STOCK_REFERENCE: "v0.1.28 stock", "peer": "peer",
              **{f"{kind}-native": f"{kind} native" for kind in native_peers.KINDS}}
STOCK_ARM = "v0128_stock"
# Worst-slot fairness metrics of the Q3 rule (q3_report.FAIRNESS) and the token rows of the tier.
FAIRNESS = {"worst_slot_ttft_p95_s": "TTFT p95", "worst_slot_sse_gap_max_s": "SSE gap max"}
TOKENS = {"prompt_tokens": "Prompt tokens per run", "completion_tokens": "Completion tokens per run",
          "output_tokens_per_s": "Completion tokens per makespan-second"}
TOKEN_ARMS = {"candidate": "hf2q experiment", STOCK_ARM: "v0.1.28 stock", "peer": "peer"}
PRECISION_PP = 2.5
RECORD_KEYS = {"verdict", "verdict_ref", "campaign_ref", "campaign_status", "halt", "primary_stop", "reasons",
               "peer_unavailable", "thermal_valid", "memory_valid", "quality_gate", "review"}
HOW_TO_READ = [
    "How to read this tier:", "",
    "- **Source.** One Q3 `verdict.json` of record per family, merged with `q3_campaign.py --rows CAMPAIGN "
    "--of-record` after independent review. `--verify` re-reads every number from its receipt and re-derives "
    "the verdict, campaign status, halt and quality gate from `verdict.json`, `campaign.json` and the quality manifests.",
    "- **Q3 verdict of record** is the family verdict of the predeclared Q3 rules (`docs/research/adr063-q3-campaign.md`): "
    "Win, Loss, Unavailable (no runnable same-GGUF peer) or Failed (hf2q task failure). A Q3 Win whose stock "
    "conditions or quality gate below are unmet shows as Withheld, with the reasons.",
    "- **Win appears only against v0.1.28 stock** (release `9a0b79c7`), and only when the makespan interval vs stock "
    "excludes a tie at half-width ≤ 2.5 points, neither stock fairness effect regresses beyond the neutral band, "
    "the primary stage reached its stopping rule with valid thermal and memory receipts, and the quality gate passed "
    "against **both** references (v0.1.28 stock and the peer; the native ds4 arm for DeepSeek-V4). Otherwise the "
    "numbers are shown with the gating reason.",
    "- **Peer: task makespan only.** The sentence names whichever arm finishes sooner and computes X on the slower "
    "arm's time, X = 1 − faster/slower; it is never a win. Unavailable families show the native-engine statement "
    "(different artifact) when one exists.",
    "- **Fairness** is the worst-slot ratio hf2q/reference with its 95% CI; above 1 means hf2q's worst slot waits longer.",
    "- **Quality gate**, per reference: Pending (manifest absent or not Complete), Pass (Q1 paired verdict Pass on "
    "the experiment binary) or Fail (any other settled Q1 verdict, shown in parentheses).",
    "- **Campaign** status includes a halt in a later stage (for example Invalid from a mixed-stage width halt); "
    "it does not change the completed primary. Controls and native-engine rows keep their own sections below.",
]


def symbolic(text, root):
    """Evidence-root paths in receipt text as `$E/...` (the manifest never carries absolute local paths)."""
    return text.replace(str(Path(root).resolve()), "$E")


def gate_status(document, reference, family, experiment_sha):
    """(status, Q1 verdict) of one quality_campaign.py manifest against the Q3 experiment binary.
    None (not written yet) or a manifest that is not Complete is Pending."""
    if document is None:
        return "Pending", None
    plan = document.get("plan") or {}
    require(plan.get("reference_kind") == reference,
            f"quality gate {reference}: manifest reference_kind is {plan.get('reference_kind')!r}")
    require(((plan.get("arms") or {}).get("subject") or {}).get("family") == family,
            f"quality gate {reference}: manifest is not for family {family}")
    q1 = (document.get("verdict") or {}).get("verdict")
    if document.get("status") != "Complete" or q1 is None:
        return "Pending", None
    subject = set()
    for run in document.get("runs", []):
        for attempt in run.get("attempts", []) if run.get("arm") == "subject" else []:
            receipt = Path(attempt["output"]) / "receipt.json"
            subject.add(json.loads(receipt.read_text()).get("binary", {}).get("sha256") if receipt.is_file() else None)
    require(subject == {experiment_sha}, f"quality gate {reference} was not run on the experiment binary")
    return ("Pass" if q1 == "Pass" else "Fail"), q1


def halt_record(campaign, root):
    halted = campaign.get("halted")
    if not halted:
        return None
    stage = next((s["workload"] for s in campaign["stages"]
                  for e in s["warmups"] + [e for b in s["blocks"] for e in b["entries"]]
                  if e["name"] == halted.get("entry")), None)
    return {"entry": halted.get("entry"), "stage": stage, "reason": symbolic(halted["reason"], root)}


def build_record(verdict, campaign, root, refs, gates, review):
    """The group record of a verdict of record. refs: (verdict, campaign) `$E/` refs; gates: [(reference,
    `$E/` manifest ref, parsed manifest or None)]; review: `$E/` ref of the independent review or None."""
    require(verdict.get("verdict_of_record", True) and not verdict.get("primary_of_record"),
            "a controls-only or native-comparison verdict is never a Q3 verdict of record")
    experiment = campaign["identities"]["arms"]["experiment"]["binary"]["sha256"]
    peer = (verdict["peer"]["unavailable"] or {}).get("peer")
    quality = []
    for reference, ref, document in gates:
        status, q1 = gate_status(document, reference, verdict["family"], experiment)
        quality.append({"reference": reference, "manifest": ref, "status": status, "q1_verdict": q1})
    return {"verdict": verdict["verdict"], "verdict_ref": refs[0], "campaign_ref": refs[1],
            "campaign_status": verdict["campaign_status"], "halt": halt_record(campaign, root),
            "primary_stop": (verdict.get("primary") or {}).get("stop"),
            "reasons": [symbolic(r, root) for r in verdict["reasons"]],
            "peer_unavailable": None if not peer else symbolic(peer["reason"], root),
            "thermal_valid": verdict["thermal_valid"], "memory_valid": verdict["memory_valid"],
            "quality_gate": quality, "review": review}


def validate(groups):
    seen = set()
    for gid, group in groups.items():
        rec = group.get("q3_record")
        if rec is None:
            continue
        require(isinstance(rec, dict) and set(rec) == RECORD_KEYS, f"{gid}: q3_record fields")
        require(group["family"] not in seen, f"{gid}: one Q3 verdict of record per family")
        seen.add(group["family"])
        require(rec["verdict"] in VERDICTS and isinstance(rec["campaign_status"], str), f"{gid}: q3_record verdict")
        require(isinstance(rec["verdict_ref"], str) and rec["verdict_ref"].startswith("$E/")
                and rec["verdict_ref"].endswith("/verdict.json") and isinstance(rec["campaign_ref"], str)
                and rec["campaign_ref"] == rec["verdict_ref"][:-len("verdict.json")] + "campaign.json",
                f"{gid}: q3_record must cite $E/.../verdict.json and the campaign.json beside it")
        require(rec["review"] is None or (isinstance(rec["review"], str) and rec["review"].startswith("$E/")),
                f"{gid}: review must be null or a $E/ reference")
        require(rec["halt"] is None or (isinstance(rec["halt"], dict) and set(rec["halt"]) == {"entry", "stage", "reason"}),
                f"{gid}: halt")
        require(isinstance(rec["reasons"], list) and all(isinstance(r, str) for r in rec["reasons"]), f"{gid}: reasons")
        require(rec["thermal_valid"] in (True, False, None) and rec["memory_valid"] in (True, False), f"{gid}: thermal/memory")
        refs = [g.get("reference") for g in rec["quality_gate"]] if isinstance(rec["quality_gate"], list) else None
        require(refs is not None and len(set(refs)) == len(refs) and set(refs) <= set(REFERENCES), f"{gid}: quality gate references")
        for g in rec["quality_gate"]:
            require(set(g) == {"reference", "manifest", "status", "q1_verdict"} and g["status"] in GATE_STATUSES
                    and isinstance(g["manifest"], str) and g["manifest"].startswith("$E/")
                    and (g["q1_verdict"] is None) == (g["status"] == "Pending")
                    and (g["status"] != "Pass" or g["q1_verdict"] == "Pass"), f"{gid}: quality gate {g.get('reference')}")


def validate_row(row, groups):
    rid = row["id"]
    require((row.get("verification") or {}).get("state") == "Verified", f"{rid}: q3_of_record rows must be Verified")
    require(row["q3_of_record"] is True and "q3_record" in groups[row["group"]], f"{rid}: q3_of_record needs a group q3_record")
    require("control_of" not in row and NATIVE_COMPARISON not in (row["variant"] or ""), f"{rid}: q3_of_record row kind")
    rec = groups[row["group"]]["q3_record"]
    require(row.get("receipt") in (rec["verdict_ref"], rec["campaign_ref"]), f"{rid}: receipt must be the record's verdict or campaign")
    if row["kind"] == "effect":
        require(row["baseline_arm"] in (STOCK_ARM, "peer") and row["metric"] in ("makespan_s", *FAIRNESS),
                f"{rid}: Q3 effects are makespan or worst-slot fairness vs v0128_stock or peer")
    else:
        require(row["metric"] in TOKENS and row["arm"] in TOKEN_ARMS, f"{rid}: Q3 measurements are token volume or rate")


def verify(manifest, evidence_root, resolve):
    """Re-derive every group record from its receipts; refuses drift (for example a gate that completed)."""
    load = lambda ref: json.loads(resolve(ref, evidence_root).read_text())
    for gid, group in manifest["groups"].items():
        rec = group.get("q3_record")
        if rec is None:
            continue
        gates = []
        for g in rec["quality_gate"]:
            exists = (Path(evidence_root) / g["manifest"][3:]).is_file()
            gates.append((g["reference"], g["manifest"], load(g["manifest"]) if exists else None))
        if rec["review"] is not None:
            resolve(rec["review"], evidence_root)
        again = build_record(load(rec["verdict_ref"]), load(rec["campaign_ref"]), evidence_root,
                             (rec["verdict_ref"], rec["campaign_ref"]), gates, rec["review"])
        for key in sorted(RECORD_KEYS):
            require(again[key] == rec[key], f"{gid}: q3_record {key} drift: receipts give {again[key]!r}")


def ratio(row):
    """hf2q/reference ratio and its CI from a '% improvement' effect row (improvement = 100 (1 - ratio))."""
    ci = row.get("ci95")
    return 1 - row["value"] / 100, None if ci is None else (1 - ci[1] / 100, 1 - ci[0] / 100)


def sooner(row, reference):
    """Task-makespan sentence on the slower arm's base: X = 1 - faster/slower, for whichever arm is faster."""
    r, ci = ratio(row)
    if r <= 1:
        who, other, x = "hf2q", reference, 100 * (1 - r)
        bounds = None if ci is None else (100 * (1 - ci[1]), 100 * (1 - ci[0]))
    else:
        who, other, x = reference, "hf2q", 100 * (1 - 1 / r)
        bounds = None if ci is None else (100 * (1 - 1 / ci[0]), 100 * (1 - 1 / ci[1]))
    text = f"{who} finishes {x:.1f}% sooner than {other}"
    if bounds is None:
        return text + " (no CI)"
    text += f" (95% CI {bounds[0]:.1f} to {bounds[1]:.1f}%"
    if row.get("half_width_pp") is not None:
        text += f", half-width {row['half_width_pp']:.2f} points"
    return text + ("; the interval includes a tie)" if bounds[0] <= 0 else ")")


def fairness(effects, baseline):
    parts = []
    for metric, label in FAIRNESS.items():
        row = effects.get((metric, baseline))
        if row is None or row.get("ci95") is None:
            parts.append(f"{label} not recorded")
        else:
            r, ci = ratio(row)
            parts.append(f"{label} {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    return ", ".join(parts)


def gate_reasons(rec):
    gates = {g["reference"]: g for g in rec["quality_gate"]}
    reasons = [] if STOCK_REFERENCE in gates else ["quality gate vs v0.1.28 stock not declared"]
    if not set(gates) - {STOCK_REFERENCE}:
        reasons.append("quality gate vs the peer not declared")
    for status in ("Pending", "Fail"):
        names = [REFERENCES[k] + (f" ({g['q1_verdict']})" if g["q1_verdict"] else "")
                 for k, g in gates.items() if g["status"] == status]
        if names:
            reasons.append(f"quality gate {status} vs " + ", ".join(names))
    return reasons


def stock_reasons(rec, effects, band):
    """Every unmet condition for showing a Win against v0.1.28 stock (empty: Win)."""
    reasons = ["the family verdict is Failed (hf2q task failure)"] if rec["verdict"] == "Failed" else []
    if not rec["primary_stop"]:
        reasons.append("the primary stage has not reached its stopping rule")
    row = effects.get(("makespan_s", STOCK_ARM))
    if row is None or row.get("ci95") is None:
        reasons.append("no makespan effect vs stock")
    else:
        if row["ci95"][0] <= 0:
            reasons.append("the makespan interval vs stock does not exclude a tie")
        if row.get("half_width_pp") is None or row["half_width_pp"] > PRECISION_PP:
            reasons.append(f"vs-stock half-width above {PRECISION_PP} points")
    for metric, label in FAIRNESS.items():
        f = effects.get((metric, STOCK_ARM))
        if f is None or f.get("ci95") is None:
            reasons.append(f"worst-slot {label} vs stock not recorded")
        elif ratio(f)[1][0] > band:
            reasons.append(f"worst-slot {label} vs stock regresses beyond the ×{band} band (ratio CI lower {ratio(f)[1][0]:.3f})")
    if rec["thermal_valid"] is False:
        reasons.append("a run observed serious or critical thermal state")
    if not rec["memory_valid"]:
        reasons.append("a passing experiment run lacks a passing Q5 memory verdict")
    return reasons + gate_reasons(rec)


def campaign_cell(rec):
    text = rec["campaign_status"]
    if rec["halt"]:
        h = rec["halt"]
        text += f": halted at {h['stage'] or 'campaign'} entry `{h['entry']}` ({h['reason']})"
    if rec["primary_stop"]:
        text += f"; the primary stopped on {rec['primary_stop']['reason']} after {rec['primary_stop']['blocks']} blocks"
    return text


def render(manifest, q3_rows, native_rows, baseline_labels):
    """Markdown lines of the tier: one line per family, a token table and each family's row ids."""
    fams, groups, band = manifest["families"], manifest["groups"], manifest["neutral_band_ratio"]
    records = {g["family"]: (gid, g["q3_record"]) for gid, g in groups.items() if "q3_record" in g}
    out = ["## Q3 verdicts of record (F2 option B; above Descriptive)", "", *HOW_TO_READ, "",
           "| Family | Q3 verdict of record | Makespan vs v0.1.28 stock | Makespan vs peer (task makespan only) | "
           "Worst-slot fairness, hf2q/reference [95% CI] | Quality gate | Campaign | Reviewed verdict |",
           "|---|---|---|---|---|---|---|---|"]
    tokens, evidence = [], []
    for fam, family in fams.items():
        if fam not in records:
            out.append(f"| {family['title']} | No Q3 verdict of record | — | — | — | — | — | — |")
            continue
        gid, rec = records[fam]
        mine = sorted((r for r in q3_rows if r["group"] == gid), key=lambda r: r["id"])
        effects = {(r["metric"], r["baseline_arm"]): r for r in mine if r["kind"] == "effect"}
        measured = {(r["metric"], r["arm"]): r for r in mine if r["kind"] == "measurement"}
        unmet = stock_reasons(rec, effects, band)
        stock = effects.get(("makespan_s", STOCK_ARM))
        stock_cell = ("—" if stock is None else sooner(stock, "v0.1.28 stock")) + (
            "; **Win**" if not unmet else "; not a win: " + "; ".join(unmet))
        verdict = rec["verdict"]
        if verdict == "Win" and unmet:
            verdict = "Withheld: " + "; ".join(unmet)
        elif verdict in ("Failed", "Inconclusive", "Pending") and rec["reasons"]:
            verdict += ": " + "; ".join(rec["reasons"])
        peer = effects.get(("makespan_s", "peer"))
        if peer is not None:
            peer_cell = sooner(peer, "the peer")
        elif rec["peer_unavailable"] or rec["verdict"] == "Unavailable":
            peer_cell = f"Unavailable: {rec['peer_unavailable'] or '; '.join(rec['reasons'])}"
            peer_cell += "".join(f". Native {baseline_labels[x['baseline_arm']]} (different artifact, task makespan only): "
                                 f"{x['note']} (`{x['id']}`)" for x in native_rows
                                 if x["family"] == fam and NATIVE_COMPARISON not in (x["variant"] or "") + x["note"])
        else:
            peer_cell = "—"
        fair = f"vs stock: {fairness(effects, STOCK_ARM)}; vs peer: " + (
            fairness(effects, "peer") if peer is not None else "—")
        gate = "; ".join(f"{REFERENCES[g['reference']]} {g['status']}"
                         + (f" ({g['q1_verdict']})" if g["q1_verdict"] not in (None, g["status"]) else "")
                         + f" `{g['manifest']}`" for g in rec["quality_gate"]) or "not declared"
        review = f"; review `{rec['review']}`" if rec["review"] else "; independent review not recorded"
        out.append(f"| {family['title']} | {verdict} | {stock_cell} | {peer_cell} | {fair} | {gate} | "
                   f"{campaign_cell(rec)} | `{rec['verdict_ref']}`{review} |")
        for arm, label in TOKEN_ARMS.items():
            cells = [measured.get((metric, arm)) for metric in TOKENS]
            if any(cells):
                tokens.append(f"| {family['title']} | {label} | " + " | ".join(
                    "not recorded" if c is None else f"{c['value']:.{c['decimals']}f}" for c in cells) + " |")
        if mine:
            evidence.append(f"- {family['title']}: " + ", ".join(f"`{r['id']}`" for r in mine))
    out += ["", "Token volume and per-output-token rate (medians over the primary stage's passing runs; descriptive, "
            "never a verdict input):", ""]
    out += (["| Family | Arm | " + " | ".join(TOKENS.values()) + " |", "|---|---|---|---|---|"] + tokens
            if tokens else ["None recorded."])
    out += ["", "Rows of record (receipt and extraction key in the rows manifest):", ""] + (evidence or ["None."])
    return out + [""]
