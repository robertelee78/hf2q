"""ADR-063 F2 tier 'Q3 verdicts of record' (F2 closure option B, operator decision 2026-10-31/11-01).

One record per family, carried by the rows group of a Q3 `verdict.json` of record (`q3_campaign.py --rows
CAMPAIGN --of-record`): the Q3 family verdict, campaign status, halt and stages, the quality gate against both
references (each bound to the campaign's arm identities), pending patch reviews and the reviewed verdict. Rows
flagged `q3_of_record` carry its numbers: makespan and worst-slot fairness effects against stock and peer, token
volumes and per-output-token rate. Validation and `--verify` re-derive the record from the verdict, campaign and
quality manifests; nothing here launches or measures.
"""
import json
from pathlib import Path

from f2_rows_extract import require
import native_peers
from q3_metrics import has_oom_marker

# q3_report rows of a native-comparison campaign carry this label; they are never a verdict of record.
NATIVE_COMPARISON = "native-comparison campaign; not the verdict of record"
VERDICTS = ("Win", "Loss", "Unavailable", "Failed", "Inconclusive", "Pending")
GATE_STATUSES = ("Pass", "Pending", "Halted", "Fail")
STOCK_REFERENCE = "stock-hf2q"
# quality_campaign.py --reference-kind vocabulary (test_f2_q3_tier binds it to quality_campaign.REFERENCE_KINDS).
REFERENCES = {STOCK_REFERENCE: "v0.1.28 stock", "peer": "peer",
              **{f"{kind}-native": f"{kind} native" for kind in native_peers.KINDS}}
STOCK_ARM = "v0128_stock"
# Worst-slot fairness metrics of the Q3 rule (q3_report.FAIRNESS) and the token rows of the tier.
FAIRNESS = {"worst_slot_ttft_p95_s": "TTFT p95", "worst_slot_sse_gap_max_s": "SSE gap max"}
SSE = "worst_slot_sse_gap_max_s"
UNMEASURABLE = "not measurable (one semantic event per streamed turn)"
TOKENS = {"prompt_tokens": "Prompt tokens per run", "completion_tokens": "Completion tokens per run",
          "output_tokens_per_s": "Completion tokens per makespan-second"}
TOKEN_ARMS = {"candidate": "hf2q experiment", STOCK_ARM: "v0.1.28 stock", "peer": "peer"}
PRECISION_PP = 2.5
RECORD_KEYS = {"verdict", "verdict_ref", "campaign_ref", "campaign_status", "halt", "primary_stop", "stages", "reasons",
               "experiment", "peer_unavailable", "unmeasurable", "thermal_valid", "memory_valid",
               "patch_review_pending", "quality_gate", "review"}
GATE_KEYS = {"reference", "manifest", "status", "q1_verdict", "reference_sha256"}
HOW_TO_READ = [
    "How to read this tier:", "",
    "- **Source.** One Q3 `verdict.json` of record per family, merged with `q3_campaign.py --rows CAMPAIGN "
    "--of-record` after independent review. `--verify` re-reads every number from its receipt and re-derives "
    "the verdict, campaign status, halt, stages and quality gate from `verdict.json`, `campaign.json` and the quality "
    "manifests. Under option B this tier is F2's qualified display; the fixed-six Qualified section below is successor scope.",
    "- **Q3 verdict of record** is the family verdict of the predeclared Q3 rules (`docs/research/adr063-q3-campaign.md`): "
    "Win, Loss, Unavailable (no runnable same-GGUF peer) or Failed (hf2q task failure). A Q3 Win whose stock "
    "conditions or quality gate below are unmet shows as Withheld, with the reasons.",
    "- **Win appears only against v0.1.28 stock** (release `9a0b79c7`), and only when the makespan interval vs stock "
    "excludes a tie at half-width ≤ 2.5 points, neither measurable stock fairness effect regresses beyond the neutral "
    "band, the primary stage reached its stopping rule with valid thermal and memory receipts, no final source awaits "
    "patch review, and the quality gate passed against **both** references (v0.1.28 stock, and the same-GGUF peer or, "
    "where it is unavailable, the family's native stand-in: ds4 for DeepSeek-V4). Otherwise the numbers are shown "
    "with the gating reason.",
    "- **Peer: task makespan only.** The sentence names whichever arm finishes sooner and computes X on the slower "
    "arm's time, X = 1 − faster/slower; it is never a win. Unavailable families show the native-engine statement "
    "(different artifact) when one exists.",
    "- **Fairness** is the worst-slot ratio hf2q/reference with its 95% CI; above 1 means hf2q's worst slot waits "
    f"longer. An SSE gap that the Q3 verdict could not estimate is {UNMEASURABLE} and does not gate, as in the Q3 rule. "
    "SSE gaps against the peer are not comparable (different event granularity).",
    "- **Quality gate**, per reference: Pending (manifest absent or still running), Halted (the campaign halted), "
    "Pass (Q1 paired verdict Pass, subject runs on the experiment binary and reference runs on that reference's "
    "binary) or Fail (any other settled Q1 verdict, shown in parentheses).",
    "- **Campaign** status includes a halt in a later stage (for example Invalid from a mixed-stage width halt), "
    "which does not change the completed primary, and lists which control stages were measured. Control effects of "
    "the record and native-engine rows keep their own sections below.",
]


def symbolic(text, root):
    """Evidence-root paths in receipt text as `$E/...` (the manifest never carries absolute local paths)."""
    return text.replace(str(Path(root).resolve()), "$E")


def attempts(document, arm):
    return [Path(a["output"]) for run in document.get("runs", []) if run.get("arm") == arm for a in run.get("attempts", [])]


def binaries(document, arm):
    """Receipt binary sha256 of every attempt of one quality-campaign arm (None for a missing receipt)."""
    return {json.loads((out / "receipt.json").read_text()).get("binary", {}).get("sha256")
            if (out / "receipt.json").is_file() else None for out in attempts(document, arm)}


def native_identities(document):
    """(engine kind, build_info) of every reference attempt's client native_readiness record."""
    out = set()
    for run_dir in attempts(document, "reference"):
        summary = run_dir / "client" / "summary.json"
        readiness = (json.loads(summary.read_text()).get("native_readiness") or {}) if summary.is_file() else {}
        out.add((readiness.get("engine_kind"), json.dumps(readiness.get("build_info"), sort_keys=True)))
    return out


def gate_status(document, reference, family, arms):
    """(status, Q1 verdict, reference binary sha256) of one quality_campaign.py manifest. A settled gate must run its
    subject on the Q3 experiment binary and its reference on the campaign's arm of that kind: the stock binary for
    stock-hf2q, the peer binary for peer, and for <kind>-native an arm of that engine kind on one native binary."""
    if document is None:
        return "Pending", None, None
    plan = document.get("plan") or {}
    require(plan.get("reference_kind") == reference,
            f"quality gate {reference}: manifest reference_kind is {plan.get('reference_kind')!r}")
    require(((plan.get("arms") or {}).get("subject") or {}).get("family") == family,
            f"quality gate {reference}: manifest is not for family {family}")
    if document.get("status") == "Halted":
        return "Halted", None, None
    q1 = (document.get("verdict") or {}).get("verdict")
    if document.get("status") != "Complete" or q1 is None:
        return "Pending", None, None
    require(binaries(document, "subject") == {arms["experiment"]["binary"]["sha256"]},
            f"quality gate {reference} was not run on the experiment binary")
    ran = binaries(document, "reference")
    if reference in (STOCK_REFERENCE, "peer"):
        arm = "stock" if reference == STOCK_REFERENCE else "peer"
        require(arm in arms, f"quality gate {reference}: the campaign has no {arm} arm")
        want = arms[arm]["binary"]["sha256"]
    else:
        kind = reference[:-len("-native")]
        require(((plan.get("arms") or {}).get("reference") or {}).get("engine_kind") == kind,
                f"quality gate {reference}: reference arm is not engine kind {kind}")
        identity = native_identities(document)
        require(len(identity) == 1 and next(iter(identity))[0] == kind and next(iter(identity))[1] != "null",
                f"quality gate {reference}: reference runs do not share one {kind} native readiness identity")
        want = next(iter(ran)) if len(ran) == 1 else None
        require(want is not None, f"quality gate {reference}: reference runs do not share one native binary")
    require(ran == {want}, f"quality gate {reference}: reference runs are not on the campaign's {REFERENCES[reference]} binary")
    return ("Pass" if q1 == "Pass" else "Fail"), q1, want


def required_references(family, peer_unavailable):
    """stock-hf2q always; the same-GGUF peer while it is available, else the family's native stand-in (ds4)."""
    if peer_unavailable is None:
        return [STOCK_REFERENCE, "peer"]
    return [STOCK_REFERENCE] + [f"{kind}-native" for kind in native_peers.KINDS
                                if native_peers.adapter(kind).SAME_GGUF_UNAVAILABLE
                                and native_peers.adapter(kind).Q3_FAMILY == family]


def halt_record(campaign, root):
    halted = campaign.get("halted")
    if not halted:
        return None
    stage = next((s["workload"] for s in campaign["stages"]
                  for e in s["warmups"] + [e for b in s["blocks"] for e in b["entries"]]
                  if e["name"] == halted.get("entry")), None)
    return {"entry": halted.get("entry"), "stage": stage, "reason": symbolic(halted["reason"], root)}


def unavailable(peer, root):
    """The peer-unavailable reason; declared evidence whose server log shows a Metal OOM reads as such."""
    if not peer:
        return None
    oom = any(Path(e["path"]).name == "server.stderr" and has_oom_marker(e["path"]) for e in peer.get("evidence") or [])
    return "peer out of memory (prior evidence)" if oom and peer["reason"] == "declared from prior evidence" \
        else symbolic(peer["reason"], root)


def build_record(verdict, campaign, root, refs, gates, review):
    """The group record of a verdict of record. refs: (verdict, campaign) `$E/` refs; gates: [(reference,
    `$E/` manifest ref, parsed manifest or None)]; review: `$E/` ref of the independent review or None."""
    require(verdict.get("verdict_of_record", True) and not verdict.get("primary_of_record"),
            "a controls-only or native-comparison verdict is never a Q3 verdict of record")
    arms = campaign["identities"]["arms"]
    quality = []
    for reference, ref, document in gates:
        status, q1, sha = gate_status(document, reference, verdict["family"], arms)
        quality.append({"reference": reference, "manifest": ref, "status": status, "q1_verdict": q1, "reference_sha256": sha})
    secondary = (verdict.get("primary") or {}).get("secondary") or {}
    declared = campaign["plan"]["declared"]["experiment"]
    return {"verdict": verdict["verdict"], "verdict_ref": refs[0], "campaign_ref": refs[1],
            "campaign_status": verdict["campaign_status"], "halt": halt_record(campaign, root),
            "primary_stop": (verdict.get("primary") or {}).get("stop"),
            "stages": [[s["workload"], s["status"]] for s in campaign["stages"]],
            "reasons": [symbolic(r, root) for r in verdict["reasons"]],
            "experiment": f"hf2q {declared['commit'][:8]}" if declared.get("commit") else declared["label"],
            "peer_unavailable": unavailable((verdict["peer"]["unavailable"] or {}).get("peer"), root),
            "unmeasurable": [f"{m}.{c}" for m in FAIRNESS for c in ("vs_stock", "vs_peer")
                             if (secondary.get(m) or {}).get(c) is not None and secondary[m][c]["improvement_percent"] is None],
            "thermal_valid": verdict["thermal_valid"], "memory_valid": verdict["memory_valid"],
            "patch_review_pending": list(verdict["quality"]["patch_review"]["pending"]),
            "quality_gate": quality, "review": review}


def validate(groups):
    """Every group q3_record; returns {verdict or campaign ref of a record: its verdict ref}."""
    seen, refs = set(), {}
    for gid, group in groups.items():
        rec = group.get("q3_record")
        if rec is None:
            continue
        require(isinstance(rec, dict) and set(rec) == RECORD_KEYS, f"{gid}: q3_record fields")
        require(group["family"] not in seen, f"{gid}: one Q3 verdict of record per family")
        seen.add(group["family"])
        require(rec["verdict"] in VERDICTS and isinstance(rec["campaign_status"], str)
                and isinstance(rec["experiment"], str), f"{gid}: q3_record verdict")
        require(isinstance(rec["verdict_ref"], str) and rec["verdict_ref"].startswith("$E/")
                and rec["verdict_ref"].endswith("/verdict.json") and isinstance(rec["campaign_ref"], str)
                and rec["campaign_ref"] == rec["verdict_ref"][:-len("verdict.json")] + "campaign.json",
                f"{gid}: q3_record must cite $E/.../verdict.json and the campaign.json beside it")
        refs.update({rec["verdict_ref"]: rec["verdict_ref"], rec["campaign_ref"]: rec["verdict_ref"]})
        require(rec["review"] is None or (isinstance(rec["review"], str) and rec["review"].startswith("$E/")),
                f"{gid}: review must be null or a $E/ reference")
        require(rec["halt"] is None or (isinstance(rec["halt"], dict) and set(rec["halt"]) == {"entry", "stage", "reason"}),
                f"{gid}: halt")
        require(isinstance(rec["stages"], list) and all(isinstance(s, list) and len(s) == 2 and all(isinstance(x, str) for x in s)
                                                       for s in rec["stages"]), f"{gid}: stages")
        for key in ("reasons", "unmeasurable", "patch_review_pending"):
            require(isinstance(rec[key], list) and all(isinstance(r, str) for r in rec[key]), f"{gid}: {key}")
        require(rec["peer_unavailable"] is None or isinstance(rec["peer_unavailable"], str), f"{gid}: peer_unavailable")
        require(rec["thermal_valid"] in (True, False, None) and rec["memory_valid"] in (True, False), f"{gid}: thermal/memory")
        names = [g.get("reference") for g in rec["quality_gate"]] if isinstance(rec["quality_gate"], list) else None
        require(names is not None and len(set(names)) == len(names) and set(names) <= set(REFERENCES), f"{gid}: quality gate references")
        for g in rec["quality_gate"]:
            settled = g.get("status") in ("Pass", "Fail")
            require(set(g) == GATE_KEYS and g["status"] in GATE_STATUSES
                    and isinstance(g["manifest"], str) and g["manifest"].startswith("$E/")
                    and (g["q1_verdict"] is not None) == settled and (g["reference_sha256"] is not None) == settled
                    and (g["status"] != "Pass" or g["q1_verdict"] == "Pass"), f"{gid}: quality gate {g.get('reference')}")
    return refs


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


def validate_foreign(row, record_refs):
    """A row outside the record may cite a record's receipts only as a control of that record (no duplicate merge)."""
    owner = record_refs.get(row.get("receipt"))
    require(owner is None or row.get("control_of") == owner,
            f"{row['id']}: cites the Q3 verdict of record {owner}; merge its --of-record fragment, not its default rows")


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
