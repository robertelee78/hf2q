"""Markdown of the F2 'Q3 verdicts of record' tier (f2_q3_tier records and rows). Pure: no receipt reads."""
import native_peers
from f2_q3_tier import (COMPARISON_FAILURES, FAIRNESS, NATIVE_COMPARISON, PEER_SPEC_OFF, PRECISION_PP, REFERENCES, SSE,
                        STOCK_ARM, STOCK_RECORD, TOKEN_ARMS, TOKENS, UNMEASURABLE, how_to_read, required_references)


def ratio(row):
    """hf2q/reference ratio and its CI from a '% improvement' effect row (improvement = 100 (1 - ratio))."""
    ci = row.get("ci95")
    return 1 - row["value"] / 100, None if ci is None else (1 - ci[1] / 100, 1 - ci[0] / 100)


def sooner(row, reference, subject="hf2q"):
    """Task-makespan sentence on the slower arm's base: X = 1 - faster/slower, for whichever arm is faster."""
    r, ci = ratio(row)
    if r <= 1:
        who, other, x = subject, reference, 100 * (1 - r)
        bounds = None if ci is None else (100 * (1 - ci[1]), 100 * (1 - ci[0]))
    else:
        who, other, x = reference, subject, 100 * (1 - 1 / r)
        bounds = None if ci is None else (100 * (1 - 1 / ci[0]), 100 * (1 - 1 / ci[1]))
    text = f"{who} finishes {x:.1f}% sooner than {other}"
    if bounds is None:
        return text + " (no CI)"
    text += f" (95% CI {bounds[0]:.1f} to {bounds[1]:.1f}%"
    if row.get("half_width_pp") is not None:
        text += f", half-width {row['half_width_pp']:.2f} points"
    return text + ("; the interval includes a tie)" if bounds[0] <= 0 else ")")


def fairness(effects, baseline, rec):
    """Worst-slot ratios against one reference. SSE gaps against the peer are never compared (engines emit
    different event granularity); an SSE gap the verdict could not estimate is shown as not measurable."""
    contrast = "vs_stock" if baseline == STOCK_ARM else "vs_splash" if baseline == "splash" else "vs_peer"
    parts = []
    for metric, label in FAIRNESS.items():
        row = effects.get((metric, baseline))
        if metric == SSE and baseline != STOCK_ARM:
            parts.append(f"{label} not comparable (different event granularity)")
        elif row is None or row.get("ci95") is None:
            parts.append(f"{label} " + (UNMEASURABLE if f"{metric}.{contrast}" in rec["unmeasurable"] else "not recorded"))
        else:
            r, ci = ratio(row)
            parts.append(f"{label} {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                         + (" (no regression demonstrated)" if metric == SSE and ci[0] <= 1 <= ci[1] else ""))
    return ", ".join(parts)


def gate_reasons(rec, family):
    """Unmet quality-gate conditions: each required reference (v0.1.28 stock, and the same-GGUF peer or the family's
    native stand-in) must be declared and Pass, and no experiment final source may await patch review."""
    gates = {g["reference"]: g for g in rec["quality_gate"]}
    required = required_references(family, rec["peer_unavailable"], rec.get("splash"))
    reasons = [f"quality gate vs {REFERENCES[k]} not declared" for k in required if k not in gates]
    if len(required) < 2:
        reasons.append("no quality-gate reference exists for the unavailable peer")
    for status in ("Pending", "Halted", "Fail"):
        names = [REFERENCES[k] + (f" ({gates[k]['q1_verdict']})" if gates[k]["q1_verdict"] else "")
                 for k in required if k in gates and gates[k]["status"] == status]
        if names:
            reasons.append(f"quality gate {status} vs " + ", ".join(names))
    if rec["patch_review_pending"]:
        reasons.append(f"patch review pending for {len(rec['patch_review_pending'])} final source(s)")
    return reasons


def stock_reasons(rec, effects, band, family):
    """Every unmet condition for showing a Win against v0.1.28 stock (empty: Win). A Failed family has one: rule 1.
    A family whose verdict is Loss vs the peer is never a Q3 Win, whatever its stock contrast."""
    if rec["verdict"] == "Failed":
        return ["the family verdict is Failed (hf2q task failure)"]
    loss = "the family verdict is Loss vs the peer" + (" or Splash" if rec.get("splash") else "")
    reasons = [loss] if rec["verdict"] == "Loss" else []
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
        if f is not None and f.get("ci95") is not None:
            if ratio(f)[1][0] > band:
                reasons.append(f"worst-slot {label} vs stock regresses beyond the ×{band} band (ratio CI lower {ratio(f)[1][0]:.3f})")
        elif f"{metric}.vs_stock" not in rec["unmeasurable"]:  # the Q3 rule passes an unestimable effect
            reasons.append(f"worst-slot {label} vs stock not recorded")
    if rec["thermal_valid"] is False:
        reasons.append("a run observed serious or critical thermal state")
    if not rec["memory_valid"]:
        reasons.append("a passing experiment run lacks a passing Q5 memory verdict")
    return reasons + gate_reasons(rec, family)


def campaign_cell(rec, effects):
    text = rec["campaign_status"]
    halt = rec["halt"]
    if halt:
        text += f": halted at {halt['stage'] or 'campaign'} entry `{halt['entry']}` ({halt['reason']})"
    stop = rec["primary_stop"]
    if stop:
        text += f"; the primary stopped on {stop['reason']} after {stop['blocks']} blocks"
        imprecise = [f"{name} half-width {row['half_width_pp']:.2f}" for name, row in
                     (("vs-stock", effects.get(("makespan_s", STOCK_ARM))), ("vs-peer", effects.get(("makespan_s", "peer"))))
                     if row is not None and row.get("half_width_pp") is not None and row["half_width_pp"] > PRECISION_PP]
        if stop["reason"] == "max_blocks" and imprecise:
            text += f", precision not met ({'; '.join(imprecise)} points)"
    controls = rec["stages"][1:]
    measured = [w for w, status in controls if status == "Complete"]
    halted = [w for w, status in controls if status not in ("Complete", "Pending", "NotRun")]
    unrun = [w for w, status in controls if status in ("Pending", "NotRun")]
    if measured:
        text += "; controls measured: " + ", ".join(measured)
    if halted:
        text += "; unmeasured: " + ", ".join(f"{w} ({'halted' if halt and halt['stage'] == w else 'stage ' + s})"
                                            for w, s in controls if w in halted)
    if unrun:
        text += "; not run: " + ", ".join(unrun)
    lost = rec.get(COMPARISON_FAILURES)
    if lost:  # the comparison arms' failure counts travel with the speed measured on their passing runs
        text += ("; comparison-arm failed runs: "
                 + ", ".join(f"{arm} {a['failures']} of {a['attempts']}" for arm, a in lost["arms"].items())
                 + f" ({lost['rule']})")
    return text


def render(manifest, q3_rows, native_rows, baseline_labels):
    """Markdown lines of the tier: one line per family, a token table and each family's row ids."""
    fams, groups, band = manifest["families"], manifest["groups"], manifest["neutral_band_ratio"]
    notices = manifest.get("q3_verdict_notices") or {}  # e.g. a record a pre-registered campaign will replace
    records = {g["family"]: (gid, g["q3_record"]) for gid, g in groups.items() if "q3_record" in g}
    same_file = native_peers.same_file_baselines()
    how = how_to_read(any(x["baseline_arm"] in same_file for x in native_rows))
    out = ["## Q3 verdicts of record (F2's qualified display under option B; above Descriptive)", "", *how, "",
           "| Family | Q3 verdict of record | Makespan vs v0.1.28 stock | Makespan vs peer (task makespan only) | "
           "Worst-slot fairness, hf2q/reference [95% CI] | Quality gate | Campaign | Reviewed verdict |",
           "|---|---|---|---|---|---|---|---|"]
    tokens, evidence = [], []
    for fam, family in fams.items():
        if fam not in records:
            out.append(f"| {family['title']} | No Q3 verdict of record | — | — | — | — | — | — |")
            continue
        gid, rec = records[fam]
        hf2q = rec["experiment"]
        mine = sorted((r for r in q3_rows if r["group"] == gid), key=lambda r: r["id"])
        effects = {(r["metric"], r["baseline_arm"]): r for r in mine if r["kind"] == "effect"}
        measured = {(r["metric"], r["arm"]): r for r in mine if r["kind"] == "measurement"}
        failures = [r for r in mine if r["status"] == "Failure" and r["metric"] == "task_pass"]
        unmet = stock_reasons(rec, effects, band, fam)
        stock = effects.get(("makespan_s", STOCK_ARM))
        # A stock-record campaign's vs-stock terms come from the same-artifact record on hf2q's artifact.
        stock_name = "v0.1.28 stock (hf2q artifact record)" if rec.get(STOCK_RECORD) else "v0.1.28 stock"
        stock_cell = (f"{hf2q}: no speed effect" if stock is None else sooner(stock, stock_name, hf2q)) + (
            "; **Win**" if not unmet else "; not a Q3 Win: " + "; ".join(unmet))
        stock_cell += "".join(f". {r['value']:.0f}/{r['n_slots']} tasks [FAIL]: {r['note']} (`{r['id']}`)" for r in failures)
        verdict = rec["verdict"]
        if verdict == "Win" and unmet:
            verdict = "Withheld: " + "; ".join(unmet)
        elif verdict in ("Failed", "Inconclusive", "Pending") and rec["reasons"]:
            verdict += ": " + "; ".join(rec["reasons"])
        elif verdict == "Unavailable" and rec["reasons"] and rec["reasons"][0].startswith("no usable stock arm"):
            verdict += ": " + "; ".join(rec["reasons"])  # the peer cell keeps its measured contrast
        if fam in notices:
            verdict += f" — **{notices[fam]}**"
        peer = effects.get(("makespan_s", "peer"))
        if peer is not None:
            peer_cell = sooner(peer, f"the {PEER_SPEC_OFF}" if PEER_SPEC_OFF in peer["note"] else "the peer", hf2q)
        elif rec["peer_unavailable"] or rec["verdict"] == "Unavailable":
            peer_cell = f"Unavailable: {rec['peer_unavailable'] or '; '.join(rec['reasons'])}"
            peer_cell += "".join(f". Native {baseline_labels[x['baseline_arm']]} "
                                 f"({'same file' if x['baseline_arm'] in same_file else 'different artifact'}, task "
                                 f"makespan only): "
                                 f"{x['note']} (`{x['id']}`)" for x in native_rows
                                 if x["family"] == fam and NATIVE_COMPARISON not in (x["variant"] or "") + x["note"])
        else:
            peer_cell = "—"
        splash, splash_rec = effects.get(("makespan_s", "splash")), rec.get("splash")
        if splash is not None:
            peer_cell += "; Splash (same GGUF): " + sooner(splash, "Splash", hf2q)
        elif splash_rec and splash_rec["required"]:
            peer_cell += "; Splash: " + (f"Unavailable: {splash_rec['unavailable']}" if splash_rec["unavailable"]
                                         else "failed: " + "; ".join(splash_rec["failures"]) if splash_rec["failures"]
                                         else "no effect")
        fair = ("— (no paired block)" if not any(m in FAIRNESS for m, _ in effects) and not rec["unmeasurable"] else
                f"{'vs stock (hf2q artifact record)' if rec.get(STOCK_RECORD) else 'vs stock'}: "
                f"{fairness(effects, STOCK_ARM, rec)}; vs peer: "
                + (fairness(effects, "peer", rec) if peer is not None else "—")
                + (f"; vs Splash: {fairness(effects, 'splash', rec)}" if splash is not None else ""))
        gate = "; ".join(f"{REFERENCES[g['reference']]} {g['status']}"
                         + (f" ({g['q1_verdict']})" if g["q1_verdict"] not in (None, g["status"]) else "")
                         + f" `{g['manifest']}`" for g in rec["quality_gate"]) or "not declared"
        if rec["patch_review_pending"]:
            gate += "; patch review pending: " + ", ".join(f"`{h[:12]}`" for h in rec["patch_review_pending"])
        review = f"; review `{rec['review']}`" if rec["review"] else "; independent review not recorded"
        out.append(f"| {family['title']} | {verdict} | {stock_cell} | {peer_cell} | {fair} | {gate} | "
                   f"{campaign_cell(rec, effects)} | `{rec['verdict_ref']}`{review} |")
        for arm, label in TOKEN_ARMS.items():
            cells = [measured.get((metric, arm)) for metric in TOKENS]
            if any(cells):
                tokens.append(f"| {family['title']} | {hf2q if arm == 'candidate' else label} | " + " | ".join(
                    "not recorded" if c is None else f"{c['value']:.{c['decimals']}f}" for c in cells) + " |")
        if mine:
            evidence.append(f"- {family['title']}: " + ", ".join(f"`{r['id']}`" for r in mine))
    out += ["", "Token volume and per-output-token rate (medians over the primary stage's passing runs; descriptive, "
            "never a verdict input):", ""]
    out += (["| Family | Arm | " + " | ".join(TOKENS.values()) + " |", "|---|---|---|---|---|"] + tokens
            if tokens else ["None recorded."])
    out += ["", "Rows of record (receipt and extraction key in the rows manifest); the record's control effects "
            "are in the Q3 controls section:", ""] + (evidence or ["None."])
    notes = manifest.get("q3_family_notes") or {}
    if notes:
        out += ["", "### Family notes", ""]
        for fam, family in fams.items():
            if fam in notes:
                out += [f"**{family['title']}**", ""] + [f"- {line}" for line in notes[fam]] + [""]
        out.pop()
    return out + [""]
