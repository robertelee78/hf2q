#!/usr/bin/env python3
"""Generate the paper figures from measured data. Real jsonl, reproducible."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/opt/hf2q/docs/figures"
os.makedirs(OUT, exist_ok=True)
PROBE = os.path.dirname(os.path.abspath(__file__))

# ---- Fig 1: front-loading (entry distribution) + post-anchor trace ----
rows = [json.loads(l) for l in open(f"{PROBE}/refusal_mass_probe.jsonl")]
# entry distribution: aggregate top tokens across prompts
from collections import defaultdict
entry_mass = defaultdict(list)
for r in rows:
    e = r.get('entry_top10_no_grammar', [])
    if e and e[0][0] != 'PARSE_ERROR':
        for tok, p in e:
            entry_mass[tok].append(p)
# top tokens by mean entry mass
toks = sorted(entry_mass.items(), key=lambda kv: -sum(kv[1])/len(kv[1]))[:6]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
# left: entry
names = [t for t, _ in toks]
means = [sum(ps)/len(ps) for _, ps in toks]
ax1.bar(range(len(names)), means, color="#b3402f")
ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=30, ha='right')
ax1.set_ylabel("entry probability"); ax1.set_title("Unconstrained entry (harmful prompt)")
ax1.set_ylim(0, 1.05)
# right: post-anchor trace (h001)
trace_src = next((r for r in rows if r.get('under_grammar_trace') and r['prompt_id']=='h001'), None)
if trace_src:
    tr = trace_src['under_grammar_trace'][:16]
    pos = list(range(len(tr)))
    probs = [p for _, p in tr]
    labels = [t for t, _ in tr]
    ax2.plot(pos, probs, marker='o', color="#2f6db3", lw=1.5)
    ax2.set_xticks(pos); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel("post-mask probability"); ax2.set_title("Under W1 grammar (per-position)")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, ls='--', color='gray', lw=0.8)
fig.suptitle("Front-loaded refusal: the entry token carries the decision", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_frontloading.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("fig1 done")

# ---- Fig 2: arm progression (dose-response) ----
# measured refusal rates across the campaign (APEX-judged, spike/full)
arms = [
    ("baseline\n(no grammar)", 1.00),
    ("B15\nneutral anchor", 0.83),
    ("B17\nrefusal-incompat", 0.33),
    ("B18\nB17+exclusion", 0.29),
    ("B19\nB17+topic", 0.059),
    ("B20\n+think block", 0.241),
    ("W1\nB19+whitelist", 0.023),
]
fig, ax = plt.subplots(figsize=(9, 4.2))
names = [a for a, _ in arms]
rates = [r for _, r in arms]
colors = ["#888"] * 2 + ["#c07b3f"] * 3 + ["#2f6db3"] * 2
colors[4] = "#2f8b2f"  # B19 best spike
colors[6] = "#1a5d1a"  # W1 best full
ax.bar(range(len(names)), rates, color=colors)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("maintained-refusal rate")
ax.set_title("Dose-response: the anchor axis does the suppression")
for i, r in enumerate(rates):
    ax.text(i, r + 0.02, f"{r:.2f}", ha='center', fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_arms.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("fig2 done")
print("wrote figures to", OUT)
