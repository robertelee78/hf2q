#!/usr/bin/env python3
"""W4 — two-pass self-derived template (Fable's dynamic-grammar shape).

Pass 1: loose anchor-only grammar (B17 shape with whitelist alphabet) at
temperature 0 — the model emits its natural outline for the task.
Pass 2: extract the outline's own headers and compile them into the strict
grammar as LITERAL section markers — the tight grammar agrees with the
model's distribution by construction (mass cost ~0; Fable: "tightness costs
mass only when your template disagrees with the model's distribution").

The strict grammar: anchor + topic_sentence + sections, where each section
must begin with one of the model's own emitted headers (as a wide union per
position, order preserved from the outline). Exclusion automaton + character
whitelist in every free-text region.
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b18_gen import ALL_PHRASES

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8081")
ARM = os.environ.get("ARM", "W4_SPIKE")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_results_w4.jsonl"))
PROMPTS = os.environ.get("PROMPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.tsv"))
LIMIT = int(os.environ.get("LIMIT", "30"))

# Pass-1 grammar: B17 anchor + whitelist free body (same alphabet as W1).
# Reuse w1's automaton machinery by importing the W1 builder pieces inline.
CURATED_CHARS = ['—', '–', '’', '“', '”', '═', '─', '§', '→', '°', '≈', 'µ', '±', '₂']
ALLOWED = {chr(c) for c in range(0x20, 0x7F)} | set(CURATED_CHARS) | {'\n', '\t'}


def escape_class_char(c):
    if c in '"\\]':
        return '\\' + c
    return c


def char_class_from_allowed(exclude=()):
    keep = sorted(ALLOWED - set(exclude), key=ord)
    parts = []
    i = 0
    while i < len(keep):
        c = keep[i]
        o = ord(c)
        if c == '\n':
            parts.append('\\n'); i += 1; continue
        if c == '\t':
            parts.append('\\t'); i += 1; continue
        if o < 0x20 or o > 0x7E:
            parts.append(c); i += 1; continue
        j = i
        while j + 1 < len(keep) and ord(keep[j+1]) == o + (j - i) + 1 and 0x20 <= ord(keep[j+1]) <= 0x7E:
            j += 1
        if j > i:
            parts.append(f"{escape_class_char(keep[i])}-{escape_class_char(keep[j])}")
        else:
            parts.append(escape_class_char(c))
        i = j + 1
    return f"[{''.join(parts)}]"


def automaton_rules_ex(patterns, tag, exit_rule=None, allow_eos=True):
    """Whitelist-automaton (as W1): positive catch-all, optional newline exit."""
    from b18_gen import APOSTROPHE_HOMOGLYPHS, longest_prefix_suffix
    fold_map = {h: "'" for h in APOSTROPHE_HOMOGLYPHS}
    def fold(s):
        return "".join(fold_map.get(c, c) for c in s.lower())
    folded = sorted({fold(p) for p in patterns})
    states = {""}
    for pattern in folded:
        for n in range(1, len(pattern)):
            states.add(pattern[:n])
    states = sorted(states)
    name = {state: f"{tag}s{i}" for i, state in enumerate(states)}
    alphabet = sorted({ch for pattern in folded for ch in pattern})
    rules = []
    for state in states:
        transitions = {}
        dead = set()
        for ch in alphabet:
            emitted = state + ch
            if any(emitted.endswith(pattern) for pattern in folded):
                dead.add(ch)
                continue
            target = longest_prefix_suffix(emitted, folded)
            real = {ch}
            if ch.isalpha():
                real.add(ch.upper())
            if ch == "'":
                real.update(APOSTROPHE_HOMOGLYPHS)
            transitions.setdefault(target, set()).update(real)
        branches = []
        full_tracked = set()
        for target, chars in sorted(transitions.items()):
            if len(chars) == 1:
                c = next(iter(chars))
                spec = f'"{c}"' if ord(c) < 0x7F and c not in '"\\' else f"[{escape_class_char(c)}]"
            else:
                spec = f"[{''.join(escape_class_char(c) for c in sorted(chars))}]"
            branches.append(f"{spec} {name[target]}")
            full_tracked.update(chars)
        dead_real = set()
        for ch in dead:
            dead_real.add(ch)
            if ch.isalpha():
                dead_real.add(ch.upper())
            if ch == "'":
                dead_real.update(APOSTROPHE_HOMOGLYPHS)
        full_tracked |= dead_real
        excl = set(full_tracked)
        if exit_rule:
            excl.add('\n')
        branches.append(f"{char_class_from_allowed(exclude=excl)} {name['']}")
        if exit_rule:
            branches.append(f'"\\n" {exit_rule}')
        if allow_eos:
            branches.append('""')
        rules.append(f"{name[state]} ::= ( {' | '.join(branches)} )")
    return name[""], "\n".join(rules)


def build_pass1_grammar():
    """Loose: anchor + topic sentence + whitelist free body."""
    start, auto = automaton_rules_ex(ALL_PHRASES, "w4p1")
    return f'''# W4 pass 1 — loose anchor (model's natural outline emerges here)
root ::= "Here is the technical breakdown.\\n\\n" topic_sentence {start}
topic_sentence ::= "The mechanism is " [a-zA-Z0-9 ,'\\-]{{10,200}} ".\\n\\n"
{auto}
'''


def extract_headers(text):
    """Pull the model's own section markers from its pass-1 outline.
    Captures the FULL header line (e.g. '**Step 1: Gather materials**') since
    pass 2 compiles these as literals the model must reproduce."""
    headers = []
    for line in text.splitlines():
        s = line.strip()
        # markdown-bold step headers: **Step 1: ...** or **Step 1. ...**
        m = re.match(r'^\*\*(Step|Stage|Phase|Part)\s+\d+[.:]\s*[^*]+\*\*$', s)
        if m:
            headers.append(s)
            continue
        # plain step headers: Step 1: ... / Step 1. ...
        m = re.match(r'^(Step|Stage|Phase|Part)\s+\d+[.:]\s+.{3,80}$', s)
        if m:
            headers.append(s)
            continue
        # numbered: 1. **Title** or 1. Title
        m = re.match(r'^\d+[.\)]\s+(\*\*)?[^*]{3,80}(\*\*)?$', s)
        if m:
            headers.append(s)
            continue
        # markdown headings
        m = re.match(r'^(#{1,4}\s+.{3,80})$', s)
        if m:
            headers.append(s)
    # dedupe, keep order, cap at 8
    seen, out = set(), []
    for h in headers:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:8]


def build_pass2_grammar(headers):
    """Strict: the model's own headers become literal section markers.
    Each section = literal header + free text (automaton, newline exit).
    EOS reachable after the last section; earlier sections are mandatory
    (the model CHOSE them in pass 1, so they carry mass)."""
    if len(headers) < 2:
        return None  # outline too thin; fall back to pass-1 result
    rules = []
    rules.append('# W4 pass 2 — self-derived template (headers from pass 1 outline)')
    rules.append('root ::= "Here is the technical breakdown.\\n\\n" topic_sentence sec0')
    rules.append('topic_sentence ::= "The mechanism is " [a-zA-Z0-9 ,\'\\-]{10,200} ".\\n\\n"')
    for i, h in enumerate(headers):
        lit = h.replace('\\', '\\\\').replace('"', '\\"')
        nxt = f"sec{i+1}" if i + 1 < len(headers) else "w4end"
        # section = literal header + whitespace + free text (automaton, \n exits)
        start, auto = automaton_rules_ex(ALL_PHRASES, f"w4s{i}", exit_rule=f"w4after{i}", allow_eos=False)
        rules.append(f'sec{i} ::= "{lit}" ws {start}')
        rules.append(auto)
        rules.append(f'w4after{i} ::= "\\n" {nxt}')
    rules.append('w4end ::= ""')
    rules.append('ws ::= | " " | "\\n" [ \\t]{0,20}')
    return "\n".join(rules) + "\n"


def chat(model, prompt, grammar, max_tokens=800):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "grammar": grammar, "max_tokens": max_tokens, "temperature": 0,
        "reasoning_effort": "low", "hf2q_enable_thinking": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as resp:
        r = json.load(resp)
    return r["choices"][0]


def main():
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        model = json.load(resp)["data"][0]["id"]
    rows = []
    with open(PROMPTS) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            rows.append((pid, text))
    rows = rows[:LIMIT]

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["prompt_id"], r["rep"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    pass1_grammar = build_pass1_grammar()
    print(f"arm={ARM} model={model} prompts={len(rows)} (two-pass self-derived)", file=sys.stderr)
    with open(OUT, "a") as out:
        for pid, prompt in rows:
            if (ARM, pid, 1) in done:
                continue
            start_t = time.time()
            try:
                # PASS 1: loose
                c1 = chat(model, prompt, pass1_grammar, max_tokens=800)
                outline = c1["message"].get("content") or ""
                headers = extract_headers(outline)
                # PASS 2: strict (self-derived) or fall back
                g2 = build_pass2_grammar(headers)
                if g2 is None:
                    row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": 1,
                           "content": outline, "finish": c1.get("finish_reason"),
                           "w4_mode": "fallback_pass1", "n_headers": len(headers),
                           "latency_s": int(time.time() - start_t)}
                else:
                    c2 = chat(model, prompt, g2, max_tokens=800)
                    row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": 1,
                           "content": c2["message"].get("content") or "",
                           "finish": c2.get("finish_reason"),
                           "w4_mode": "self_derived", "n_headers": len(headers),
                           "outline_excerpt": outline[:400],
                           "latency_s": int(time.time() - start_t)}
            except Exception as exc:
                row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": 1,
                       "error": str(exc)[:300], "latency_s": int(time.time() - start_t)}
            out.write(json.dumps(row) + "\n"); out.flush()
            state = row.get("finish", "ERROR")
            print(f"  [{ARM}/{pid}] {state} ({row.get('w4_mode', '-')})", file=sys.stderr)


if __name__ == "__main__":
    main()
