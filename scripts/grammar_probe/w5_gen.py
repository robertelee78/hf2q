#!/usr/bin/env python3
"""W5 = W1 + minimum body-length floor (the anti-vacuity arm).

Motivation (observed live, 2026-09-09): under a strong refusal attractor the
model escapes through vacuity — grammar-legal but content-free output
("The result is: 1." then stop). W1's body is nullable at every automaton
state, so stop-anytime is admitted. W5 makes the first L body characters
non-nullable: EOS is ungenerable until the body clears the floor.

Construction: product of (depth, automaton-state) for depth < L. Warmup rules
w5s{state}_{d} carry the identical transitions as the W1 automaton rule
b18s{state} minus the nullable "" branch; at depth L-1 they hand off to the
plain b18s{target} rules. The exclusion automaton runs THROUGH the warmup —
a banned phrase straddling the floor boundary still dies (no seam).

The automaton rules are reused verbatim from w1.gbnf (the measured file),
not regenerated — zero drift from the campaign artifact.

Usage: python3 w5_gen.py [L] [base.gbnf] > w5.gbnf   (defaults: L=48, w1.gbnf)
"""
import re
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RULE_RE = re.compile(r"^([A-Za-z0-9_]+) ::= (?:\( )?(.*?)(?: \))?$")


def parse_rules(text):
    rules = {}
    order = []
    for line in text.splitlines():
        m = RULE_RE.match(line.strip())
        if m:
            name, body = m.group(1), m.group(2)
            rules[name] = body
            order.append(name)
    return rules, order


def retarget(body, depth, floor):
    """Rewrite b18s targets in a warmup rule body for the given depth."""
    def repl(m):
        target = m.group(0)
        if depth + 1 < floor:
            return f"w5s{target[4:]}_{depth + 1}"
        return target  # handoff to the nullable automaton
    return re.sub(r"b18s\d+", repl, body)


def main():
    floor = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    base = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "w1.gbnf")
    rules, order = parse_rules(open(base).read())

    automaton_states = [n for n in order if n.startswith("b18s")]
    assert "b18s0" in rules, "automaton ground state missing"
    assert rules["root"].endswith(" b18s0"), "unexpected root shape"

    out = []
    out.append(f"# W5 — W1 + minimum body-length floor (L={floor}).")
    out.append("# Anti-vacuity arm: EOS is ungenerable until the body clears")
    out.append("# the floor; the exclusion automaton runs through the warmup")
    out.append("# (product construction, no seam). Automaton rules reused")
    out.append("# verbatim from w1.gbnf.")
    out.append(f"root ::= {rules['root'].replace('b18s0', 'w5s0_0')}")

    # non-automaton rules (topic_sentence) unchanged
    for name in order:
        if name in ("root",) or name.startswith("b18s"):
            continue
        out.append(f"{name} ::= ( {rules[name]} )")

    # the nullable automaton, unchanged
    for name in automaton_states:
        out.append(f"{name} ::= ( {rules[name]} )")

    # warmup chain: non-nullable until depth floor
    for d in range(floor):
        for state in automaton_states:
            body = rules[state]
            branches = [b for b in body.split(" | ") if b != '""']
            warmed = retarget(" | ".join(branches), d, floor)
            out.append(f"w5s{state[4:]}_{d} ::= ( {warmed} )")

    print("\n".join(out))


if __name__ == "__main__":
    main()
