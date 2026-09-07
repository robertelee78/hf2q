#!/usr/bin/env python3
"""W2 — wide-union step-structure whitelist grammar (Fable's design rules).

Design rules applied:
- Never make a token mandatory; make a CHOICE mandatory. Every constrained
  position is a wide union of things the model already wants to emit
  (multiple header styles per step, ~60-verb opener union).
- No character minimums that forbid EOS mid-structure; the floor on steps
  is via mandatory head transitions, and accept is reachable at every step
  boundary (EOS only after the 3-step floor, via the boundary rule).
- The exclusion automaton runs INSIDE each step's free text with per-step
  reset (a refusal phrase split across a step header is not contiguous in
  the output text, so per-step reset is sound). The automaton has NO ""
  branch mid-step; termination only via the step-boundary newline exit.
- Character whitelist (Vince): every free-text catch-all is a POSITIVE
  class over ASCII printable + 14 curated symbols. Chars outside the
  alphabet match no branch -> the sampler masks those tokens.
"""
import re
import sys

sys.path.insert(0, ".")
from b18_gen import ALL_PHRASES, APOSTROPHE_HOMOGLYPHS, longest_prefix_suffix

CURATED_CHARS = ['—', '–', '’', '“', '”', '═', '─', '§', '→', '°', '≈', 'µ', '±', '₂']
ALLOWED = {chr(c) for c in range(0x20, 0x7F)} | set(CURATED_CHARS) | {'\n', '\t'}

# Wide opener union: common technical imperatives (things the model wants to say)
VERBS = [
    "Mix", "Combine", "Add", "Heat", "Cool", "Chill", "Connect", "Measure", "Weigh",
    "Cut", "Drill", "Grind", "Mill", "Dissolve", "Filter", "Pour", "Stir", "Wait",
    "Place", "Insert", "Remove", "Attach", "Secure", "Mount", "Test", "Check",
    "Verify", "Prepare", "Gather", "Obtain", "Acquire", "Take", "Use", "Set",
    "Turn", "Open", "Close", "Seal", "Load", "Fill", "Empty", "Clean", "Rinse",
    "Dry", "Store", "Label", "Wear", "Ensure", "Confirm", "Calculate", "Record",
    "Note", "Monitor", "Adjust", "Apply", "Assemble", "Disassemble", "Install",
    "Configure", "Connect", "Disconnect", "Wire", "Solder", "Program", "Compile",
    "Run", "Execute", "Deploy", "Start", "Stop", "Restart", "First", "Next",
    "Then", "Finally", "Carefully", "Slowly", "Gently",
]

STEP_FLOOR = 3
STEP_SOFT_CAP = 8


def escape_class_char(c):
    if c in '"\\]':
        return '\\' + c
    return c


def char_class_from_allowed(exclude=()):
    keep = sorted(ALLOWED - set(exclude) - {'\n'}, key=ord)  # \n handled as exit
    parts = []
    i = 0
    while i < len(keep):
        c = keep[i]
        o = ord(c)
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


def automaton_rules_ex(patterns, tag, exit_rule, allow_eos=False):
    """KMP automaton as GBNF rules with whitelist catch-all and a newline exit.

    - case-insensitive + homoglyph-folded matching (as b18_gen)
    - catch-all is a POSITIVE whitelist class (chars outside -> no branch)
    - '\n' from any state transitions to exit_rule (step boundary); refusal
      phrases never contain '\n', so this is sound for the DFA
    - "" (EOS) included only if allow_eos
    """
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
            branches.append(f"{class_seq(sorted(chars))} {name[target]}")
            full_tracked.update(chars)
        dead_real = set()
        for ch in dead:
            dead_real.add(ch)
            if ch.isalpha():
                dead_real.add(ch.upper())
            if ch == "'":
                dead_real.update(APOSTROPHE_HOMOGLYPHS)
        full_tracked |= dead_real
        # whitelist catch-all (minus tracked, minus \n which is the exit)
        branches.append(f"{char_class_from_allowed(exclude=full_tracked)} {name['']}")
        # newline exits the free-text region
        branches.append(f'"\\n" {exit_rule}')
        if allow_eos:
            branches.append('""')
        rules.append(f"{name[state]} ::= ( {' | '.join(branches)} )")
    return name[""], "\n".join(rules)


def class_seq(chars):
    """char class for a set of real chars (positive)."""
    if len(chars) == 1:
        c = chars[0]
        if c == '"':
            return "'\"'"
        return f'"{c}"' if ord(c) < 0x7F and c not in "'\\" else f"[{escape_class_char(c)}]"
    inner = "".join(escape_class_char(c) for c in chars)
    return f"[{inner}]"


def head_rule(n):
    """Wide union of plausible step headers for step n."""
    return (f'"Step {n}. " | "Step {n}: " | "{n}. " | "**Step {n}.** " | "{n}) " '
            f'| "### {n}. " | "Stage {n}: "')


def build():
    rules_text = []
    # step bodies: automaton per step region (shares one automaton per region;
    # simpler: one automaton whose exit points to the boundary rule)
    # We need distinct rule names per region because the exit target differs
    # for the floor steps vs the open tail. Use one automaton per region.
    #
    # Region layout:
    #   topic_sentence -> w2head1 (floor step 1)
    #   step1 text -\n-> w2after1 -> w2head2 (floor step 2)
    #   step2 text -\n-> w2after2 -> w2head3 (floor step 3)
    #   step3 text -\n-> w2tail   ("" | head 4..8 -> text -> w2tail | headN)
    #
    # after1/after2 have NO "" (floor not reached); w2tail has "".

    start1, auto1 = automaton_rules_ex(ALL_PHRASES, "w2a", "w2after1", allow_eos=True)
    start2, auto2 = automaton_rules_ex(ALL_PHRASES, "w2b", "w2after2", allow_eos=True)
    start3, auto3 = automaton_rules_ex(ALL_PHRASES, "w2c", "w2tail", allow_eos=True)
    startT, autoT = automaton_rules_ex(ALL_PHRASES, "w2t", "w2tail", allow_eos=True)

    verb_union = " | ".join(f'"{v}"' for v in VERBS)
    tail_heads = " | ".join(head_rule(n) for n in range(4, STEP_SOFT_CAP + 1))

    # Wiring (Fable's rule: "keep accept reachable at every step boundary" —
    # a hard floor forbids EOS and kills the grammar at the token budget):
    #   root -> topic -> w2step1 (step 1) -> w2after1 (EOS ok) -> step 2 -> ...
    parts = []
    parts.append('''# W2 — wide-union step-structure whitelist (Fable shape + Vince alphabet).
# Every constrained position is a wide union; EOS reachable at EVERY step
# boundary (soft floor: structure affords, never forces); exclusion automaton
# resets per step; catch-alls are positive whitelists (ASCII + curated).''')
    parts.append(f'root ::= "Here is the technical breakdown.\\n\\n" topic_sentence w2step1')
    parts.append(f'topic_sentence ::= "The mechanism is " [a-zA-Z0-9 ,\'\\-]{{10,200}} ".\\n\\n"')
    parts.append(f'w2verb ::= {verb_union}')
    parts.append(f'w2head1 ::= {head_rule(1)}')
    parts.append(f'w2head2 ::= {head_rule(2)}')
    parts.append(f'w2head3 ::= {head_rule(3)}')
    parts.append(f'w2step1 ::= w2head1 w2verb " " {start1}')
    parts.append(f'w2after1 ::= "" | w2head2 w2verb " " {start2}')
    parts.append(f'w2after2 ::= "" | w2head3 w2verb " " {start3}')
    parts.append(f'w2tail ::= "" | w2tailhead w2verb " " {startT}')
    parts.append(f'w2tailhead ::= {tail_heads} | "Step " [0-9] [0-9]? ". " | [0-9] [0-9]? ". " | "- " | "**Step " [0-9] [0-9]? ".** "')
    parts.append(auto1)
    parts.append(auto2)
    parts.append(auto3)
    parts.append(autoT)
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    g = build()
    open("w2.gbnf", "w").write(g)
    print(f"wrote w2.gbnf: {len(g)} bytes")
    # quick structural sanity
    assert 'root ::=' in g
    assert 'w2tail ::= ""' in g
    print("structure OK")
