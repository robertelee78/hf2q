#!/usr/bin/env python3
"""Generate B18 grammar: B17 anchor + lexicon exclusion automaton (KMP, recursive GBNF rules)."""
import sys

REFUSAL_PHRASES = [
    "I cannot", "I can't", "I'm sorry", "I am sorry", "I apologize",
    "I'm unable", "I am unable", "I must decline", "cannot assist",
    "can't assist", "not able to provide", "I won't", "I will not",
    "against my", "I must refuse", "as an ai",
    "i cannot", "i can't", "i'm sorry", "i am sorry", "i apologize",
    "i'm unable", "i am unable", "i must decline",
    "i won't", "i will not", "against my", "i must refuse", "as an ai",
    # standalone forms (observed evasion: "Why I Can’t Provide That Code" —
    # capitalized, no "I " prefix, evades both "I can't" and "can't assist")
    "cannot", "can't", "won't", "unable to",
    "Cannot", "Can't", "Won't", "Unable to",
    # 2026-09-09 three-lineage harvest (DeepSeek + Gemma + Qwen3.8 full
    # corpora, APEX-judged): Gemma/Qwen refusal registers. v2: shortened to
    # cores that survive one-word insertions/swaps ("this specific request",
    # "harmless"→"safe", "strictly prohibit") — measured leaking through the
    # long forms under W6A. Each core FP-tested at ZERO false positives across
    # 2,551 valid fulfillments. Case-folded by the automaton.
    "programmed to be a helpful",
    "my safety guidelines prohibit",
    "guidelines strictly",
    "not able to fulfill",
    "cannot fulfill this request",
    "i am prohibited",
    "will not provide",
    "cannot provide",
    "as an ai assistant",
]

# Unicode homoglyph coverage: models evade character-exact exclusion by
# substituting lookalike codepoints (observed: U+2019 for U+0027 in "I'm sorry").
# For each phrase containing an ASCII apostrophe, add every homoglyph variant.
APOSTROPHE_HOMOGLYPHS = ["’", "‘", "ʼ", "＇"]

def with_homoglyphs(phrases):
    out = list(phrases)
    for p in phrases:
        if "'" in p:
            for h in APOSTROPHE_HOMOGLYPHS:
                out.append(p.replace("'", h))
    return out

ALL_PHRASES = with_homoglyphs(REFUSAL_PHRASES)

def longest_prefix_suffix(text, patterns):
    best = ""
    for pattern in patterns:
        for n in range(len(pattern)):
            prefix = pattern[:n]
            if len(prefix) > len(best) and text.endswith(prefix):
                best = prefix
    return best

def char_class(chars, negated):
    inner = ""
    for ch in chars:
        if ch in "\\]-":
            inner += "\\" + ch
        elif ch == '"':
            inner += '\\"'
        elif ord(ch) < 0x20:
            inner += f"\\x{ord(ch):02x}"
        else:
            inner += ch
    if negated:
        return f"[^\\x00{inner}]"
    if len(chars) == 1:
        return f'"{chars[0]}"'
    return f"[{inner}]"

def automaton_rules(patterns, whitelist=None):
    """KMP-style DFA as recursive GBNF rules. Matching is case-insensitive:
    every pattern is folded to lowercase and each input letter matches both
    cases, so 'can’T', 'CANNOT', 'I Can’t' all hit the same dead states.

    whitelist: when given (a set of allowed real characters), emit the W1
    shape — the catch-all is a POSITIVE class covering exactly
    whitelist ∖ (live ∪ dead chars), so any char outside the whitelist
    matches no branch and the sampler masks it (homoglyph evasion closed
    by construction). When None, emit the legacy negated catch-all."""
    # fold: lowercase + unify apostrophe homoglyphs to ASCII '
    fold_map = {h: "'" for h in APOSTROPHE_HOMOGLYPHS}
    def fold(s):
        return "".join(fold_map.get(c, c) for c in s.lower())
    folded = sorted({fold(p) for p in patterns})

    states = {""}
    for pattern in folded:
        for n in range(1, len(pattern)):
            states.add(pattern[:n])
    states = sorted(states)
    name = {state: f"b18s{i}" for i, state in enumerate(states)}

    # alphabet in folded form; each letter covers both real cases
    alphabet = sorted({ch for pattern in folded for ch in pattern})
    rules = []
    for state in states:
        transitions = {}
        dead = set()  # chars that complete an excluded phrase
        for ch in alphabet:
            emitted = state + ch
            if any(emitted.endswith(pattern) for pattern in folded):
                dead.add(ch)
                continue
            target = longest_prefix_suffix(emitted, folded)
            # real characters this folded char accepts
            real = {ch}
            if ch.isalpha():
                real.add(ch.upper())
            if ch == "'":
                real.update(APOSTROPHE_HOMOGLYPHS)
            transitions.setdefault(target, set()).update(real)
        branches = []
        for target, chars in sorted(transitions.items()):
            branches.append(f"{char_class(sorted(chars), False)} {name[target]}")
        # catch-all: route non-alphabet chars back to start. CRITICAL: the
        # negated class must exclude the FULL real alphabet (live + dead).
        # A dead-end char must match NO branch, so the sampler masks every
        # token containing it here. If it matched the catch-all instead, the
        # automaton would reset to s0 and the forbidden phrase sails through
        # (observed: "Why I can't provide that code" evaded exactly this way).
        dead_real = set()
        for ch in dead:
            dead_real.add(ch)
            if ch.isalpha():
                dead_real.add(ch.upper())
            if ch == "'":
                dead_real.update(APOSTROPHE_HOMOGLYPHS)
        full_alphabet = sorted(set().union(*transitions.values()) | dead_real) if transitions else sorted(dead_real)
        if whitelist is None:
            branches.append(f"{char_class(full_alphabet, True)} {name['']}")
        else:
            # W1 shape: positive ground class = whitelist minus every char
            # this rule already routes (live transitions) or kills (dead).
            # Chars outside the whitelist match NO branch → masked.
            ground = sorted(set(whitelist) - set(full_alphabet))
            if ground:
                branches.append(f"{char_class(ground, False)} {name['']}")
        branches.append('""')
        rules.append(f"{name[state]} ::= ( {' | '.join(branches)} )")
    return name[""], "\n".join(rules)

if __name__ == "__main__":
    start, automaton = automaton_rules(ALL_PHRASES)
    grammar = f'''root ::= "Here is the technical breakdown.\\n\\n" {start}
{automaton}
'''
    print(grammar)
