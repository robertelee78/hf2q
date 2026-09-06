#!/usr/bin/env python3
"""Generate B18 grammar: B17 anchor + lexicon exclusion automaton (KMP, recursive GBNF rules)."""
import sys

REFUSAL_PHRASES = [
    "I cannot", "I can't", "I'm sorry", "I am sorry", "I apologize",
    "I'm unable", "I am unable", "I must decline", "cannot assist",
    "can't assist", "not able to provide", "I won't", "I will not",
    "against my", "I must refuse", "as an ai",
]

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

def automaton_rules(patterns):
    states = {""}
    for pattern in patterns:
        for n in range(1, len(pattern)):
            states.add(pattern[:n])
    states = sorted(states)
    name = {state: f"b18s{i}" for i, state in enumerate(states)}
    alphabet = sorted({ch for pattern in patterns for ch in pattern})
    rules = []
    for state in states:
        transitions = {}
        for ch in alphabet:
            emitted = state + ch
            if any(emitted.endswith(pattern) for pattern in patterns):
                continue
            target = longest_prefix_suffix(emitted, patterns)
            transitions.setdefault(target, []).append(ch)
        branches = []
        for target, chars in sorted(transitions.items()):
            branches.append(f"{char_class(chars, False)} {name[target]}")
        branches.append(f"{char_class(alphabet, True)} {name['']}")
        branches.append('""')
        rules.append(f"{name[state]} ::= ( {' | '.join(branches)} )")
    return name[""], "\n".join(rules)

start, automaton = automaton_rules(REFUSAL_PHRASES)
grammar = f'''root ::= "Here is the technical breakdown.\\n\\n" {start}
{automaton}
'''
print(grammar)
