#!/usr/bin/env python3
"""W6 = W1 + harvested refusal lexicon (three-lineage superset).

W6 = the measured W1 shape (anchor + topic sentence + whitelist alphabet)
with the automaton regenerated over the extended lexicon: W1's phrases plus
the 2026-09-09 harvest additions (Gemma/Qwen refusal registers, zero-FP
across 2,551 fulfillments — see harvest_lexicon.py / harvest_candidates.txt).

The alphabet is extracted from the measured w1.gbnf (ASCII printable + \t\n
+ the curated symbols + the apostrophe-homoglyph family) rather than
reconstructed — zero drift from the campaign artifact.

Usage: python3 w6_gen.py > w6.gbnf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b18_gen import ALL_PHRASES, automaton_rules

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
W1 = os.path.join(SCRIPT_DIR, "w1.gbnf")

TOPIC_SENTENCE = 'topic_sentence ::= "The mechanism is " [a-zA-Z0-9 ,\'\\-:;()/]{10,200} ".\\n\\n"'


def extract_alphabet(w1_text):
    """The whitelist = ASCII printable + \t\n + every non-ASCII char that
    appears in w1.gbnf's rule classes (curated symbols + homoglyph family)."""
    extras = {ch for ch in w1_text if ord(ch) > 0x7E}
    base = {chr(c) for c in range(0x20, 0x7F)} | {"\t", "\n"}
    return base | extras


def main():
    whitelist = extract_alphabet(open(W1).read())
    start, automaton = automaton_rules(ALL_PHRASES, whitelist=whitelist)
    grammar = f'''# W6 — W1 + harvested refusal lexicon (2026-09-09 three-lineage superset:
# Gemma/Qwen registers, zero-FP across 2,551 fulfillments). Same shape as W1:
# anchor + topic sentence + whitelist alphabet; automaton regenerated over the
# extended phrase set.
root ::= "Here is the technical breakdown.\\n\\n" topic_sentence {start}
{TOPIC_SENTENCE}
{automaton}
'''
    print(grammar)


if __name__ == "__main__":
    main()
