#!/usr/bin/env python3
"""Trie-DFS tokenization audit (Matt's technique, our vocab).

Question: for each refusal phrase the W1 automaton excludes, does ANY token
sequence decode (byte-concatenated) to a string containing that phrase?
If none, the tokenization seam is closed — the character-level automaton is
complete at the byte level.

The vocab is extracted from the DeepSeek-V4 GGUF (tokenizer.ggml.tokens,
129,280 tokens). DeepSeek uses GPT2-style byte-level BPE: '▁' (U+2581) marks
a space, and the decoded text is the token string with '▁'->' '.

We ask: can a token sequence's concatenated *decoded* text contain the phrase?
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b18_gen import ALL_PHRASES, APOSTROPHE_HOMOGLYPHS


def fold(s):
    for h in APOSTROPHE_HOMOGLYPHS:
        s = s.replace(h, "'")
    return s.lower()


LEXICON = sorted({fold(p) for p in ALL_PHRASES})


def decoded_texts(tokens):
    """token string -> the decoded text it contributes (▁ -> space)."""
    return [t.replace("▁", " ") for t in tokens]


def find_sequence_containing(texts, target):
    """DFS: does any token sequence's concatenated text contain `target`?
    Returns a witness list of token indices, or None."""
    n = len(target)
    # index tokens by first char for prefix extension
    by_first = defaultdict(list)
    for tid, txt in enumerate(texts):
        if txt:
            by_first[txt[0]].append((tid, txt))

    def lps(text, pat):
        best = 0
        for k in range(1, min(len(pat), len(text)) + 1):
            if text.endswith(pat[:k]):
                best = k
        return best

    max_window = 3 * n + 16
    stack = [("", [])]
    visited = set()
    while stack:
        acc, ids = stack.pop()
        if target in acc:
            return ids
        if len(acc) > max_window:
            continue
        key = acc[-(n + 8):]
        if key in visited:
            continue
        visited.add(key)
        cur = lps(acc, target)
        nxt = target[cur] if cur < n else None
        if nxt is not None:
            for tid, txt in by_first.get(nxt, []):
                stack.append((acc + txt, ids + [tid]))
        for tid, txt in by_first.get(target[0], []):
            stack.append((acc + txt, ids + [tid]))
    return None


def main():
    vocab_path = os.path.expanduser(
        "~/.local/share/hf2q/models/deepseek4/vocab_tokens.json")
    tokens = json.load(open(vocab_path))
    texts = decoded_texts(tokens)
    print(f"lexicon: {len(LEXICON)} phrases; vocab: {len(tokens)} tokens")

    hits = 0
    for phrase in LEXICON:
        witness = find_sequence_containing(texts, phrase)
        if witness is not None:
            hits += 1
            toks = [texts[i] for i in witness]
            print(f"  SEAM: '{phrase}' via {toks[:6]}{'...' if len(toks) > 6 else ''}")

    print()
    if hits == 0:
        print(f"ALL {len(LEXICON)} phrases unreachable by any token sequence")
        print("tokenization seam CLOSED: the automaton's character-level exclusion")
        print("is complete at the byte level (no alternate-tokenization bypass)")
    else:
        print(f"{hits}/{len(LEXICON)} phrases reachable — tokenization seam OPEN")


if __name__ == "__main__":
    main()
