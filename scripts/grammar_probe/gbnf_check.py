#!/usr/bin/env python3
r"""gbnf_check.py — offline grammar testing (Vince's #1 rule: test the grammar
without the LLM; inference calls are for prompts, not grammar logic).

Parses the GBNF subset our generators emit and runs accept/reject batteries
with zero inference calls. What is asserted is the LANGUAGE DENOTED (same
doctrine as gcd-tool-calling-benchmark's acceptance.rs).

Supported subset: string literals, char classes (ranges, negation, \xNN, \n \t
escapes), rule refs, `|` alternation, `( ... )` groups, repetition suffixes
{m,n} / {n} / ? / * / + on classes/literals/groups, and "" (empty alternative).

Usage:
  python3 gbnf_check.py GRAMMAR --accept-inline "text" --reject-inline "text"
  python3 gbnf_check.py GRAMMAR --accept file.txt --reject file.txt
Exit 0 = all expectations met.
"""
import re
import sys
from functools import lru_cache

# ---------------- parser ----------------

def parse_gbnf(text):
    rules = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^(\w+)\s*::=\s*(.*)$', line)
        if not m:
            raise ValueError(f"unparseable line: {line[:100]}")
        name, body = m.groups()
        rules[name] = parse_alts(body)
    if "root" not in rules:
        raise ValueError("no root rule")
    return rules

def parse_alts(body):
    alts, cur, depth, inq, i = [], [], 0, False, 0
    while i < len(body):
        c = body[i]
        if c == '\\' and i + 1 < len(body):
            cur.append(body[i:i+2]); i += 2; continue
        if c == '"':
            inq = not inq; cur.append(c); i += 1; continue
        if not inq:
            if c == '[':
                # char class is atomic: copy through its closing ']'
                j = i + 1
                while j < len(body):
                    if body[j] == '\\':
                        j += 4 if (j + 1 < len(body) and body[j+1] == 'x') else 2; continue
                    if body[j] == ']':
                        break
                    j += 1
                cur.append(body[i:j+1]); i = j + 1; continue
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == '|' and depth == 0:
                alts.append("".join(cur).strip()); cur = []; i += 1; continue
        cur.append(c); i += 1
    alts.append("".join(cur).strip())
    return [parse_seq(a) for a in alts]

def parse_seq(s):
    terms = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1; continue
        if c == '"':
            j = i + 1; buf = []
            while j < len(s) and s[j] != '"':
                if s[j] == '\\':
                    buf.append(unescape(s[j:j+2])); j += 2
                else:
                    buf.append(s[j]); j += 1
            term = ("lit", "".join(buf)); i = j + 1
        elif c == '[':
            # scan to the closing ']', honoring escapes (\], \x5d, \\, etc.)
            j = i + 1
            while j < len(s):
                if s[j] == '\\':
                    j += 4 if (j + 1 < len(s) and s[j+1] == 'x') else 2
                    continue
                if s[j] == ']':
                    break
                j += 1
            term = ("class",) + parse_class(s[i:j+1]); i = j + 1
        elif c == '(':
            depth, j = 1, i + 1
            inq = False
            while j < len(s) and depth:
                if s[j] == '"':
                    inq = not inq; j += 1; continue
                if inq:
                    j += 1; continue
                if s[j] == '\\':
                    j += 4 if (j + 1 < len(s) and s[j+1] == 'x') else 2; continue
                if s[j] == '[':
                    # skip char class
                    j += 1
                    while j < len(s):
                        if s[j] == '\\':
                            j += 4 if (j + 1 < len(s) and s[j+1] == 'x') else 2; continue
                        if s[j] == ']':
                            break
                        j += 1
                    j += 1; continue
                if s[j] == '(': depth += 1
                if s[j] == ')': depth -= 1
                j += 1
            term = ("group", parse_alts(s[i+1:j-1])); i = j
        elif c.isalnum() or c == '_':
            m = re.match(r'\w+', s[i:])
            term = ("ref", m.group(0)); i += len(m.group(0))
        else:
            raise ValueError(f"cannot parse at: {s[i:i+40]!r}")
        # repetition suffix
        lo = hi = None
        if i < len(s) and s[i] == '{':
            j = s.index('}', i)
            spec = s[i+1:j]
            if ',' in spec:
                a, b = spec.split(',', 1)
                lo = int(a) if a else 0
                hi = int(b) if b else None
            else:
                lo = hi = int(spec)
            i = j + 1
        elif i < len(s) and s[i] == '?':
            lo, hi = 0, 1; i += 1
        elif i < len(s) and s[i] == '*':
            lo, hi = 0, None; i += 1
        elif i < len(s) and s[i] == '+':
            lo, hi = 1, None; i += 1
        if lo is not None:
            term = ("rep", term, lo, hi)
        terms.append(term)
    return terms

def parse_class(spec):
    negated = spec.startswith("[^")
    inner = spec[2:-1] if negated else spec[1:-1]
    chars = set()
    i = 0
    while i < len(inner):
        if inner[i] == '\\' and i+1 < len(inner) and inner[i+1] == 'x':
            chars.add(chr(int(inner[i+2:i+4], 16))); i += 4; continue
        if inner[i] == '\\' and i+1 < len(inner):
            chars.add(unescape(inner[i:i+2])); i += 2; continue
        if i + 2 < len(inner) and inner[i+1] == '-':
            for c in range(ord(inner[i]), ord(inner[i+2]) + 1):
                chars.add(chr(c))
            i += 3; continue
        chars.add(inner[i]); i += 1
    return chars, negated

def unescape(s):
    return {'\\n': '\n', '\\t': '\t', '\\r': '\r', '\\"': '"', "\\'": "'", '\\\\': '\\'}.get(s, s[-1])

# ---------------- matcher (memoized reachability over (rule, pos)) ----------------

class Matcher:
    def __init__(self, rules, text):
        self.rules = rules
        self.text = text
        self.memo = {}
        self.inflight = set()

    def match_rule(self, name, pos):
        key = (name, pos)
        if key in self.memo:
            return self.memo[key]
        if key in self.inflight:
            return set()  # left-recursion guard
        self.inflight.add(key)
        outs = set()
        for alt in self.rules.get(name, []):
            outs |= self.match_seq(alt, pos)
        self.inflight.discard(key)
        self.memo[key] = outs
        return outs

    def match_seq(self, terms, pos):
        positions = {pos}
        for t in terms:
            nxt = set()
            for p in positions:
                nxt |= self.match_term(t, p)
            positions = nxt
            if not positions:
                break
        return positions

    def match_term(self, t, pos):
        kind = t[0]
        if kind == "lit":
            lit = t[1]
            return {pos + len(lit)} if self.text.startswith(lit, pos) else set()
        if kind == "class":
            chars, negated = t[1], t[2]
            if pos >= len(self.text):
                return set()
            inch = self.text[pos] in chars
            return {pos + 1} if (inch != negated) else set()
        if kind == "ref":
            return self.match_rule(t[1], pos)
        if kind == "group":
            outs = set()
            for alt in t[1]:
                outs |= self.match_seq(alt, pos)
            return outs
        if kind == "rep":
            _, sub, lo, hi = t
            hi_cap = hi if hi is not None else len(self.text) - pos + 1
            results = set()
            frontier = {pos}
            count = 0
            while frontier:
                if count >= lo:
                    results |= frontier
                if count >= hi_cap:
                    break
                nxt = set()
                for p in frontier:
                    for q in self.match_term(sub, p):
                        if q > p or sub[0] == "group":  # progress guard
                            nxt.add(q)
                frontier = nxt
                count += 1
            return results
        return {pos}

def match(rules, text):
    m = Matcher(rules, text)
    return len(text) in m.match_rule("root", 0)

def decode_case(s):
    """Battery files are line-based; \\n \\t \\\\ are decoded."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            nxt = s[i+1]
            if nxt == 'n': out.append('\n'); i += 2; continue
            if nxt == 't': out.append('\t'); i += 2; continue
            if nxt == '\\': out.append('\\'); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)

def main():
    path = sys.argv[1]
    rules = parse_gbnf(open(path).read())
    print(f"parsed {path}: {len(rules)} rules", file=sys.stderr)

    accepts, rejects = [], []
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--accept-inline":
            accepts.append(args[i+1]); i += 2
        elif args[i] == "--reject-inline":
            rejects.append(args[i+1]); i += 2
        elif args[i] == "--accept":
            accepts.extend(l.rstrip("\n") for l in open(args[i+1]) if l.strip() and not l.startswith("#")); i += 2
        elif args[i] == "--reject":
            rejects.extend(l.rstrip("\n") for l in open(args[i+1]) if l.strip() and not l.startswith("#")); i += 2
        else:
            i += 1

    fails = 0
    for t in accepts:
        t = decode_case(t)
        ok = match(rules, t)
        print(f"  [{'OK' if ok else '!!!'}] accept: {t[:70]!r}")
        fails += 0 if ok else 1
    for t in rejects:
        t = decode_case(t)
        ok = not match(rules, t)
        print(f"  [{'OK' if ok else '!!!'}] reject: {t[:70]!r}")
        fails += 0 if ok else 1
    print(f"{'ALL PASS' if fails == 0 else f'{fails} FAILURES'} ({len(accepts)} accept, {len(rejects)} reject)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
