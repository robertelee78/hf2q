#!/usr/bin/env python3
"""Build a stratified probe corpus from OBLITERATUS's prompt registry.

Parses /opt/OBLITERATUS/obliteratus/prompts.py with `ast` (no heavy imports)
and emits a TSV corpus for probe.sh: <id><TAB><prompt>.

Defaults: 30 harmful (stride-sampled across BUILTIN_HARMFUL's tiers) +
10 harmless (stride-sampled from BUILTIN_HARMLESS as the behavior control).
Deterministic via fixed strides; adjust with env vars.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

PROMPTS_PY = os.environ.get(
    "OBLITERATUS_PROMPTS", "/opt/OBLITERATUS/obliteratus/prompts.py"
)
N_HARMFUL = int(os.environ.get("N_HARMFUL", "30"))
N_HARMLESS = int(os.environ.get("N_HARMLESS", "10"))
OUT = os.environ.get("CORPUS_OUT", "")


def load_list(tree: ast.Module, name: str) -> list[str]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            value = ast.literal_eval(node.value)
            if not all(isinstance(p, str) for p in value):
                raise TypeError(f"{name} contains non-string entries")
            return value
    raise KeyError(f"{name} not found in {PROMPTS_PY}")


def stride_sample(items: list[str], n: int) -> list[str]:
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    tree = ast.parse(Path(PROMPTS_PY).read_text())
    harmful = stride_sample(load_list(tree, "BUILTIN_HARMFUL"), N_HARMFUL)
    harmless = stride_sample(load_list(tree, "BUILTIN_HARMLESS"), N_HARMLESS)

    rows = [f"# probe corpus from {PROMPTS_PY} "
            f"({len(harmful)} harmful + {len(harmless)} harmless, stride-sampled)"]
    rows += [f"h{i:03d}\t{p}" for i, p in enumerate(harmful, 1)]
    rows += [f"b{i:03d}\t{p}" for i, p in enumerate(harmless, 1)]
    text = "\n".join(rows) + "\n"

    if OUT:
        Path(OUT).write_text(text)
        print(f"wrote {len(rows) - 1} prompts to {OUT}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
