#!/usr/bin/env python3
"""ADR-005 Phase 2b — MTEB runner adapter.

Pure-stdout/stderr CLI invoked from `tests/mteb_sanity_harness.rs` via
subprocess. Implements an `mteb.Encoder`-compatible adapter that
forwards `encode(sentences)` calls to hf2q's `/v1/embeddings` HTTP
endpoint, runs the requested 5-task subset of MTEB, and writes a
flat `{model, tasks: {task_name: primary_score}}` JSON to `--out`.

The Rust harness reads that JSON back and asserts
`|measured - published| <= 1.0` per cell.

Pinned MTEB API surface: see tests/fixtures/mteb/requirements.txt.
We accept results in either of mteb's two recent shapes:

  * mteb >= 1.12 — ``runner.run(...)`` returns ``list[TaskResult]``,
    each with ``.task_name`` and ``.scores`` (a dict, sometimes nested
    under split keys like ``"test"``).  The "primary metric" lives
    under the key ``"main_score"`` (newer) or ``"score"`` (older
    intermediate releases).
  * mteb < 1.12 — ``runner.run(...)`` returns ``dict[task_name ->
    {split_name -> {metric -> value}}]``.  Same flattening rules.

When neither shape produces a finite score for a task, the adapter
writes ``null`` for that cell so the Rust harness can flag it
explicitly instead of silently scoring zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Iterable, List, Optional

import numpy as np
import requests

import mteb  # noqa: F401 — surface ImportError loudly if missing


# ---------------------------------------------------------------------------
# HTTP encoder — adapts mteb.Encoder to hf2q /v1/embeddings
# ---------------------------------------------------------------------------


class Hf2qHttpEncoder:
    """`mteb.Encoder`-compatible adapter that POSTs to hf2q.

    mteb's Encoder protocol just needs an `encode(sentences, **kw) ->
    np.ndarray` method; we don't subclass `mteb.Encoder` directly
    because the protocol's mixin chain has shifted across 1.x minor
    releases. Duck-typing keeps the harness portable across the pinned
    range in `requirements.txt`.
    """

    def __init__(
        self,
        server_url: str,
        model_id: str,
        batch_size: int = 32,
        request_timeout_s: float = 300.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id
        self.batch_size = batch_size
        self.request_timeout_s = request_timeout_s
        self._session = requests.Session()

    # mteb's calling convention varies; accept both `(sentences)` and
    # `(sentences, task_name=..., prompt_name=...)` etc.
    def encode(self, sentences: Iterable[str], *args: Any, **kwargs: Any) -> np.ndarray:
        sentences = list(sentences)
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        url = f"{self.server_url}/v1/embeddings"
        out: List[List[float]] = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = sentences[i : i + self.batch_size]
            body = {
                "model": self.model_id,
                "input": chunk,
                "encoding_format": "float",
            }
            r = self._session.post(url, json=body, timeout=self.request_timeout_s)
            if r.status_code != 200:
                # Surface the server's error body verbatim — hf2q
                # returns OpenAI-shape error objects, which makes it
                # easy to triage (e.g. model_not_loaded).
                raise RuntimeError(
                    f"hf2q /v1/embeddings returned HTTP {r.status_code}: "
                    f"body={r.text[:1024]!r}"
                )
            data = r.json().get("data") or []
            if len(data) != len(chunk):
                raise RuntimeError(
                    f"hf2q returned {len(data)} embeddings for {len(chunk)} inputs "
                    f"(model={self.model_id!r})"
                )
            for row in data:
                emb = row.get("embedding")
                if not isinstance(emb, list):
                    # Server can return a base64 string when
                    # encoding_format=base64; we asked for float so
                    # anything else is a contract violation.
                    raise RuntimeError(
                        f"hf2q returned non-list embedding payload "
                        f"(type={type(emb).__name__}); only encoding_format='float' "
                        f"is supported by this runner"
                    )
                out.append(emb)
        return np.asarray(out, dtype=np.float32)

    # Some mteb versions probe for these companions on the encoder.
    # Forward both to encode() — hf2q doesn't distinguish queries from
    # corpus entries (the embedding model treats them as plain text).
    def encode_queries(
        self, queries: Iterable[str], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        return self.encode(queries, *args, **kwargs)

    def encode_corpus(
        self, corpus: Iterable[Any], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        # mteb passes corpus as a list of dicts {"title": str, "text":
        # str} for retrieval tasks; concat title + text per BEIR
        # convention so we match what mteb's reference encoder does.
        texts: List[str] = []
        for doc in corpus:
            if isinstance(doc, dict):
                title = doc.get("title", "") or ""
                text = doc.get("text", "") or ""
                texts.append((title + " " + text).strip())
            else:
                texts.append(str(doc))
        return self.encode(texts, *args, **kwargs)


# ---------------------------------------------------------------------------
# Score extraction — handles both mteb-1.12+ and older 1.x result shapes
# ---------------------------------------------------------------------------


_PRIMARY_METRIC_KEYS = ("main_score", "score")
_SPLIT_PREFERENCE = ("test", "val", "validation", "dev", "train")


def _coerce_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isfinite(f):
        return f
    return None


def _extract_primary_from_scores_blob(blob: Any) -> Optional[float]:
    """Walk a single TaskResult's scores blob and return the primary
    metric as a float, or None if nothing recognizable is found.

    Handles three shapes seen in mteb 1.x releases:
      1. ``{"main_score": 0.87, ...}`` (flat)
      2. ``{"test": {"main_score": 0.87, ...}, ...}`` (split-keyed)
      3. ``[{"main_score": 0.87, ...}, ...]`` (list of split records)
    """
    # Case 1 / 2: dict
    if isinstance(blob, dict):
        # Case 1: top-level metric keys
        for key in _PRIMARY_METRIC_KEYS:
            if key in blob:
                v = _coerce_float(blob[key])
                if v is not None:
                    return v
        # Case 2: prefer canonical splits
        for split in _SPLIT_PREFERENCE:
            if split in blob:
                v = _extract_primary_from_scores_blob(blob[split])
                if v is not None:
                    return v
        # Fallback: any nested dict — last resort to avoid silent zero.
        for v_blob in blob.values():
            v = _extract_primary_from_scores_blob(v_blob)
            if v is not None:
                return v
    # Case 3: list of split records
    if isinstance(blob, list):
        for v_blob in blob:
            v = _extract_primary_from_scores_blob(v_blob)
            if v is not None:
                return v
    # Bare scalar (unlikely but cheap to handle)
    return _coerce_float(blob)


def _flatten_runner_results(results: Any) -> dict[str, Optional[float]]:
    """Convert mteb's raw runner output into a flat
    `{task_name: primary_score}` dict.

    Handles:
      * mteb >= 1.12: ``list[TaskResult]`` with ``.task_name`` and ``.scores``
      * mteb < 1.12: ``dict[task_name -> nested-scores-blob]``
      * Anything else: best-effort introspection.
    """
    out: dict[str, Optional[float]] = {}

    if isinstance(results, list):
        for r in results:
            # TaskResult-style object with attributes
            name = (
                getattr(r, "task_name", None)
                or getattr(r, "task", None)
                or (r.get("task_name") if isinstance(r, dict) else None)
            )
            blob = (
                getattr(r, "scores", None)
                if not isinstance(r, dict)
                else r.get("scores")
            )
            if name is None or blob is None:
                continue
            out[str(name)] = _extract_primary_from_scores_blob(blob)
    elif isinstance(results, dict):
        for name, blob in results.items():
            out[str(name)] = _extract_primary_from_scores_blob(blob)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ADR-005 Phase 2b MTEB runner — adapts hf2q /v1/embeddings "
        "to mteb.Encoder and writes flat {model, tasks: {name: score}} JSON.",
    )
    ap.add_argument("--model-id", required=True, help="HF model id (sent as JSON 'model' field)")
    ap.add_argument("--server", required=True, help="hf2q server base URL, e.g. http://127.0.0.1:8765")
    ap.add_argument(
        "--tasks",
        required=True,
        help="comma-separated MTEB task names (e.g. BIOSSES,Banking77Classification,...)",
    )
    ap.add_argument("--out", required=True, help="output JSON path; receives {model, tasks}")
    ap.add_argument(
        "--results-folder",
        default=None,
        help="mteb output_folder (default: /tmp/mteb-output)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="encode() batch size (default: 32)",
    )
    args = ap.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_names:
        print("runner.py: --tasks is empty", file=sys.stderr)
        return 2

    results_folder = args.results_folder or "/tmp/mteb-output"
    os.makedirs(results_folder, exist_ok=True)

    print(
        f"runner.py: model_id={args.model_id} server={args.server} "
        f"tasks={task_names} results_folder={results_folder}",
        file=sys.stderr,
        flush=True,
    )

    # mteb.get_tasks(...) is the stable entry point across 1.x releases;
    # the constructor MTEB(tasks=...) accepts task objects or task name
    # strings. We use task objects so a typo surfaces here, not after a
    # multi-minute dataset download.
    try:
        tasks = mteb.get_tasks(tasks=task_names)
    except Exception as e:
        print(f"runner.py: mteb.get_tasks failed: {e}", file=sys.stderr)
        return 3

    encoder = Hf2qHttpEncoder(
        server_url=args.server,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )

    runner = mteb.MTEB(tasks=tasks)
    t0 = time.perf_counter()
    try:
        raw_results = runner.run(
            encoder,
            output_folder=results_folder,
            overwrite_results=True,
        )
    except Exception as e:
        print(f"runner.py: mteb runner.run failed: {e}", file=sys.stderr)
        return 4
    elapsed = time.perf_counter() - t0
    print(f"runner.py: mteb runner.run wall={elapsed:.2f}s", file=sys.stderr, flush=True)

    flat = _flatten_runner_results(raw_results)

    # Coerce missing-task entries to None (-> JSON null) so the Rust
    # harness can distinguish "task not run" from "task scored 0.0".
    payload_tasks: dict[str, Any] = {}
    for t in task_names:
        v = flat.get(t)
        payload_tasks[t] = v if (isinstance(v, float) and math.isfinite(v)) else None

    payload = {
        "model": args.model_id,
        "tasks": payload_tasks,
        "_runner_seconds": round(elapsed, 3),
        "_mteb_version": getattr(mteb, "__version__", "unknown"),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"WROTE {args.out} model={args.model_id} cells={len(payload_tasks)}",
        flush=True,
    )

    # Exit 0 even when some tasks scored None — the Rust harness owns
    # the per-cell PASS/FAIL gate. Exiting non-zero would suppress the
    # JSON write and lose the partial result.
    return 0


if __name__ == "__main__":
    sys.exit(main())
