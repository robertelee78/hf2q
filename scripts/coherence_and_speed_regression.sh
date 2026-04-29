#!/usr/bin/env bash
# scripts/coherence_and_speed_regression.sh
#
# ADR-015 iter41 — combined coherence + speed regression gate.
#
# Pipeline:
#   1. Run `cargo test --test coherence_smoke --release` (always; <60s).
#      Detects degenerate-pattern decode regressions BEFORE any benchmark.
#   2. Run `scripts/bench-matrix.sh` (only if smoke PASS).
#      Captures hf2q vs llama tok/s across the (model × quant) matrix.
#   3. Parse the matrix output and compare per-cell ratio against
#      `tests/perf_baseline.json::cells[<cell>].ratio_floor`. Cell fails if
#      `measured_ratio < floor - tolerance_pp/100`.
#
# Exit codes:
#   0  — all gates PASS
#   1  — smoke FAIL (gibberish detected; no bench run)
#   2  — bench script FAIL (missing binary, OOM, etc.)
#   3  — perf regression: at least one cell dropped below baseline
#   4  — environmental (jq/python missing, baseline json malformed)
#
# Standing rules (from ADR-015 §Lessons learned):
#   - Coherence gate is mandatory before any perf bench. A pre-iter40
#     coherence-blind workflow shipped 4 commits against broken decode.
#   - Peer-parity is the truth-of-the-day. baseline ratios are renewed
#     whenever llama.cpp peer drifts (project_end_gate_reality_check).
#
# Env:
#   HF2Q_BIN=/path/to/hf2q          (default: target/release/hf2q)
#   LLAMA_BENCH_BIN=...              (default: /opt/homebrew/bin/llama-bench)
#   CELLS=cell1,cell2                (subset; default: all)
#   SKIP_BENCH=1                     (only run smoke)
#   N_TRIALS=3 NGEN=256              (passed to bench-matrix.sh)
#
# Usage:
#   ./scripts/coherence_and_speed_regression.sh
#   CELLS=qwen3.6-27b-dwq46 ./scripts/coherence_and_speed_regression.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASELINE_JSON="${BASELINE_JSON:-tests/perf_baseline.json}"
SKIP_BENCH="${SKIP_BENCH:-0}"
OUT_DIR="${OUT_DIR:-/tmp/adr015-iter41-regression}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$OUT_DIR"

log() {
  echo "[$(date -u +%H:%M:%S)] $*" >&2
}

# ─── prereqs ────────────────────────────────────────────────────────────
for tool in cargo python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    log "ERROR: $tool not found in PATH"
    exit 4
  fi
done

if [[ ! -f "$BASELINE_JSON" ]]; then
  log "ERROR: baseline JSON not found at $BASELINE_JSON"
  exit 4
fi

# ─── PHASE 1 — coherence smoke ──────────────────────────────────────────
log "PHASE 1 — cargo test --test coherence_smoke --release"
SMOKE_LOG="$OUT_DIR/smoke-${DATE_TAG}.log"

if cargo test --test coherence_smoke --release -- --nocapture 2>&1 | tee "$SMOKE_LOG"; then
  log "PHASE 1 PASS — no degenerate-pattern decode detected"
else
  log "PHASE 1 FAIL — gibberish decode pattern matched; see $SMOKE_LOG"
  log "REFUSING to run perf bench against broken decode (ADR-015 §Lessons learned)."
  exit 1
fi

if [[ "$SKIP_BENCH" == "1" ]]; then
  log "SKIP_BENCH=1 — exiting after smoke PASS"
  exit 0
fi

# ─── PHASE 2 — bench matrix ─────────────────────────────────────────────
log "PHASE 2 — scripts/bench-matrix.sh"
BENCH_LOG="$OUT_DIR/bench-${DATE_TAG}.log"

if ! bash scripts/bench-matrix.sh 2>&1 | tee "$BENCH_LOG"; then
  log "PHASE 2 FAIL — bench-matrix.sh exited non-zero; see $BENCH_LOG"
  exit 2
fi

# Find the most-recent matrix-*.md the script emitted
MATRIX_MD="$(ls -t /tmp/adr015-iter18/bench/matrix-*.md 2>/dev/null | head -1)"
if [[ -z "$MATRIX_MD" || ! -f "$MATRIX_MD" ]]; then
  log "PHASE 2 FAIL — could not find matrix-*.md emitted by bench-matrix.sh"
  exit 2
fi
log "Matrix output: $MATRIX_MD"

# ─── PHASE 3 — compare against baseline ─────────────────────────────────
log "PHASE 3 — compare per-cell ratio against $BASELINE_JSON"

python3 - "$MATRIX_MD" "$BASELINE_JSON" <<'PY'
import json
import re
import sys

matrix_path, baseline_path = sys.argv[1], sys.argv[2]

with open(baseline_path) as f:
    baseline = json.load(f)
floors = {k: v["ratio_floor"] for k, v in baseline["cells"].items()}
tolerance = baseline.get("_tolerance_pp", 1.0) / 100.0

# Parse the markdown matrix.  bench-matrix.sh writes a row per cell with
# columns: cell | hf2q tok/s | llama tok/s | ratio | Δpp.
# We only need cell+ratio.
cells_seen = {}
row_re = re.compile(
    r"^\|\s*(?P<cell>[a-zA-Z0-9._\-]+)\s*\|"
    r"\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*"
    r"(?P<ratio>[\d.]+)\s*\|"
)
with open(matrix_path) as f:
    for line in f:
        m = row_re.match(line.strip())
        if m:
            try:
                cells_seen[m.group("cell")] = float(m.group("ratio"))
            except ValueError:
                pass

if not cells_seen:
    print(f"PHASE 3 FAIL — no cell rows parsed from {matrix_path}",
          file=sys.stderr)
    sys.exit(2)

failures = []
print()
print(f"{'cell':<32}  {'measured':>9}  {'floor':>7}  {'verdict':<8}")
print("-" * 64)
for cell, ratio in cells_seen.items():
    floor = floors.get(cell)
    if floor is None:
        print(f"{cell:<32}  {ratio:>9.4f}  {'(none)':>7}  SKIP")
        continue
    threshold = floor - tolerance
    if ratio < threshold:
        verdict = "REGRESS"
        failures.append((cell, ratio, floor, threshold))
    else:
        verdict = "PASS"
    print(f"{cell:<32}  {ratio:>9.4f}  {floor:>7.3f}  {verdict}")

print()
if failures:
    print("PHASE 3 FAIL — perf regression:", file=sys.stderr)
    for cell, ratio, floor, threshold in failures:
        print(
            f"  - {cell}: measured {ratio:.4f} < threshold {threshold:.4f} "
            f"(floor {floor:.3f} − tolerance {tolerance*100:.1f}pp)",
            file=sys.stderr,
        )
    sys.exit(3)

print("PHASE 3 PASS — all cells at-or-above ratio floor")
PY
RC=$?

if [[ $RC -ne 0 ]]; then
  log "PHASE 3 — exit $RC"
  exit $RC
fi

log "ALL PHASES PASS — no coherence or speed regression detected"
exit 0
