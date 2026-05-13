#!/bin/bash
# ADR-029 iter-146: per-CB GPU wall-time attribution via xctrace.
#
# Captures a Metal System Trace of hf2q decode, extracts
# `metal-application-command-buffer-submissions` schema (which has
# submission-to-completion durations per CB), and prints summary stats.
#
# Works on Apple Silicon without needing:
#   - commit_labeled wiring (gemma4 uses plain encoder.commit())
#   - MTLCounterSampleBuffer (hardware-blocked at per-dispatch granularity)
#   - Shader Timeline (requires custom .tracetemplate)
#
# Usage:
#   scripts/adr029_xctrace_per_cb_gpu_time.sh [MAX_TOKENS=200]
#
# Output: per-CB GPU wall time stats + raw submissions XML in /tmp.

set -euo pipefail

MAX_TOKENS="${1:-200}"
TRACE_DIR=/tmp/adr029_xctrace
mkdir -p "$TRACE_DIR"
TRACE="$TRACE_DIR/hf2q_$(date +%Y%m%d_%H%M%S).trace"
XML="$TRACE_DIR/submissions_$(date +%Y%m%d_%H%M%S).xml"

GGUF=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf
HF2Q=/opt/hf2q/target/release/hf2q

echo "[iter-146] capturing trace ($MAX_TOKENS decode tokens)..."
xcrun xctrace record \
    --template "Metal System Trace" \
    --output "$TRACE" \
    --launch -- \
    "$HF2Q" generate \
        --model "$GGUF" \
        --prompt "Q." \
        --max-tokens "$MAX_TOKENS" \
        --ignore-eos \
        --temperature 0 \
    2>&1 | tail -5

echo "[iter-146] exporting submissions schema..."
xcrun xctrace export \
    --input "$TRACE" \
    --xpath '/trace-toc/run/data/table[@schema="metal-application-command-buffer-submissions"]' \
    > "$XML" 2>/dev/null

echo "[iter-146] analyzing..."
python3 -c "
import re
with open('$XML') as f:
    xml = f.read()
rows = re.findall(r'<row>(.*?)</row>', xml, re.S)
durations_us = []
for row in rows:
    m = re.search(r'<duration[^>]*>(\d+)</duration>', row)
    if m:
        durations_us.append(int(m.group(1)) / 1000.0)
if not durations_us:
    print('no rows extracted')
    exit(1)
ds = sorted(durations_us)
print(f'Total CBs:    {len(ds)}')
print(f'Sum (all):    {sum(ds)/1000:.2f} ms')
print(f'Mean:         {sum(ds)/len(ds):.1f} us')
print(f'Median:       {ds[len(ds)//2]:.1f} us')
print(f'P95:          {ds[int(len(ds)*0.95)]:.1f} us')
print(f'P99:          {ds[int(len(ds)*0.99)]:.1f} us')
print(f'Max:          {ds[-1]:.1f} us')
print(f'')
print(f'Top 10 longest CBs:')
for d in ds[-10:][::-1]:
    print(f'  {d:>8.1f} us')
print(f'')
print(f'Histogram (us bins):')
import collections
bins = collections.Counter()
edges = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000]
for d in ds:
    for i, e in enumerate(edges[1:], 1):
        if d < e:
            bins[edges[i-1]] += 1
            break
for e in edges[:-1]:
    print(f'  [{e:>5}-{(edges[edges.index(e)+1]):>5}] us: {bins[e]:>4} CBs')
"
echo ""
echo "Trace: $TRACE"
echo "XML:   $XML"
