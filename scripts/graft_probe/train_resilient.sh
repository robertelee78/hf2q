#!/bin/bash
# Resilient training wrapper (ADR-059 graft lane).
#
# The shared 128GB host jetsams big training runs when co-tenant waves
# land (three kills measured: one from own concurrency, two from
# co-tenants — resident VMs, builds, other agents' spikes). The trainer
# is now resumable (resume.pt every --ckpt-every steps, fail-closed on
# config mismatch), so this wrapper just retries until ckpt_final.pt
# exists: each kill costs <= ckpt_every steps, not the run.
#
# Per-attempt memory guard: require >= 60GB free before loading the
# model; otherwise wait for co-tenents to recede.
#
# Usage: train_resilient.sh <model-key> [max-attempts]
#   e.g. train_resilient.sh gemma4-26b
set -u
MODEL_KEY="${1:?usage: train_resilient.sh <model-key> [max-attempts]}"
MAX_ATTEMPTS="${2:-50}"
PY=/opt/phantom-kv/.venv/bin/python
PORT=scripts/graft_probe/phantom_port.py
ART=/opt/hf2q/artifacts/grafts/$MODEL_KEY

mkdir -p "$ART/train"
attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    if [ -f "$ART/train/ckpt_final.pt" ]; then
        echo "[resilient] ckpt_final.pt exists — training complete"
        break
    fi
    FREE=$(memory_pressure -Q 2>/dev/null | grep -oE 'free percentage: [0-9]+' | grep -oE '[0-9]+')
    if [ -n "$FREE" ] && [ $((128 * FREE / 100)) -lt 60 ]; then
        echo "[resilient] attempt $attempt: only ${FREE}% free — co-tenant pressure; waiting 120s"
        sleep 120
        continue
    fi
    echo "[resilient] attempt $attempt at $(date) (${FREE:-?}% free)"
    if "$PY" "$PORT" train --model="$MODEL_KEY"; then
        echo "[resilient] training exited cleanly at $(date)"
        break
    fi
    echo "[resilient] attempt $attempt died at $(date) — resume.pt holds the progress; retrying in 30s"
    sleep 30
done

if [ ! -f "$ART/train/ckpt_final.pt" ]; then
    echo "[resilient] FAILED: no ckpt_final.pt after $MAX_ATTEMPTS attempts"
    exit 1
fi

cd /opt/hf2q || exit 1
"$PY" "$PORT" compile --model="$MODEL_KEY" \
    && "$PY" "$PORT" eval --model="$MODEL_KEY" \
    && "$PY" "$PORT" to-gguf --model="$MODEL_KEY" \
    && echo "[resilient] PIPELINE_DONE at $(date)"
