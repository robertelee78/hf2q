#!/bin/bash
# iter83 default bench: pp4123 walkbar (default workload, NOT chunk-engaged).
#
# pp4123 % 64 != 0 → chunk_path_eligible fails → autoregressive prefill
# path runs (NO chunk-pipeline orchestrator dispatch). Expected NEUTRAL
# (zero wall delta) since iter83's ChunkInternalArena lives entirely in
# the chunk-pipeline path that doesn't fire here. This bench enforces
# "do no harm" on default workloads.
#
# Same harness pattern as iter78/iter82.
set -euo pipefail

ITER83_BIN=/tmp/cfa-iter83/hf2q/target/release/hf2q
ITER82_BIN=/opt/hf2q/target/release/hf2q
LOGDIR=/tmp/cfa-iter83/impl-bench/logs/default
COOLDOWN_S=60
HEARTBEAT=/tmp/cfa-iter83/.heartbeat
PROMPT=/tmp/walkbar-pp4096-prompt.txt   # default pp4123 (not chunk-engaged)

mkdir -p "${LOGDIR}"
cd /tmp

run_one() {
    local label="$1"
    local bin="$2"

    HF2Q_PROFILE_GPU_TS=1 \
    HF2Q_PROFILE_W5B8=1 \
    HF2Q_QWEN36_AUTOREG=1 \
    HF2Q_UNSAFE_EXPERIMENTS=1 \
    HF2Q_CHUNK_SCAN_PREFILL=1 \
    "${bin}" generate \
        --model /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf \
        --tokenizer /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat/tokenizer.json \
        --config /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat/config.json \
        --prompt-file "${PROMPT}" \
        --max-tokens 4 \
        --temperature 0.0 \
        > "${LOGDIR}/${label}.log" 2>&1
    grep "^prefill:" "${LOGDIR}/${label}.log" | tail -1 | tee -a "${HEARTBEAT}" || true
}

echo "=== default Warmup iter83 at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
run_one warmup-iter83 "${ITER83_BIN}"
sleep ${COOLDOWN_S}

echo "=== default Warmup iter82 at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
run_one warmup-iter82 "${ITER82_BIN}"
sleep ${COOLDOWN_S}

for i in 1 2 3; do
    echo "=== Default A${i} (iter83) at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
    run_one A${i}-iter83 "${ITER83_BIN}"
    sleep ${COOLDOWN_S}

    echo "=== Default B${i} (iter82) at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
    run_one B${i}-iter82 "${ITER82_BIN}"
    sleep ${COOLDOWN_S}
done
echo "=== DEFAULT ALL DONE at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
