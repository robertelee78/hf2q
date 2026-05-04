#!/bin/bash
# iter83 chunk-engaged bench: 4096-token prompt that engages the chunk path.
#
# 22884 bytes of /tmp/iter78-chunk-22884.txt tokenises to exactly 4096
# tokens (4096 % 64 = 0), engaging chunk_path_eligible. This fixture
# exercises both:
#   - iter78 ChunkAllocsArena (hf2q-side, 7 wrapper scratches)
#   - iter83 ChunkInternalArena (mlx-native-side, 7 large + 5 small scratches)
#
# Pre-registered hypothesis: -50 to -100 ms wall improvement on chunk-
# engaged 4096-token workload (mirrors iter78's pattern at the next
# nesting level). Falsification: <30ms wall.
#
# Same harness pattern as iter78 cold-loop.sh: 3 alternating cold trials
# with 60s cooldown, mandatory warmup per binary.
set -euo pipefail

ITER83_BIN=/tmp/cfa-iter83/hf2q/target/release/hf2q
ITER82_BIN=/opt/hf2q/target/release/hf2q   # iter82 NEUTRAL → SCOPE EXHAUSTION baseline (HEAD 02f062f)
LOGDIR=/tmp/cfa-iter83/impl-bench/logs/chunk-engaged
COOLDOWN_S=60
HEARTBEAT=/tmp/cfa-iter83/.heartbeat
PROMPT=/tmp/iter78-chunk-22884.txt   # 4096 tokens (chunk-eligible)

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

echo "=== chunk-engaged Warmup iter83 at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
run_one warmup-iter83 "${ITER83_BIN}"
sleep ${COOLDOWN_S}

echo "=== chunk-engaged Warmup iter82 at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
run_one warmup-iter82 "${ITER82_BIN}"
sleep ${COOLDOWN_S}

for i in 1 2 3; do
    echo "=== Chunk-Engaged A${i} (iter83) at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
    run_one A${i}-iter83 "${ITER83_BIN}"
    sleep ${COOLDOWN_S}

    echo "=== Chunk-Engaged B${i} (iter82) at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
    run_one B${i}-iter82 "${ITER82_BIN}"
    sleep ${COOLDOWN_S}
done
echo "=== CHUNK-ENGAGED ALL DONE at $(date '+%H:%M:%S') ===" | tee -a "${HEARTBEAT}"
