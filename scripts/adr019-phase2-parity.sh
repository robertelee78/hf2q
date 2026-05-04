#!/usr/bin/env bash
# ADR-019 Phase 2 iter89e2-F — 4-fixture decode parity harness
#
# Runs the production hf2q binary at /opt/hf2q/target/release/hf2q on each
# of the 4 canonical fixtures, with --max-tokens 32 --temperature 0
# --top-k 1, and SHA256s the UTF-8 decoded text.
#
# Inputs:
#   $1 = label (e.g. "baseline" or "phase2-iter89e2f")
#
# Outputs (under /tmp/adr019_phase2_parity/):
#   <fixture>_<label>.{out,err,text,sha}
set -euo pipefail
LABEL="${1:?usage: $0 <label>}"
OUTDIR=/tmp/adr019_phase2_parity
mkdir -p "$OUTDIR"

PROMPT="Explain transformer neural networks in one short paragraph."
HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"

declare -a FIXTURES=(
  "qwen3.6-27b-dwq46:/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf"
  "qwen3.6-35b-a3b-apex:/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf"
  "qwen3.6-35b-a3b-q4_0-flat:/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat/qwen3.6-35b-a3b-abliterix-ega-abliterated-q4_0-flat.gguf"
  "gemma-4-26B-A4B-dwq:/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
)

for entry in "${FIXTURES[@]}"; do
  NAME="${entry%%:*}"
  PATH_GGUF="${entry##*:}"
  echo "=== $NAME ($LABEL) ===" >&2
  STAMP="${OUTDIR}/${NAME}_${LABEL}"
  "$HF2Q" generate \
    --model "$PATH_GGUF" \
    --prompt "$PROMPT" \
    --max-tokens 32 \
    --temperature 0 \
    --top-k 1 \
    > "${STAMP}.out" 2> "${STAMP}.err" || {
      echo "  FAIL: $NAME (see ${STAMP}.err)" >&2
      continue
    }
  # Extract just the generated text after the load banner. The banner ends
  # with the "prefill: N tok in Mms (T tok/s)" line, then a blank line, then
  # the decoded text.
  awk '
    /^prefill:/ { in_text = 1; next }
    in_text == 1 && NF == 0 { next }
    in_text == 1 { print }
  ' "${STAMP}.out" > "${STAMP}.text"
  shasum -a 256 "${STAMP}.text" | tee "${STAMP}.sha" >&2
done
