#!/usr/bin/env bash
# byte_cmp_full_pipeline.sh — re-validate hf2q convert against current canonical SHA
#
# Usage: scripts/byte_cmp_full_pipeline.sh [<hf_model_dir>] [<output_dir>]
#
# Validates that hf2q's full convert pipeline produces byte-equivalent output
# to canonical /opt/llama.cpp's convert+quantize chain. Reports per-tensor-type
# residuals plus overall percentage.
#
# Prerequisite: /opt/llama.cpp HEAD matches `data/llama_cpp_pin.txt` (or be
# prepared to regenerate the canonical reference).
#
# Run from /opt/hf2q root.

set -euo pipefail

HF_DIR=${1:-/opt/hf2q/models/google-gemma-4-26b-a4b-it}
OUT_DIR=${2:-/tmp/hf2q_byte_cmp}
QUANT=${QUANT:-q4_k_m}

mkdir -p "$OUT_DIR"

NAME=$(basename "$HF_DIR")
CANON_F16="$OUT_DIR/${NAME}_canonical_f16.gguf"
CANON_Q="$OUT_DIR/${NAME}_canonical_${QUANT}.gguf"
HF2Q_Q="$OUT_DIR/${NAME}_hf2q_${QUANT}.gguf"

# Step 1: canonical convert F32→F16 if not exists
if [ ! -f "$CANON_F16" ]; then
    echo "→ Step 1/3: canonical convert F32 → F16"
    python3 /opt/llama.cpp/convert_hf_to_gguf.py "$HF_DIR" --outtype f16 --outfile "$CANON_F16"
else
    echo "[skip] canonical F16 already at $CANON_F16"
fi

# Step 2: canonical llama-quantize F16 → target quant if not exists
if [ ! -f "$CANON_Q" ]; then
    echo "→ Step 2/3: canonical llama-quantize F16 → $QUANT"
    /opt/llama.cpp/build/bin/llama-quantize "$CANON_F16" "$CANON_Q" "${QUANT^^}"
else
    echo "[skip] canonical $QUANT already at $CANON_Q"
fi

# Step 3: hf2q convert
echo "→ Step 3/3: hf2q convert HF_DIR → $QUANT"
time /opt/hf2q/target/release/hf2q convert "$HF_DIR" --quant "$QUANT" --output "$HF2Q_Q"

# Compare
echo "→ Byte-cmp:"
python3 -c "
import gguf
canon = gguf.GGUFReader('$CANON_Q', 'r')
hf2q = gguf.GGUFReader('$HF2Q_Q', 'r')
c_dict = {t.name: t for t in canon.tensors}
h_dict = {t.name: t for t in hf2q.tensors}
totals = {}
for name in c_dict:
    if name not in h_dict: continue
    t_type = c_dict[name].tensor_type.name
    c = c_dict[name].data.tobytes()
    h = h_dict[name].data.tobytes()
    if len(c) != len(h): continue
    diff = sum(1 for a, b in zip(c, h) if a != b)
    if t_type not in totals: totals[t_type] = [0, 0]
    totals[t_type][0] += diff
    totals[t_type][1] += len(c)
overall_d, overall_t = 0, 0
for k, (d, t) in sorted(totals.items()):
    if t > 0:
        print(f'  {k}: {d}/{t} = {100*d/t:.6f}%')
        overall_d += d; overall_t += t
print(f'  OVERALL: {overall_d}/{overall_t} = {100*overall_d/overall_t:.6f}%')
"
