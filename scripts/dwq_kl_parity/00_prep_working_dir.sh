#!/usr/bin/env bash
#
# ADR-020 iter-19b prep — build a working dir for mlx_lm.dwq that:
#  1. Symlinks the cached BF16 safetensors weights (no copy)
#  2. Patches tokenizer_config.json to inject Qwen3.6-27B's
#     chat_template (the abliterated maintainer stripped the
#     default; mlx_lm.dwq's tulu-3-sft-mixture corpus needs
#     apply_chat_template).
#  3. Patches config.json model_type from "qwen3_5_moe_text" to
#     "qwen3_5_moe" (mlx-lm 0.31.2's MODEL_REMAPPING table doesn't
#     resolve the "_text" suffix; the qwen3_5_moe.py module's
#     ModelArgs.from_dict already handles flat-config via the
#     `if "text_config" not in params` branch, so the remap is
#     safe).
#
# Run before 01_run_dwq.sh.

set -euo pipefail
cd "$(dirname "$0")"

REF_SNAP="${REF_SNAP:-$HOME/.cache/huggingface/hub/models--jenerallee78--Qwen3.6-35B-A3B-Abliterix-EGA-abliterated/snapshots}"
TPL_SNAP="${TPL_SNAP:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots}"
WORK="${WORK:-./abliterix_with_chat_template}"

# Pick first snapshot dir under each.
SNAP=$(ls -d "$REF_SNAP"/*/ | head -1)
QSNAP=$(ls -d "$TPL_SNAP"/*/ | head -1)
echo "[prep] reference: $SNAP"
echo "[prep] template src: $QSNAP"

mkdir -p "$WORK"

# Symlink all .safetensors + json files (no copy).
for f in "$SNAP"*.safetensors "$SNAP"*.json; do
    [ -e "$f" ] && ln -sf "$f" "$WORK/$(basename "$f")"
done

# Replace symlinks with real copies for files we need to edit.
for f in tokenizer.json tokenizer_config.json config.json; do
    rm -f "$WORK/$f"
    cp "$SNAP$f" "$WORK/$f"
done

# Inject chat_template + remap model_type.
python3 -c "
import json, sys
src_cfg = json.load(open('$WORK/tokenizer_config.json'))
qwen_cfg = json.load(open('$QSNAP/tokenizer_config.json'))
src_cfg['chat_template'] = qwen_cfg['chat_template']
json.dump(src_cfg, open('$WORK/tokenizer_config.json','w'), indent=2)
print('[prep] injected chat_template (len=' + str(len(qwen_cfg['chat_template'])) + ')')

cfg = json.load(open('$WORK/config.json'))
print('[prep] model_type before:', cfg.get('model_type'))
cfg['model_type'] = 'qwen3_5_moe'
json.dump(cfg, open('$WORK/config.json','w'), indent=2)
print('[prep] model_type after:', cfg['model_type'])
"
echo "[prep] working dir ready: $WORK"
