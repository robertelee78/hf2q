# Qwen3.8 Transformers reference

This directory contains the pinned, validation-only external reference for
ADR-046. It is not an hf2q runtime dependency. The script accepts the exact
token ids and prediction schedule emitted by the hidden `source-teacher`
operator; it never tokenizes or re-renders a prompt.

Run hf2q and the external reference sequentially so two 27B models are never
resident at once. Every destination must be fresh:

```bash
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo build --release --locked

target/release/hf2q source-teacher \
  --model-dir "$MODEL_DIR" \
  --output "$NATIVE_TARGET" \
  --execute >"$NATIVE_SUMMARY"

uv run --frozen --project scripts/reference/qwen38_transformers \
  scripts/reference/qwen38_transformers/run_reference.py \
  --model-dir "$MODEL_DIR" \
  --source-teacher-summary "$NATIVE_SUMMARY" \
  --output-target "$REFERENCE_TARGET" \
  --output-evidence "$REFERENCE_EVIDENCE" \
  --device mps

target/release/hf2q source-teacher-reference \
  --model-dir "$MODEL_DIR" \
  --native-summary "$NATIVE_SUMMARY" \
  --native-target "$NATIVE_TARGET" \
  --external-evidence "$REFERENCE_EVIDENCE" \
  --external-target "$REFERENCE_TARGET" >"$COMPARISON_RECEIPT"
```

The first matched canary characterizes backend drift. It deliberately has no
quality threshold. Thresholds must be declared from Calibration and
PolicyValidation evidence before AcceptanceHoldout is opened. The external
target, evidence JSON, and comparison JSON cannot recreate source-teacher,
sensitivity, allocator, selector, autoquant, or replay authority.

The lightweight framing tests do not load the model:

```bash
uv run --frozen --project scripts/reference/qwen38_transformers \
  python -m unittest scripts/reference/qwen38_transformers/test_reference.py
```
