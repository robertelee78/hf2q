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
  --evaluation-split "$EVALUATION_SPLIT" \
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

Run this sequence once with `EVALUATION_SPLIT=calibration` and once with
`EVALUATION_SPLIT=policy-validation`, using fresh destinations throughout.
These characterization runs deliberately have no quality threshold.
Thresholds were declared from both receipts before AcceptanceHoldout was
opened. The one-time holdout transaction completed at hf2q commit
`07b59ba806f273ae8bb9eebf079277a317831a51`; its raw comparison and passing
quality receipt are now checked in. The execution and comparison-minting routes
were removed after that run. Verify the closed evidence against the exact local
source with:

```bash
target/release/hf2q source-teacher-acceptance-verify \
  --model-dir "$MODEL_DIR"
```

The verifier authenticates the exact source, reconstructs the sealed one-row,
one-trajectory plan, byte-verifies both embedded receipts, and checks that the
quality receipt contains the exact raw comparison. It performs no Metal model
load and cannot mint new evidence. The checked-in raw comparison retains every
authority flag as false; the quality receipt grants only predeclared-threshold
quality-gate authority, never source-teacher, sensitivity, allocator, selector,
autoquant, runtime-dependency, DWQ, or replay authority.

The lightweight framing tests do not load the model:

```bash
uv run --frozen --project scripts/reference/qwen38_transformers \
  python -m unittest scripts/reference/qwen38_transformers/test_reference.py
```
