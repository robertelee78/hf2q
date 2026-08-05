# hf2q contributor and agent guide

## Project identity

hf2q is a Rust CLI for converting Hugging Face model weights into GGUF,
quantizing them, and running local inference on Apple Silicon through an
OpenAI-compatible HTTP API. The production inference backend is
`mlx-native`; this repository is not a TypeScript or Node.js application.

The crate targets Rust 1.88 or newer. `Cargo.toml`, `Cargo.lock`, the source
tree, and the checked-in ADRs are the authoritative description of the
implementation.

Start with these documents when working in an unfamiliar area:

- `README.md` for supported commands and operator-facing behavior.
- `docs/ARCHITECTURE.md` for the source-grounded module map.
- `docs/arch-onboarding.md` for adding a model family.
- `docs/shipping-contract.md` for experimental-feature safety gates.
- The relevant `docs/ADR-*.md` before changing an architectural decision.

## Non-negotiable product boundaries

- hf2q owns conversion and quantization. Production code must not shell out
  to Python, llama.cpp, mlx-lm, vendor converters, or another quantization
  tool as an implementation or fallback.
- hf2q owns inference. Production serving and generation must run through the
  Rust and `mlx-native` paths, not an external inference process.
- Downloads are the narrow exception. Rust code may use `hf-hub`, and
  operator or fixture scripts may use Hugging Face tooling, `curl`, or `wget`
  to retrieve source weights or test data. Retrieval must not outsource
  conversion, quantization, or inference.
- Benchmark and parity harnesses may execute external reference programs when
  the comparison explicitly requires them. Those programs must never become
  runtime dependencies.
- Do not download a pre-quantized model to satisfy a conversion request.
  Given Hugging Face source weights, `hf2q convert` must produce the requested
  quantized artifact itself.
- Model-family behavior is explicit. Never silently route an unsupported
  architecture through an approximately compatible loader, tokenizer,
  template, cache, or forward graph.
- Large weights, GGUFs, caches, profiles, and generated artifacts are local
  data. Never commit them.

## Agentic serving contract

The OpenAI-compatible server is used by OpenCode and other agentic coding
clients, so correctness is broader than producing one completion:

- Preserve OpenAI chat request and SSE response compatibility.
- Support tool definitions, assistant tool calls, tool-result messages, and
  the model family's native tool-call encoding.
- Reuse the unchanged prompt prefix and its KV state across turns. A normal
  follow-up must not recompute the full conversation.
- Invalidate or rebuild cache state safely when the prefix, template, model,
  or inference configuration is incompatible. Never leak state between
  conversations.
- Treat time to first streamed token as a required behavior. A GPU workload
  with no timely response is a failure even when the process remains alive.
- Keep family-specific cache behavior explicit. Qwen cache mechanisms are not
  an automatic fallback for DeepSeek or another architecture.
- Launcher scripts under `scripts/serve_*_opencode.sh` are canonical operator
  entry points. Validate ports and prerequisites before loading a large model,
  and ensure test servers are always stopped during cleanup.

Changes to serving, templates, tool calling, or caching require a multi-turn
test that proves both valid output and prefix reuse. For performance work,
report cached-token counts, prefill/decode rates, and time to first token when
those metrics are relevant.

## Source layout

- `src/arch/`: architecture registry and conformance metadata.
- `src/input/`: Hugging Face configuration, tokenizer, and safetensors input.
- `src/models/`: per-family tensor mapping and conversion behavior.
- `src/quantize/`: quantization codecs and policies.
- `src/backends/`: GGUF and other output writers.
- `src/inference/`: MLX-native model loading, forward graphs, and KV caches.
- `src/serve/`: OpenAI-compatible HTTP, streaming, scheduling, and caching.
- `src/quality/`: parity and quality measurements.
- `tests/`: integration and regression tests.
- `scripts/`: reproducible launchers, benchmarks, smoke tests, and runbooks.
- `docs/`: architecture, ADRs, operator guidance, and experiment records.

New model families follow the established order: registry and tensor catalog,
conversion mapping, inference graph, smoke/parity proof, then serving. Do not
open the serve path before conversion and forward-pass correctness are proven.

## Working method

1. Inspect `git status`, the relevant implementation, its tests, and the
   governing ADR before editing.
2. State the observable contract and identify the failure path. Add or update
   the smallest test that can prove it.
3. Make the minimal coherent change. Avoid unrelated cleanup and mass
   formatting.
4. Run focused tests first, then the applicable regression gate.
5. For behavior or architecture changes, update the governing ADR or operator
   documentation in the same change.
6. Report what was actually tested, including skipped hardware or real-model
   validation.

Use parallel agents only when work has genuinely independent lanes. Never let
two writers modify one worktree. One integration owner reconciles shared
manifests, lockfiles, and final commits.

## Build and validation

Use locked dependencies for validation:

```bash
cargo check --locked --all-targets --all-features
cargo build --release --locked
cargo test --locked
```

Prefer a focused command while iterating, for example:

```bash
cargo test --locked --bin hf2q deepseek4
cargo test --locked --test convert_integration
cargo run --locked -- doctor
```

CI treats `cargo check`, the release build, hosted-safe tests, and the unsafe
experiment activation matrix as blocking. Clippy and whole-tree rustfmt are
currently informational because the legacy tree has existing debt. Keep edited
code rustfmt-compatible, but do not reformat unrelated files.

Metal execution, real-model generation, and parity/performance checks require
an appropriate Apple Silicon host and local model artifacts. Unit tests alone
do not prove a model-serving change.

## Performance and parity rules

- Establish a source-bound baseline before optimizing.
- Compare identical model artifacts, prompts, sampling settings, context, and
  hardware conditions.
- Warm up when appropriate and report multiple-run medians rather than a
  favorable single run.
- Verify output parity or the documented quality threshold before accepting a
  speedup.
- Preserve failed experiments in their isolated branch or worktree when they
  contain useful evidence; do not land them on `main`.
- Do not publish performance claims that cannot be reproduced from a checked-in
  script or documented command.

## Code and git hygiene

- Preserve unrelated user changes in a dirty worktree.
- Do not add secrets, credentials, `.env` files, local paths, model artifacts,
  or generated machine configuration.
- Validate paths and untrusted API input at boundaries.
- Avoid growing legacy files beyond roughly 500 lines; extract a focused module
  when practical.
- Use focused conventional commits such as `feat(scope): ...`,
  `fix(scope): ...`, `perf(scope): ...`, and `test(scope): ...`.
- Do not add `Co-Authored-By` trailers unless the repository explicitly
  authorizes them.
- Do not commit, push, merge, release, delete worktrees, or discard changes
  unless the user has authorized that action.

`main` is the integration branch. A local passing commit is not a published
release, and an unpushed commit must be described as local.
