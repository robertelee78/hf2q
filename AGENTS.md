# hf2q contributor and agent guide

## Project identity

hf2q is a Rust CLI for converting Hugging Face model weights into GGUF,
quantizing them, and running local inference on Apple Silicon through an
OpenAI-compatible HTTP API. The production inference backend is
`mlx-native`; this repository is not a TypeScript or Node.js application.

The crate targets Rust 1.88 or newer. `Cargo.toml`, `Cargo.lock`, the source
tree, checked-in ADRs, and exact validation evidence are authoritative for the
implementation.

Start with these sources when working in an unfamiliar area:

- `README.md` for supported commands and operator-facing behavior.
- `docs/ARCHITECTURE.md` for the source-grounded module map.
- `docs/arch-onboarding.md` for adding a model family.
- `docs/shipping-contract.md` for experimental-feature safety gates.
- The relevant `docs/ADR-*.md` before changing an architectural decision.
- Ruflo AgentDB memory for prior decisions, evidence, and failure patterns.

## Ruflo and ADR memory workflow

Ruflo is the coordination and recall ledger; it does not replace the Rust
implementation, checked-in ADRs, tests, or source-bound evidence.

Before architecture, performance, model-family, serving, or cache work:

1. Run `ak status` to inspect the integrated project state. It reports the
   active native AgentDB writer, learning state, MCP registration, host
   routing, and OpenCode convergence.
2. Use the live Ruflo MCP `memory_search` or `memory_search_unified` across
   relevant namespaces, then retrieve important entries by exact key.
3. Read the ADR named by the recalled decision and verify it against current
   source. Memory is evidence and context, not authority to override source or
   user instructions.
4. Record validated decisions and evidence only after focused tests or
   benchmarks. Keep claims bound to exact commits, model artifacts, prompts,
   settings, and hardware.

The generic `ruflo status` command primarily reports active swarm/session
state. It may show no active memory backend when no swarm is running even while
the project's native AgentDB is healthy. Do not infer that project memory is
empty from that view; use `ak status` and the live memory MCP.

Established memory namespaces include `decisions`, `evidence`, `hf2q`, and
`hf2q-patterns`. For DeepSeek-V4, the decision key
`deepseek-v4-flash-architecture-contract` points to
`docs/ADR-042-deepseek-v4-flash-rust-native.md`; the agentic acceptance contract
is stored as `hf2q/deepseek4-agentic-serving-contract-2026-08-05`.

Use Ruflo routing or swarms only when work has genuinely independent lanes and
the added coordination is useful. Never allow two writers in one worktree.
One integration owner reconciles manifests, lockfiles, ADRs, and final commits.
Machine-managed Claude/Codex dual-host guidance belongs in machine-scoped
configuration and must not be copied into this checked-in file.

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
- Treat time to first semantic streamed token as required behavior. An early
  empty SSE role event does not make a long silent prefill acceptable.
- Keep family-specific cache behavior explicit. Qwen cache mechanisms are not
  an automatic fallback for DeepSeek or another architecture.
- Launcher scripts under `scripts/serve_*_opencode.sh` are canonical operator
  entry points. Validate ports and prerequisites before loading a large model,
  and ensure test servers are always stopped during cleanup.

Changes to serving, templates, tool calling, or caching require a realistic
multi-turn coding test that proves correct tool semantics, tool-result
continuation, valid unary/SSE output, and prefix reuse. For performance work,
report cached-token counts, prefill/decode rates, time to first token, model
quality, exact prompt/settings, and the matched reference result.

On a 128 GiB host, never co-reside hf2q and llama.cpp instances of a roughly
90 GiB DeepSeek artifact. Verify memory and listeners before each load, run one
full-model runtime at a time, and unload it before starting the reference.

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

Apply the hf2q operating Kata before and throughout the numbered workflow:

- Need it? If no, it is out of scope; do not spend effort on it.
- If yes, ask whether it is possible. If possible, do it. If it is not yet
  possible, research exhaustively until there is an evidence-backed path.
- Execute the complete loop: hypothesis -> smallest spike or test ->
  reformulate from measured results -> update the governing ADR/docs ->
  implement -> prove correctness, quality, and performance -> commit -> push
  -> merge.

A failed spike is useful evidence, not a shippable change. Remove it from the
landing diff and record the measured conclusion before pursuing the revised
hypothesis. Stopping short of the complete loop is a Kata violation.

1. Recall Ruflo decisions/evidence and read the governing ADR.
2. Inspect `git status`, relevant source, tests, dependency pins, runtime state,
   and any existing user changes before editing.
3. State the observable contract and failure path. Add or update the smallest
   test that can prove it.
4. Make the minimal coherent change. Avoid unrelated cleanup and mass
   formatting.
5. Run focused tests first, then the applicable regression and hardware gate.
6. For behavior or architecture changes, update the governing ADR and memory
   evidence without turning a failed experiment into an accepted claim.
7. Report exactly what was tested, including skipped hardware or real-model
   validation and any unpublished dependency commit.

Do not rely on an ignored local Cargo patch for a landed result. A change that
needs a new `mlx-native` revision is reproducible only after that revision is
published, pinned in `Cargo.toml`/`Cargo.lock`, and verified from a clean source
tree.

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
- Compare identical model artifacts, prompts, sampling settings, context,
  batch sizes, and hardware conditions.
- Warm up when appropriate and report multiple-run medians rather than a
  favorable single run.
- Verify output parity or the documented quality threshold before accepting a
  speedup. Structurally valid JSON with the wrong tool or arguments is a fail.
- Preserve failed experiments in their isolated branch or worktree when they
  contain useful evidence; do not land them on `main`.
- Do not publish performance claims that cannot be reproduced from a
  checked-in script or documented command.

## Code and git hygiene

- Preserve unrelated user changes in a dirty worktree.
- Do not add secrets, credentials, `.env` files, local paths, model artifacts,
  generated machine configuration, or memory databases.
- Validate paths and untrusted API input at boundaries.
- Avoid growing legacy files beyond roughly 500 lines; extract a focused
  module when practical.
- Use focused conventional commits such as `feat(scope): ...`,
  `fix(scope): ...`, `perf(scope): ...`, and `test(scope): ...`.
- Do not add `Co-Authored-By` trailers unless the repository explicitly
  authorizes them.
- Do not commit, push, merge, release, delete worktrees, or discard changes
  unless the user has authorized that action.

`main` is the integration branch. A local passing commit is not a published
release, and an unpushed commit must be described as local.
