# hf2q

[![CI](https://github.com/robertelee78/hf2q/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robertelee78/hf2q/actions/workflows/ci.yml)
[![License: Apache-2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](#license)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)](#install)
[![Backend: mlx-native](https://img.shields.io/badge/backend-mlx--native-purple.svg)](https://crates.io/crates/mlx-native)

Pure-Rust CLI for converting HuggingFace models to hardware-optimized
formats — and serving them through an OpenAI-compatible HTTP API on
Apple Silicon. **No C++ at build, test, or runtime** (ADR-008
sovereignty rule); the inference path runs entirely on `mlx-native`
Metal kernels we own end-to-end.

> **Serving reliability is part of correctness.** The canonical SlotAware
> launchers for native Qwen, Gemma, and DeepSeek give each agent an independent
> full logical context while sharing model weights. Qwen SlotAware prefill advances in bounded GPU
> transactions so active streams can decode and cancellation can be observed
> between chunks. A fatal Metal command-buffer/watchdog/ignored-submission
> error, or an independently observed transaction deadline that never
> returns, fails the affected Qwen, Gemma, or DeepSeek worker closed;
> the process must be recreated rather than submitting more work to a poisoned
> queue. See [Full-context agentic serving](#full-context-agentic-serving),
> [the shipping contract](docs/shipping-contract.md), and the family ADRs for
> the exact supported surface and current evidence.

| | |
|---|---|
| **License** | Apache-2.0 OR MIT (dual) |
| **Rust** | 1.88+ |
| **Inference backend** | Exact [`mlx-native`](https://crates.io/crates/mlx-native) registry pin in `Cargo.toml` (Apple Metal) — ADR-008 |
| **Output formats** | GGUF (`llama.cpp` consumers), mlx-lm safetensors |
| **Status** | hf2q 0.1.6 is the release line described by this checkout and resolves published, checksum-pinned `mlx-native 0.10.8`. Public availability is authoritative only when the `v0.1.6` tag, GitHub artifact, and crates.io bytes match the exact main-branch release SHA. Support is family- and scheduler-specific; see `docs/shipping-contract.md`. |

```bash
# Convert a HuggingFace model to a Q4_K_M GGUF (auto-downloads via --repo)
hf2q convert \
  --repo google/gemma-4-26b-it \
  --quant q4_k_m \
  --output models/gemma-4-26b-it-q4_k_m/out.gguf

# Serve it over an OpenAI-compatible HTTP API
hf2q serve --model models/gemma-4-26b-it-q4_k_m/out.gguf --port 8080
```

---

## What it does

`hf2q` is two tools fused into one binary:

1. **A conversion pipeline.** Read HuggingFace `config.json` +
   `*.safetensors`, normalize tensor names per architecture, run
   quantization (legacy block `Q4_0` / `Q8_0`, K-quants
   `Q{2..6}_K_{S,M,L}`, imatrix-weighted K-quants including
   `imatrix-adaptive`, or mixed-bit `dynamic-quant-*`) and emit GGUF
   or mlx-lm safetensors.
   No `llama.cpp` or `candle` is involved at build, test or runtime
   (ADR-008 — "candle divorce"; sovereignty rule in
   `docs/arch-onboarding.md`).

2. **An inference + serving engine.** Load a GGUF, run prefill +
   speculative-or-vanilla decode on the GPU via `mlx-native`, expose
   it through an OpenAI-style `/v1/chat/completions`, `/v1/embeddings`
   and `/v1/models` HTTP API. Supports tools / function-calling,
   streaming SSE, vision (`qwen3vl`), grammar-constrained sampling,
   and a persistent block-prefix KV cache.

Supported architectures today: **Gemma 4 (dense + MoE)**, **Qwen 3.5 /
3.6 (dense + MoE + multi-token-prediction)**, **Qwen 3.8-27B text
(dense + multi-token-prediction)**, **DeepSeek-V4-Flash-0731
(compressed-attention MoE)**, **Qwen 3-VL (vision + text)**, and **BERT /
Nomic-BERT** (embedding-only). Each lives under a single
`src/inference/models/<arch>/` module — the arch-registry (`src/arch/`)
is the single source of truth for tensor catalogs, quality thresholds,
smoke prompts and MTP/vision flags.

## Install

`hf2q` is a Cargo crate. Apple Silicon is currently the only supported
target — the inference path is Metal-only.

```bash
git clone git@github.com:robertelee78/hf2q.git
cd hf2q
cargo build --release
./target/release/hf2q --help
```

The exact `mlx-native` declaration in `Cargo.toml` resolves from `crates.io`.
For local mlx-native development place a path
override in a gitignored `.cargo/config.toml` (template at
`Cargo.toml:217+`) — out-of-the-box `cargo build` does NOT path-pin
to a sibling checkout.

`cargo build` requires:

- macOS with Metal Performance Shaders (M1 or newer).
- A working Rust toolchain at the version pinned in `Cargo.toml`
  (`rust-version = "1.88.0"`).
- Per-arch disk floor for convert (`src/arch/entries/`): **100 GB** for
  Qwen 3.5 dense, **150 GB** for Qwen 3.5 MoE. Smoke preflight refuses
  to start below `disk_floor_gb + 10`.

`hf2q doctor` enumerates the runtime checks (hardware detection, disk
space, optional RuVector backend); run it after `cargo install` if
anything misbehaves.

## CLI subcommands

| Command | What it does |
|---|---|
| `hf2q convert` | HuggingFace safetensors → GGUF (streaming convert, ADR-033 unified pipeline). |
| `hf2q gguf-patch` | Rewrite a GGUF's metadata in place (e.g. inject a chat template). |
| `hf2q info` | Inspect a GGUF model without loading weights. |
| `hf2q generate` | Single-shot text generation from a GGUF on the local GPU. |
| `hf2q serve` | OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/embeddings`). |
| `hf2q parity` | ADR-009 parity validation against locked reference outputs. |
| `hf2q smoke` | ADR-012 end-gate smoke test for a registered architecture. |
| `hf2q cache` | Manage `~/.cache/hf2q/` (list / size / clear). |
| `hf2q doctor` | Diagnose hardware, cache, RuVector, disk. |
| `hf2q completions` | Generate shell completions. |

Run `hf2q <command> --help` for the full flag surface.

### Quantization variants

The `hf2q convert` pipeline accepts two families of `--quant <name>`
values, parsed via
[`QuantSelector::from_name`](src/convert/quant_selector.rs):

| Family | Variants | Notes |
|---|---|---|
| Standard llama.cpp ftypes | `f32`, `f16`, `bf16`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_k`, `q3_k_{s,m,l}`, `q4_k_{s,m}`, `q5_k_{s,m}`, `q6_k`, `iq4_nl` | Byte-identical to stock `llama-quantize` output for the same ftype. |
| APEX algorithmic tiers (MoE arches only) | `apex-quality`, `apex-i-quality`, `apex-balanced`, `apex-i-balanced`, `apex-compact`, `apex-i-compact`, `apex-mini` | Per-tier overlay derived from `mudler/apex-quant`. Auto-detects against the per-model fingerprint manifest at [`data/apex-references/manifest.json`](data/apex-references/manifest.json) (ADR-033 §9). I-tier variants require imatrix data via `--imatrix <file>` or `--imatrix-corpus <name>` (Pi shipped 2026-05-19 — see [I-tier APEX](#i-tier-apex-imatrix-aware-quantization) below). |

Reserved names surface as typed errors with actionable hints:
`--quant dwq` → "reserved for the future DWQ-train pipeline";
`--quant apex` (unqualified) → suggests `apex-balanced` etc.;
`--quant tq1_0`/`tq2_0` → "recognized ftype but out of v1 scope".

## Quick start: convert + serve a model

The `hf2q convert` pipeline reads a HuggingFace model directory
(config.json + safetensors + tokenizer.json) and emits a single GGUF
that loads in stock `llama.cpp` and in `hf2q serve`. The source can
be a path that already exists on disk OR a `--repo <hf_repo>` that
the driver auto-downloads via `huggingface-cli`.

At serve time, Qwen3.5/Qwen3.6 reads its tokenizer and chat template from
the GGUF metadata; a sibling `tokenizer.json` is not required. Vision uses a
separate projector GGUF. Compatible externally produced text/projector pairs
are accepted from standard architecture, multimodal-token, profile, width,
tensor, and forward-warmup checks. When exact source or artifact hashes are
present, hf2q additionally requires those identities to match.

When `serve --mmproj` successfully binds that projector to the loaded chat
model, `/v1/models` advertises `input_modalities: ["text", "image"]` and the
attached `vision_projector` on the chat-model row. The projector is not a
separate selectable language model. Stock OpenCode custom-provider entries
still need the equivalent local `modalities.input: ["text", "image"]`
declaration because its generic OpenAI-compatible provider does not infer
custom model capabilities from `/v1/models`.

```bash
# 1. Pre-download the HF source explicitly:
huggingface-cli download google/gemma-4-26b-a4b-it \
  --local-dir ./models/google-gemma-4-26b-a4b-it

# 2. Convert to Q5_K_M. Streaming convert keeps peak memory ~5 GB
#    even on a 48 GB-source 26 B-param model. ~8-15 min on M-series.
hf2q convert ./models/google-gemma-4-26b-a4b-it \
  --quant q5_k_m \
  -o ./out/gemma4-26b-q5_k_m.gguf

# Alternative: --repo auto-downloads via huggingface-cli into
# ~/.cache/hf2q/repos/google__gemma-4-26b-a4b-it/ and then converts.
# Mutually exclusive with the positional path form above.
hf2q convert --repo google/gemma-4-26b-a4b-it \
  --quant q5_k_m \
  -o ./out/gemma4-26b-q5_k_m.gguf

# 3a. Test load with stock llama.cpp (single-shot generation):
llama-cli -m ./out/gemma4-26b-q5_k_m.gguf \
  -p "What is the capital of France?" -n 64 --temp 0 --seed 42

# 3b. Serve with hf2q's OpenAI-compatible HTTP API:
hf2q serve --model ./out/gemma4-26b-q5_k_m.gguf --port 8080

# 4. Use it (OpenAI SDK works out of the box)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"hello"}]}'
```

### Full-context agentic serving

The native Gemma 4, Qwen 3.6/3.8, and DeepSeek-V4 workers are intended for
OpenAI-compatible coding clients such as OpenCode. Their canonical launchers
default to four independent agent slots. Every slot receives the complete
configured logical context; model weights are shared, while KV, recurrent
state, token ledgers, template state, and tool-call state remain isolated per
conversation. One shared physical KV budget governs demand-grown residency—it
never divides the advertised context by the slot count.

Start the launcher for the model family you want to serve:

```bash
# Gemma 4 (default port 8082)
./scripts/serve_gemma4_opencode.sh

# Qwen 3.6 (default port 8081)
./scripts/serve_qwen36_opencode.sh

# Qwen 3.8-27B multimodal (default port 8081)
./scripts/serve_qwen38_opencode.sh

# DeepSeek-V4 (default port 8081; run one large family at a time)
./scripts/serve_deepseek4_opencode.sh

# A different explicitly supported GGUF can be served without hf2q provenance:
MODEL=./out/DeepSeek-V4-Flash-0731.gguf PORT=8090 \
  ./scripts/serve_deepseek4_opencode.sh

curl http://127.0.0.1:8081/v1/models
```

For DeepSeek-V4 through OpenCode's generic OpenAI-compatible provider, use an
explicit agent profile instead of relying on client defaults. The locally
validated starting point is `temperature=0.55`, `top_p=0.95`, and the model's
`max` reasoning variant:

```json
{
  "agent": {
    "build": {"temperature": 0.55, "top_p": 0.95, "variant": "max"},
    "plan": {"temperature": 0.55, "top_p": 0.95, "variant": "max"}
  },
  "provider": {
    "local": {
      "models": {
        "Deepseek v4 Flash 0731 Source": {
          "reasoning": true,
          "interleaved": "reasoning_content",
          "variants": {"max": {"reasoningEffort": "max"}}
        }
      }
    }
  }
}
```

hf2q accepts `reasoning_effort` (`low`, `high`, or `max`) directly on a
DeepSeek chat-completion request and retains the older
`chat_template_kwargs.reasoning_effort` form for compatibility. A supplied
integer `seed` now drives a decode-step-indexed deterministic sampler; identical
rendered prompts and sampling settings reproduce across worker threads.

Create the Qwen3.8 artifact natively from its immutable source revision before
using that launcher:

```bash
hf2q convert --repo Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --quant q4_k_m \
  --output /opt/hf2q/models/qwen3.8/Qwen3.8-27B-Q4_K_M.gguf
```

Native text conversion and serving are accepted. Vision is a separate
candidate surface: hf2q converts the 333-tensor tower into a paired projector,
binds it to the text architecture and width, and runs the production image
graph before language-model admission. The exact candidate has passed unary
and SSE image chat, two-image ordering, an image-driven tool call and
tool-result continuation, same-image single-flight/cache reuse, disconnect
isolation, and the official 4,096 by 4,096 processor maximum. Vision is not yet
performance-accepted—the official-maximum cold path remains materially slower
than desired—so those results are candidate evidence, not a release-wide speed
claim. Speculative MTP decode remains disabled and outside the accepted server
surface.

Foreground launchers use the live operator dashboard automatically when
stderr is an interactive terminal. Runtime work stays in place instead of
forming a log wall: each request shows its slot and phase, cached/new prompt
tokens, prefill percentage and ETA, and decode rate. Use
`--operator-ui plain` for the traditional log stream, or
`--operator-ui dashboard` to require the dashboard and fail early when the
terminal cannot support it. Pipes, CI, services, and `--log-format json`
remain plain and machine-readable automatically.

Point the client's OpenAI-compatible base URL at the selected launcher's
`http://127.0.0.1:<port>/v1` endpoint and select the model ID returned by
`/v1/models` (normally the GGUF file stem). Set `MAX_SLOTS=1` for one agent or
`MAX_SLOTS=8` for an eight-slot hardware experiment; four is the
release-validated default. DeepSeek's `CONTEXT_LEN` override changes the full
logical context of each slot; Gemma and Qwen use the context declared by their
GGUF. `KV_CACHE_BUDGET_BYTES` independently caps aggregate physical KV
high-water. Requests that cannot safely fit wait or fail explicitly instead of
silently receiving a shorter context.

Use `/readyz`, not merely `/health` or `/v1/models`, as the generation
readiness probe. `/health` is process liveness. Present by public 0.1.5 and
strengthened in the 0.1.6 candidate, a fatal Metal
command-buffer/watchdog/ignored-submission error
(including device-loss reports), or an independently observed transaction
deadline, terminates every active and queued request for the affected Qwen,
Gemma, or DeepSeek worker once, rejects new work with HTTP 503, and keeps
`/readyz` unavailable. A slow SSE consumer is cancelled locally instead of
blocking other slots.
A supervisor must recreate the process/device generation; an in-process slot
reset is not safe recovery from a poisoned Metal queue.

Qwen3.5/Qwen3.6/Qwen3.8 SlotAware chat uses at most 2,048 prompt tokens per GPU
prefill transaction. Active decoders run before the next cold transaction,
multiple cold prompts rotate fairly, and cache/ledger state advances only
after the verifier full-attention cursors agree. The optional MTP cursor is
tracked independently until speculative decoding runs. For a bound Qwen3.8
projector, image requests carry soft-token embeddings, the explicit DeepStack
layout, and 3D positions through the same scheduler-yielding prefill state;
the payload is validated before its first GPU transaction. Image-bearing KV
remains exact-image isolated. A later first-image turn may reuse only the
causally earlier text-only snapshot when every soft/DeepStack position is in
the suffix and all four cached mRoPE axes are proven to be ordinary text.

The canonical Qwen launcher also bounds a still-open reasoning span at 2,048
tokens and continues decoding the answer. For smaller `max_tokens` values the
default adapts to retain answer capacity. Set `THINKING_TOKEN_BUDGET=0` to
disable this policy, or send the vLLM-compatible `thinking_token_budget` field
per request. Qwen3-VL remains a distinct model family rather than an
approximate fallback through another Qwen text family.

Long Gemma 4 text prefills use 4,096-token transactions and split at
the stable-prefix boundary. Decode runs before each `Mixed` prefill step, and
all configured HB, hybrid, dense, and MLX per-slot cursors are committed only
after the complete transaction succeeds. Cross-slot cold and retained-prefix
batches share the same 4,096-row aggregate Metal-transaction ceiling; lanes
over that bound remain FIFO and return to scheduler-backed resumable states.
When several compatible long-text states are installed, one transaction
shares the 4,096 rows across those lanes instead of multiplying the bound by
the number of slots. The 4,096-token ceiling is present by public 0.1.5 and
must pass exact fresh-versus-reused bounded real-model parity again from the packed
0.1.6 candidate before release; it is not inherited from Qwen. Long Gemma soft-token
prefill remains fail-closed until it has a resumable graph.

On the target M5 Max host, the launcher defaults to the schema-v2,
source-bound `deepseek4-agentic-q2` reproduction that passed the strict
coherence, throughput, tool-use, and long-prefix cache gates. It enables
operator progress telemetry and rejects unsafe port or memory state before
mapping the approximately 100 GiB model. When a competing process exceeds the
8 GiB RSS ceiling, the macOS launcher refines that value with physical
`footprint` so reclaimable WebKit/IOAccelerator mappings do not create a false
positive. If the probe is unavailable or malformed, the original RSS upper
bound remains authoritative and the launcher still fails closed.

Unary and streaming chat completions support reasoning content, OpenAI tools,
required/automatic tool choice, parallel DSML invokes, cancellation, and usage
telemetry. Growing transcripts reuse the live native KV/recurrent prefix;
DeepSeek's old-reasoning canonicalization restores a prompt-tail checkpoint, so
normal agent turns do not prefill the full context again. DeepSeek serving is
slot-aware and uses bounded admission/decode waves so several agents make
progress without duplicating model weights. Embeddings and multimodal messages
remain unsupported for DeepSeek and fail explicitly rather than selecting
another family or runtime.

DeepSeek production prefill now uses the exact gathered-attention path for
every nonempty prompt and retained-prefix suffix. The older dense kernel is a
diagnostic oracle only: it produced a different, incoherent first-token
ordering on stock-client structured-tool prompts and did not provide a speed
advantage. Native template rendering emits the model's canonical JSON spacing,
and the DSML grammar accepts both that spaced form and compact JSON while still
rejecting `null` for required strings. This covers nested `question` and
`todowrite` payloads in required and automatic tool-choice modes, including
recovery after prior invalid calls, SSE, and tool-result continuation.

The canonical launcher does not inject a repetition penalty when the client
omits one (`HF2Q_DEFAULT_REPETITION_PENALTY=1.0`). The former hidden `1.05`
default distorted constrained strings and did not stop client-side action
loops. Set `REP_PENALTY` only for a measured workload; an explicit request
value still wins.

With an already-running DeepSeek server, run the focused real-model gate:

```bash
BASE_URL=http://127.0.0.1:8081 \
  ./scripts/test_deepseek4_structured_tools.sh
```

The gate defaults to three temperature-0.55 repetitions per required and
automatic `question`/`todowrite` case, validates meaningful non-null nested
strings, exercises two-prior-failure recovery, checks SSE and continuation,
and requires retained-prefix cache use. An isolated stock OpenCode 1.18.18
coding gate additionally completed a Rust edit and same-session continuation
without a repeated tool loop. These are candidate-source and local-artifact
proofs; publication still requires the protected exact-package hardware gate.

DeepSeek cold and meaningful retained-prefix suffix work advances at native
atomic verifier boundaries. At most two cold prefills own the single scratch
arena concurrently. In a lopsided cohort with a runnable decoder, mixed work
caps the next prefill slice at two 128-token native windows and runs up to the
normal eight-token decode quantum before the next slice. When a filling cold
cohort still has another cold request queued, unary cold-wave decode is
deferred through the draining phase while any cold prefill remains because its
output cannot be delivered before the cohort barrier; full 2,048-token prefill
transactions resume. Streaming and warm decode remain responsive. Once no
runnable visible decoder remains, prefill also returns to the full transaction.
If a decoder becomes terminal, completion stays parked until the barrier lifts
so its physical cache cannot be reused before a tool-result continuation.
Cached-suffix work is not counted as cold-cohort work. With no cold barrier
active, staggered warm work may join an existing decoder whenever another
physical slot is free. Cancelling a
cached suffix rolls back to a valid, position-consistent pre-request turn
anchor; poisoned or inconsistent state still resets fully.

After prefill drains, pure decode advances in 64-token slot quanta to amortize
session swaps and scheduler publication across a full cohort. This does not
widen genuinely mixed work: a runnable decoder beside prefill remains clamped
to the eight-token/two-window interactive budget above.

Large DeepSeek MoE prefills also pair the routed expert gate and up
projections through the family-neutral `mlx-native 0.10.8` schedule primitive.
That primitive constructs the expert routing schedule once, then encodes the
two existing quantized projections; it is not a new approximate arithmetic
kernel. Decode-sized work, forced matvec/slotted diagnostics, calls without
scratch, and threshold-override diagnostics retain the independent projection
path. This is a candidate prefill optimization until the exact packed hf2q
hardware gates below prove end-to-end quality and latency; the native
primitive's focused benchmark is not a substitute.

The calibrated DeepSeek release envelope measures macOS thermal state through
the four atomic cold receipts, which is the phase that exercises large
prefill. It does not pause or reorder the agents: cached requests may still
overlap the cold tail exactly as in the frozen workload. The same live server
then completes cached unary/SSE, automatic tool choice, and tool-result
continuation under their unchanged latency and semantic limits. Receipt names
and hashes bind the thermal boundary; any non-Nominal sample before all four
cold receipts still fails closed.

`scripts/test_deepseek4_cached_suffix.sh` is the focused Apple-Silicon gate for
that contract. It overlaps a three-transaction cached tool-result suffix with
a live SSE decoder, then disconnects a separate cached suffix at transaction
three and requires bounded stop, one cancellation count, no terminal Done,
post-cancellation prefix reuse, readiness, and a clean fatal-log delta. Its
focused receipt complements rather than replaces the unchanged four-agent
agentic gate.

`scripts/run_deepseek4_matched_peer.sh` is the developer-only same-input
llama.cpp discriminator for the frozen four-agent cold workload. It starts a
fresh pinned peer for each wave, disables prompt caching, binds binary/model/
fixture/request identity, requires exact `read_file` semantics and zero-cache
usage, and records monotonic response/cohort timing under continuous AC and
thermal telemetry. The peer renders the byte-identical request as 6,695 prompt
tokens versus hf2q's 6,685, so both runtime-specific counts remain explicit.
The script is reference evidence only; it never participates in production
serving and cannot replace hf2q's exact packed-artifact cache gate.

The Qwen watchdog acceptance scripts are reproducible operator gates, not
startup defaults. Existing receipts are causal local dependency-spike evidence;
they are not final hf2q artifact authority. Release requires rerunning the same
gates from a clean hf2q package resolving published `mlx-native 0.10.8`:

- `scripts/test_qwen36_prefill_watchdog.sh` enqueues the deterministic
  552-token SSE lane immediately before the public 87,972-token/347-tool lane,
  requires decode-first progress, and validates the exact 44-transaction
  stable-boundary plan plus the complete tool/SSE response.
- `scripts/test_qwen36_prefill_cancellation.sh` runs with `MAX_SLOTS=1`, drops
  the long stream at a transaction boundary, and proves exact slot reuse.
- `scripts/test_qwen36_watchdog_harness_contract.sh` is the model-free negative
  test for the receipt parser.
- `scripts/test_deepseek4_interactive_overlap.sh` pairs a short decoder with
  the public 347-tool cold prompt, requires an eight-token interactive quantum
  before a legacy 2,048-token turn can monopolize the worker, and validates
  the complete long tool/SSE result under an uninterrupted AC-power window.
- `scripts/test_agentic_cache_lifecycle.sh` is the unchanged cross-family
  cache gate. Against a fresh Qwen, Gemma, or DeepSeek process it creates a
  long tool conversation, queues an exact retry while the strongest retained
  prefix is active, cancels the owner, requires the retry to reuse
  the restored checkpoint, and checks that an unrelated conversation cannot
  inherit private history. Run it once per process; never co-reside the large
  family artifacts on a 128 GiB host.
- `scripts/run_agentic_cache_release_gate.sh` is the release wrapper used by
  the manual `Cache lifecycle` workflow. It packages the exact main commit,
  runs DeepSeek, Gemma, and Qwen sequentially under continuous AC and
  `caffeinate` guards, verifies each GGUF against a protected SHA-256, and
  emits a source/crate/binary/model-bound manifest that the publication
  workflow must download and verify. Its calibrated four-slot Gemma waves run
  before the long overlap/lifecycle soak, retain the default latency limits,
  and bind continuous Nominal thermal telemetry across every cold, cached,
  automatic-tool, and tool-result turn.

The governing decisions and the old-failure-versus-final-artifact distinction
are recorded in `docs/ADR-019-mlx-native-encoder-architecture.md`,
`docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md`, and
`docs/ADR-040-continuous-batching-reopen.md`.

#### Test the 0.1.6 serving release

Build and verify the exact checkout before loading a model:

```bash
cargo check --locked --all-targets --all-features
cargo build --release --locked
cargo audit

# These are the focused serving contracts. CI also runs the library,
# conversion, LCP, fixture, readiness, and parser-negative suites listed in
# .github/workflows/ci.yml.
cargo test --locked --bin hf2q --all-features \
  qwen35_bounded_prefill_watchdog_tests -- --test-threads=1
cargo test --locked --bin hf2q --all-features \
  prompt_cache_ -- --test-threads=1
cargo test --locked --bin hf2q --all-features \
  gemma4_bounded_prefill_tests -- --test-threads=1
cargo test --locked --bin hf2q --all-features \
  engine_supervisor::tests -- --test-threads=1
cargo test --locked --bin hf2q --all-features deepseek4 -- \
  --skip real_artifact_tests
bash scripts/test_qwen36_watchdog_harness_contract.sh
```

Then start exactly one family from the same checkout. Setting `HF2Q_BIN`
prevents a launcher from accidentally selecting an older repository build:

```bash
# Choose one launcher and leave it in the foreground.
HF2Q_BIN="$PWD/target/release/hf2q" ./scripts/serve_qwen36_opencode.sh
HF2Q_BIN="$PWD/target/release/hf2q" MMPROJ=/nonexistent \
  ./scripts/serve_gemma4_opencode.sh
HF2Q_BIN="$PWD/target/release/hf2q" ./scripts/serve_deepseek4_opencode.sh
```

In another terminal, verify readiness and run the matching four-agent gate:

```bash
curl --fail http://127.0.0.1:8081/readyz
BASE_URL=http://127.0.0.1:8081 FAMILY=qwen36 AGENTS=4 \
  ./scripts/test_full_context_agent_slots.sh

curl --fail http://127.0.0.1:8082/readyz
BASE_URL=http://127.0.0.1:8082 FAMILY=gemma4 AGENTS=4 \
  ./scripts/test_full_context_agent_slots.sh

curl --fail http://127.0.0.1:8081/readyz
BASE_URL=http://127.0.0.1:8081 FAMILY=deepseek4 AGENTS=4 \
  ./scripts/test_full_context_agent_slots.sh
```

Run one model at a time. A battery-powered run is useful for functional
testing but is not performance authority; the release latency gates require
AC power, clear thermal status, and the exact artifact/power receipts described
in `docs/shipping-contract.md`.

For MoE models, pass an APEX tier instead of a standard ftype:

```bash
hf2q convert ./models/Qwen3.5-35B-A3B \
  --quant apex-balanced \
  -o ./out/qwen35-apex-balanced.gguf
```

The driver looks up the fingerprint manifest and, on match, logs
`[hf2q apex] auto-detected APEX config: vendor/apex-quant/configs/<file>`
before quantizing — confirming the exact per-tensor overlay in use.

### I-tier APEX (imatrix-aware quantization)

The `apex-i-*` tiers (`apex-i-quality`, `apex-i-balanced`,
`apex-i-compact`) require per-row activation-importance data
(imatrix). Two ways to supply it:

```bash
# A. In-tree: hf2q runs its own forward pass over a calibration corpus.
#    Stage 3.0 supports Gemma 4 only; other arches use option B.
hf2q convert ./models/google-gemma-4-26b-a4b-it \
  --quant apex-i-balanced \
  --imatrix-corpus cdv3 \
  -o ./out/gemma4-26b-apex-i-balanced.gguf

# B. Pre-computed: pass an external `.imatrix.gguf` (works for any
#    supported arch — Qwen 3.5/3.6 MoE included).
llama-imatrix -m ./out/qwen35-f16.gguf \
  -f data/calibration/cdv3.txt \
  -o /tmp/qwen35.imatrix.gguf
hf2q convert ./models/Qwen3.5-35B-A3B \
  --quant apex-i-balanced \
  --imatrix /tmp/qwen35.imatrix.gguf \
  -o ./out/qwen35-apex-i-balanced.gguf
```

The in-tree path (option A) writes a temporary F16 GGUF, drives
the forward pass over `cdv3` (bartowski's calibration corpus, baked
into the binary), and consumes the resulting per-tensor
sum-of-squared-activations to choose the per-layer mix. Wall time
is dominated by the forward pass: roughly seconds per 512-token
chunk × ~100 chunks on a 26B-A4B Gemma 4 model = operator-coffee-time,
not CI-time.

Optional flags:
- `--imatrix-out <path>` — write the computed (or loaded) imatrix
  to disk for reuse across multiple `--quant apex-i-*` runs against
  the same base model.
- `--imatrix-n-ctx <N>` — override the default 512-token chunk size
  (matches stock `llama-imatrix -c 512`). Larger `N` means fewer,
  longer chunks per forward-pass loop; useful when matching imatrices
  produced by stock `llama-imatrix -c <other>`. Must be `> 0`;
  passing `0` surfaces a typed `ConvertError::ImatrixNCtxInvalid`.

## Architecture

A full source-grounded architecture map lives in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). One-paragraph version:

```
   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
HF │ input/       │ -> │ models/<arch>/   │ -> │ backends/    │
   │ - safetensors│    │ - tensor rename  │    │ - gguf       │
   │ - config     │    │ - MoE merge      │    │ - safetensors│
   └──────────────┘    │ - DWQ targets    │    └──────────────┘
                       └──────────────────┘            │ GGUF
                                                       v
                                              ┌──────────────────┐
                                              │ inference/       │
                                              │ - load + warmup  │
                                              │ - forward (mlx)  │
                                              │ - KV cache (TQ)  │
                                              │ - spec-decode    │
                                              └──────────────────┘
                                                       │
                                              ┌──────────────────┐
                                              │ serve/           │
                                              │ - OpenAI HTTP    │
                                              │ - SSE streaming  │
                                              │ - block-prefix$  │
                                              │ - multi-model    │
                                              └──────────────────┘
```

## Historical performance snapshot

The following numbers are the matched 2026-05-17 M5 Max snapshot, not a claim
about every later commit or model artifact. Re-run the linked protocol for a
current purchasing or deployment decision; correctness and release gates do
not treat these historical medians as continuously verified.

Re-bench at the recorded HEAD on M5 Max against `llama.cpp` peer
(build `389ff61d7`, `-fa 1`) with identical GGUFs.  3-run median;
hf2q uses default config including the HF2Q_NO_FA hybrid-attn
fix from commit `03328ee5`.  See
[`docs/peer-parity-baselines-2026-04-26.md`](docs/peer-parity-baselines-2026-04-26.md)
for the full thermal-fair alt-pair protocol used by ADR-029 baselines.

- **Decode (Gemma-4 26B-A4B Q6_K)** — `tg200` **1.01× peer-FA AHEAD**
  (hf2q 105.2 t/s vs llama-bench 104.32 t/s); `tg2000` **0.97× peer-FA**
  (hf2q 93.5 t/s vs 96.69 t/s).  The historical ADR-029 iter-175
  `~1.05× AHEAD across tg200/tg2000/tg5000` claim was measured at a
  pre-HF2Q_NO_FA HEAD; re-bench at current main shows it holding at
  tg200 only.
- **Prefill (Gemma-4 26B)** — crossover regime: `pp1800` **0.96×
  peer-FA** (hf2q 2734 t/s vs llama-bench 2837 t/s); `pp3700`
  **1.24× peer-FA AHEAD** (hf2q 2703 t/s vs 2181 t/s).  hf2q's
  prefill rate drops ~1% from pp1800→pp3700 while llama's drops
  ~23% (FA tile-skip helps less at longer K), so the cross-over
  sits early in this range.  The historical `1.07-1.09× AHEAD`
  claim across the whole range no longer holds at current main.
- **Decode (Qwen 3.6 35B-A3B APEX-Q5_K_M)** — `tg200` **1.29× peer-FA
  AHEAD** (hf2q 130.6 t/s vs 101.31 t/s).  Historical ADR-028
  `~1.34×` measurement is within ~4% of current re-bench (thermal /
  build drift).
- **KV-cache footprint** — TurboQuant 8-bit (ADR-007 + ADR-027 iter-34)
  drops F32 K/V allocations entirely on Qwen 3.6 35B-A3B at 32K
  context, **340 MiB vs 1.34 GiB F32-only baseline = 3.94× memory
  savings**.  This is the only major performance claim with an
  in-tree regression pin (`tests/qh35_no_f32_kv_alloc_with_tq_kv.rs`).

Regression protection for the decode path: 8 parity tests
(V2/V3 unbatched + V3 batched), `coherence_smoke` (2 cells),
200-token byte-identity verification.  No automated bench-vs-peer
gate is currently in CI — these numbers are operator-driven
re-bench, not continuously verified.

Note: DWQ at the production-default `perturb=1.0` is mathematically
equivalent to the underlying K-quant baseline (ADR-020 finding
2026-05-08); DWQ wins materialize only at lower perturb values that
move the scales/biases off the K-quant projection.

Performance work is investigation-driven and tracked in numbered
ADR-029 (Gemma 4 decode), ADR-028 (peer-parity baseline), ADR-030
(speculative decode) iter-logs under `docs/`.

## Repository layout

```
src/
├── arch/          single source of truth for per-arch conformance
├── backends/      GGUF + mlx-lm safetensors writers
├── calibrate/     DWQ training, autograd, imatrix
├── inference/     per-arch forward graphs, spec-decode, vision
├── input/         HF config + safetensors loaders, HF Hub download
├── intelligence/  hardware probe, auto-quant heuristics, RuVector
├── ir/            internal tensor / metadata representation
├── models/        per-arch tensor rename + MoE merge
├── quality/       cosine / KL / perplexity scorers
├── quantize/      Q-format codecs (legacy / K-quant / DWQ / mixed)
└── serve/         OpenAI HTTP API, block-prefix KV cache, multi-model
docs/              architectural decisions + operator/runbook evidence
tests/             integration, parity, packaging, and regression gates
scripts/           launchers, benchmarks, incident repros, and runbooks
```

## Development

```bash
cargo build              # debug build
cargo test               # full test suite
cargo build --release    # release binary
cargo run -- doctor      # diagnostic
```

The project is TDD-heavy: every ADR closes only when its acceptance
tests + smoke prompts pass. New architectures must be onboarded via
the checklist in `docs/arch-onboarding.md` — registry entry + tensor
catalog + smoke prompt before any forward-pass code lands.

## Documentation index

- `docs/ARCHITECTURE.md` — source-grounded architecture map.
- `docs/converting-a-model.md` — generic convert reference.
- `docs/converting-qwen35.md` — Qwen 3.5/3.6 specifics.
- `docs/operating-kv-cache.md` — TurboQuant KV cache operator guide.
- `docs/operator-env-vars.md` — every `HF2Q_*` env var, what it gates.
- `docs/ADR-043-foreground-serve-dashboard.md` — live foreground serve UX,
  nonblocking telemetry, privacy, and terminal acceptance contract.
- `docs/shipping-contract.md` — default, supported, experimental, and
  investigation-only product surfaces.
- `docs/ADR-019-mlx-native-encoder-architecture.md` — Metal encoder ownership
  and pool-less worker lifetime contract.
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` — Qwen hybrid cache,
  bounded prefill, cancellation, and watchdog containment.
- `docs/ADR-040-continuous-batching-reopen.md` — full-context slot scheduling.
- `docs/ADR-*.md` — architectural decisions, rationale, failed spikes, and verification status.

## License

Dual-licensed under Apache-2.0 OR MIT (`Cargo.toml` `license` field;
`LICENSE-APACHE` and `LICENSE-MIT` files at repo root).  See
`docs/ADR-008-candle-divorce.md` for the dependency philosophy.
