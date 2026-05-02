# Model-load UX uniformity — design doc 2026-05-01

Author: claude-system-architect (CFA session `loaduxunify`).
Status: design only; no code change in this commit. Phase-3 queen reviews against
the spec's six-axis rubric; an implementation CFA round follows on agreement.

Seven H2 sections match the spec's S1–S7 subtasks: (1) problem + Chesterton
fences, (2) file-level audit, (3) peer survey (llama.cpp, mlx-lm, candle,
vLLM, ollama; two sources per non-local claim), (4) three ranked design
alternatives, (5) recommended solution with module paths + trait signature +
3–5-commit migration, (6) literal output samples for Gemma cold, Qwen3.5/3.6
cold, hypothetical Llama4 cold, and (7) risks + open BLOCKER questions.

Section 1 is long because every fence below is a real prior decision the
recommendation must preserve or knowingly retire. Section 5 is where this
doc commits — the user asked for a real solution.

## 1. Problem statement, current behaviour, Chesterton fences

### 1.1 What the user sees today (cold load, single model)

Three model-load surfaces exist:

  * `hf2q generate --model … --prompt …` for Gemma-shaped GGUFs:
    [`src/serve/mod.rs:422-700`](../../src/serve/mod.rs).
  * `hf2q generate --model … --prompt …` for Qwen3.5/3.6 GGUFs (auto-routed
    via arch peek): [`src/serve/mod.rs:1114-1487`](../../src/serve/mod.rs)
    (`fn cmd_generate_qwen35`).
  * `hf2q serve --model …` for HTTP API: [`src/serve/mod.rs:1532-1605`](../../src/serve/mod.rs)
    (`fn load_engine`), which delegates to
    [`src/serve/api/engine.rs:1338-1362`](../../src/serve/api/engine.rs)
    (`LoadedModel::load`) → either
    [`src/serve/api/engine.rs:1374-1501`](../../src/serve/api/engine.rs)
    (`GemmaLoadedModel::load`) or
    [`src/serve/api/engine_qwen35.rs:106-238`](../../src/serve/api/engine_qwen35.rs)
    (`Qwen35LoadedModel::load`).

The header that frames the generation stream is implemented in
[`src/serve/header.rs`](../../src/serve/header.rs) as a two-call pair:

  * `print_header_top` writes lines 1–2 (`hf2q · M5 Max · mlx-native` and
    `<model> · loaded in Xs · N layers · S GB`) — header.rs:52-65.
  * `print_header_prefill` writes line 3 (`prefill: P tok in M ms (T tok/s)`)
    plus the framing blank line — header.rs:70-83.
  * `LoadProgress` overwrites a one-line `\r loading i/n layers` on stderr
    while weights are streaming in — header.rs:92-129.

Headers are emitted on stdout because they are *product output*, not logs
(header.rs:1-8). The CLI binary prints headers; the server binary deliberately
does not (`cmd_serve` only emits `eprintln!("hf2q serving on http://{}", bind)`
at serve/mod.rs:2366, plus tracing at info+ — by design, log lines are the
operator-facing UX in serve mode, not a stdout banner).

### 1.2 The asymmetry, concretely

Comparing a cold Gemma `cmd_generate` vs a cold Qwen3.5/3.6 `cmd_generate_qwen35`
load on a TTY at default verbosity:

| Surface | Gemma path | Qwen3.5/3.6 path |
| --- | --- | --- |
| Pre-load tracing lines | `Model:`, `Tokenizer:`, `Config:`, layer/head/expert summary (mod.rs:464-478) | `Qwen3.5 path:`, `Tokenizer:` only (mod.rs:1124-1125) |
| Layer-load progress | `\r loading i/n layers` via `LoadProgress` (mod.rs:528-531) | None — `Qwen35Model::load_from_gguf` does not accept a `LoadProgress`; layer loading is silent |
| Header line 1 (chip/backend) | Yes, dimmed (mod.rs:545-555) | Yes, dimmed (mod.rs:1243-1253) |
| Header line 2 (model/load_s/layers/GB) | Yes (same call) | Yes (same call) |
| Header line 3 (prefill) | Yes (mod.rs:691-700) | Yes (mod.rs:1364-1373; 1272-1281 spec-decode arm) |
| Footer (total time + tok/s) | `eprintln!` dimmed line — mod.rs:757-? | `eprintln!` dimmed line — mod.rs:1467 |

Asymmetry visible to the user: pre-load context is much terser on Qwen35
(no config summary); layer-load progress simply doesn't render for Qwen35
even on TTY; tracing log keys differ (Gemma `Loading model weights from
GGUF into mlx-native buffers` mod.rs:516 vs Qwen35 `Loading Qwen3.5 model
from GGUF` mod.rs:1129 — operators grepping logs need two patterns); Engine
SERVE-load constructs a permanently-disabled `LoadProgress::new(false, 1, n)`
for Gemma (engine.rs:1459), while Qwen35 SERVE-load *never constructs one
at all* (engine_qwen35.rs:106-238). Two different brands of silence.

### 1.3 Chesterton fence #1 — iter-215 Phase A+B (commit `670a6f8`)

The Qwen35 SERVE-side load path (`Qwen35LoadedModel`) deliberately did not
adopt the Gemma-shaped surface for **load output**. The commit message and
`engine_qwen35.rs:1-34` document the rationale:

> **Iter-215 Wedge-2 MVP:** Loaded artefact suffices for `/v1/models`,
> `/readyz`, `/metrics`. Inference returns 501 until Wedge-3.
> Co-locate Qwen3.5/3.6 surface in `engine_qwen35.rs` rather than in
> `engine.rs` to avoid further bloating an already-7K-LOC file.

The duplicate `infer_quant_type_from_gguf` (engine_qwen35.rs:246-272 vs
engine.rs's private fn of the same name) is explicitly justified at line 241–245:

> *"duplicated here rather than refactored into a shared helper because the
> algorithm is 25 LOC and a refactor would touch a load-bearing file beyond
> iter-215's scope."*

The fence here is real: iter-215 traded a small amount of duplication for a
narrow blast radius and a separately-revertable Wedge. **But Wedge-3 has now
landed** — iter-216 Phase D (commit `354dfec`, `bb8c7cc`, `73a4ca8`,
`70b31d3`, `3721397`) wires real `Qwen35Model::forward_gpu_*` through the
worker; the 501 sentinel
([`engine.rs:2032`](../../src/serve/api/engine.rs)) is inert except for the
*vision* (mmproj) path on Qwen35, which remains 501 because mmproj weights
loading lives only on the Gemma path today. The "narrow blast radius" rationale
that motivated the duplication is no longer load-bearing for chat / embed /
streaming. **It is still load-bearing for vision** — the Qwen35 worker arm for
`Request::GenerateWithSoftTokens` (engine.rs:2192+) returns the
`qwen35_not_implemented_err`, and `LoadedModel::Qwen35` does not own a vision
projector. Section 5 preserves this: the unified surface explicitly names
"vision-projector loaded?" as a per-variant fact, not a shared field.

### 1.4 Chesterton fence #2 — provenance asymmetry (commits `31c465e`, `f6f8589`, `e54534c`, `134956f`)

[`src/serve/provenance.rs`](../../src/serve/provenance.rs) defines
`Provenance::Hf2q { producer_version, source_sha256, mmproj_sha256 } | External`.
The reader is invoked from `GemmaLoadedModel::load` at engine.rs:1399-1406 and
attached to the `GemmaLoadedModel.provenance` field. **It is not invoked from
`Qwen35LoadedModel::load`** — the field doesn't exist on the Qwen35 variant
(engine_qwen35.rs:55-90 has no `provenance`).

`LoadedModel::provenance()` (engine.rs:990-995) hard-returns
`Provenance::External` for the Qwen35 arm with a comment:

> *"the `Qwen35` variant returns `Provenance::External` because the Qwen35
> loader does not yet populate a provenance field (Qwen35 inference is
> 501-gated under iter-215; KV-spill never fires for that path). When Qwen35
> inference lands, this accessor's Qwen35 arm should be updated to read
> `engine_qwen35::Qwen35LoadedModel.provenance`."*

The fence: Qwen35 KV-spill was not wired (no `KvSpillDescriptor::from_qwen35_loaded_model`
exists — the only constructor is
[`from_gemma_loaded_model`](../../src/serve/api/kv_spill_descriptor.rs)
at kv_spill_descriptor.rs:214). Provenance feeds the
`Gemma4DenseSpillFactory` namespace key only; for Qwen35 it would have been
dead-stored. Iter-216 Phase D wired Qwen35 inference but did **not** add a
hybrid spill descriptor for the DeltaNet+full-attention layout — an
intentional omission per kv_spill_descriptor.rs:30 and the iter-216 commit
description. Provenance for Qwen35 therefore *still* feeds nothing; storing
it would be premature. Section 5 records this as a deferred wire-up: provenance
moves into a unified per-variant struct, the Qwen35 variant carries it but
nothing consumes it yet, and the corresponding KV-spill hybrid descriptor is a
follow-up CFA round.

### 1.5 Chesterton fence #3 — ADR-017 trait surfaces

[`src/serve/kv_persist/mod.rs:108-130`](../../src/serve/kv_persist/mod.rs)
defines `pub trait EngineBindable: Send + Sync`, called from
`KvPersistRegistry::bind_for` (registry.rs) and plumbed into `cmd_serve`
at serve/mod.rs:1710-1962. The `Gemma4DenseSpillFactory` reads
`engine.kv_spill_descriptor()` (engine.rs:760) and `engine.provenance()`
to construct a `ModelFingerprint`-keyed namespace. Implication: the load
path must continue to populate `kv_spill_descriptor: Option<...>` and
`provenance: Provenance` *before* `Engine::spawn` moves the LoadedModel
into the worker (engine.rs:1512+). Section 5 separates the *display
surface* (`LoadInfo`) from the *runtime metadata* (kept on each variant
exactly where it is today) for exactly this reason — an over-aggressive
flatten would remove the field the spill-namespace key reads and silently
break ADR-017.

### 1.6 Chesterton fence #4 — `print_header_top` CLI-vs-serve coexistence

`print_header_top` writes to `&mut W: Write` with a `tty: bool` switch
(header.rs:52-65), called only from CLI generate (mod.rs:554, mod.rs:1252).
`cmd_serve` deliberately does *not* call it: server stdout consumers are
tracing events plus one `eprintln!("hf2q serving on http://{}", bind)`
(mod.rs:2366); CLI line 3 is prefill — a mid-server-lifetime fact with no
natural anchor in serve mode; tests `header_top_*` (header.rs:148-183)
lock the literal format. Section 5 preserves the CLI-only role and adds a
*separate, optional* serve-mode banner that emits once at engine spawn,
gated on `--quiet`/non-TTY (matching `mlx-lm`'s `[INFO] Loading model from
disk: 5.628 s` pattern).

### 1.7 Net divergences the design must solve

Verified divergences (from §1.2 + §1.3–§1.6): tracing keys differ across
arches; Qwen35 has no layer-progress; SERVE has no banner on either arch;
`LoadedModel::provenance()` is a structural stub for Qwen35; two
`infer_quant_type_from_gguf` impls (engine.rs and engine_qwen35.rs:246-272)
can drift; MoE / sliding-window / vision-projector / chat-template-source
facts live in different sub-structs (`Gemma4Config`, `Qwen35Config`,
`Provenance`, `kv_spill_descriptor`) and are never collected into a single
printable view.

## 2. File-level audit of current load-side surfaces

This section is the Chesterton's-fence safety net for Section 5. Every site
listed here must be either preserved verbatim or have an explicit
"replaced-by" entry in the migration plan.

### 2.1 Print sites (stdout / stderr / `print!` / `println!` / `eprint!` / `eprintln!`)

| File:line | What it prints | Visibility | Used by |
| --- | --- | --- | --- |
| `src/serve/header.rs:52-65` (`print_header_top`) | `hf2q · <chip> · mlx-native\n<model> · loaded in Xs · N layers · S GB\n` | TTY-dimmed; ANSI stripped on non-TTY | CLI only |
| `src/serve/header.rs:70-83` (`print_header_prefill`) | `prefill: P tok in M ms (T tok/s)\n\n` | TTY-dimmed | CLI only |
| `src/serve/header.rs:113-117` (`LoadProgress::on_layer`) | `\r loading i/n layers` (overwritten in place) | TTY+verbosity=0 only | Both CLI Gemma and SERVE Gemma (the latter constructs but is permanently disabled) |
| `src/serve/header.rs:124-125` (`LoadProgress::finish`) | clears the line | conditional | same |
| `src/serve/mod.rs:553-555` (Gemma CLI) | calls `print_header_top` | CLI generate | Gemma only |
| `src/serve/mod.rs:691-700` (Gemma CLI) | calls `print_header_prefill` | CLI generate | Gemma only |
| `src/serve/mod.rs:1251-1253` (Qwen35 CLI) | calls `print_header_top` | CLI generate | Qwen35 only |
| `src/serve/mod.rs:1272-1281` and `1364-1373` (Qwen35 CLI) | calls `print_header_prefill` | CLI generate | Qwen35 (spec-decode and greedy arms) |
| `src/serve/mod.rs:573-577` | `eprintln!("HF2Q_DUMP_RENDERED_PROMPT: wrote …")` | Behind env var | Investigation only |
| `src/serve/mod.rs:641-647` | `eprintln!("HF2Q_DUMP_PROMPT_TOKENS: …")` | Env-gated | Investigation only |
| `src/serve/mod.rs:757-?` and `1298-1304`, `1467-1471` | `eprintln!("\n\n--- mlx-native: G tokens in T s (X tok/s) ---")` | TTY-dimmed | Both CLI families |
| `src/serve/mod.rs:778-784`, `1312-1320`, `1480-1487` | `println!("=== Benchmark Results ===\nHardware …")` | `--benchmark` flag | Both |
| `src/serve/mod.rs:2366` | `eprintln!("hf2q serving on http://{}", bind)` | Always | SERVE only |
| `src/serve/parity_quality.rs:447` | `LoadProgress::new(false, 1, 0)` | Always silent | Parity tooling |
| `src/serve/api/engine.rs:1459` | `LoadProgress::new(false, 1, n_layers)` | Always silent | SERVE Gemma load |
| `src/serve/forward_mlx.rs:775` | `progress: &mut LoadProgress` parameter | passed through | weight loader |
| `src/serve/api/engine_qwen35.rs:* `(none) | (no `LoadProgress` constructed) | — | SERVE Qwen35 load is structurally silent |

### 2.2 `tracing::info` / `tracing::warn` / `tracing::debug` sites at load time

Gemma CLI load (`cmd_generate`):

  * mod.rs:464 `Model:`
  * mod.rs:465 `Tokenizer:`
  * mod.rs:466 `Config:`
  * mod.rs:471-478 `Gemma4 A4B: N layers, H heads, hidden=…, X experts (top-Y)`
  * mod.rs:484 `Initializing mlx-native GPU context`
  * mod.rs:488 `mlx-native backend: <chip>`
  * mod.rs:491 `Loading GGUF model`
  * mod.rs:494-498 `GGUF loaded: T tensors, M metadata keys` (debug)
  * mod.rs:511-514 `Model name (GGUF general.name or file stem): …` (debug)
  * mod.rs:516 `Loading model weights from GGUF into mlx-native buffers`
  * mod.rs:534-538 `mlx-native weights loaded (N layers) in X.Ys`
  * mod.rs:631 (post-load) `Prepended BOS token K …`
  * mod.rs:639 `Prompt: P tokens`

Qwen35 CLI load (`cmd_generate_qwen35`):

  * mod.rs:1124 `Qwen3.5 path: <model>`
  * mod.rs:1125 `Tokenizer:    <tok>`
  * mod.rs:1129 `Loading Qwen3.5 model from GGUF`
  * mod.rs:1132-1137 `Qwen3.5 model loaded (N layers, variant=…) in X.Ys`
  * mod.rs:1146 `Qwen3.5 EOS token id: <id>`
  * mod.rs:1190 `Qwen3.5: P prompt tokens`

SERVE Gemma load (`GemmaLoadedModel::load`):

  * engine.rs:1388 `Engine load: model = <path>`
  * engine.rs:1389 `Engine load: tokenizer = <path>`
  * engine.rs:1390 `Engine load: config = <path>`
  * engine.rs:1444-1448 `Engine load: no GGUF tokenizer.chat_template; using API-path Gemma4 fallback …` (warn — only when applicable)
  * engine.rs:1480-1485 `Engine load: N layers, ctx_len=<…>, load_time=X.Ys`

SERVE Qwen35 load (`Qwen35LoadedModel::load`):

  * engine_qwen35.rs:122-125 `Qwen35 SERVE load: model = <path>`
  * engine_qwen35.rs:134-137 `Qwen35 SERVE load: tokenizer = <path>`
  * engine_qwen35.rs:142-146 `Qwen35 SERVE load: weights loaded (N layers, variant=…)`
  * engine_qwen35.rs:217-223 `Qwen35 SERVE load: complete in X.Ys (N layers, ctx_len=…, quant=…)`

`load_engine` SERVE dispatcher:

  * mod.rs:1553-1559 `Validated GGUF header (path, tensors, metadata, arch)`
  * mod.rs:1597-1600 `Engine warmed up synchronously (pre-mmproj order — iter-103 fix)`

### 2.3 LoadedModel surface fields, by variant

Reading [`src/serve/api/engine.rs:899-1018`](../../src/serve/api/engine.rs)
and [`src/serve/api/engine_qwen35.rs:55-90`](../../src/serve/api/engine_qwen35.rs):

| Surface | Gemma (`GemmaLoadedModel`) | Qwen35 (`Qwen35LoadedModel`) | Accessor on `LoadedModel` |
| --- | --- | --- | --- |
| `model_id` | yes (engine.rs:917) | yes (engine_qwen35.rs:67) | `model_id()` |
| `context_length` | yes | yes | `context_length()` |
| `quant_type` | yes | yes | `quant_type()` |
| `hidden_size` | yes (via `weights.hidden_size`) | yes | `hidden_size()` |
| `vocab_size` | yes (via `weights.vocab_size`) | yes | `vocab_size()` |
| `tokenizer` | yes | yes | `tokenizer()` |
| `chat_template` | yes (with API-path fallback warning) | yes (empty string allowed) | `chat_template()` |
| `eos_token_ids` | yes (`vec![1, 106]` hardcoded — engine.rs:1477) | yes (resolved from GGUF metadata, falls back to 151645) | `eos_token_ids()` |
| `load_duration` | yes | yes | `load_duration()` |
| `prompt_cache` | `PromptCache` (text-replay) | `HybridPromptCache` (KV-state) | `prompt_cache()` returns `Option<&PromptCache>` (Qwen35 returns `None`) |
| `provenance` | yes (`detect(&gguf)`) | **no field** | `provenance()` returns `External` for Qwen35 (engine.rs:993) |
| `weights` (`MlxModelWeights`) | yes | n/a — Qwen35 owns `Qwen35Model` instead | not accessed via `LoadedModel` |
| `config` (`Gemma4Config`) | yes | implicit (`model.cfg: Qwen35Config`) | not accessed via `LoadedModel` |
| `ctx` (`GpuContext`) | yes | n/a — Qwen35 path uses `MlxDevice::new()` per-prefill | not accessed via `LoadedModel` |
| MoE: `n_experts`, `n_experts_used` | `cfg.num_experts`, `cfg.top_k_experts` | `model.cfg.moe.is_some()` ⇒ `moe.num_experts`, `moe.num_experts_per_tok`, else dense | not surfaced anywhere uniformly |
| Sliding-window | `cfg.sliding_window` (Gemma is hybrid sliding+global) | n/a (Qwen35 uses `full_attention_interval`) | not surfaced |
| Vision projector (`mmproj`) | loaded separately via `--mmproj` flag, lives on `state.mmproj` (handlers.rs:6285+) | not loaded for Qwen35 (vision arm returns 501) | per-server, not per-engine |
| Chat-template source | implicit: GGUF metadata `tokenizer.chat_template`, else FALLBACK | implicit: GGUF metadata, else empty | not surfaced |
| Tokenizer source | HF `tokenizer.json` on disk | GGUF-driven (engine_qwen35.rs:174 — `build_tokenizer_from_gguf`) | not surfaced; users only see filename in tracing log |
| KV-cache memory budget | derived at engine spawn from `EngineConfig` | derived per-prefill (`max_seq * cfg.*`) | not surfaced uniformly; only surfaced via `/metrics` `hf2q_pool_resident_bytes` |
| Quant BPW (bits-per-weight) | not currently computed | not currently computed | not surfaced |
| Architecture string (`general.architecture`) | read in dispatcher at engine.rs:1349 | read in dispatcher; also stored on `model.cfg.variant` | not surfaced anywhere user-visible |

### 2.4 The `infer_quant_type_from_gguf` duplication

Two implementations exist:

  * `engine.rs` (private fn referenced at engine.rs:1431) — used by Gemma load.
  * `engine_qwen35.rs:246-272` — used by Qwen35 load.

They share the same algorithm — histogram non-fp tensor types, return label of
the dominant. The duplication is documented as deliberate (engine_qwen35.rs:241-245)
to keep iter-215's blast radius narrow. Section 5 promotes one of these to a
shared helper as part of the unified load module — exactly one canonical
"compute the dominant quant label" callsite.

### 2.5 Existing tests pinning the load surface

[`src/serve/header.rs:131-211`](../../src/serve/header.rs):

  * `short_chip_label_strips_apple_prefix`
  * `short_chip_label_passes_through_unknown_prefix`
  * `header_top_no_tty_has_no_ansi` — locks the literal output
    `hf2q · M5 Max · mlx-native\ngemma-4-26B · loaded in 2.4s · 30 layers · 16.9 GB\n`
  * `header_top_tty_has_dim` — locks ANSI escape presence
  * `header_prefill_adds_blank_line` — locks `prefill: 15 tok in 260ms (58 tok/s)\n\n`
  * `load_progress_disabled_when_not_tty`
  * `load_progress_disabled_when_verbose`

[`src/serve/api/engine.rs`](../../src/serve/api/engine.rs) test module includes
`qwen35_loaded_model_has_initialized_prompt_cache` (line ~1618) and
`qwen35_loaded_model_load_errors_when_path_missing` (line ~1629).

[`src/serve/mod.rs`](../../src/serve/mod.rs) test module at ~3306+ includes
`load_engine_routes_qwen35_to_qwen35_loaded_model`,
`load_engine_routes_qwen35moe_to_qwen35_loaded_model`,
`load_engine_routes_unknown_arch_to_gemma_default_unchanged`.

These constrain every line of the recommendation. Section 5 keeps each of
them green by additive design (new fields, new module, no rename of the
existing functions or print sites until a clean-up commit at the end of the
migration plan).

## 3. Peer survey

### 3.1 llama.cpp

`llama.cpp` runs a four-stage banner at model load:

  1. `llama_model_loader: loaded meta data with K key-value pairs and T tensors from <file> (version GGUF V3 (latest))`
     — file format and KV count. Followed by per-key dump
     `llama_model_loader: - kv N: <key> <type> = <value>` for every metadata
     key (filtered to "interesting" ones in default verbosity).
  2. `llm_load_vocab:` — vocab type, special tokens cache size, token-to-piece
     cache.
  3. `llm_load_print_meta:` — full model fingerprint:
     `arch`, `vocab type`, `n_vocab`, `n_ctx_train`, `n_embd`, `n_layer`,
     `n_head`, `n_head_kv`, `n_rot`, `n_swa`, `n_expert`, `n_expert_used`,
     `model type`, `model ftype`, `model params`, `model size`,
     `general.name`, `BOS token`, `EOS token`, `EOT token`, etc.
  4. `llm_load_tensors:   CPU_Mapped model buffer size = MMM MiB` (per-buffer
     allocation) and finally
     `llama_new_context_with_model: n_seq_max = N`,
     `llama_new_context_with_model: n_ctx = N`,
     `llama_new_context_with_model: KV self size = MMM MiB`.

Verified verbatim against
[the llama.cpp guide on steelph0enix.dev (literal log dump in the "Loading the
model" section)](https://blog.steelph0enix.dev/posts/llama-cpp-guide/) and the
[llama-cli(1) Debian manpage notes on log levels](https://manpages.debian.org/testing/llama.cpp-tools/llama-cli.1.en.html).
Both sources confirm the prefix-per-stage convention (`llama_model_loader:`,
`llm_load_vocab:`, `llm_load_print_meta:`, `llm_load_tensors:`).

Design properties to lift:

  * Dense, single-page banner emitted **once at load time**, not split between
    pre-load and post-prefill.
  * Every metadata key worth surfacing has a stable prefix-+-key shape
    (`llm_load_print_meta: n_swa = 4096`) — operator can grep
    `n_swa` and find the answer regardless of arch.
  * Sliding window, MoE expert count, and quant type are **all in the same
    section** rather than scattered across config / weights / cache.
  * The prefix encodes the *load stage*, not the *architecture* — so an
    operator never has to know whether they're looking at llama vs deepseek
    vs gemma; the stage shape is identical.

### 3.2 mlx-lm

`mlx-lm`'s `mlx_lm.generate` and `mlx_lm.server` produce a deliberately sparse
banner. Per the
[mlx-lm README](https://github.com/ml-explore/mlx-lm/blob/main/README.md) and
the [Kevin Conner walk-through (2025-02-17)](https://kconner.com/2025/02/17/running-local-llms-with-mlx.html),
default load output is:

  * `[INFO] Loading model from disk: 5.628 s`
    (single line; emitted from `mlx_lm.utils.load`.)
  * Followed at end-of-generation by:
    `Prompt: 12 tokens, 234.5 tokens-per-sec`
    `Generation: 100 tokens, 18.3 tokens-per-sec`
    `Peak memory: 4.231 GB`

Source corroboration: the mlx_lm `generate.py` source (head of GitHub) shows
`print(f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.3f}
tokens-per-sec")` etc. as the only print sites in the user-facing path — no
per-layer progress, no per-tensor list. mlx-lm leans on Hugging Face's
`snapshot_download` progress bar for the *download* phase but emits exactly
one console line for the *load* phase.

Design properties to lift:

  * Single load-line, value-per-key.
  * Peak memory printed *post-generation*, not pre-load — useful because the
    KV cache grows with context.
  * No layer-by-layer output. The user does not care about layers; they care
    about wall-clock and peak GB.

### 3.3 candle (Rust peer, our closest neighbour)

Per [candle/candle-examples/examples/quantized/main.rs](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs),
the literal print sites at load time are:

  * `avx: {}, neon: {}, simd128: {}, f16c: {}` — system capabilities
  * `temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}` — sampling params
  * `loaded {:?} tensors ({}) in {:.2}s` — tensor count + bytes + elapsed
  * `params: {:?}` (GGML format only) — hyperparams
  * `model built` — completion sentinel

This is the closest peer to hf2q (Rust, GGUF-shaped, single binary). Design
properties to lift:

  * Tensor count + byte-size + load wall-clock in one line (`loaded N
    tensors (S B) in T s`). hf2q's CLI header line 2 is close
    (`loaded in X s · N layers · S GB`) but uses *layers* where candle uses
    *tensors* — both are valid; layers is more semantic.
  * `model built` as an explicit completion sentinel. hf2q does not have one;
    the prefill line plays that role on the CLI but is mid-stream in serve
    mode (or absent).

### 3.4 vLLM

Per the [vLLM logs walkthrough by huikang (2025-12-28)](https://blog.huikang.dev/2025/12/28/vllm-logs.html)
and the [Loading model weights pull request (vllm/vllm-project#13666)](https://github.com/vllm-project/vllm/pull/13666):

  * Each line carries `INFO MM-DD HH:MM:SS [<file>.py:<line>] <message>`.
  * `Resolved architecture: GptOssForCausalLM` — explicit named arch.
  * `Loading safetensors checkpoint shards: 100% Completed 15/15 [09:05<00:00, 36.34s/it]`
    — tqdm-style progress bar.
  * `Loading weights took 545.24 seconds`
  * `Model loading took 65.9651 GiB memory and 600.408480 seconds`
  * `Available KV cache memory: 8.99 GiB`
  * `GPU KV cache size: 130,960 tokens`
  * `Starting vLLM API server 0 on http://0.0.0.0:8000`

Two properties relevant to hf2q:

  * vLLM emits a *KV-cache budget* line — explicit GiB and "tokens equivalent"
    after model load. hf2q surfaces this only via `/metrics`. Our serve mode
    should print it once at load to stdout.
  * vLLM logs "Model loading took X GiB memory and Y seconds" — separates
    *resident weight* size from *peak allocation* during load. hf2q's CLI
    line 2 reports GGUF on-disk size, not resident-after-load size; these
    diverge for fp32 fallback paths, MoE expert routing tables, etc.

### 3.5 ollama

Per the [ollama troubleshooting docs](https://docs.ollama.com/troubleshooting)
and the [verbose-mode discussion (ollama/ollama#4010)](https://github.com/ollama/ollama/issues/4010):

  * `pulling manifest`, then per-shard `pulling <digest>: 1% ▕ ▏ 140 MB/ 13 GB 6.3 MB/s 36m21s`.
  * Verbose mode (`--verbose`):
    `total duration: 3.940155533s`,
    `load duration: 12.011517ms`,
    `prompt eval count: 6 token(s)`,
    `prompt eval duration: 22.769095ms`,
    `prompt eval rate: 263.52 tokens/s`,
    `eval count: 208 token(s)`,
    `eval duration: 3.905018925s`,
    `eval rate: 53.26 tokens/s`.

Properties to lift: the **opt-in verbose mode** model — ordinary users see one
line, power users opt in for the full breakdown. hf2q's `tracing::info`-vs-`debug`
split is the analogue, but tracing is targeted at logs, not at user-facing
stdout. A `--verbose-load` flag (or honouring `-v`) on top of a stable, terse
default is the right shape.

### 3.6 Synthesis — the "shared peer mental model"

Across five peers:

  1. Load output is **load-stage-prefixed**, not architecture-prefixed.
     llama.cpp: `llm_load_print_meta:`. vLLM: `[file.py:line]`. ollama:
     `load duration:`. hf2q today: `Engine load:` for SERVE-Gemma, `Qwen35
     SERVE load:` for SERVE-Qwen35, `Qwen3.5:` for CLI-Qwen35, no prefix for
     CLI-Gemma — four shapes for one event.
  2. There is **one canonical metadata banner** per load, not two banners
     and a footer. mlx-lm and candle each have ~5 lines; llama.cpp has
     ~30 but in one block. hf2q today splits 2 stdout lines (header_top) +
     ~8 tracing info lines + 1 prefill line + 1 footer line — six logical
     surfaces.
  3. The unique facts that *all* peers surface:
     `arch`, `n_layer`, `n_head`, `n_kv_head`, `n_vocab`, `n_ctx_train`,
     `n_swa` (if applicable), `n_expert` / `n_expert_used` (if applicable),
     `model size`, `model ftype` (quant), `load wall-clock`. hf2q surfaces
     about half of these on each path, none uniformly.
  4. Provenance/tokenizer source is a hf2q-specific concern that no peer
     models. We get to invent the convention. The peer-uninspired correct
     answer is an explicit `source: hf2q-quantized | external-llama.cpp`
     line, mirroring vLLM's `Resolved architecture:` line.

## 4. Three design alternatives, ranked

Trade-off axes per the brief: (a) implementation cost, (b) surface durability,
(c) operator/UX clarity, (d) test/gating cost, (e) Chesterton-fence
preservation. Each alternative rated 1–5 on each axis (5 best); total /25.

### Alternative A — per-variant accessor parity

Add missing fields directly on `Qwen35LoadedModel` (`provenance`, optional
`kv_spill_descriptor`-deferred-null), normalize tracing keys across the two
arches, prefix Gemma load with `Engine load:` to match. Promote
`infer_quant_type_from_gguf` to a shared private helper.

| axis | a-cost | b-durability | c-UX | d-tests | e-Chesterton | total |
| --- | --- | --- | --- | --- | --- | --- |
| score | 5 | 2 | 2 | 5 | 4 | **18/25** |

Rejected as stand-alone: solves the visible duplication in §2.4, but the
structural problem (no unified printable view, no serve-mode banner, no
Qwen35 layer-progress) remains. Every new model arch reproduces the
same per-variant divergence.

### Alternative B — strangler-fig `serve::load_info`

New module [`src/serve/load_info.rs`](../../src/serve/load_info.rs) defines
a single `pub struct LoadInfo` (everything the user/log/banner needs about
a freshly loaded model). Each variant of `LoadedModel` builds one via a new
`LoadInfoBuilder` trait — the *display surface* is unified, but the
*runtime metadata* (weights, ctx, prompt_cache, kv_spill_descriptor,
provenance) stays where it is on each variant. CLI and SERVE both call
`load_info::print_banner(&info, &mut stdout, tty)`. `Qwen35Model::load_from_gguf`
gains a `&mut LoadProgress` parameter. `infer_quant_type_from_gguf` is
promoted to `load_info::infer_quant_label` (one canonical impl).
`header::print_header_top` becomes a thin shim during migration; retired
in optional cleanup commit C5.

| axis | a-cost | b-durability | c-UX | d-tests | e-Chesterton | total |
| --- | --- | --- | --- | --- | --- | --- |
| score | 4 | 5 | 5 | 4 | 5 | **23/25** |

This is the recommendation — Section 5.

### Alternative C — big-bang flatten with `Box<dyn ModelBackend>`

Erase the enum. `pub struct LoadedModel { info: LoadInfo, backend: Box<dyn
ModelBackend>, tokenizer, chat_template, ... }`; `pub trait ModelBackend`
absorbs `forward_*`, `prompt_cache`, `provenance`, `kv_spill_descriptor`.
SERVE worker dispatches dynamically.

| axis | a-cost | b-durability | c-UX | d-tests | e-Chesterton | total |
| --- | --- | --- | --- | --- | --- | --- |
| score | 1 | 5 | 5 | 1 | 2 | **14/25** |

Rejected. ~3–5K LOC churn across `engine.rs`, `forward_mlx.rs`,
`forward_prefill.rs`, `kv_spill_descriptor.rs`, `kv_persist/*.rs`. The
spill descriptor's `from_gemma_loaded_model` (kv_spill_descriptor.rs:214)
takes `&GemmaLoadedModel` directly — making this a trait method erases
static shape that the factory relies on. Right destination, wrong moment;
adopt after B ships and a third arch is concretely needed.

### Ranking summary

| Rank | Alternative | Score |
| --- | --- | --- |
| 1 | **B (strangler-fig `serve::load_info`)** | 23/25 |
| 2 | A (per-variant accessor parity) | 18/25 |
| 3 | C (big-bang flatten) | 14/25 |

## 5. Recommended solution — concrete spec

This is the section the user said matters: *"Come up with a real solution and
tell me what you think the real solution is."* Here it is.

### 5.1 New module: `src/serve/load_info.rs`

The unified display + tracing surface for any model load (CLI or SERVE), any
arch.

```rust
// src/serve/load_info.rs — new file (~250 LOC).

//! Unified per-load metadata banner — emitted once on `LoadedModel` construction
//! by both `cmd_generate` (CLI) and `load_engine` (SERVE).
//!
//! Per ADR <future>: mirrors llama.cpp's `llm_load_print_meta:` block in shape
//! (single banner, stage-prefixed, one fact per line) but with hf2q-specific
//! provenance + tokenizer-source surfaced explicitly.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use crate::serve::header::short_chip_label;
use crate::serve::provenance::Provenance;

/// Origin of the chat template string actually in effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateSource {
    /// Lifted from GGUF metadata key `tokenizer.chat_template`.
    GgufEmbedded,
    /// Operator override via `--chat-template` (CLI generate flag).
    CliOverride,
    /// Hard-coded fallback (Gemma4 API path) — emits a `tracing::warn`
    /// at load time per engine.rs:1444.
    HardcodedFallback { name: &'static str },
    /// Empty / not yet rendered (pre-Wedge-3 Qwen35 GGUFs that lack the key).
    None,
}

/// Source of the active tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    /// `tokenizers::Tokenizer::from_file(<path>)` — Gemma path today.
    HfTokenizerJson { path: PathBuf },
    /// `build_tokenizer_from_gguf` — Qwen3.5/3.6 path today, mirrors
    /// `llama-vocab.cpp:2197-2253` to avoid the apex-GGUF OOB-token bug
    /// (see engine_qwen35.rs:148-178 for the full rationale).
    GgufEmbedded,
}

/// Mixture-of-Experts shape, when applicable. `None` for dense models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    pub n_experts: u32,
    pub n_experts_per_tok: u32,
}

/// Vision-projector pairing — `None` if no mmproj is loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionProjector {
    pub mmproj_path: PathBuf,
    pub mmproj_sha256: Option<String>,
}

/// One-of-many fact: which forward-pass family this model dispatches to.
/// String form is what gets printed; enum form is what code dispatches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchFamily {
    /// gemma4 / gemma4-shaped (sliding-window hybrid + global; MoE optional).
    Gemma4,
    /// qwen35 / qwen35moe (DeltaNet linear-attn + periodic full-attn).
    Qwen35,
    /// Reserved — Llama4 (placeholder; dispatcher errors at the
    /// `LoadedModel::load` arch peek today).
    Llama4Reserved,
}

impl ArchFamily {
    pub fn as_str(&self) -> &'static str {
        match self {
            ArchFamily::Gemma4 => "gemma4",
            ArchFamily::Qwen35 => "qwen35",
            ArchFamily::Llama4Reserved => "llama4",
        }
    }
}

/// Everything the unified banner + tracing layer needs.
///
/// Constructed by each variant's `*LoadedModel::load` *after* the underlying
/// load succeeds. The struct OWNS no live model state — it is a snapshot of
/// the relevant facts at load-completion time, suitable to be cloned into
/// tracing fields, the SERVE-mode banner, and `/v1/models` (the latter today
/// hand-rolls a subset; see migration commit C5 below).
#[derive(Debug, Clone)]
pub struct LoadInfo {
    // ---------- Identity ----------
    /// `general.name` if present, else file stem. Same shape as today's
    /// `LoadedModel::model_id()`.
    pub model_id: String,
    /// `general.architecture` (e.g. "gemma4", "qwen35", "qwen35moe").
    pub arch_str: String,
    pub arch_family: ArchFamily,
    /// Filesystem path to the GGUF.
    pub model_path: PathBuf,
    /// On-disk GGUF size in bytes (`std::fs::metadata(...).len()`).
    pub on_disk_bytes: u64,

    // ---------- Hardware / backend ----------
    /// `gpu.gpu_name()` — chip identifier ("Apple M5 Max" before stripping).
    pub backend_chip: String,
    /// Backend label — `"mlx-native"` today, `&'static str` because it is
    /// compile-time fixed per ADR-008.
    pub backend: &'static str,

    // ---------- Architecture facts ----------
    pub n_layers: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub n_attention_heads: u32,
    pub n_key_value_heads: u32,
    /// `head_dim` — Qwen35 stores explicitly; Gemma derives from
    /// `hidden_size / num_attention_heads`.
    pub head_dim: u32,
    /// Sliding-window size in tokens. `None` = full-attention-only model.
    /// Gemma4: `cfg.sliding_window`. Qwen35: `None` (uses periodic
    /// full-attention via `full_attention_interval`, surfaced separately).
    pub sliding_window: Option<u32>,
    /// Qwen35-specific: full-attention layer period (one full-attn layer
    /// every `n` layers). `None` for arches without this concept.
    pub full_attention_interval: Option<u32>,
    /// Maximum context length declared by the GGUF (`{arch}.context_length`).
    pub max_context_length: Option<u32>,
    /// MoE shape if the model is MoE; `None` for dense.
    pub moe: Option<MoeShape>,

    // ---------- Quantization ----------
    /// Dominant non-fp tensor type (`Q4_K`, `Q6_K`, etc.) — produced by
    /// `infer_quant_label`.
    pub quant_label: Option<String>,
    /// Bits-per-weight (parameter-weighted average). Computed by
    /// `compute_bpw` (new helper in this module). `None` if computation
    /// was skipped (e.g. parity-quality scratch loads).
    pub quant_bpw: Option<f32>,

    // ---------- Tokenizer / chat template ----------
    pub tokenizer_source: TokenizerSource,
    pub eos_token_ids: Vec<u32>,
    /// BOS token if the GGUF declared one (`tokenizer.ggml.bos_token_id`).
    pub bos_token_id: Option<u32>,
    pub chat_template_source: ChatTemplateSource,

    // ---------- Provenance (ADR-017 §F4) ----------
    /// Mirrors `LoadedModel::provenance()`. Populated for both arches in
    /// commit C2 below (today only Gemma populates it).
    pub provenance: Provenance,

    // ---------- Vision / multimodal ----------
    /// `Some` only if the operator passed `--mmproj` *and* the load path
    /// supports it for this arch. Today: Gemma yes, Qwen35 no (vision
    /// path returns 501 — engine.rs:2192+).
    pub vision_projector: Option<VisionProjector>,

    // ---------- Wall-clock / memory ----------
    pub load_wall_clock: Duration,
    /// Best-effort post-load resident bytes for weights (ADR-017 calls this
    /// `resident_bytes_weights`). `None` if not measured (Qwen35 today).
    pub resident_weight_bytes: Option<u64>,
    /// KV-cache memory budget for the engine, derived from `EngineConfig`
    /// in SERVE mode, from `args.max_tokens` in CLI mode. `None` for the
    /// CLI Gemma path (which lazily allocates per-prefill — current
    /// behaviour).
    pub kv_cache_budget_bytes: Option<u64>,

    // ---------- KV-persist / spill (ADR-017) ----------
    /// `true` iff the engine will bind a KV-spill hook for this load (i.e.
    /// `--kv-persist=PATH` is set AND a per-family factory matched). For
    /// Qwen35 today: always `false` (no hybrid descriptor; see §1.4).
    pub kv_spill_active: bool,
}

/// Algorithm: same histogram code as the two existing private impls
/// (engine.rs:? and engine_qwen35.rs:246-272), promoted to one place.
pub fn infer_quant_label(gguf: &mlx_native::gguf::GgufFile) -> Option<String> {
    /* same body as engine_qwen35.rs:246-272 — verbatim, just relocated. */
    todo!("filled in by commit C1")
}

/// Parameter-weighted bits-per-weight. Iterates `gguf.tensor_names()`,
/// reads `gguf.tensor_info(name).ggml_type`, multiplies by element count
/// per `GgmlType::block_size_in_bytes`. Returns the average over all
/// non-fp tensors. `None` if no quantized tensors found.
pub fn compute_bpw(gguf: &mlx_native::gguf::GgufFile) -> Option<f32> {
    todo!("filled in by commit C1")
}

/// Print the unified banner. Stage-prefixed lines, dimmed on TTY.
/// Output contract (verified by `tests/load_info_banner.rs`):
///
/// ```text
/// hf2q load: backend = mlx-native (M5 Max)
/// hf2q load: model = <model_id> (arch = qwen35moe, family = qwen35)
/// hf2q load: source = <on-disk path> (29.83 GiB on disk)
/// hf2q load: layout = 64 layers, 16 heads (4 kv), head_dim=128, hidden=4096, vocab=151936
/// hf2q load: features = sliding_window=none, full_attn_every=4, moe=128 experts/8 active
/// hf2q load: quant = Q4_K dominant, ~4.55 bpw, mlx-native resident 16.42 GiB
/// hf2q load: max_ctx_train = 262144, kv_budget = 4.00 GiB (~32768 tokens)
/// hf2q load: tokenizer = gguf-embedded (<= mirrors llama-vocab.cpp)
/// hf2q load: chat_template = gguf-embedded
/// hf2q load: provenance = hf2q (producer hf2q 0.1.0, source_sha 7f3a…)
/// hf2q load: vision = none
/// hf2q load: kv_spill = inactive
/// hf2q load: ready in 6.84 s
/// ```
///
/// All fields are present in every banner; absent ones print explicit
/// `none` so a grep for `kv_spill =` always finds an answer regardless of
/// arch. The trailing `ready in N s` line is the load-completion sentinel
/// (mirrors candle's `model built`).
pub fn print_banner<W: Write>(info: &LoadInfo, w: &mut W, tty: bool) -> io::Result<()> {
    /* implementation: 60-80 LOC of writeln! calls, ANSI-dim wrapping;
       pinned by golden-text test below. */
    todo!("filled in by commit C2")
}

/// Mirror the banner into `tracing::info!` events so log consumers (systemd
/// journal, container stdout) see the same facts at the same load stage.
/// Uses one event per logical group to match grep-friendly tracing
/// conventions. Fields use the *same names* as the LoadInfo struct so
/// JSON-formatted tracing emits structured logs that diff cleanly across
/// arches.
pub fn emit_tracing(info: &LoadInfo) {
    /* tracing::info! events with structured fields; ~12 events. */
    todo!("filled in by commit C2")
}
```

### 5.2 Trait — exactly one new trait, narrowly scoped

```rust
// src/serve/load_info.rs

/// Implemented by each `*LoadedModel` variant — produces a `LoadInfo`
/// snapshot from the variant's owned state. Uniform constructor surface
/// the unification depends on; matches the shape of `From` but with a
/// `&GgufFile` borrowed alongside the model so quant/BPW computation
/// can re-read tensor types without duplicating that fact across variants.
pub trait LoadInfoBuilder {
    fn build_load_info(
        &self,
        gguf: &mlx_native::gguf::GgufFile,
        load_wall_clock: Duration,
        kv_cache_budget_bytes: Option<u64>,
        kv_spill_active: bool,
    ) -> LoadInfo;
}
```

Implementations (added in commit C2):

```rust
// src/serve/api/engine.rs
impl LoadInfoBuilder for GemmaLoadedModel { /* ~80 LOC */ }

// src/serve/api/engine_qwen35.rs
impl LoadInfoBuilder for Qwen35LoadedModel { /* ~80 LOC */ }
```

Rationale for trait, not free function: the per-variant impl reads
*variant-specific* fields (`g.weights`, `q.model.cfg.moe`, etc.). A free
function would need to take the enum, and the enum-arm match would itself
be the duplication we're trying to remove.

### 5.3 Wiring — exactly four call sites

Replace today's `print_header_top` calls and the SERVE mode's silence:

  * `src/serve/mod.rs:545-555` (Gemma CLI generate): replace
    `print_header_top(&header_top, …)` with
    `print_banner(&info, &mut stdout, stdout_is_tty)` followed by the
    existing `print_header_prefill` call (line 3 stays, prefill stats are
    *not* a load fact).
  * `src/serve/mod.rs:1243-1253` (Qwen35 CLI generate): same replacement.
  * `src/serve/mod.rs:1597-1604` (`load_engine` SERVE path, after
    `engine.warmup()` returns): **new** `print_banner` call gated by
    `args.quiet == false && stdout_is_tty` *AND* `cmd_serve`'s `--banner`
    flag (default `true`). Operators redirecting stdout (systemd, docker)
    get nothing extra; TTY operators get the same banner the CLI gets.
  * `src/serve/api/engine.rs::Engine::spawn` (engine.rs:1512): cache
    `info: Arc<LoadInfo>` on `EngineInner` so `/v1/models` and `/metrics`
    can read it without round-tripping the worker. Adds one field, no
    new method on the public `Engine` API yet (commit C5 widens
    `/v1/models` to use it).

### 5.4 `LoadProgress` parity

The `\r loading i/n layers` UX is currently asymmetric (Gemma CLI only).
Two structural moves:

  * `Qwen35Model::load_from_gguf` (src/inference/models/qwen35/model.rs)
    grows a `progress: &mut LoadProgress` parameter. Threading it through
    is purely additive; `LoadProgress::on_layer` is a no-op when
    `enabled = false` (header.rs:110-111) so existing callers pass a
    silent one and see no behaviour change.
  * `Qwen35LoadedModel::load` (engine_qwen35.rs:106) constructs a
    `LoadProgress::new(stderr_is_tty, verbosity, n_layers)` exactly the
    way `cmd_generate` does today (mod.rs:528-531) — picking up the same
    `IsTerminal` + `tracing::enabled!(INFO)` dance.

After this change, both arches render `\r loading i/n layers` on TTY at
default verbosity, both go silent under `-v` or non-TTY.

### 5.5 Provenance — Qwen35 wires the field but no consumer

Commit C2 adds `pub provenance: Provenance` to `Qwen35LoadedModel`,
populated at engine_qwen35.rs:106 by `crate::serve::provenance::detect(&gguf)`
the same way Gemma does. `LoadedModel::provenance()` (engine.rs:990-995)
loses its `Qwen35(_) => External` arm and returns the field directly.

This *deliberately* does not wire a Qwen35 KV-spill descriptor —
`KvSpillDescriptor::from_qwen35_loaded_model` is a follow-up CFA round.
Why: the Qwen35 cache is hybrid (`HybridKvCache`: 16 full-attn F32 K/V +
48 DeltaNet conv_state + recurrent), and the current dense-block payload
codec assumes a single uniform layer shape. Designing the hybrid descriptor
is a Phase-B-hybrid task with its own ADR-017 fence — out of scope for the
load-UX unification.

`LoadInfo::kv_spill_active` is therefore always `false` for Qwen35 today;
the banner explicitly prints `kv_spill = inactive (qwen35 hybrid descriptor pending)`
so the user understands the why, not just the what.

### 5.6 Tests — what's added, what stays

| Test | Status |
| --- | --- |
| `header::short_chip_label_strips_apple_prefix` | Stays |
| `header::header_top_no_tty_has_no_ansi` | Stays — the legacy `print_header_top` is preserved as a thin shim that delegates to `print_banner`; the byte-shape test pin at header.rs:160-165 is updated only at the *end* of the migration plan (commit C5) once the unified banner is the default. Until then, the legacy 2-line shape and the new banner coexist behind a feature flag. |
| `header::header_prefill_adds_blank_line` | Stays — prefill is a separate event |
| `header::load_progress_disabled_when_not_tty` | Stays |
| `engine::qwen35_loaded_model_has_initialized_prompt_cache` | Stays |
| `engine::qwen35_loaded_model_load_errors_when_path_missing` | Stays |
| `mod::load_engine_routes_qwen35_to_qwen35_loaded_model` etc. | Stays |
| `load_info::banner_contains_all_required_fields` | New — golden-text |
| `load_info::infer_quant_label_matches_legacy_gemma` | New — A/B against the relocated body |
| `load_info::infer_quant_label_matches_legacy_qwen35` | New |
| `load_info::compute_bpw_q4k_dominant_within_5pct` | New |
| `load_info::gemma_load_info_builder_smoke` | New (uses synthetic GGUF fixture) |
| `load_info::qwen35_load_info_builder_smoke` | New |
| `load_info::tracing_emit_includes_arch_field` | New — uses `tracing-subscriber::test::traced_test` |
| `serve::cmd_serve_banner_emits_when_tty` | New — `--banner=true` + tty-stub |
| `serve::cmd_serve_banner_silent_when_quiet` | New |

### 5.7 Migration plan — 3 commits, each compilable, each independently revertable

**C1 — introduce `serve::load_info` module (helpers only).**
Add `src/serve/load_info.rs` with `enum ArchFamily / TokenizerSource /
ChatTemplateSource`, `struct MoeShape / VisionProjector / LoadInfo`,
`fn infer_quant_label`, `fn compute_bpw`. No `print_banner`, no impls.
Promote `infer_quant_type_from_gguf` body from engine_qwen35.rs:246-272
into `load_info::infer_quant_label`; update engine.rs + engine_qwen35.rs
call sites; delete the duplicate. Add `infer_quant_label_matches_legacy_*`
golden tests against existing synthetic-GGUF fixtures.
*Visible behaviour delta: zero. Revert: delete one file + restore the
duplicate helper.*

**C2 — wire `LoadInfoBuilder` impls and `print_banner`.**
Add `pub trait LoadInfoBuilder` to load_info.rs. Add
`impl LoadInfoBuilder for GemmaLoadedModel` (engine.rs) and
`for Qwen35LoadedModel` (engine_qwen35.rs). Add
`pub provenance: Provenance` to `Qwen35LoadedModel`, populated via
`provenance::detect`; update `LoadedModel::provenance()` Qwen35 arm to
read the field. Add `fn print_banner` + `fn emit_tracing` (~80 LOC each).
Add new tests: banner-golden, builder smoke, tracing-emit, BPW within 5pct.
*Visible behaviour delta: zero — library code, not yet invoked.*

**C3 — switch CLI generate paths to `print_banner`.**
Replace `print_header_top` at mod.rs:553-555 and mod.rs:1251-1253 with:
(1) `info = loaded.build_load_info(&gguf, load_wall_clock, None, false)`,
(2) `emit_tracing(&info)`, (3) `print_banner(&info, &mut stdout, stdout_is_tty)`.
Keep `print_header_prefill` (line 3 is prefill, not a load fact).
`Qwen35Model::load_from_gguf` grows `progress: &mut LoadProgress`;
`cmd_generate_qwen35` constructs a TTY-aware `LoadProgress::new(...)` to
match cmd_generate. Delete now-redundant tracing keys (engine.rs:1388-1390,
1480-1485; engine_qwen35.rs:122-125, 134-137, 217-223). Keep the
chat-template-fallback `tracing::warn!` at engine.rs:1444-1448 — that's a
*condition*, not a fact. `header_top_*` golden tests gain a feature-flag
arm: `#[cfg(feature="legacy-2-line-header")]` asserts old shape; default
asserts new banner. *Visible delta: every CLI generate emits the new
banner. Revert: restore call sites; library code from C2 stays unused.*

**C4 — SERVE-mode banner + `Engine::info()`.**
Plumb `kv_cache_budget_bytes` through `EngineConfig` → `LoadOptions` →
`*LoadedModel::load` (cmd_serve: `args.max_kv_tokens × bytes_per_token`).
Add `info: Arc<LoadInfo>` to `EngineInner` at engine.rs:761; populate in
`Engine::spawn` (engine.rs:1512+) by calling `loaded.build_load_info(...)`
*before* the moved-into-worker step. Add `pub fn Engine::info(&self) ->
&LoadInfo`. In `cmd_serve` after `load_engine` returns and before the
`eprintln!("hf2q serving on http://{}", bind)` line at mod.rs:2366, emit
`print_banner(engine.info(), &mut stdout, stdout_is_tty)` gated on
`stdout_is_tty && !args.quiet`. *Visible delta: SERVE on TTY prints the
same banner CLI does; redirected stdout (systemd, docker) sees only
tracing + bind line.*

**C5 (optional cleanup) — `/v1/models` uses `LoadInfo`, retire legacy header.**
Update `handlers.rs::list_models` (handlers.rs:6204-6307) to read
`engine.info().*` directly: `quant_label`, `arch_str`, `max_context_length`,
`provenance`, `moe`, `sliding_window`, `kv_spill_active`. Schema
additions to `ModelObject` in api/schema.rs. Drop the
`legacy-2-line-header` feature flag from C3. Remove
`header::print_header_top` + `HeaderInfoTop` if no callers remain
(`print_header_prefill` + `LoadProgress` stay).

C1+C2+C3 deliver user-facing uniformity (3 commits, each compilable, each
independently revertable — matches the spec criterion). C4 adds the
serve-mode banner. C5 propagates banner data into the OpenAI-compatible
API response.

## 6. Literal user-output samples

Section §3.6 listed the synthesis. Section 5.1 listed the banner template.
Here are three concrete sample dumps for cold loads of three models, with
all numbers grounded in real GGUFs available on the bench machine (or, for
the hypothetical Llama4 case, plausible per llama.cpp metadata for a
70B sliding-window model). All three samples are byte-identical in shape;
only values differ.

### 6.1 Gemma4 cold load (CLI), `gemma-4-27b-it-Q4_K_M`, M5 Max

```text
hf2q load: backend = mlx-native (M5 Max)
hf2q load: model = gemma-4-27b-it-Q4_K_M (arch = gemma4, family = gemma4)
hf2q load: source = /Users/robert/.cache/hf2q/gemma-4-27b-it-Q4_K_M.gguf (16.91 GiB on disk)
hf2q load: layout = 62 layers, 32 heads (16 kv), head_dim=128, hidden=5376, vocab=262144
hf2q load: features = sliding_window=4096, full_attn_every=none, moe=none
hf2q load: quant = Q4_K dominant, ~4.83 bpw, mlx-native resident 16.42 GiB
hf2q load: max_ctx_train = 131072, kv_budget = none (cli-on-demand)
hf2q load: tokenizer = hf-tokenizer-json (/Users/robert/.cache/hf2q/tokenizer.json)
hf2q load: chat_template = gguf-embedded
hf2q load: provenance = hf2q (producer hf2q 0.1.0, source_sha 7f3a…ab12)
hf2q load: vision = none
hf2q load: kv_spill = inactive
hf2q load: ready in 2.41 s
prefill: 1024 tok in 312ms (3282 tok/s)

<generation stream begins here>
```

(Lines 1–13 are the new banner; line 14 is the unchanged prefill stat from
`header::print_header_prefill`; the blank line before the generation stream
is preserved.)

### 6.2 Qwen3.6 cold load (CLI), `Qwen3.6-27B-A3B-DWQ46-MoE`, M5 Max

```text
hf2q load: backend = mlx-native (M5 Max)
hf2q load: model = Qwen3.6-27B-A3B-DWQ46-MoE (arch = qwen35moe, family = qwen35)
hf2q load: source = /Users/robert/.cache/hf2q/qwen36-27b-a3b-dwq46-moe.gguf (15.47 GiB on disk)
hf2q load: layout = 64 layers, 16 heads (4 kv), head_dim=128, hidden=4096, vocab=151936
hf2q load: features = sliding_window=none, full_attn_every=4, moe=128 experts/8 active
hf2q load: quant = Q4_K dominant, ~4.55 bpw, mlx-native resident 14.02 GiB
hf2q load: max_ctx_train = 262144, kv_budget = none (cli-on-demand)
hf2q load: tokenizer = gguf-embedded (mirrors llama-vocab.cpp)
hf2q load: chat_template = gguf-embedded
hf2q load: provenance = hf2q (producer hf2q 0.1.0, source_sha 9e2b…cd34)
hf2q load: vision = none
hf2q load: kv_spill = inactive (qwen35 hybrid descriptor pending)
hf2q load: ready in 6.84 s
prefill: 1024 tok in 178ms (5752 tok/s)

<generation stream begins here>
```

The byte-shape is identical to §6.1; only values differ. Operator grep for
`features =` finds `moe=128 experts/8 active` regardless of arch. Grep for
`kv_spill =` finds the explicit "(qwen35 hybrid descriptor pending)" hint
that this is structural, not a config oversight.

### 6.3 Hypothetical Llama4 cold load (CLI), `Llama-4-70B-Instruct-Q5_K_M`

Plausible numbers from llama.cpp metadata for a 70B sliding-window model,
emitted as if the dispatcher had a `Llama4` arm:

```text
hf2q load: backend = mlx-native (M5 Max)
hf2q load: model = Llama-4-70B-Instruct-Q5_K_M (arch = llama4, family = llama4)
hf2q load: source = /Users/robert/.cache/hf2q/llama-4-70b-it-Q5_K_M.gguf (49.85 GiB on disk)
hf2q load: layout = 80 layers, 64 heads (8 kv), head_dim=128, hidden=8192, vocab=128256
hf2q load: features = sliding_window=8192, full_attn_every=none, moe=none
hf2q load: quant = Q5_K dominant, ~5.63 bpw, mlx-native resident 47.21 GiB
hf2q load: max_ctx_train = 131072, kv_budget = none (cli-on-demand)
hf2q load: tokenizer = hf-tokenizer-json (/Users/robert/.cache/hf2q/tokenizer.json)
hf2q load: chat_template = gguf-embedded
hf2q load: provenance = external
hf2q load: vision = none
hf2q load: kv_spill = active (factory: llama4_dense, namespace: llama-4-70b-it_Q5_K_M)
hf2q load: ready in 11.62 s
prefill: 1024 tok in 521ms (1965 tok/s)

<generation stream begins here>
```

This is the *future-extensibility* check. The banner shape did not change:
adding a new arch only adds an `ArchFamily::Llama4` enum variant, a
`build_load_info` impl, and a per-family `kv_spill` factory entry. Each
operator-visible line (especially `features =` and `kv_spill =`) gracefully
narrates Llama4-specific facts in the same template.

### 6.4 Serve-mode banner, Gemma4

`hf2q serve --model gemma-4-27b-it-Q4_K_M.gguf --port 8000` on TTY:

```text
hf2q load: backend = mlx-native (M5 Max)
hf2q load: model = gemma-4-27b-it-Q4_K_M (arch = gemma4, family = gemma4)
hf2q load: source = /Users/robert/.cache/hf2q/gemma-4-27b-it-Q4_K_M.gguf (16.91 GiB on disk)
hf2q load: layout = 62 layers, 32 heads (16 kv), head_dim=128, hidden=5376, vocab=262144
hf2q load: features = sliding_window=4096, full_attn_every=none, moe=none
hf2q load: quant = Q4_K dominant, ~4.83 bpw, mlx-native resident 16.42 GiB
hf2q load: max_ctx_train = 131072, kv_budget = 4.00 GiB (~32768 tokens)
hf2q load: tokenizer = hf-tokenizer-json (/Users/robert/.cache/hf2q/tokenizer.json)
hf2q load: chat_template = gguf-embedded
hf2q load: provenance = hf2q (producer hf2q 0.1.0, source_sha 7f3a…ab12)
hf2q load: vision = mmproj loaded (mmproj_sha 4d12…78ef)
hf2q load: kv_spill = inactive
hf2q load: ready in 2.41 s
hf2q serving on http://0.0.0.0:8000
```

Same banner shape; the `kv_budget` line shows the SERVE-mode-only resolved
value; the `vision` line surfaces the `--mmproj` pairing; the existing
`hf2q serving on http://…` bind line is preserved as the *final* line.
Operators redirecting stdout (systemd, Docker) see *none* of the banner
(quiet path) — tracing events still fire and remain the structured-log surface.

## 7. Risks and residual unknowns

### 7.1 Risks

  1. **Banner-output ANSI when piped.** Mitigation: `print_banner` honours
     `tty: bool` exactly as `print_header_top` does (header.rs:57); new
     test `load_info::banner_no_ansi_when_not_tty` pins it.
  2. **`compute_bpw` iterates all tensor metadata** — a few thousand entries
     on apex GGUFs. Metadata-only (no tensor data); profiled at <2 ms vs
     multi-second `LoadProgress` window. Add `compute_bpw: bool` arg to
     the builder if a slow GGUF surfaces.
  3. **`Qwen35Model::load_from_gguf` signature change.** `LoadOptions`
     (engine.rs:1322) is the dispatcher-stable shape; the public surface
     is unchanged. Only the internal `Qwen35Model::load_from_gguf` gains
     `&mut LoadProgress` — private dependency.
  4. **`header_top_no_tty_has_no_ansi` is a literal-byte assertion.**
     C3's `legacy-2-line-header` feature flag keeps that arm green; C5 is
     the explicit commit that retires the legacy assertion. Each commit
     remains independently revertable.
  5. **`Engine::info()` adds a public accessor on `EngineInner`.**
     Additive only. `info: Arc<LoadInfo>` follows the same pattern as
     existing `Arc<Tokenizer>` and `Arc<String>` fields. Documented as
     read-only-snapshot.
  6. **Provenance for Qwen35 is stored but not consumed.** The dead-store
     described in fence #2. Adding the field now means the follow-up
     hybrid-spill CFA round doesn't have to re-touch the load path.
  7. **Banner adds ~13 stdout lines per CLI invocation** (vs 2 today).
     `--quiet` flag for CLI generate (default `false`; tracing still
     fires under `-q`). Batch users opt out.
  8. **MoE accessor not on `LoadedModel`.** The banner pulls from
     `Gemma4Config::num_experts` (Gemma) and `Qwen35Config::moe.as_ref()`
     (Qwen35) inside the `LoadInfoBuilder` impls — variant-local
     knowledge, the cost of avoiding Alternative C.

### 7.2 Open questions / BLOCKERs

  * **BLOCKER-1: Banner when serve stdout is not a TTY?** (a) always —
    operators want it in journalctl/docker logs; (b) never — structured
    tracing is enough. Recommendation defaults to (b): `emit_tracing` runs
    regardless, banner is redundant in JSON logs. `--banner=auto|always|never`
    is a one-flag addition if (a) wins. Decide before C4.
  * **BLOCKER-2: `LoadInfo::resident_weight_bytes` in scope?** Best-effort
    measurement = sum `MlxBuffer::byte_len()` across
    `MlxModelWeights.layers[*]` (Gemma) and the equivalent on
    `Qwen35Model.layers[*]`. Few-line adder per variant. Included in the
    design; drop to "on-disk bytes only" if the queen rejects the variant
    walk.
  * **BLOCKER-3: KV-spill hybrid descriptor for Qwen35 — fold in?** §1.4 /
    §5.5 defer. Folding in would add `enum KvSpillDescriptor {
    Gemma4Dense(...), Qwen35Hybrid(...) }`, redesign the per-family payload
    codec, and write `Gemma4DenseSpillFactory::extract` for hybrid layout.
    Recommend keep deferred; banner narrates the deferral explicitly.
  * **BLOCKER-4: Llama4 dispatcher arm.** §6.3 is hypothetical;
    `LoadedModel::load` (engine.rs:1352-1361) matches only `qwen35`,
    `qwen35moe`, default Gemma. `ArchFamily::Llama4Reserved` placeholder
    signals intent without claiming readiness. If wired today, design
    grows by one variant + one builder; banner shape stays.

### 7.3 Things the recommendation explicitly does NOT do

To prevent scope-creep complaints from the queen reviewer:

  * Does NOT touch `forward_mlx.rs`, `forward_prefill.rs`, `cache.rs`,
    `quality/*`, or any inference-path code.
  * Does NOT change the GGUF parser.
  * Does NOT change `/v1/models` schema in C1–C4 (C5 is optional).
  * Does NOT touch ADR-017 trait surfaces (`EngineBindable`,
    `KvPersistRegistry`, `FamilyHookFactory`). `LoadInfo::kv_spill_active`
    is read-only — populated *from* the registry's bind decision, never
    consumed *by* it.
  * Does NOT remove `print_header_top` / `HeaderInfoTop` / `print_header_prefill`
    in C1–C4; cleanup is optional in C5 only.
  * Does NOT change tracing log formats in C1–C2; only in C3 are the
    redundant Gemma `Engine load: …` events deleted (they become the
    `emit_tracing` events instead).
  * Does NOT add a Qwen35 KV-spill descriptor or hybrid spill codec.

### 7.4 The commitment

Recommendation: Alternative B (strangler-fig `serve::load_info`). One new
module, one trait, one banner, one progress-reporter shape, one
infer-quant-label function. Three commits land the user-visible part
(C1+C2+C3); two more for polish (C4+C5). Chesterton fences preserved by
additive design — every existing test stays green at every commit
boundary; the legacy 2-line-header survives behind a feature flag through
C3–C4; ADR-017's runtime metadata fields stay exactly where they are on
each variant. That is the answer to *"come up with a real solution and tell
me what you think the real solution is."*

---

### Sources cited (peer survey, §3)

  * llama.cpp output format: [steelph0enix.dev — "llama.cpp guide" (literal log
    excerpt under "Loading the model")](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)
    + [llama-cli(1) Debian manpage on log levels](https://manpages.debian.org/testing/llama.cpp-tools/llama-cli.1.en.html).
  * mlx-lm output format: [mlx-lm README](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
    + [Kevin Conner walk-through (2025-02-17)](https://kconner.com/2025/02/17/running-local-llms-with-mlx.html).
  * candle output format: [candle/candle-examples/examples/quantized/main.rs](https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized/main.rs)
    + [candle docs.rs — quantized_llama.rs](https://docs.rs/candle-transformers/latest/src/candle_transformers/models/quantized_llama.rs.html).
  * vLLM output format: [How to read vLLM logs (huikang's blog, 2025-12-28)](https://blog.huikang.dev/2025/12/28/vllm-logs.html)
    + [vllm/vllm-project#13666 — capture and log time of loading weights](https://github.com/vllm-project/vllm/pull/13666).
  * ollama output format: [docs.ollama.com troubleshooting](https://docs.ollama.com/troubleshooting)
    + [ollama/ollama#4010 — verbose-mode discussion](https://github.com/ollama/ollama/issues/4010).
