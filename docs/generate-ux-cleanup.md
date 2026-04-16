# Serve UX cleanup + tracing migration

**Status:** ready-for-dev
**Author:** Party-mode session (Amelia, Winston, Barry, Sally, Paige, Bob — facilitated by Robert)
**Created:** 2026-04-16
**Mantra:** No shortcuts. Measure 3x, cut once. Chesterton's fence on every print.

---

## 1. Context

### 1.1 The problem
`hf2q generate` (invoked colloquially as "serve") currently emits ~40+ stderr lines during boot and prefill, rendered with the same visual weight as the actual model generation. A developer watching the terminal during kernel/GPU perf work has to scroll to find where the generation stream starts. Example (truncated):

```
Initializing mlx-native GPU context...
mlx-native backend: Apple M5 Max
Loading GGUF model...
GGUF loaded: 663 tensors, 39 metadata keys
Loading model weights from GGUF into mlx-native buffers...
  Loading mlx-native weights directly from GGUF...
  Loading embed_weight...
  Loading final_norm...
  Quantizing lm_head to Q8_0 (forced — F16 size 1476.4 MB)...
  Q8_0 lm_head created (784.3 MB, 1.88× smaller than F16).
  GGUF layer 1/30: loading weights...
  GGUF layer 1/30: MoE experts loaded (stacked, 416.3 MB + 269.6 MB)
  GGUF layer 2/30: loading weights...
  ... [28 more layer lines, 5 more MoE lines] ...
  Loaded 30/30 mlx-native layer weights from GGUF (including MoE).
mlx-native weights loaded (30 layers).
Running mlx-native forward pass...
Prefill: KV cache dtype = F32
Prefill: 22 tokens × 30 layers (dense SDPA)
Prefill complete (dense SDPA): 22 tokens in 342.7 ms (64.2 tok/s), first decode token = 41950
Making sourdough bread is a blend of science (fermentation) and art ...
```

The generation text collides with diagnostic output on the line above. There is no visual frame.

### 1.2 Primary audience
The developer (Robert) using `generate` as a debug/dev harness while working on GPU kernel perf. Full current verbosity must remain reachable on demand.

### 1.3 Why one PR (not two)
Doing the UX cleanup with a `Verbosity` enum first and migrating to `tracing` later means every new boot-path line written in between is authored against the wrong contract, then rewritten. The tracing migration *is* the reclassification engine — they are one piece of work.

### 1.4 Non-goals
- Implementing the stubbed `Serve` subcommand (main.rs:139). Separate work.
- Changing prefill/decode *algorithm* output (generation stream itself).
- Changing `cmd_parity`, `cmd_convert`, `cmd_info`, `cmd_validate`, `cmd_doctor` output formats. Scope is the generate boot/run path.
- Structured JSON log output. Prepared for (via `tracing-subscriber`) but not enabled by default.
- Renaming the `src/serve/` folder. Naming cleanup is deferred; this PR keeps the folder where it is.

---

## 2. Current-state audit (Chesterton's fence)

Every existing print site on the `generate` boot/run path, with its reason for existing. **No site is deleted outright — each gets reclassified.**

### 2.1 `src/serve/mod.rs` — `cmd_generate` + chat-template resolution

| Line | Current | Purpose | Target |
|------|---------|---------|--------|
| 132  | `tracing::info!("Chat template: using CLI --chat-template override")` | Chat template source decision | `info!` — no change |
| 138  | `tracing::info!("Chat template: loading from --chat-template-file {}", ...)` | Same | `info!` — no change |
| 146  | `tracing::info!("Chat template: using GGUF metadata tokenizer.chat_template ({} chars)", ...)` | Same | `info!` — no change |
| 154  | `tracing::warn!("Chat template: no GGUF metadata ... falling back ...")` | Chat template fallback warning | `warn!` — no change |
| 192  | `tracing::info!("Model: {}", ...)` | Path echo for debugging mis-resolution | `info!` — no change |
| 193  | `tracing::info!("Tokenizer: {}", ...)` | Same | `info!` — no change |
| 194  | `tracing::info!("Config: {}", ...)` | Same | `info!` — no change |
| 199  | `tracing::info!("Gemma4 A4B: {} layers, {} heads, hidden={}, {} experts (top-{})", ...)` | Model arch summary | `info!` — no change |
| 206  | `eprintln!("Initializing mlx-native GPU context...")` | Step marker for GPU init failures | `info!("Initializing mlx-native GPU context")` |
| 209  | `eprintln!("mlx-native backend: {}", ctx.gpu_name())` | Record chip for bug reports | **Source for header line 1** + `info!` |
| 212  | `eprintln!("Loading GGUF model...")` | Step marker for GGUF parse failures | `info!` |
| 215  | `eprintln!("GGUF loaded: {} tensors, {} metadata keys", ...)` | GGUF sanity check | `debug!` |
| 218  | `eprintln!("Loading model weights from GGUF into mlx-native buffers...")` | Step marker | `info!` |
| 220  | `eprintln!("mlx-native weights loaded ({} layers).", ...)` | Completion signal for weight load | `info!` + contributes to header line 2 |
| 249  | `tracing::info!("Prompt: {} tokens", ...)` | Tokenization sanity check | `info!` — no change |
| 268  | `eprintln!("Running mlx-native forward pass...")` | Step marker pre-prefill | `info!` |
| 339  | `eprintln!("\n\n--- mlx-native: ... tok/s ---")` | Post-generation footer with decode speed | **Trailer: stays as stderr product output** (dim on TTY) |

### 2.1b `src/serve/gpu.rs`

| Line | Current | Purpose | Target |
|------|---------|---------|--------|
| 55   | `tracing::info!("mlx-native GpuContext initialized on {}", gpu_name)` | GPU context init ack | `info!` — no change |

### 2.2 `src/serve/forward_mlx.rs` — weight loading

| Line | Current | Purpose | Target |
|------|---------|---------|--------|
| 534  | `eprintln!("  Loading mlx-native weights directly from GGUF...")` | Step marker | `debug!` |
| 537  | `eprintln!("  Loading embed_weight...")` | Progress on single large tensor | `debug!` |
| 542  | `eprintln!("  Loading final_norm...")` | Same | `debug!` |
| 597  | `eprintln!("  Quantizing lm_head to Q8_0 ({} — F16 size {:.1} MB)...", ...)` | Signal quantization choice | `info!` |
| 635  | `eprintln!("  Q8_0 lm_head created ({:.1} MB, {:.2}× smaller than F16){}.", ...)` | Confirm quantization success | `info!` |
| 697  | `eprintln!("  GGUF layer {}/{}: loading weights...", i+1, num_layers)` (× 30) | Per-layer progress, useful on load failure | `debug!` + **single `\r` progress line in default mode** |
| 755  | `eprintln!("  GGUF layer {}/{}: MoE experts loaded (stacked, {:.1} MB + {:.1} MB)", ...)` (× ~5) | MoE expert stack diagnostic | `debug!` |
| 885  | `eprintln!("\r  Loaded {}/{} mlx-native layer weights from GGUF (including MoE).    ")` | Final `\r` overwrite of progress | `info!` + **contributes to header line 2 (total layers / total MB)** |

### 2.3 `src/serve/forward_prefill.rs` — prefill

| Line | Current | Purpose | Target |
|------|---------|---------|--------|
| 95   | `eprintln!("Prefill: KV cache dtype = {:?}", kv_dtype)` | ADR-009 dtype diagnostic | `debug!` |
| 130  | `eprintln!("Prefill: {} tokens × {} layers (dense SDPA)", ...)` | Prefill path marker | `debug!` |
| 848  | `eprintln!("Prefill complete (dense SDPA): {} tokens in {:.1} ms ({:.1} tok/s), first decode token = {}")` | **Perf number Robert watches** | **Source for header line 3** + `debug!` emits `first decode token` only |

### 2.4 Generation stream (do-not-touch zone)

| Line | Current | Status |
|------|---------|--------|
| 294  | `print!("{}", token_str); stdout().flush()` (first decode token) | **Stays** — product output on stdout |
| 320, 333 | Same for subsequent decode tokens | **Stays** |

### 2.5 Benchmark mode (do-not-touch zone)

| Line | Current | Status |
|------|---------|--------|
| 358–364 | `println!` × 6 (gated on `args.benchmark`) | **Stays** — benchmark mode is product output by design |

### 2.6 Investigation-env debug dumps (do-not-touch zone)

| Line | Current | Status |
|------|---------|--------|
| 238  | `HF2Q_DUMP_RENDERED_PROMPT: wrote ...` | **Stays as `eprintln!`** — env-var contract per `docs/operator-env-vars.md` |
| 251  | `HF2Q_DUMP_PROMPT_TOKENS: first10=..."` | **Stays** — env-var contract |

*These env-var-gated dumps are stable operator-facing output and must not be reclassified without updating the operator contract doc.*

### 2.7 Library-crate prints (mlx-native, out of scope to rewrite here)

`/opt/mlx-native/src/graph.rs` has 5 `eprintln!` sites (lines 645, 1466, 1471, 1806, 1831) and `/opt/mlx-native/src/lib.rs:122` has 1 `println!`. All are env-var-gated diagnostics (GROUP_STATS, GRAPH_OPT, DIAG). **Out of scope for this PR** — mlx-native is a separate crate, and its diagnostic env-var contract is stable. Follow-up ticket may migrate mlx-native to `tracing` separately.

---

## 3. Target default output spec

### 3.1 Byte-level layout (stdout)

When stdout is a TTY:

```
\x1b[2mhf2q · M5 Max · mlx-native\x1b[0m
\x1b[2mgemma-3-27b-it-Q4_K_M · loaded in 8.3s · 30 layers · 12.4 GB\x1b[0m
\x1b[2mprefill: 22 tok in 343ms (64 tok/s)\x1b[0m

Making sourdough bread is a blend of science (fermentation) and art...
```

When stdout is not a TTY (pipe, file redirect):

```
hf2q · M5 Max · mlx-native
gemma-3-27b-it-Q4_K_M · loaded in 8.3s · 30 layers · 12.4 GB
prefill: 22 tok in 343ms (64 tok/s)

Making sourdough bread is a blend of science (fermentation) and art...
```

### 3.2 Header line construction

- **Line 1** — `hf2q · {chip} · {backend_name}`
  - `chip` from `ctx.gpu_name()` (already exists, mod.rs:209). Rename backend form to short chip label (e.g., `"Apple M5 Max"` → `"M5 Max"`). Add a small helper `short_chip_label(&str) -> String` — strips vendor prefix ("Apple "). Fall back to full string if prefix absent.
  - `backend_name` = literal `"mlx-native"` for now (ADR-008 sole backend).
- **Line 2** — `{model} · loaded in {load_s:.1}s · {n_layers} layers · {total_gb:.1} GB`
  - `model` from GGUF metadata `general.name` (fall back to `model_path.file_stem()` if absent).
  - `load_s` timed from start of `Initializing mlx-native GPU context` to completion of `mlx-native weights loaded`.
  - `n_layers` from `mlx_w.layers.len()` (already computed, mod.rs:220).
  - `total_gb` from sum of tensor bytes loaded (needs a `bytes_loaded: u64` accumulator on `MlxModelWeights`).
- **Line 3** — `prefill: {n_tokens} tok in {ms:.0}ms ({tok_per_s:.0} tok/s)`
  - All three numbers already computed at forward_prefill.rs:848. Re-use.

### 3.3 Progress during load (TTY only, default mode)

Before line 2 is printed, show a single `\r`-overwritten progress line on stderr (not stdout):

```
loading... 12/30 layers (5.1 GB)
```

Final `\r` overwrite clears the line before line 2 renders to stdout. On non-TTY, the progress line is suppressed entirely; user sees lines 1→2→3→blank→generation with no progress noise.

### 3.4 Generation

Untouched. Existing `print!` → stdout at mod.rs:294/320/333.

### 3.5 Trailer

Existing post-generation trailer (`--- mlx-native: N tokens in Ts (X tok/s) ---`, mod.rs:339) stays on **stderr**, dimmed on TTY. Moves out of stdout's path entirely so redirected stdout captures only model output.

---

## 4. Target `-v` / `-vv` / `RUST_LOG` behavior

### 4.1 Flag mapping (main.rs / cli.rs)

**Already present** (Chesterton's fence — pre-existing at cli.rs:14-17):

```rust
#[arg(short, long, action = clap::ArgAction::Count, global = true)]
pub verbose: u8,
```

Global so it applies to `generate`, `parity`, `info`, etc.

### 4.2 Subscriber init (main.rs, in `fn main` after `Cli::parse`)

```rust
let filter = match cli.verbose {
    0 => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("hf2q=warn")),
    1 => EnvFilter::new("hf2q=info,mlx_native=info"),
    2 => EnvFilter::new("hf2q=debug,mlx_native=debug"),
    _ => EnvFilter::new("hf2q=trace,mlx_native=trace"),
};
tracing_subscriber::fmt()
    .with_env_filter(filter)
    .with_writer(std::io::stderr)
    .without_time()  // match current UX; add timestamps via RUST_LOG only
    .init();
```

Rationale for each choice:
- `with_writer(std::io::stderr)`: logs never touch stdout. Stdout stays reserved for the three header lines + generation + trailer.
- `without_time()`: current output has no timestamps; AC-5 requires reproducing current output modulo format-wrapper differences. Timestamps would change every line.
- `RUST_LOG` wins at verbosity 0: if the user sets `RUST_LOG=hf2q=debug` without `-v`, they get debug output. `-v` only kicks in if they pass it explicitly.
- Default level `hf2q=warn`: silent in default mode. The three header lines bypass tracing entirely (they are product output).
- **`INVESTIGATION_ENV.activate()` runs before `Cli::parse`** — it uses direct `eprintln!` (see §2.6), so it doesn't depend on the subscriber. Ordering preserves the current contract: investigation-var warnings render even on `--help` / `--version` early exit.

### 4.3 `-v` contract (phase markers)
Shows high-level phase markers only (tracing `info!` events): init start, backend name, GGUF load, weight-load start/end, quantization decisions, forward-pass start, prompt token count. Cleaner than today's default. ~8–10 lines on a typical run.

### 4.4 `-vv` contract (detail — reproduces current default)
Adds `debug!` events: per-layer weight-load progress, MoE stack sizes, KV cache dtype, prefill markers. Approximately reproduces today's boot output line-for-line. This is the level golden-tested in §6.2.

### 4.5 `-vvv` contract (trace — everything)
Adds `trace!` events: first decode token ID and any future granular diagnostics.

---

## 5. Acceptance criteria

- **AC-1** Default `hf2q generate` stdout is exactly: 3 dim header lines (non-dim when not a TTY), blank line, generation stream. Trailer line goes to stderr. Nothing else on stdout.
- **AC-2** All boot-path sites enumerated in §2.1, §2.2, §2.3 are reclassified to the `tracing` level specified in the "Target" column. No non-enumerated print sites remain on the generate path. (Audit enforced by §6.2 test.)
- **AC-3** `tracing-subscriber` with `EnvFilter` initialized once in `main`. `-v` → `hf2q=info,mlx_native=info`. `-vv` → `hf2q=debug,mlx_native=debug`. `-vvv+` → `hf2q=trace,mlx_native=trace`. `RUST_LOG` respected at verbosity 0.
- **AC-4** On non-TTY stdout (pipe/redirect), ANSI escape codes are stripped from the 3 header lines. Tracing events continue to route to stderr regardless of stdout TTY state. Progress line (§3.3) is suppressed on non-TTY.
- **AC-5** `hf2q generate ... -vv 2>stderr.txt >stdout.txt` produces a `stderr.txt` whose tracing-wrapper-stripped form matches the pre-PR stderr line-by-line (modulo env-var-gated dump lines which stay byte-identical). `-vv` is the level at which reclassified `debug!` events fire, so it maps to today's default boot volume. Enforced by golden test in §6.
- **AC-6** Library crates used by the generate path (starting with the `serve/` module's sibling crates inside `hf2q`) emit **only** `tracing` events for boot-path diagnostics — no direct `eprintln!`/`println!` survives except §2.4/§2.5/§2.6 do-not-touch zones.
- **AC-7** Three header lines use `general.name` from GGUF metadata when present; fall back to file stem when absent. Unit-testable.

---

## 6. Regression guards

### 6.1 Manual smoke
```
hf2q generate <model> --max-tokens 10 --prompt "Test."           # default: 3 dim lines + blank + text
hf2q generate <model> --max-tokens 10 --prompt "Test." 2>/dev/null  # plain lines, no ANSI
hf2q generate <model> --max-tokens 10 --prompt "Test." >out.txt  # headers stripped of ANSI in out.txt
hf2q generate <model> --max-tokens 10 --prompt "Test." -v         # phase markers (cleaner than current default)
hf2q generate <model> --max-tokens 10 --prompt "Test." -vv        # current default volume (debug events)
hf2q generate <model> --max-tokens 10 --prompt "Test." -vvv       # trace-level (first decode token, etc.)
RUST_LOG=hf2q=debug hf2q generate <model> --max-tokens 10 ...     # debug events without -v flag
```

### 6.2 Golden-output test (`tests/serve_ux.rs`, new file)

Three test cases, all using a short deterministic seed+prompt:

1. `default_stdout_has_three_header_lines_and_generation` — captures stdout, strips ANSI, asserts line count == `3 + 1 + len(generation_tokens)`, asserts lines 0..3 match header regexes.
2. `default_stderr_empty_except_trailer` — captures stderr in default mode, asserts only the post-gen trailer line appears.
3. `vv_flag_reproduces_current_boot_output` — runs with `-vv`, strips tracing wrapper format, compares against a checked-in `tests/fixtures/generate_boot_log.golden.txt` that was captured pre-PR. CI fails if boot lines drift.

The golden fixture is captured **once before any refactor**, locked, and only mutated by intentional boot-output changes (which require updating the fixture in the same commit).

### 6.3 Unit tests

- `short_chip_label("Apple M5 Max") == "M5 Max"`
- `short_chip_label("AMD Radeon Pro") == "AMD Radeon Pro"` (no "Apple " prefix)
- `header_line_1(...)` shape for a fake ctx
- `header_line_2(...)` with `general.name` present vs absent
- ANSI stripping on non-TTY path

---

## 7. Implementation plan (ordered)

Execute in this order; each step leaves the build green and tests passing.

**Step 0 — Capture golden fixture.**
Before any boot-path reclassification, run `hf2q generate <model> --max-tokens 8 --prompt "Test." 2>tests/fixtures/generate_boot_log.golden.txt`. Commit the fixture on its own.

**Step 1 — Tracing subscriber scaffold. [DONE]**
- `-v` flag already on `Cli` (cli.rs:14-17) pre-existing.
- Reordered `main.rs`: `INVESTIGATION_ENV.activate()` → `Cli::parse()` → `tracing-subscriber` init with verbosity-aware `EnvFilter`.
- No behavior change at verbosity 0 (`hf2q=warn` default, all existing `eprintln!` still fire as before).
- Step 0's fixture capture can happen after Step 1 (Step 1 doesn't change any boot-path `eprintln!` output).

**Step 2 — Reclassify `src/serve/forward_prefill.rs` (§2.3).**
3 sites. Smallest blast radius. Golden fixture at `-v` should still match.

**Step 3 — Reclassify `src/serve/forward_mlx.rs` (§2.2).**
8 sites. Keep the `\r` layer-progress trick but move it behind a new helper that emits via `eprint!` only on TTY. Export `bytes_loaded` for header line 2.

**Step 4 — Reclassify `src/serve/mod.rs` (§2.1).**
Add `LoadTiming` struct capturing instants; compute `loaded in Xs`. Add `general.name` extraction from GGUF metadata with filename-stem fallback.

**Step 5 — Header printer.**
New small module `src/serve/header.rs` (< 120 lines) with:
- `short_chip_label`
- `HeaderInfo { chip, backend, model, load_s, n_layers, total_gb, prefill_n, prefill_ms, prefill_tok_s }`
- `print_header(w: &mut impl Write, info: &HeaderInfo, tty: bool)` — renders lines 1–3 with/without ANSI.

Called from `cmd_generate` at two points:
- After weight load (lines 1–2, before prefill starts).
- After prefill completes (line 3, before generation begins).

Rationale for two-phase print: the spec shows all three as a block, but line 3 depends on prefill having run. Print lines 1–2 as soon as known so the user sees progress; when prefill finishes, print line 3 followed by the blank. This matches current latency characteristics — prefill takes <1s, so visual grouping still reads as a block.

*Alternative considered:* buffer all three lines until prefill completes. Rejected because it hides the "model loaded" signal behind prefill latency, worse for slow-load scenarios.

**Step 6 — Trailer.**
Dim the existing trailer on TTY. One-line change.

**Step 7 — Tests.**
Wire in §6.2 and §6.3. Make CI fail if `tests/fixtures/serve_boot_log.golden.txt` drifts.

**Step 8 — Docs (follow-up, separate PR).**
Paige drafts user-facing docs update describing default output, `-v`/`-vv`/`RUST_LOG` after this PR merges.

---

## 8. Chesterton's fence — what not to touch

1. **`src/serve/forward_mlx.rs:885` `\r` overwrite pattern** — reuse it for §3.3 progress, don't replace with `indicatif`. The existing pattern works, is zero-dep beyond console escape sequences, and `indicatif` has its own stderr/stdout interaction quirks.
2. **Env-var-gated dumps (§2.6)** — do not touch without opening `docs/operator-env-vars.md` and `docs/shipping-contract.md` first.
3. **Benchmark mode stdout prints (§2.5)** — `--benchmark` is a separate product mode; its stdout output is a contract.
4. **`mlx-native` crate prints (§2.7)** — separate crate, separate diagnostic contract. Out of scope.
5. **`cmd_parity`** boot-path prints (if any) — parity has its own contract used by CI fixtures.
6. **`src/serve/` folder naming** — it houses `generate`; rename is a separate concern.

---

## 9. Out-of-scope / follow-ups

1. Docs update for `-v` / `-vv` / `RUST_LOG` semantics. Separate PR, Paige.
2. Migrating `mlx-native` crate to `tracing`. Separate ticket, separate PR. Coordinates with ADR-010.
3. Implementing the stubbed `Serve` subcommand (HTTP server). Will reuse this subscriber.
4. Structured JSON log output (for when `Serve` becomes a real HTTP server).
5. Renaming `src/serve/` → `src/inference/` (or similar) to match what it actually contains.

---

## 10. Effort estimate

- Step 0: 10 min (capture golden)
- Step 1: 30 min (subscriber + flag plumbing)
- Steps 2–4: ~1 hr (mechanical reclassification, 20 sites total)
- Step 5: 1–2 hr (header module + two-phase print)
- Step 6: 10 min (trailer dim)
- Step 7: 1–2 hr (tests, golden fixture discipline)

Total: **~4–6 hours** of focused work, one PR, ~600 LOC changed, ~200 LOC added (tests + header module).

---

## 11. Approvals captured in party-mode session

- **Sally (UX):** Visual-frame requirement. Dim on TTY, plain on pipe.
- **Amelia (Dev):** Spec locked at the byte level. AC-5 is load-bearing.
- **Winston (Architect):** Tracing migration is the right sequencing; library crates emit events, product code emits stdout.
- **Barry (Quick Flow):** AC-5 golden test is the gate that prevents this from becoming a multi-day refactor.
- **Paige (Tech Writer):** Docs update after merge.
- **Bob (SM):** Single story, Story A (one-shot). Scope tracked here.

*Mantra upheld: no shortcuts, no stubs, Chesterton's fence observed for every print site.*
