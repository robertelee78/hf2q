# BUG — Gemma CLI silently ignores sampling flags

**Discovered:** 2026-05-16
**Severity:** HIGH — explains user-reported "looping garbage output" failure mode
**Trigger:** any user running `hf2q generate --model <gemma-gguf>` with
`--temperature`, `--top-k`, `--top-p`, or `--repetition-penalty`
**Affected arch:** Gemma 4 (and any other arch routed through
`cmd_generate` → `mlx_w.forward_decode`).  Qwen 3.5 / 3.6 path is
correct (uses `sample_qwen35_logits_for_generate`).

## Proof

Three byte-identical SHA256 hashes from runs with materially different
sampling parameters on the same model + prompt:

```
PROMPT="Write a 200-word essay on photosynthesis. Be detailed."
MODEL=models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf

# Run A — default (no sampling flags)
hf2q generate --model $MODEL --prompt "$PROMPT" --max-tokens 200
# → 769ecb98ba77b2d32b5f1974c7089d119073cc847d10b655f67014dd82ea58bb

# Run B — --repetition-penalty 1.5 (extreme penalty)
hf2q generate --model $MODEL --prompt "$PROMPT" --max-tokens 200 \
    --repetition-penalty 1.5
# → 769ecb98ba77b2d32b5f1974c7089d119073cc847d10b655f67014dd82ea58bb

# Run C — --temperature 0.7
hf2q generate --model $MODEL --prompt "$PROMPT" --max-tokens 200 \
    --temperature 0.7
# → 769ecb98ba77b2d32b5f1974c7089d119073cc847d10b655f67014dd82ea58bb
```

All three outputs are byte-identical (1143 bytes).  The CLI flags have
zero effect on Gemma decode.

## Root cause

`src/serve/mod.rs:1557` calls
`mlx_w.forward_decode(next_token, pos, &mut ctx, &mut p)`.

`forward_decode` signature at `src/serve/forward_mlx.rs:5986`:

```rust
pub fn forward_decode(
    &mut self,
    input_token: u32,
    seq_pos: usize,
    gpu: &mut GpuContext,
    profile: &mut Option<TokenProfile>,
) -> Result<u32>
```

**No sampling parameters in the function signature.**  It does a
pure GPU-argmax greedy decode and returns the argmax token id.  The
sampling params parsed from CLI (`args.temperature`,
`args.repetition_penalty`, `args.top_k`, `args.top_p`) are read into the
local `params` struct at `mod.rs:1244` and never reach the decode
function.

The acknowledging comment is at `src/serve/mod.rs:1586-1588`:

> "Sampling (temperature/top_k/top_p/repetition_penalty) is wired in
> forward_decode but the CLI does not pass them through; use the
> chat-completion API for non-deterministic decoding."

The comment claims sampling is "wired in forward_decode" — but the
function signature has no sampling params, so the comment is wrong about
that part.  The runtime sampling code lives at
`crate::serve::sampler_pure::sample_token` and is only called from the
HTTP API path (`src/serve/api/engine.rs:65, 282, 300, 452`).

## Why this causes "looping garbage" reports

Greedy decode + zero repetition penalty + zero sampling on a model that
produces near-tied logits at a degenerate template entrance = the
classic decoder loop trap.  Once the model picks a self-reinforcing
pattern, every subsequent step picks the same template continuation
because argmax has no escape mechanism.

The existing greedy-loop detector at `mod.rs:2030`
(`detect_greedy_repetition_loop`) checks n-gram sizes `{8, 12, 16, 20,
24}` × ≥4 repetitions.  Any loop with cycle length not in that set
slips through and runs to `--max-tokens`.

User has no CLI escape hatch:
- `--repetition-penalty 1.1` ignored
- `--temperature 0.3` ignored
- `--top-p 0.9` ignored
- `--top-k 40` ignored

## Fix plan

**Phase A (minimum-correct):** wire sampling through `cmd_generate`
mirroring the Qwen35 pattern.

1. Add a `pub fn forward_decode_with_sampling(&mut self,
   input_token, seq_pos, gpu, profile, sample_params)` to
   `MlxModelWeights` that downloads last-token logits, applies repetition
   penalty against `decoded_tokens` history, then calls
   `sampler_pure::sample_token` with `(temperature, top_k, top_p, min_p)`.
2. In `cmd_generate` (mod.rs:1542 loop): branch on
   `params.requires_sampling()` (cf. mod.rs:2122 helper for qwen35).
   - true → call new sampling decode
   - false → keep existing GPU-argmax fast path (perf-critical)
3. Unit test: same SHA256 proof above must show A ≠ B ≠ C after fix.
4. Integration test: a known greedy-loop prompt with
   `--repetition-penalty 1.1` must escape the loop and produce diverse
   output.

**Phase B (UX safety net):** if `requires_sampling()` returns true but
the code path doesn't support it yet (any unsupported arch), emit a
loud `eprintln!` warning at CLI entry — never silently ignore.

**Phase C (loop-detector hardening):** replace fixed n-gram set with a
sliding-window entropy check OR a generic compression-ratio check.
Catches cycles of any length.  Independent of Phase A; lands separately.

## Cross-arch impact

| Arch | CLI greedy path | CLI sampling path | Status |
|---|---|---|---|
| Gemma 4 | `forward_decode` (no params) | none — flags silently ignored | **BROKEN** |
| Qwen 3.5/3.6 | `forward_gpu_greedy` | `forward_gpu_last_logits + sample_qwen35_logits_for_generate` | OK |
| Qwen 3-VL | unknown — needs audit | unknown | unaudited |
| BERT / NomicBert | n/a (embeddings) | n/a | n/a |

## Standing rule going forward

CLI flags that are read from the command line but not honored by the
runtime are a contract violation.  Either honor them or refuse to
accept them at parse time.  No silent ignores.
