# `tests/fixtures/ppl-corpus/` — perplexity smoke corpus

ADR-014 P10 iter-2a: peer-side `llama-perplexity` subprocess wrapper +
smoke-tier fixture for the harness plumbing only. Production parity
gates run against the **full** ~280 k-token wikitext-2 split (Decision
16, fetched at iter-3 / P11 close).

## Files

### `wikitext2-smoke.tokens`

- **Format:** raw little-endian `u32` stream, **no header**, no
  delimiters. Reader pseudocode:
  ```rust
  let bytes = std::fs::read("wikitext2-smoke.tokens")?;
  assert_eq!(bytes.len(), 512 * 4);
  let tokens: Vec<u32> = bytes
      .chunks_exact(4)
      .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
      .collect();
  ```
- **Length:** exactly 512 tokens ⇒ exactly 2048 bytes on disk.
- **Generation rule (deterministic, reproducible from a single seed —
  no fetch step, no tokenizer dependency):**
  ```text
  for i in 0..512 {
      tokens[i] = ((i as u32) * 17 + 3) % 32000
  }
  ```
  First four tokens: `[3, 20, 37, 54]`.
  Last four tokens: `[8639, 8656, 8673, 8690]`.
  First 16 bytes (hex): `03000000 14000000 25000000 36000000`.

  The modulus 32000 is a conservative lower bound on the vocabulary
  size of every Qwen / Gemma / Llama model the harness benchmarks
  against (Qwen3.6 = 152064, Gemma 3.4 = 256000, Llama 3.x = 128256);
  every generated id is in-vocab for every peer.

  17 is coprime with 32000 (`gcd(17, 32000) = 1`) so the sequence
  walks the residue class deterministically without trivial repeats
  inside the 512-token window.

## Role

This fixture exists **only** to exercise the iter-2a harness plumbing:
- `tests/common/llama_cpp_runner.rs::run_llama_perplexity` is wired
  end-to-end (resolves the binary, captures stderr, parses the
  upstream `Final estimate: PPL = <f32>` line);
- `tests/peer_parity_gates.rs::emit_markdown_table` grows three
  PPL columns (`hf2q PPL`, `peer PPL`, `PPL ratio`);
- the smoke tests cover (a) deterministic-content load, (b)
  missing-binary sentinel, (c) markdown rendering of half-measured
  rows, (d) markdown rendering of fully-measured rows + verdict logic
  on `ppl_tolerance`.

It is **not** a parity-grade corpus. The 512-token window is too
short for statistically meaningful PPL — Decision 16 (lines ~602 of
`docs/ADR-014-streaming-convert-pipeline.md`) locks the gate corpus
to wikitext-2 full test split (~280 k tokens), which iter-3 fetches
via a checked-in fetch script (deferred to iter-3 + P11 close).

## What lands when

| Iter | Lands |
|------|-------|
| **iter-2a** (this) | Peer-side `run_llama_perplexity` wrapper + this 512-token smoke fixture + 3 PPL columns in `emit_markdown_table` + 4 always-on smoke tests. The 8 `#[ignore]`-gated cells now record `peer_ppl` for real; hf2q-side stays `Verdict::NotMeasured`. |
| **iter-2b** (next) | hf2q-side `measure_ppl_qwen35_dense_gguf(model, tokens) -> f32` driver wrapping `Qwen35Dense::from_gguf` + chunked forward-pass + `compute_perplexity`. Flips the 8 cells to record hf2q PPL too. |
| **iter-3** (after P11) | Full wikitext-2 ~280 k token split fetched via fetch script + the 8 cells run end-to-end on real models per Decision 15. |
