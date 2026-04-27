# `tests/fixtures/ppl-corpus/` — perplexity smoke corpus

ADR-014 P10: peer-side `llama-perplexity` subprocess wrapper +
smoke-tier fixture for harness plumbing, plus the iter-3 fetcher for
the production parity corpus. P11 parity gates run against the
**full** wikitext-2 raw test split (Decision 16), not the smoke fixture.

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

### `wikitext2-full.tokens`

- **Format:** same as the smoke fixture: raw little-endian `u32`
  stream, no header, no delimiters.
- **Source:** Stephen Merity/Salesforce `wikitext-2-raw-v1.zip`
  artifact, downloaded by `scripts/fetch_wikitext2.sh` from the
  llama.cpp CI HuggingFace mirror:
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`.
  The script locks the zip SHA-256 to
  `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`
  and extracts `wikitext-2-raw/wiki.test.raw`.
- **Generation:** `llama-tokenize --model <gguf> --file <wiki.test.raw>
  --no-bos --ids --log-disable`, parsed to little-endian `u32`.
  The installed llama.cpp tokenizer uses `--file`; it does not expose
  a `--prompt-file` alias.
- **Validation:** token count must be at least 280,000 and binary
  file size must be at least 1 MiB. SHA mismatch, tokenizer failure,
  or undersized output aborts the script.
- **Git:** `*.tokens` is ignored in this directory, with
  `!wikitext2-smoke.tokens` preserving the checked-in smoke fixture.
  Do not commit `wikitext2-full.tokens`.

One-line fetch example:

```bash
HF2Q_QWEN_VOCAB_GGUF=/path/to/qwen-tokenizer.gguf bash scripts/fetch_wikitext2.sh
```

The default cache is
`${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/wikitext2`; override with
`--cache-dir <path>`. Use `--force` to redownload and rebuild after
inspecting a failed or stale cache entry.

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
to the full wikitext-2 raw test split. The peer-parity loader now
prefers `wikitext2-full.tokens` when present and valid, and logs a
fallback to `wikitext2-smoke.tokens` otherwise.

## What lands when

| Iter | Lands |
|------|-------|
| **iter-2a** | Peer-side `run_llama_perplexity` wrapper + this 512-token smoke fixture + 3 PPL columns in `emit_markdown_table` + 4 always-on smoke tests. |
| **iter-2b** | hf2q-side `measure_ppl_qwen35_dense_gguf(model, tokens) -> f32` driver wrapping `Qwen35Dense::from_gguf` + chunked forward-pass + `compute_perplexity`. |
| **iter-2c** | Renamed the driver to variant-agnostic `measure_ppl_qwen35`; all 8 cells route through it. |
| **iter-3** (this) | `scripts/fetch_wikitext2.sh` builds `wikitext2-full.tokens`; `.gitignore` keeps generated corpora out of git; the corpus loader auto-picks full when present and falls back to smoke otherwise. |
