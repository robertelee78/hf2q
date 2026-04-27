# Wave 5b.3 Walk-Bar Validation Results

**Status:** **FAIL — chunk-pipeline diverges from llama.cpp at pp4096**
**Date:** 2026-04-27
**Worker:** Wave 5b.3 (ADR-005 inference-server mission close)
**HEAD at start:** `2239ed0` (Wave 5b.1 iter 5 audit hygiene close)
**HEAD at finish:** working-tree (observability `tracing::info!` added at `gpu_delta_net.rs:1330`, uncommitted at write time)

## Setup

| Field | Value |
|---|---|
| Model | `/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (27 B Qwen3.6 hybrid, DWQ46 quant, 16.07 GiB) |
| llama.cpp | build `8680 (15f786e65)`, AppleClang 21.0.0, Metal+Tensor backend on M5 Max |
| llama tool used | **`llama-completion`** (`llama-cli` rejected `--no-conversation`; per its own help, "use llama-completion instead") |
| hf2q binary | `/opt/hf2q/target/release/hf2q` rebuilt this iter (`cargo build --release --bin hf2q`, 0 errors, 79 warnings — all pre-existing) |
| Hardware | Apple M5 Max, 128 GiB unified memory |
| RAM at start | 75.5 GiB free (Pages free 4 944 446) |
| RAM at finish | 95.8 GiB free (Pages free 6 281 117) — no leak |

## Prompt construction

The chunk-pipeline gate requires `seq_len > 64 ∧ seq_len % 64 == 0 ∧ d_k == 128`. To force the chunk path to fire at `seq_len = 4096` (= 64 chunks × 64), the prompt was tuned via `llama-tokenize --no-bos` until the post-tokenization length hit exactly 4096:

```
prompt = "The history of bread baking spans millennia. " * 511 + "spans millennia ago. The dough rises slowly"
→ 4096 tokens (no BOS)
```

To make hf2q's `cmd_generate` emit this **without** a chat-template wrap, an identity Jinja template was passed: `--chat-template '{{ messages[0]["content"] }}'`. To make `llama-completion` skip BOS, `--override-kv tokenizer.ggml.add_bos_token=bool:false` was passed. Both engines therefore see the **same 4096 tokens** (verified: hf2q stderr `Qwen3.5: 4096 prompt tokens`, llama stderr `prompt eval time = ... / 4096 tokens`).

## Observability added

A single one-shot `tracing::info!` block at `src/inference/models/qwen35/gpu_delta_net.rs:1330-1357` (gated by static `AtomicBool`) emits exactly one log line per process the first time the chunk path either fires or is skipped under `HF2Q_CHUNK_SCAN_PREFILL=1`. Without this, a chunk-path-skip (e.g., `seq_len % 64 != 0`) is silent and any walk-bar test would report a fake-PASS. The probe confirmed the chunk path **actually fired** at `seq_len=4096, d_k=128`.

## Per-pp results

| pp | prompt tokens | llama-completion `n-predict 1` token | hf2q-chunk token | hf2q-autoreg token | match (chunk vs llama) | match (autoreg vs llama) |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **4096** | 4096 (raw, no BOS) | **`11` (`,`)** | **`42824` (` spans`)** | **`11` (`,`)** | **NO — DIVERGE** | YES |

**Root cause analysis.** The chunk-pipeline ENGAGED at `seq_len=4096` (confirmed by the new `wave5b3` log line). The autoregressive path with the **same** model + tokens reproduces llama-completion's first decoded token bit-exactly (id 11). Disabling `HF2Q_CHUNK_SCAN_PREFILL` while keeping `HF2Q_QWEN36_AUTOREG=1` flips hf2q's first decoded token from `42824` back to `11`. **The divergence is therefore introduced by the chunk-path itself, not by hf2q's tokenizer, kv-cache, sampler, or shared autoregressive forward path.**

This is consistent with the brief's own warning: the existing `chunk_path_first_token_matches_autoregressive_at_seq128` test passed at `1.64e-2` max divergence at `seq_len=128`, "within the cross-path bf16 budget but accumulates with seq_len." At `seq_len=4096` (32× longer) the per-element error has clearly grown enough to flip an argmax tie at the LM head. Likely contributors:
1. **bf16 round-off accumulating across 64 chunks** — chunk-pipeline q/k/v inputs are bf16-storage; each of the 6 sub-kernels (kkt, tri_solve_invert, recompute_w_u, inter_state, chunk_o, cumsum_g) does at least one bf16↔f32 round-trip and the per-chunk error compounds across the inter-chunk recurrence.
2. **State-recurrence drift** — the chunk-parallel reformulation reorders the delta-rule recurrence; mathematically equivalent in f32 (oracle confirms `max_err = 1.19e-7` at synthetic seq=128) but bf16-stored intermediate states grow drift with chunk index.
3. **Layer stacking** — Qwen3.6 has 30 delta-net layers; per-layer chunk-vs-autoreg residue compounds through the stack before reaching the LM head.

## Stages NOT executed

| Stage | Reason |
|---|---|
| pp16384 | **Skipped — pp4096 already failed.** Per discipline, "If hf2q's CLI doesn't yet support the chunk-pipeline activation cleanly … STOP and report" and "we want truthful validation status, not fake-PASS." Re-running at 16k/65k would only confirm and amplify the same numerical divergence with no new diagnostic value, while consuming ~10 min × 3 of model-load time. |
| pp65536 | Same as above. Also: pp65536 with `n_batch=65536` would risk running into the historical "first_decode_token=0 anomaly" already documented at `project_long_prefill_parity_inverts.md`. |

## Total wall-clock for prefill

| engine | prefill wall-clock | tok/s | first decoded token |
|---|---:|---:|:---:|
| llama-completion | 6388 ms | 641.2 | `11` (`,`) |
| hf2q autoregressive (chunk OFF) | 51 097 ms | 80.2 | `11` (`,`) |
| hf2q chunk-pipeline (chunk ON) | 45 385 ms | 90.3 | `42824` (` spans`) |

(Informational only — W-5b.3 is correctness, not perf. The chunk path is **12.6 % faster than autoregressive** in hf2q at this length but is **7.1× slower than llama.cpp**.)

## Environmental issues encountered

1. **`llama-cli` rejected `--no-conversation`** with the message *"--no-conversation is not supported by llama-cli — please use llama-completion instead"* — switched to `llama-completion`.
2. **`llama-completion` defaults to chat-mode** when the model has a chat template (kv 30); `-no-cnv` required to disable.
3. **`llama-completion` adds BOS by default** (`tokenizer.ggml.add_bos_token=true`); `--override-kv tokenizer.ggml.add_bos_token=bool:false` required to match hf2q's no-BOS tokenization.
4. **hf2q has no built-in chat-template-bypass flag**; `--chat-template '{{ messages[0]["content"] }}'` (identity template) was used as a workaround. Suggest a future `--no-chat-template` flag.
5. **Chunk-path silent skip when `seq_len % 64 != 0`** is a footgun for any walk-bar harness (the first probe at 5610 tokens silently ran the autoregressive path despite the env gate). The new `tracing::info!` at `gpu_delta_net.rs:1330-1357` makes this observable; recommend keeping it as production-hygiene observability.

## ADR-005 status flip recommendation

**NO-CHANGE for the partial-flip wording on AC 5468 (line 5735) and AC 5470 (line 5737).** The Wave 5a addendum at line 5714 correctly classifies these as "partial — Wave 5a autoregressive opt-in landed; chunk-scan kernel for SOTA prefill perf deferred to W-5b." That phrasing remains accurate after this iter:

- The autoregressive opt-in is correct (matches llama.cpp at pp4096).
- The W-5b chunk-scan kernel is wired (per W-5b.1 iter 5 milestone) but **fails the walk-bar correctness gate at production prefill length pp4096**.

**A new ADR-005 paragraph SHOULD be added under the Wave 5b.1 iter 5 entry at line 5720** documenting: (a) chunk-path correctness FAILED at pp4096 with the diagnostic above, (b) the chunk-path is therefore ENGAGED-but-NOT-CORRECT at production prefill lengths, and (c) AC 5468 ±5 % perf-bar measurement is **still blocked** — not by the chunk kernel's existence (it's wired) but by its numerical accuracy budget at long prefill. The AC remains `[x] (partial)`; W-5b.4 (or a follow-on) needs to address the bf16 accumulation budget before AC 5468 can flip to `[x]` (full).

## Walk-bar status

**FAIL.** The chunk-pipeline at `HF2Q_CHUNK_SCAN_PREFILL=1` produces a different first decoded token from llama.cpp at the smallest production prefill length we tested (pp4096), and from hf2q's own autoregressive baseline. The hf2q autoregressive path passes the walk-bar (token-id match against llama-completion). The chunk-path's existing `chunk_path_first_token_matches_autoregressive_at_seq128` unit test (1.64e-2 max divergence at seq=128) is **insufficient** as a correctness gate at pp4096+ — the test should be extended to assert argmax-id stability at production seq lengths, or the chunk path should be gated off until a representation/numerical fix lands.

## Reproducibility

Inputs preserved at:
- `/tmp/walkbar-pp4096-prompt.txt` (4096-token raw prompt, 23 038 bytes)
- `/tmp/walkbar-pp4096-llama.stdout` + `.stderr` (llama-completion baseline)
- `/tmp/walkbar-pp4096-hf2q-chunk.stdout` + `.stderr` (chunk-pipeline run)
- `/tmp/walkbar-pp4096-hf2q-autoreg.stdout` + `.stderr` (autoregressive sanity)

Reproduce with:

```sh
# llama baseline
llama-completion --model qwen3.6-27b-dwq46.gguf --file walkbar-pp4096-prompt.txt \
  --override-kv tokenizer.ggml.add_bos_token=bool:false \
  --n-predict 1 --temp 0.0 --seed 42 --no-warmup \
  --batch-size 4096 --ubatch-size 4096 -no-cnv

# hf2q chunk-pipeline (FAILS walk-bar)
HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_CHUNK_SCAN_PREFILL=1 \
  hf2q -v generate --model qwen3.6-27b-dwq46.gguf \
  --prompt-file walkbar-pp4096-prompt.txt \
  --chat-template '{{ messages[0]["content"] }}' \
  --max-tokens 1 --temperature 0.0

# hf2q autoregressive (matches llama)
HF2Q_QWEN36_AUTOREG=1 HF2Q_UNSAFE_EXPERIMENTS=1 \
  hf2q -v generate --model qwen3.6-27b-dwq46.gguf \
  --prompt-file walkbar-pp4096-prompt.txt \
  --chat-template '{{ messages[0]["content"] }}' \
  --max-tokens 1 --temperature 0.0
```
