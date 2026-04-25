# Converting Qwen3.5 / Qwen3.6 Models with hf2q

Canonical reference for converting the Qwen3.5 / Qwen3.6 model family to DWQ GGUF.
Two variants are covered: the 35B MoE and the 27B dense.

See `docs/ADR-012-qwen35moe-conversion.md` for the full architectural rationale.

---

## Prerequisites

- `hf2q` built from this repo (`cargo build --release --bin hf2q`).
- An HuggingFace token with access to the target repo, either via
  `HF_TOKEN=hf_xxxx` env var or `~/.huggingface/token`.
- Sufficient free disk space (see below).

---

## Disk preflight

hf2q checks available disk space before starting any download. If the check
fails, the download is aborted immediately with a user-actionable error.

| Model | Minimum free disk |
|---|---|
| Qwen3.5-MoE 35B | 150 GB |
| Qwen3.5 27B dense | 55 GB |
| Other models | 100 GB |

The check uses `--cache-dir` (or `HF_HUB_CACHE` / `~/.cache/huggingface/hub`)
as the target path. To use a different disk, set:

```bash
export HF_HUB_CACHE=/path/on/large/disk
```

Example error when disk space is insufficient:

```
Error: Qwen3.5-MoE 35B requires ≥150 GB free in /home/user/.cache/huggingface/hub; found 80 GB. Free space or change --cache-dir.
```

---

## Shard resumption

hf-hub skips shards that have already been fully downloaded. If a download is
interrupted (e.g. `Ctrl+C`), re-invoking `hf2q` with the same `--repo` will
re-fetch only the in-flight shard and resume from there. Completed shards are
not re-downloaded.

Manual verification: after `Ctrl+C`, inspect
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/` — fully
downloaded shards are present; the interrupted shard may be partial. On
re-invoke, only the partial shard is re-fetched.

---

## Sidecar files

After writing the `.gguf`, hf2q copies these files from the HF source
directory into the output directory alongside the model file, byte-identical:

- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`

Files missing from the source are silently skipped. The sidecar set is the
same for all model classes (Gemma, Qwen3.5-MoE, Qwen3.5 dense).

---

## Converting Qwen3.6-35B-A3B (MoE variant)

> **Pre-P12 status (as of 2026-04-24):** DWQ on `qwen35` / `qwen35moe`
> requires real forward-pass activations from ADR-013 P12's
> `RealActivationCapture` impl. Until P12 ships, the command below
> **fails fast** with the structured `NoActivationCapture` error
> (see `feedback_never_ship_fallback_without_rootcause.md`). For the
> shipping path today, use `--quant q4_0` — see the smoke section
> below — and switch to DWQ once ADR-013 P12 lands. There is no
> weight-space fallback.

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/out.gguf
```

**Expected output size:** 18–22 GB (DWQ-4-6 of a 35B MoE; expert-merged).

Alternative with 4-bit base / 8-bit sensitive layers:

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/out.gguf
```

**Expected output size:** 22–28 GB (DWQ-4-8; sensitive layers promoted to 8-bit).

---

## Converting Qwen3.5-27B (dense variant)

```bash
hf2q convert \
  --repo <qwen35-dense-repo> \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --output models/qwen35-27b-dense-dwq48/out-dwq48.gguf
```

Replace `<qwen35-dense-repo>` with the HuggingFace repo ID of the 27B dense
model (e.g. `Qwen/Qwen3.5-27B-Instruct` or a fine-tune).

**Expected output size:** 14–18 GB (DWQ-4-8 of a 27B dense).

---

## Manual smoke test

After conversion, verify that llama.cpp can load the file without errors:

```bash
llama-cli --model models/.../out.gguf -p "Hello" -n 8
```

Expected: llama.cpp prints the model load summary and emits 8 tokens without
error. Inference coherence is out of scope for the convert acceptance contract
(see ADR-013).

---

## Running `hf2q smoke` (ADR-012 Decision 16)

`hf2q smoke` is the automated end-gate harness. Preflights the environment,
then runs convert + 8-token inference deterministically, asserts transcript
integrity, and lands the transcript under `tests/fixtures/smoke-transcripts/`.

```bash
# Dry-run just runs preflight; useful in CI to catch missing prerequisites.
cargo run --release -- smoke --arch qwen35 --quant q4_0 --dry-run

# Full smoke (requires HF_TOKEN, free disk per arch floor, /opt/llama.cpp built).
cargo run --release -- smoke --arch qwen35 --quant q4_0
cargo run --release -- smoke --arch qwen35moe --quant q4_0
```

**Preflight exit codes (single-line failure mode naming the prerequisite):**

| Exit | Meaning |
|---|---|
| 2 | `HF_TOKEN` missing or empty |
| 3 | Insufficient free disk (`disk_floor_gb + 10 GB` buffer) |
| 4 | `llama-cli` not at `/opt/llama.cpp/build/bin/` or in `$PATH` |
| 5 | hf2q not built in release mode |
| 6 | HF repo unresolvable (private / bad token / network) |
| 7 | Unknown arch — known arches: `qwen35`, `qwen35moe` |
| 8 | Smoke transcript assertion failed (tensor count / n_eval / regression pattern) |

Determinism is real (`--seed 42 --temp 0 --no-warmup`): two fresh runs on the
same host produce byte-identical transcripts.

**CI note:** the full smoke path is NOT invoked by CI (disk + HF + wall-clock
requirements). `tests/smoke_conformance.rs` exercises every preflight exit
code via `assert_cmd` + `env_remove` — zero disk / HF / token dependency.

---

## How P11 catches MTP regressions (manual bisection)

`tests/convert_qwen35_mtp_roundtrip.rs` (ADR-012 Decision 19, landed with
ADR-013 P14 cross-link) converts a synthetic `mtp_num_hidden_layers: 1`
model and asserts the 4 MTP tensors land at the exact GGUF names ADR-013's
loader + llama.cpp expect:

```
blk.{num_hidden_layers}.nextn.enorm.weight        (llama-arch.cpp:449)
blk.{num_hidden_layers}.nextn.hnorm.weight        (llama-arch.cpp:450)
blk.{num_hidden_layers}.nextn.embed_tokens.weight (llama-arch.cpp:448)
blk.{num_hidden_layers}.nextn.eh_proj.weight      (llama-arch.cpp:447)
```

### Bisection: renaming any suffix trips the gate

To confirm the gate is live, introduce a one-letter regression in
`src/backends/gguf.rs:hf_name_to_gguf`:

```diff
-                    "embed_tokens.weight" => Some("nextn.embed_tokens.weight"),
+                    "embed_tokens.weight" => Some("nextn.emb_tokens.weight"),
```

then:

```bash
cargo test --test convert_qwen35_mtp_roundtrip qwen35_mtp_roundtrip
```

Expected failure message names the missing tensor exactly:

```
missing MTP tensor "blk.4.nextn.embed_tokens.weight" in converted GGUF.
Found names: ["blk.4.nextn.emb_tokens.weight", ...]
```

Revert the diff when done.

### Bisection: re-introducing the P4 stub form trips the gate

Earlier P4 emission used `blk.mtp{idx}.nextn.*` as a literal-"mtp" placeholder
for the block index. The P11 gate's negative assertion forbids that — if a
future refactor re-introduces it, the round-trip fails with `P4 stub MTP
placeholder ... should never reach GGUF — see ADR-012 P11`.

---

## Acceptance criteria (shipping contract)

A converted qwen35 / qwen35moe GGUF is accepted when:

1. `.gguf` is structurally valid per hf2q's reader (magic `GGUF`, version 3,
   tensor_count > 0, kv_count > 0).
2. Every metadata key in the ADR-012 catalog is present.
3. Every tensor name follows the ADR-012 P4 naming spec (Decision 8).
4. `llama-cli --model ...out.gguf -p "Hello" -n 8` loads without error.

Inference coherence (sourdough gate, sliding window parity) is delegated to
ADR-013 (Qwen3.5 inference engine).

---

## References

- `docs/ADR-012-qwen35moe-conversion.md` — full decision record.
- `docs/converting-a-model.md` — generic convert-command reference.
- `docs/shipping-contract.md` — overall product contract including qwen35 section.
- `docs/operator-env-vars.md` — HF_TOKEN, HF_HUB_CACHE, and other env vars.
