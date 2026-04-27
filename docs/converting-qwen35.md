# Converting Qwen3.5 / Qwen3.6 Models with hf2q

Canonical reference for converting the Qwen3.5 / Qwen3.6 model family.
Two variants are covered: the **27B dense** (`qwen35`) and the **35B-A3B
MoE** (`qwen35moe`). See `docs/ADR-012-qwen35moe-conversion.md` for the
full architectural rationale and `docs/ADR-014-streaming-convert-pipeline.md`
for the streaming-pipeline + peer-parity gate context.

---

## Prerequisites

- `hf2q` built from this repo (`cargo build --release --bin hf2q`).
- A HuggingFace token with access to the target repo, either via
  `HF_TOKEN=hf_xxxx` env var or `~/.huggingface/token`.
- Sufficient free disk space (see "Disk preflight" below).
- For DWQ / imatrix variants: ADR-013's `RealActivationCapture` forward
  driver — already in tree for `qwen35` and `qwen35moe`. **No
  weight-space fallback** when the driver is missing for an
  architecture (per `feedback_never_ship_fallback_without_rootcause.md`).

---

## Disk preflight

hf2q checks available disk space before starting any download. If the
check fails, the download is aborted immediately with a user-actionable
error.

| Model | Minimum free disk |
|---|---|
| Qwen3.5-MoE 35B (`qwen35moe`) | 150 GB |
| Qwen3.5 27B dense (`qwen35`) | 55 GB |
| Other models | 100 GB |

The check uses the HF cache root (`HF_HUB_CACHE` →
`~/.cache/huggingface/hub` → `~/.huggingface/hub`) as the target path.
To use a different disk:

```bash
export HF_HUB_CACHE=/path/on/large/disk
```

Example error when disk space is insufficient:

```
Error: Qwen3.5-MoE 35B requires ≥150 GB free in /home/user/.cache/huggingface/hub; found 80 GB. Free space or change --cache-dir.
```

---

## Shard resumption

hf-hub skips shards that have already been fully downloaded. If a
download is interrupted (e.g. `Ctrl+C`), re-invoking `hf2q` with the
same `--repo` will re-fetch only the in-flight shard and resume from
there. Completed shards are not re-downloaded.

Manual verification: after `Ctrl+C`, inspect
`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/` —
fully downloaded shards are present; the interrupted shard may be
partial. On re-invoke, only the partial shard is re-fetched.

---

## Sidecar files

After writing the `.gguf` (or after writing the safetensors directory),
hf2q copies these files from the HF source directory into the output
directory alongside the model file, byte-identical:

- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `generation_config.json`
- `special_tokens_map.json`

Files missing from the source are silently skipped. The sidecar set is
the same for all model classes (Gemma-4, Qwen3.5-MoE, Qwen3.5 dense).

---

## DWQ variant recommendations

Both `qwen35` and `qwen35moe` are supported through the full DWQ
variant menu (ADR-014 P11-prereq Iter C). All four variants emit
**Q4_K_M-family GGUFs** by default (commit `975a67a`); the legacy
Q4_0-base path is preserved behind `HF2Q_USE_LEGACY_DWQ_Q4_0=1`.

| Variant | Base target | Sensitive target | Recommended for |
|---|---|---|---|
| `dwq-4-6` | Q4_K | Q6_K | **Default starting point**. Best size/quality trade-off; matches the ADR-012 reference deliverable for both `qwen35` and `qwen35moe`. |
| `dwq-4-8` | Q4_K | Q8_0 | Higher fidelity on the sensitive layers; ~30 % larger than `dwq-4-6`. Recommended when latency budget permits and PPL gap to `q8` matters. |
| `dwq-6-8` | Q6_K | Q8_0 | Both buckets at ≥ 6-bit; closest DWQ variant to a `q6_k` baseline with the bit-pair savings on non-sensitive layers. |
| `dwq-2-8` | Q2_K (codec port pending) | Q8_0 | **Currently partial**: sensitive-target tensors quantize correctly under Q8_0; base-target tensors surface a typed `QuantizeError::TensorQuantizeFailed` because the Q2_K codec port has not landed yet. No panics, no silent fallback — see `src/quantize/dwq_k_quantizer.rs::P28` doc. |

For each variant, hf2q runs the **two-pass activation calibration** end
to end (ADR-012 P9 + P9b, shipped 2026-04-25):

1. Emit intermediate F16 GGUF from the in-memory tensor map
   (`backends::gguf::emit_gguf_from_tensor_map`).
2. Construct `RealActivationCapture::new(intermediate_gguf, tokenizer)`
   which loads via `Qwen35Model::load_from_gguf` (ADR-013).
3. Run `quantize::dwq_activation::run_dwq_activation_calibration`
   which generates calibration tokens, runs the CPU forward pass
   through the loaded model, computes per-layer sensitivity, and
   produces an activation-driven sensitive-layer set.
4. Final GGUF (or safetensors directory) is emitted at the
   user-specified output path. The intermediate is dropped via
   `tempfile::TempDir` RAII.

**No weight-space fallback for these architectures** (Decision 13). If
the forward driver fails, calibration surfaces
`CalibrationError::ForwardPassUnavailable` and the convert exits non-zero.

---

## MoE expert handling (`qwen35moe`)

The MoE convert pipeline quantizes each expert's per-tensor weights
(`*_exps.weight` and `*_shexp.weight`) under the same per-tensor
classifier as dense models (`TensorCategory::classify`,
`src/quantize/layer_mix.rs:115-150` ported from
`llama-quant.cpp:115-150`):

- `blk.X.ffn_down_exps.weight` classifies as `FfnDown` and gets the
  `use_more_bits` Q6_K bump on appropriate layers (closes the
  Q4_K_M parity gap on MoE that pre-`c4dcb0e` had).
- `blk.X.ffn_up_exps.weight` and `blk.X.ffn_gate_exps.weight` classify
  as `FfnUp` / `FfnGate` and route to the variant's base target.
- `blk.X.ffn_*_shexp.weight` (shared-expert weights) classify under
  the same canonical category as the per-expert tensors.

The **MoE router gate** `ffn_gate_inp.weight` is always preserved at
original precision. The `should_skip_quantization` predicate
(`src/quantize/layer_mix.rs:281`) matches `llama-quant.cpp:307`:

```rust
pub fn should_skip_quantization(tensor_name: &str) -> bool {
    tensor_name.contains("ffn_gate_inp.weight")
}
```

`VariantKQuantizer` honours the predicate via passthrough — the router
gate is small (`[n_experts, hidden_size]`) and quantizing it would
introduce routing noise out of proportion to the size saving.

---

## Converting Qwen3.6-35B-A3B (MoE variant)

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-4-6 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/out.gguf
```

**Expected output size:** 18–22 GB (DWQ-4-6 of a 35B MoE; expert-merged).

Alternative with 4-bit base / 8-bit sensitive layers:

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-4-8 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/out.gguf
```

**Expected output size:** 22–28 GB.

For a safetensors directory (mlx-lm / serve-loader consumer):

```bash
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format safetensors \
  --quant dwq-4-6 \
  --shard-size-gb 5.0 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/
```

`--format safetensors` with a quantized variant emits the mlx-lm-style
directory layout (`model-NNNNN-of-MMMMM.safetensors` +
`model.safetensors.index.json` + injected `config.json` with the
mlx-lm `quantization` block). See `docs/converting-a-model.md`
"Safetensors directory layout" for the on-disk schema.

---

## Converting Qwen3.5-27B (dense variant)

```bash
hf2q convert \
  --repo <qwen35-dense-repo> \
  --format gguf \
  --quant dwq-4-6 \
  --output models/qwen35-27b-dense-dwq46/out-dwq46.gguf
```

Replace `<qwen35-dense-repo>` with the HuggingFace repo ID of the 27B
dense model (e.g. `Qwen/Qwen3.5-27B-Instruct` or a fine-tune).

**Expected output size:** 14–18 GB (DWQ-4-6 of a 27B dense).

---

## ADR-012 reference DWQ artefacts + P11 re-emit plan

ADR-012's closure milestone (commit cohort `38d2f3c`, 2026-04-26)
shipped four reference DWQ GGUFs:

- `qwen3.6-27b-dwq46.gguf`
- `qwen3.6-27b-dwq48.gguf`
- `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf`
- `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf`

ADR-014 P11 re-emits these four GGUFs (and four DWQ safetensors twins)
through the post-Iter-C streaming pipeline so the artefacts incorporate
the K-quant base from the start instead of the legacy Q4_0 base. The
re-emit is **pending hardware** as of 2026-04-27 — it requires the
M5 Max apex MoE GPU + ~150 GB free disk + a Metal-validated llama.cpp
build that supports the `qwen35moe` MoE expert-routing kernel for
post-emit verification.

The forward-pointer for the actual numbers:

- **Peer-parity gate results** will land at
  `docs/peer-parity-results-<YYYY-MM-DD>.md` (date is the P11 close
  date; emitted by `tests/peer_parity_gates.rs::write_results_to_dated_doc`).
- **Per-cell verdicts** (8 cells: 27B dense × 4 + apex MoE × 4) will
  surface as the markdown table from
  `tests/peer_parity_gates.rs::emit_markdown_table`.
- **Final P12 close** (this doc's next revision) will replace the
  forward-pointer block above with a "Re-emitted artefacts" block
  citing the new commit hash + the per-cell PPL / wall / RSS numbers
  from the gate run.

Until P11 closes, the four ADR-012 reference GGUFs in `models/`
remain the shipping artefacts.

---

## Manual smoke test

After conversion, verify that llama.cpp can load the file without errors:

```bash
llama-cli --model models/.../out.gguf -p "Hello" -n 8
```

Expected: llama.cpp prints the model load summary and emits 8 tokens
without error. Inference coherence (sourdough gate, sliding-window
parity) is **out of scope** for the convert acceptance contract — see
ADR-013.

---

## Running `hf2q smoke` (ADR-012 Decision 16)

`hf2q smoke` is the automated end-gate harness. Preflights the
environment, then runs convert + 8-token inference deterministically,
asserts transcript integrity, and lands the transcript under
`tests/fixtures/smoke-transcripts/`.

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

Determinism is real (`--seed 42 --temp 0 --no-warmup`): two fresh runs
on the same host produce byte-identical transcripts.

**CI note:** the full smoke path is NOT invoked by CI (disk + HF +
wall-clock requirements). `tests/smoke_conformance.rs` exercises every
preflight exit code via `assert_cmd` + `env_remove` — zero disk / HF /
token dependency.

### Smoke against an already-converted GGUF (skip the convert step)

If you already have a converted `.gguf` on disk (e.g. produced by an
earlier `hf2q convert` run), you can skip the convert step entirely
and only validate the inference + transcript half of the pipeline:

```bash
# 1. Symlink the canonical filename smoke expects (`{arch}-{quant}.gguf`).
ln -sf qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf \
       /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen35moe-q4_0.gguf

# 2. Run smoke with --skip-convert + --convert-output-dir pointing at the
#    same dir as --local-dir.
./target/release/hf2q smoke \
    --arch qwen35moe \
    --skip-convert \
    --local-dir /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex \
    --convert-output-dir /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex
```

`--local-dir` skips the HF_TOKEN preflight (exit code 2), and
`--skip-convert` skips the convert subprocess. Disk floor is still
checked; preflight exit code 3 fires if free space <
`disk_floor_gb + 10 GB` even though no convert is actually run.

**Wall-clock budget on M5 Max** (apex MoE Q4_0, ~25 GB GGUF):
empirically **>10 minutes** for the 8-token decode (validated
2026-04-25). Even with `/opt/llama.cpp/build/bin/llama-cli` having
Metal kernels available, the apex MoE path appears to fall back to
single-thread CPU under heavy memory paging (~4M syscalls/sec,
single-core saturation, 30 GB RSS). **Apex-scale q4_0 smoke validation
is impractical without further tuning** (e.g. explicit `-ngl 99` + a
Metal-validated llama.cpp build that supports the `qwen35moe` MoE
expert-routing kernel).

For now, real-model q4_0 smoke transcripts at apex scale are deferred:
the smoke pipeline DESIGN is correct (synthetic small-arch testing in
`tests/smoke_conformance.rs` covers all preflight + dispatch
contracts); the GAP is environment / kernel availability for inference
at the apex size class on this hardware. Smaller real models would
complete the smoke in seconds — no design change required.

---

## How P11 catches MTP regressions (manual bisection)

`tests/convert_qwen35_mtp_roundtrip.rs` (ADR-012 Decision 19, landed
with ADR-013 P14 cross-link) converts a synthetic
`mtp_num_hidden_layers: 1` model and asserts the 4 MTP tensors land at
the exact GGUF names ADR-013's loader + llama.cpp expect:

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

Earlier P4 emission used `blk.mtp{idx}.nextn.*` as a literal-"mtp"
placeholder for the block index. The P11 gate's negative assertion
forbids that — if a future refactor re-introduces it, the round-trip
fails with `P4 stub MTP placeholder ... should never reach GGUF — see
ADR-012 P11`.

---

## Acceptance criteria (shipping contract)

A converted `qwen35` / `qwen35moe` GGUF is accepted when:

1. `.gguf` is structurally valid per hf2q's reader (magic `GGUF`,
   version 3, `tensor_count > 0`, `kv_count > 0`).
2. Every metadata key in the ADR-012 catalog is present.
3. Every tensor name follows the ADR-012 P4 naming spec (Decision 8).
4. `llama-cli --model out.gguf -p "Hello" -n 8` loads without error.

Inference coherence (sourdough gate, sliding-window parity) is
delegated to ADR-013 (Qwen3.5 inference engine).

The full table — including MTP tensors, mmproj emission, sidecar set,
and the smoke-harness gate — lives at `docs/shipping-contract.md`
"Qwen3.5 / Qwen3.6 conversion acceptance".

---

## References

- `docs/ADR-012-qwen35moe-conversion.md` — full decision record for the
  Qwen3.5 / Qwen3.6 convert pipeline.
- `docs/ADR-014-streaming-convert-pipeline.md` — streaming pipeline +
  peer-parity gate matrix (8 cells, 4 dense + 4 MoE).
- `docs/converting-a-model.md` — generic convert-command reference (full
  17-variant menu, format options, env vars, error catalog).
- `docs/calibrator-onboarding.md` — developer guide for adding new
  Calibrator implementations.
- `docs/shipping-contract.md` — overall product contract including
  qwen35 acceptance section + peer-parity gates.
- `docs/operator-env-vars.md` — `HF_TOKEN`, `HF_HUB_CACHE`,
  `HF2Q_USE_LEGACY_DWQ_Q4_0`, and other env vars.
