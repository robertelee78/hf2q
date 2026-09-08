# Hosted resolution Kata — 2026-09-05

Status: investigation and implementation in progress; no serving proof yet.

Base hf2q source: `626a62e31c5b782c0decfdb68386382f9a726684`.
Installed operator executable reports 0.1.20; the checkout is 0.1.21. Relevant
resolver sources are unchanged from tag v0.1.20. The integration checkout is
isolated from the operator's dirty main checkout.

## Hypothesis and measured spike

Hypothesis: the two errors arise from treating filename hints as model role
and runtime quantization identity, with compatibility checked too late.

`CARGO_BUILD_JOBS=2 cargo test --locked --bin hf2q hosted_resolution_kata -- --nocapture`
at the base plus two regression tests: **0 passed, 2 failed**, 0 ignored.

- `hosted_resolution_kata_probes_candidates_before_declaring_ambiguity` fails
  before invoking its metadata probe: two Q8_0 candidates remain ambiguous even
  when the first candidate's runtime contract rejects it.
- `hosted_resolution_kata_bf16_geometry_uses_the_shared_wire_type` fails with
  `unsupported GGML tensor type 30`.

The installed CLI also reproduces the unknown-quant rejection for
`unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q8_K_XL` before repository resolution.

## Exact artifact header evidence

Repository: `unsloth/gemma-4-26B-A4B-it-GGUF`.
Immutable revision: `c099eb48e663fd284577b04978a94ffccb261841`.
Each probe fetched only bytes 0–16777215 with HTTP 206. These are header-prefix
observations, not full-payload digest or inference proofs.

| Filename | Logical bytes | Architecture | File type | Tensor inventory |
|---|---:|---|---:|---|
| `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` | 461766816 | `gemma4-assistant` | 7 | 26 F32, 23 Q8_0 |
| `gemma-4-26B-A4B-it-Q8_0.gguf` | 26859861728 | `gemma4` | 7 | 392 F32, 266 Q8_0 |
| `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` | 27636232928 | `gemma4` | 7 | 392 F32, 258 Q8_0, 8 BF16 |

The UD artifact's BF16 tensors are all in block 29: `attn_k`, `attn_output`,
`attn_q`, `ffn_down`, `ffn_down_exps`, `ffn_gate`, `ffn_gate_up_exps`, and
`ffn_up` weights. Both dense projections and routed expert products must work.

Prefix SHA-256, in table order:

```text
2351145e89fe26988f8b5faea65ab7ebe8e03f172dead27cc8212856a2a1f4c4
9236fdb22e2871c6b8c8886d97be10f8180d896d11004729d2317b4a57c959f3
11cbe24ce3b869f4ad9b0eef7aadf438af637e2ed71d13a22582a610fe11b12e
```

## Reformulation

The fix must preserve exact hosted identity independently of runtime quant,
probe compatibility before ambiguity, and derive admission from runtime
contracts. Name-based MTP exclusions and publisher-specific quant aliases are
rejected approaches. A parser-only BF16 fix is insufficient: the 0.11.2 runtime
cannot read BF16 GGUF. Published mlx-native 0.15.1 contains BF16 GGUF loading,
native dense BF16 capability, and the separate `dense_matmul_id` scalar-expert
primitive. Its quantized expert entrypoint correctly rejects scalar storage;
that rejection does not establish absence of a native scalar-expert route.

Implementation, full regression results, artifact load proof, quality, and
latency evidence will be recorded here as the Kata progresses.

## Follow-up spikes and implementation evidence (2026-09-06)

The first native expert Metal test rejected its own incorrect BF16 fixture
stride: a 2x2 BF16 matrix occupies 8 bytes, not 16. The corrected fixture passes;
the loader derives real expert stride from the original tensor extent divided
by expert count. A separate test now verifies shared per-token inputs with
multiple selected experts, as well as flattened down-projection rows and
repeated expert IDs.

Source review identified a real dense dispatch gap beyond the original two
errors: hf2q's shared dispatcher handled F32/F16 explicitly but routed BF16 into
a quantized API. The new branch consumes the native BF16 buffer through
`dense_matmul_bf16_f32_tensor`. Decode and batched Metal tests compare numeric
outputs and assert original BF16 bits/dtype remain unchanged.

A pre-existing Gemma head policy dequantized and requantized an embedding even
when it was already Q8. The Q8-selected/on-disk-Q8 case now loads the original
blocks directly; this avoids introducing a second quantization into either
requested artifact. It does not change BF16 tensors into Q8 tensors.

Cache review found same-basename collisions between repository subdirectories
and loss of the original subpath during adoption. New destinations use a full
filename digest directory, and sidecars preserve the original filename.
Regression coverage proves separate identities/destinations and exact-selector
reuse. Literal labels spanning multiple runtime file types remain ambiguous;
quality ranking cannot silently choose between those explicitly matched files.

Ownership remains explicit: hf2q owns resolution, identity, model topology and
dispatch; mlx-native owns GGUF scalar storage and native execution. The required
backend functionality is already published in 0.15.1, so no local mlx-native
source patch or unpublished dependency is used. Its changed APIs require sticky
expert-ID status propagation in DeepSeek, retirement of a removed Qwen Q5_K
fused entrypoint, and updated dependency-provenance receipts.

Completed checks at the evolving integration tree (not an immutable release):

- `cargo check --locked --all-targets --all-features`: pass.
- `cargo build --release --locked`: pass; the final commit needs its own recorded binary digest.
- Managed-artifact unit suite: 103 passed, 0 failed.
- Download/admission suite: 78 passed, 0 failed, 1 ignored pinned-header test.
- BF16 filtered suite: 26 passed, 0 failed, including native Metal dense tests.
- Native scalar expert tests: 3 passed, 0 failed, no Metal skips on this host.
- Native Q8 head policy/storage tests: 2 passed, 0 failed.
- Model selector tests: 6 passed, 0 failed.
- Runtime Qwen storage predicate regression: 1 passed.
- Explicit pinned-header invocation: 1 passed, 0 failed, proving both primary
  artifact prefixes admitted and the assistant architecture rejected.
- CLI/build/completion/conversion integration suites: pass; the conversion suite
  reports one existing ignored real-model test.

Pending: full payload download/digests, real-model unary/SSE/tool-result/prefix-cache quality
and latency validation, matched peer runs, and commit/push/merge. The host's
existing DeepSeek workload is preserved while an exclusive validation window
is requested. No full-model success or performance improvement is claimed.

Additional validation: eight synthetic Gemma admission tests passed after the
positive fixture was corrected to include the runtime-required native tool
chat template. A direct BF16 GGUF loader test also passed, comparing original
u16 storage bits with the loaded Metal buffer. The hosted-safe CI sequence
passed through its required checks, after correcting test lock placement to
satisfy the repository's per-test GPU-lock discipline. Its final DeepSeek
subset reported 120 passed, 0 failed, 8 existing ignored tests.

Reference preparation found the existing reference executable reports commit
`5e6a37cb1`, different from both the local source HEAD
`74a7c897f049c17e7080423aa2111776eff6ebbf` and the repository pin
`e15384a5cb092b080c2a01c0b9e3f8635079d6df`. The pin is absent locally; an exact
fetch from the canonical upstream remote returned `not our ref`. The existing
`../../scripts/fixtures/grammars/peer/PROVENANCE.md` also records this unavailable
pin. Any alternative reference run must identify its actual immutable source
and binary digest explicitly; it cannot be claimed as the pinned baseline.

The final integrated locked all-target/all-feature check, release build, and
clean/REFUSED/UNSAFE activation matrix passed. The three new hosted-safe CI
entries (runtime registry, synthetic Gemma admission, Qwen storage admission)
were also executed explicitly and passed (1 + 8 + 1 tests). Full unfiltered
`cargo test --locked` and real-model gates have not run because that wider suite
can load local models while the host's separate DeepSeek workload is active.
The peer reference was built without source modifications in a separate
checkout at `74a7c897f049c17e7080423aa2111776eff6ebbf`; no reference model has
been loaded yet.
