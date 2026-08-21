# Converting a model with hf2q

This is the operator reference for the current Rust-native `hf2q convert`
command. The converter emits GGUF; it does not delegate downloading,
conversion, or quantization to Python, `hf`, llama.cpp, or mlx-lm.

For architecture-specific behavior, also see:

- Qwen3.5/Qwen3.6: `docs/converting-qwen35.md`
- the conversion architecture and evidence contract: ADR-033
- the end-user installation and setup journey: ADR-045

## Quick start

Resolve a public Hub repository to an immutable commit, download and verify
the exact source inventory, then convert it:

```bash
hf2q convert google/gemma-4-26b-a4b-it \
  --quant q4_k_m \
  --output models/gemma-4-26b-a4b-it-q4_k_m.gguf
```

Convert an existing local directory by using an existing path or explicit
path syntax:

```bash
hf2q convert ./models/google-gemma-4-26b-a4b-it \
  --quant q4_k_m \
  --output models/gemma-4-26b-a4b-it-q4_k_m.gguf
```

Both successful remote-source forms write the GGUF and a sibling
`<output>.receipt.json`. The receipt binds the original reference, canonical
repository identity, exact 40-hex revision, sorted source files and local
SHA-256 values, converter commit, selected quantization, and output bytes.

## Synopsis

```text
hf2q convert [OPTIONS] --quant <QUANT> --output <OUTPUT> [HF_DIR]
```

Exactly one source is required:

- positional `[HF_DIR]`, which may be an explicit local path or a Hub model
  reference; or
- `--repo <REFERENCE>`, retained as a compatibility spelling for a remote
  model reference.

Run `hf2q convert --help` for the complete current flag surface.

## Local and remote source classification

An existing path, absolute path, `./...`, `../...`, `.` or `..` is local.
Non-path positional text is parsed as a Hugging Face model reference. Use
explicit path syntax for a local directory that does not exist yet; otherwise
`owner/repo` intentionally means a remote identity.

Accepted remote forms are:

```text
owner/repository
https://huggingface.co/owner/repository
https://huggingface.co/owner/repository/tree/<revision>
https://huggingface.co/owner/repository/blob/<revision>/<filename>
https://huggingface.co/owner/repository/resolve/<revision>/<filename>
```

File-specific `blob` and `resolve` URLs are structurally recognized for the
shared input identity grammar, but model conversion rejects them because
conversion requires the repository's complete index-selected source set.

`--revision <REVISION>` may name a branch, tag, or exact commit. It must equal
any URL-embedded revision. hf2q asks the official Hub endpoint for repository
information and seals the result to the returned exact 40-hex commit before
any selected file transfer. `HF_ENDPOINT` cannot redirect this production
path to another origin.

For a pre-downloaded local directory that still needs remote provenance, use
an exact immutable revision:

```bash
hf2q convert ./models/example \
  --source-repo owner/repository \
  --source-revision 0123456789abcdef0123456789abcdef01234567 \
  --quant q4_k_m \
  --output models/example-q4_k_m.gguf
```

## Native Hub download and integrity

Remote conversion uses the standard `hf-hub` cache and token discovery:

1. `HF_TOKEN`
2. `HUGGING_FACE_HUB_TOKEN`
3. `~/.cache/huggingface/token`
4. `~/.huggingface/token`

The cache directory follows `HF_HUB_CACHE`, then `HF_HOME`, then
`XDG_CACHE_HOME`, then `~/.cache/huggingface/hub`. Tokens are never copied
into hf2q configuration or receipts.

Before each selected transfer, hf2q requires the file metadata to name the
already-resolved commit and a supported immutable identity. Safetensors must
be LFS objects with a SHA-256 identity. Git-managed configuration/tokenizer
assets are verified by their canonical Git blob SHA-1 and then recorded in
the conversion receipt with a local SHA-256. Same-size rewrites fail.

For sharded checkpoints, `model.safetensors.index.json` is authenticated and
bounded before parsing. Its JSON structure, duplicate tensor names, tensor
count, paths, and selected shard count are bounded. Only shards named by its
`weight_map` are downloaded; unrelated safetensors, `.bin`, ONNX, and
pre-quantized GGUF artifacts have no conversion authority. A monolithic model
must contain exactly the selected `model.safetensors` source file.

Current hostile-input bounds are:

- repository inventory: at most 4,096 files;
- selected relative path: at most 1,024 bytes and 64 components;
- safetensors index: at most 16 MiB and 262,144 tensor entries;
- configuration and other small metadata: at most 16 MiB each; and
- tokenizer/vocabulary assets: at most 512 MiB each.

The accepted Qwen3.8 reference defaults to the ADR-044 accepted revision.
Other references default to `main`, which is still resolved to an immutable
commit before transfer.

## Quantization

`--quant` is currently required. Supported standard names include:

```text
f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0
q2_k q3_k_s q3_k_m q3_k_l q4_k_s q4_k_m
q5_k_s q5_k_m q6_k iq4_nl
```

MoE architectures also expose the independently defined APEX tiers
`apex-quality`, `apex-balanced`, `apex-compact`, `apex-mini`, and the
imatrix-backed `apex-i-*` variants. DeepSeek-V4 additionally has its explicit
`deepseek4-agentic-q2` profile. Unsupported or reserved names fail with a
typed explanation; there is no approximate fallback.

I-tier APEX conversion accepts either `--imatrix <FILE>` or an in-process
`--imatrix-corpus <NAME>` for architectures whose native calibration driver
has landed. See `hf2q convert --help` for the exact supported corpus surface.

## Output and projector mode

`--output <PATH>` names the text GGUF and is required. For a supported
multimodal source, the default command also writes an F16 projector beside it
as `<output-stem>-mmproj.gguf`. `--mmproj-output <PATH>` selects a different
name in the same directory. Each remote-source artifact has its own receipt,
and the text GGUF binds the exact projector digest.

Pair publication is one recoverable transaction. hf2q stages and fully syncs
both artifacts, assigns the same generation UUID to their GGUF metadata, and
publishes the projector and receipts before publishing the text GGUF as the
commit marker. A private sibling journal and backup preserve the entire old
pair until that commit. An ordinary error rolls back immediately; after a
process or machine interruption, the next conversion recovers the journal
before starting. Serving takes the matching shared lock and rejects mixed
generations or projector bytes that do not match the digest embedded in the
text artifact. Older locally converted pairs keep working, but any embedded
projector digest is still enforced even when no remote-source receipt exists.

`--text-only` explicitly suppresses the companion projector. `--mmproj`
retains the projector-only expert path for repairing or replacing a sidecar;
it does not mean the ordinary multimodal workflow needs two invocations.
Unsupported projector families and incomplete multimodal sources—including
vision tensors without a usable `vision_config`—fail before writing rather
than silently dropping vision.

`--dry-run` creates no GGUF. For a multimodal source it reports both planned
paths while the detailed tensor/type/byte plan remains the text plan. A remote
dry run can still resolve, download, and hash the complete source because the
plan must be based on authenticated bytes.

## Disk preflight and resumability

Before Hub lookup or a large transfer, the current downloader checks the
filesystem containing the Hub cache. Existing class floors are 150 GiB for
Qwen 35B MoE sources, 55 GiB for dense 27B Qwen sources, and 100 GiB for other
models. These are conservative transfer safeguards; the operator still chooses
the model, output path, and quantization explicitly or through `hf2q setup`'s
consumed default.

`hf-hub` reuses complete cache objects. An interrupted in-flight object is
retried by the client; already completed objects are not downloaded again.
hf2q never deletes source data from the shared Hub cache.

## Current boundary

The canonical identity parser, immutable resolution, exact selected download,
integrity checks, receipt schema v3, and automatic source-bound multimodal
pairs have landed. Model selection, revision, and text output path remain
explicit operator choices; the projector path is deterministic unless
overridden. Setup can provide a default quant selector, but it does not
download, convert, register, retain, or delete model files. Unsupported inputs
fail visibly rather than being routed through an implicit orchestration or
external tool path.
