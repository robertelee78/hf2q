# ADR-063 F1: initial identity inventory

- Date: 2026-10-23
- Work key: `robertelee78/hf2q:ADR-063:F1` (#356)
- Status: In progress. This is an inventory checkpoint, not a complete campaign freeze, model qualification or performance result.
- Scope: Qwen3.8 and DeepSeek-V4 first, then Qwen3.6 and Gemma 4; final acceptance requires separate speed wins for all four.

## Source and public package identities

| Surface | Observed identity | Evidence and limit |
| --- | --- | --- |
| Stock hf2q | `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`, v0.1.27 | Clean primary checkout and published release target agree. No experimental binary is frozen yet. |
| Local mlx-native | `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`, 0.15.1 | Clean native checkout; release target agrees. |
| hf2q native dependency | exact `=0.15.1` registry requirement | `Cargo.toml`, `Cargo.lock`, and `scripts/verify_release_dependency_provenance.sh` agree. |
| Downloaded native crate | SHA-256 `76ce4c8d5773c72554a98020aadd330e566792dd27f843451ac5dd567bb6b5dd` | The public crates.io archive was downloaded and hashed; digest matches Cargo.lock and native GitHub release asset metadata. No build/test was run in this inventory. |
| Published stock hf2q crate | release asset digest `98aa74c97c268286c29ee25643a562025b8480e1837d270f5778f71e0d760985` | GitHub metadata only; downloading and independently checking the stock archive/binary remains required for the campaign. |
| Published stock standalone | release asset digest `1ab3ac7570d8afbda97781542d8da3cd5c8f212964668223e30f496680b32d65` | GitHub metadata only; not a locally executed binary identity. |

Sources: [native v0.15.1 release](https://github.com/robertelee78/mlx-native/releases/tag/v0.15.1), [hf2q v0.1.27 release](https://github.com/robertelee78/hf2q/releases/tag/v0.1.27). Public archive verification used `curl --fail --location` against the crates.io static archive endpoint followed by `shasum -a 256`; compare the result with the locked registry checksum before accepting it.

## Peer identities remain distinct

- Product/historical pin: `e15384a5cb092b080c2a01c0b9e3f8635079d6df` from `data/llama_cpp_pin.txt`. Its earlier reachability problem remains unresolved; this inventory does not replace the pin.
- Existing local peer source: `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`. The expected server executable is absent. Source identity cannot establish executable identity.
- Current upstream source observed through the GitHub commit API: `4231b23dc2c46661d01157aac5245bdcf03ce4b1`, committed 2026-10-23 12:56:59 UTC. This is a candidate campaign reference, not a built or qualified binary. Resolve latest again at actual campaign start, isolate its checkout/build, and freeze its own version output plus binary hash.

The October 22 research screening cutoff must not silently turn the historical/local source into the current-peer runtime arm. The revised ADR distinguishes these clocks.

## Hardware and target inventory

Observed hardware is Apple M5 Max, 137,438,953,472 physical bytes (128 GiB), macOS 27.2 build 26B5112a. `sysctl` and `sw_vers` supplied these values. Metal/compiler identity, live pressure, competing use, thermals and a disjoint runtime memory budget remain campaign inputs; no model was loaded.

Source-directory download metadata records these candidate HF revisions:

| Target | Config download revision | Qualification limit |
| --- | --- | --- |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Download metadata is not full shard integrity, official repository authentication, hf2q conversion provenance or a GGUF identity. |
| Qwen3.6-35B-A3B | `995ad96eacd98c81ed38be0c5b274b04031597b0` | Same limitation; stock weights must not be confused with the default APEX artifact. |
| Gemma 4 26B | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | Same limitation; select the explicit official artifact and quantization before comparison. |
| DeepSeek-V4 | Not established in this inventory | Resolve official source, local availability and conversion provenance before admitting the lane. |

### Official source metadata verification (2026-10-23)

Read-only HTTPS requests to the Hugging Face model revision API resolved each exact revision above to its official repository. Local `config.json` Git-blob hashes (`SHA1("blob " + byte_length + NUL + bytes)`) match the API's blob IDs:

| Repository | Config Git blob | Expected safetensors shards / bytes | Local inventory |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `706cebd746c4b6f2b1d1f892630867acfdfd3df8` | 18 / 55,563,006,776 | All present, lengths match |
| `Qwen/Qwen3.6-35B-A3B` | `9c2aacf8e27267fadb4f6bf8209e7f3c26f5dd66` | 26 / 71,903,776,776 | All present, lengths match |
| `google/gemma-4-26B-A4B-it` | `f42adcca257bb0865c506947671e7d0af52ac9c4` | 2 / 51,612,009,916 | All present, lengths match |

Reproduce metadata lookup with `https://huggingface.co/api/models/OWNER/REPO/revision/REVISION?blobs=true`, using the exact repository/revision pairs in this table and the preceding table. Compare returned `id`/`sha`, the `config.json` sibling's `blobId`, and every safetensors sibling's expected size against local file metadata. These checks read only configuration bytes and file lengths; **no weight-content hash sweep ran**. Correct lengths do not establish correct weight bytes. Full source integrity, tokenizer/template identities and reproducible hf2q conversion remain required. No new model was downloaded.

The canonical launcher defaults are operator entry points, not automatic campaign identities. In particular, the Qwen3.8 default names an abliterated SFT artifact; it cannot substitute for the required vanilla target. The other default filenames similarly do not prove source revision or conversion settings. A directory-name or download-metadata hit is never promoted to a model hash.

## Native and model responsibilities

The native crate already exposes shared-weight batched GGML matrix APIs, dense/expert routing policies, dispatch tracing, GPU timing/counters, and explicit graph/session mechanisms. hf2q consumes those operations through family graph adapters, including its Gemma batched decode body and Qwen execution-evidence dispatcher. S3 and K1 must profile these paths before proposing a new batching implementation. Source ownership and the per-story delivery split are specified by ADR-063; this inventory does not assume an existing API is fast enough for the declared workload.

A winning native change must be reviewed, merged, published and archive-verified first; hf2q then pins the exact registry version/checksum and qualifies its clean build. Final delivery additionally publishes the hf2q crate and signed standalone artifacts. Both repositories' release workflows are required evidence, not interchangeable milestones.

## Remaining F1 work and failure path

1. Restore a valid release-coordination read before any host-heavy build, hash sweep or model load. The native memory MCP currently returns `policy-ledger-anchor-missing`; do not establish a trust anchor or bypass integrity checks without determining whether this is a legitimate legacy ledger.
2. Choose and authenticate all four exact source/artifact identities, prove source shard integrity and hf2q conversion provenance, retain MTP tensors where needed, and freeze templates/rendered tokens and production sampling.
3. Populate the versioned comparison manifest. The initial offline validator and negative controls now live in `scripts/four_agent/`; they require all four models and three arms, actual file hashes, native registry/archive agreement and explicit matched/peer-best settings. Its success deliberately leaves `campaign_qualified` false; receipt semantics and live binding remain separate gates. Bind both repository/source and registry identities, binary version/hash, model/draft hashes, hardware/software/Metal, exact settings and matched/peer-best capabilities. Missing fields must reject campaign admission rather than inherit launcher defaults.
4. Build/identify the isolated current peer after coordination preflight, screen the dated alternative comparators, and record unavailable capabilities explicitly. Do not mutate the existing peer checkout or silently replace the product pin.
5. Independently verify frozen stock and experimental binaries and all public package bytes; freeze the manifest at campaign start. Real model loads additionally require Q5 admission and the applicable qualification gates.

No runtime, conversion, model hashing sweep, Metal test, performance campaign or package publication ran during this checkpoint. F1 remains incomplete.

Validation update: `python3 -m unittest discover -s scripts/four_agent -p 'test_*.py' -v` passes 16 synthetic identity controls. These test changed bytes, missing files/models, baseline/peer/native archive/source/version mismatches and omitted configuration evidence; they do not complete the F1 campaign freeze.
