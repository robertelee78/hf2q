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
| Downloaded stock hf2q crate | SHA-256 `98aa74c97c268286c29ee25643a562025b8480e1837d270f5778f71e0d760985` | Downloaded crates.io archive digest matches release metadata. Embedded VCS commit matches stock main; packaged Cargo.toml pins native =0.15.1 and packaged Cargo.lock matches the verified native checksum. Binary identity remains separate. |
| Published stock standalone | release asset digest `1ab3ac7570d8afbda97781542d8da3cd5c8f212964668223e30f496680b32d65` | GitHub metadata only; not a locally executed binary identity. |

Sources: [native v0.15.1 release](https://github.com/robertelee78/mlx-native/releases/tag/v0.15.1), [hf2q v0.1.27 release](https://github.com/robertelee78/hf2q/releases/tag/v0.1.27). Both native and hf2q public archive verification used `curl --fail --location` against the crates.io static archive endpoint followed by `shasum -a 256`; compare the result with the locked registry checksum before accepting it.

## Peer identities remain distinct

- Product/historical pin: `e15384a5cb092b080c2a01c0b9e3f8635079d6df` from `data/llama_cpp_pin.txt`. Its earlier reachability problem remains unresolved; this inventory does not replace the pin.
- Existing local peer source: `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`. The expected server executable is absent. Source identity cannot establish executable identity.
- Current upstream source observed through the GitHub commit API: `4231b23dc2c46661d01157aac5245bdcf03ce4b1`, committed 2026-10-23 12:56:59 UTC. This is a candidate campaign reference, not a built or qualified binary. Resolve latest again at actual campaign start, isolate its checkout/build, and freeze its own version output plus binary hash.

The October 22 research screening cutoff must not silently turn the historical/local source into the current-peer runtime arm. The revised ADR distinguishes these clocks.

### Isolated current-peer executable (2026-10-23)

An isolated shallow checkout at `4231b23dc2c46661d01157aac5245bdcf03ce4b1` now has a successfully built server; the existing peer checkout and accepted product pin remain unchanged. Configuration: `cmake -S . -B build-adr063 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON`; build: `cmake --build build-adr063 --target llama-server -j 6`. CMake detected the Metal backend and ggml 0.27.0. SDK: 27.0; compiler: AppleClang 21.0.0.21000334, Darwin arm64.

Executable SHA-256: `98ee8ebf20710ff9b1752b0a4807031b454e927e6aabbaf9e256205927178695`. Its own `--version` reports `0.7.0-dev (build 1, commit 4231b23)`. The build number is 1 because this is a shallow checkout; identify it by the immutable source and binary hashes, not by treating that count as an upstream release number. The local build receipt additionally freezes CMakeCache bytes, dynamic linkage and 25 built runtime-library file hashes. Later campaign admission must recheck those files, not only the main executable.

The executable's `--help` advertises `draft-mtp`, `draft-dflash`, `draft-dspark`, explicit server slot count, continuous batching, prefix-cache reuse and a slots-monitoring endpoint. This is an executable capability inventory, not a successful model-level compatibility or speed result. Freeze matched-settings and peer-best configurations only after exact-artifact probes. No model was loaded and no peer test/performance suite ran in this build.

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

## Existing managed artifact verification (2026-10-23)

After the operator identified the managed model directory, read-only inventory found reusable converted artifacts. No replacement download or conversion was needed for these three candidates. Sequential SHA-256 reads of each complete GGUF matched its adjacent schema-v3 conversion receipt and byte length:

| Candidate | Bytes | Verified GGUF SHA-256 | Receipt converter commit / quantization |
| --- | --- | --- | --- |
| Vanilla Qwen3.8-27B | 16,810,714,976 | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` | `9c42dc1fc19b4dbe94c63c6c4bfef7f990c8f856` / Q4_K_M |
| Official Gemma 4 26B A4B IT | 16,796,018,144 | `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53` | `2c5931c3ddbcc141de40fe40e24dae3af9177321` / Q4_K_M |
| DeepSeek-V4-Flash-0731 | 107,431,343,264 | `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` | `64a2344f2b519955e7bc7a13b48c450a7caa2901` / deepseek4-agentic-q2 |

Receipts bind the official source revisions listed above; DeepSeek's full revision is `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, also resolved independently through the official HF API. These are verified file/receipt bindings, not independent proof of every historical conversion operation or model quality.

The signed stock v0.1.27 standalone was downloaded from its release. Its SHA-256 is `1ab3ac7570d8afbda97781542d8da3cd5c8f212964668223e30f496680b32d65`, strict codesign verification passes, and execution of `--version` reports `hf2q 0.1.27`. Static `info` with an isolated state root, inflight-batched scheduling and four slots passes on the Qwen3.8 candidate: 866 tensors, 49 metadata entries, embedded tokenizer/template and required directory/shape checks. No tensor decoding/upload or Metal initialization occurred. Its full-context static estimate is 52.56 GiB model plus four-slot KV before scratch/allocator overhead; this is not Q5 memory admission.

Directory-only GGUF inspection finds one next-token layer in both existing Qwen3.8 and stock Qwen3.6 candidates, including explicit next-token projection/norm tensors. The DeepSeek candidate has no next-token metadata or explicit MTP/DSpark tensor names; its conversion receipt says 4,705 DSpark tensors were excluded from the base GGUF. F1/D2 must qualify compatible separate draft artifacts or record the unavailable arm; do not silently handicap or invent peer capabilities. The existing stock Qwen3.6 artifact lacks an adjacent conversion receipt in the inspected directory, so its provenance needs recovery or a reproducible conversion before campaign acceptance.

All 18 vanilla Qwen3.8 source-weight shards were additionally hashed against the official pinned revision's LFS SHA-256 identities and matched. This strengthens that source's earlier size-only inventory. Other source-weight content verification and complete tokenizer/template receipt review remain open.

The operator disabled the memory-tool workflow. From that point, coordination uses current GitHub Actions/PR evidence plus direct host process, listener and memory checks; do not invoke the disabled tooling as a prerequisite. Before the sequential hashes, no local model/runtime/compiler process or test-model listener was observed, GitHub showed hosted CI only, and memory pressure was low. Recheck before each heavy action. Runtime experiments still require Q5 and the applicable independent-QE gates.

### Fresh Qwen3.6 provenance (2026-10-23)

The stock hf2q 0.1.27 signed executable completed a new conversion from existing official `Qwen/Qwen3.6-35B-A3B` source revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, using `--source-repo`, `--source-revision`, `--quant q4_k_m`, `--text-only` and an isolated state root. The original receipt-less artifact remains untouched. No source or prequantized artifact was downloaded.

The native converter verified source bytes and emitted a schema-v3 receipt identifying converter commit `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`. Independent post-conversion file hashing matches the receipt: 21,712,410,208 bytes, SHA-256 `815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220`. Static `info` with four inflight-batched slots passes without initializing Metal. This is the Qwen3.6 campaign candidate; final runtime/capability/quality qualification remains open.

MTP was retained. The converter prints a legacy warning that stable peers lack a Qwen MTP loader. That warning is not a current-peer capability result: at frozen peer commit `4231b23`, `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` contain explicit next-token tensor loading, and `common/speculative.cpp` contains Qwen MTP handling. Do not strip MTP to satisfy the old warning; probe exact-artifact compatibility after Q5 admission. Conversion and subsequent hashes ran sequentially with no co-resident inference model and no observed memory-pressure breach.

### Client wire capture and settings contract (2026-10-23)

Installed OpenCode reports v2.1.1, independently of the earlier tool-inventory version. Its currently configured local Qwen3.8 SFT provider/model was copied into an isolated profile with routing/credentials replaced by a synthetic localhost stub. The original configuration was not changed and no inference model ran. Both an initial capture and a reproduction using `scripts/four_agent/capture_client.py` observed a separate tool-free title request followed by the actual agent request: 12 tools, streaming enabled, `max_completion_tokens=8192`, and omitted temperature/top-p/top-k/seed/penalty fields. This is not capability evidence for the vanilla target; target-specific configurations still need capture.

The source-bound effective settings and matched/peer-best distinction are documented in `scripts/four_agent/SETTINGS.md`. hf2q's request handler resolves the observed omitted sampling controls to greedy temperature 0, top-p 1 and top-k 0, while the current peer has different defaults. Qwen repetition-history handling also differs across the two sources; equal penalty numbers are not enough to declare a matched distribution. No production setting is silently changed to manufacture agreement or speed.

Validation: the real OpenCode client completed the synthetic localhost exchange twice; an in-checkout output directory was rejected before creation. Two additional subprocess/HTTP boundary controls prove that source credentials/plugins/unrelated providers and secret environment variables do not reach the isolated profile, and that a timed-out owned client is reaped without a success receipt. The complete focused suite now passes 18 tests. These are harness controls, not a real-model or four-agent result.

## Remaining F1 work and failure path

1. Use the operator-authorized direct coordination path: inspect live Actions jobs and local processes/listeners/memory before every heavy operation. The earlier memory-tool trust-anchor failure is historical context; the operator disabled that tooling, and it must not block or be invoked for subsequent work.
2. Choose and authenticate all four exact source/artifact identities, prove source shard integrity and hf2q conversion provenance, retain MTP tensors where needed, and freeze templates/rendered tokens and production sampling.
3. Populate the versioned comparison manifest. The initial offline validator and negative controls now live in `scripts/four_agent/`; they require all four models and three arms, actual file hashes, native registry/archive agreement and explicit matched/peer-best settings. Its success deliberately leaves `campaign_qualified` false; receipt semantics and live binding remain separate gates. Bind both repository/source and registry identities, binary version/hash, model/draft hashes, hardware/software/Metal, exact settings and matched/peer-best capabilities. Missing fields must reject campaign admission rather than inherit launcher defaults.
4. Build/identify the isolated current peer after coordination preflight, screen the dated alternative comparators, and record unavailable capabilities explicitly. Do not mutate the existing peer checkout or silently replace the product pin.
5. Independently verify frozen stock and experimental binaries and all public package bytes; freeze the manifest at campaign start. Real model loads additionally require Q5 admission and the applicable qualification gates.

No inference runtime, Metal execution, performance campaign or package publication ran during this checkpoint. The fresh Qwen3.6 conversion, source/model hashing and static GGUF inspection are documented above. F1 remains incomplete.

Validation update: `python3 -m unittest discover -s scripts/four_agent -p 'test_*.py' -v` passes 16 synthetic identity controls. These test changed bytes, missing files/models, baseline/peer/native archive/source/version mismatches and omitted configuration evidence; they do not complete the F1 campaign freeze.


### Reviewed capture provenance update (2026-10-23)

The clean committed renderer capture at `d40521439b3449e363407cd83823975953f4c2ed` passed 28 cases: cold plus six canonical tool-result continuations for each of four models. Its receipt records source cleanliness, configuration SHA-256 `dd1a2d6ece9cc1621f822d3e742471112e3401b95a206aa33613dfde8b734870`, and capture-binary SHA-256 `c2ace022f3f7cb639323d766af475ae09daea924e3b7b31286aaf4b25eac29b2`. The exact binary, configuration, aggregate receipt and prompt/token sidecars are retained in private campaign evidence. Raw model payloads were not rehashed by the renderer; separate full-GGUF verification establishes those identities.

The helper uses the production HTTP tokenizer with postprocessor special-token insertion disabled and the effective serving template, including DeepSeek's native pinned encoding. Two focused Rust controls verify BOS/EOS postprocessor behavior and rejection of foreign embedded DeepSeek template selection. These are offline synthetic renderer fixtures, not successful generated coding tasks, full API admission or F2 runtime qualification. Earlier captures were superseded conservatively after review; the current DeepSeek artifact's embedded template matched the native template, so the review did not prove that its earlier token stream was wrong.

The 34 Python controls pass after review fixes for nested provider overrides, ancestor-checkout output isolation, subprocess descendant cleanup, setup/thread failure cleanup, serialized request logs and manifest/sidecar bindings. Independent Opus review found a further raw-versus-semantic configuration identity defect in Python equality (`0 == 0.0`, `1 == True`); commit `130909ae` compares canonical JSON bytes and adds both negative controls. The manifest records raw and canonical digests separately. An independent Opus follow-up confirmed that this specific blocker is resolved; it did not certify full F1. Remaining nonblocking capture improvements are tracked below rather than promoted to guarantees.

Capture limitations: endpoint redirection is not an outbound network sandbox; process cleanup covers the owned process group and cannot cover descendants that create another session. Request classification must be inspected to establish an agent request rather than a title-only exchange. Executable identity must be frozen separately from client version text. A future hardened capture should authenticate the localhost stub, retain rejected-request counts and fail on logging errors; none of these fixtures is performance evidence.


### Supplemental comparator and peer-pin dispositions

The [independent comparator source screen](adr063-f1-comparator-dispositions-2026-10-23.md) records BaseRT's `.base` conversion route, Kapitsa's Qwen import/precision changes and the separate MLX core/serving-adapter identities. They remain source-inspected supplemental arms with explicit applicability and later quality probes. None replaces the mandatory latest-peer same-GGUF arm.

The [peer-pin replacement proposal](adr063-f1-peer-pin-proposal-2026-10-23.md) preserves the unreachable historical pin and proposes reachable source `4231b23dc2c46661d01157aac5245bdcf03ce4b1` subject to quantizer/fixture and runtime qualification. It identifies hard-coded shared-checkout paths, unkeyed cached references and silent test skips that must be resolved before adoption. The current product pin remains unchanged. The isolated server build proves executable identity; it does not prove quantizer or four-family runtime equivalence.
