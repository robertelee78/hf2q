# Four-agent campaign identities (ADR-063 F1)

Requires Python 3.11 or newer. `identity.py` checks a locally frozen campaign manifest and hashes every referenced file. It does not run inference, build software, contact an upstream service or certify performance. Invoke it only after release-coordination preflight: hashing large models is host work too.

```sh
python3 scripts/four_agent/identity.py /path/to/campaign.json
python3 -m unittest discover -s scripts/four_agent -p 'test_*.py' -v
```

Paths in the manifest are absolute or relative to the manifest directory. Keep campaign JSON and referenced model/binary/receipt files outside Git. Do not put credentials in version output, settings or receipt files. Every file identity is an object with `path`, positive integer `bytes` and lowercase SHA-256 `sha256`. Files must exist and their actual length and digest must match. There is no skip-hashing option.

The manifest uses schema `hf2q.four-agent-identities.v1` and these required fields:

| Object | Required fields |
| --- | --- |
| Root | `schema`, `campaign_id`, timezone-qualified `frozen_at`, `coordination_ref`, `hardware`, `engines`, `models` |
| Hardware | `chip` = `Apple M5 Max`, `physical_bytes` = 137438953472, `os_version`, `os_build`, `metal_version`, `compiler_version` |
| Engines | Exactly `stock`, `experiment`, `latest_peer` |
| Every engine | 40-character `source_commit`, captured executable `version_output`, `binary` file identity, `build_receipt` file identity |
| Stock and experiment | `native` object and `dependency_provenance_receipt` file identity |
| Native | 40-character `source_commit`, published `version`, registry `source` = `registry+https://github.com/rust-lang/crates.io-index`, locked `checksum`, downloaded `crate` file identity matching that checksum |
| Experiment | `baseline_commit` equal to stock `source_commit`; explicit `identity_mode: stock_control` if its executable bytes equal stock |
| Latest peer | `resolved_upstream_commit` equal to its built `source_commit`, `upstream_resolution_receipt` file identity |
| Models | Exactly `qwen38`, `deepseek4`, `qwen36`, `gemma4` |
| Every model | `hf_repository`, 40-character `hf_revision`, `variant` = `vanilla`, `quantization`; file identities for `gguf`, `conversion_receipt`, `source_integrity_receipt`, `tokenizer`, `template`, `rendered_tokens`, `production_sampling`; `arms` object |
| Every model's arms | Exactly the three engine arm names, each with `model_sha256` equal to the shared GGUF digest, `matched_settings`, `resolved_production_sampling` and `capability_receipt` file identities, explicit `drafts` list of file identities (empty when no separate draft artifact) |
| Every model's latest peer binding | Additional `peer_best_settings` file identity, separate from matched-settings diagnostics |

The downloaded native archive must also contain a unique regular-file Cargo manifest with the declared package/version and VCS metadata matching the declared clean native source commit; metadata is read without extraction. Stock and experiment can pin different published native versions. Their common baseline is explicit; source ancestry and the exact build inputs must be established by review of build/dependency receipts, not inferred from matching text fields. Different draft artifacts are allowed, but each is hashed and capabilities/settings must justify its use. Models with different GGUF bytes belong in separately labeled native-format diagnostics, not this same-artifact manifest.

A successful result states `offline_file_identities_verified: true` and **always** `campaign_qualified: false`. This distinguishes file integrity from semantic proof. The validator does not authenticate HF ownership, infer conversion settings, parse arbitrary build/provenance receipts, establish Git ancestry, prove upstream recency from a self-declared commit, or execute the recorded version command. Independent review must inspect those hashed receipts. F2/Q2 must additionally bind the frozen manifest to live runtime attestations, exact inputs/settings, memory/thermal and coordination admission, real slot/cohort progress and task/tool outcomes. Q3/Q4 require paired independent four-model speed verdicts.

The existing `scripts/grammar_probe/runtime_identity.py` supplies the live runtime-attestation checks. This module adds the campaign-wide cross-arm/file bindings and does not replace that live check or existing thermal/process guards. Synthetic fixtures in `test_identity.py` exercise rejection paths; their passing result is not model or hardware evidence.

See [SETTINGS.md](SETTINGS.md) for source-bound client wire capture, effective sampling controls, repetition-history differences and the separation between production, matched-diagnostic and peer-best arms. `capture_client.py` provides a model-free localhost capture with an isolated OpenCode profile.

The four GGUF digests must be distinct. Each model's tokenizer, template and rendered-input receipt must have distinct digests, and the peer executable must differ from the hf2q executables. An explicit stock-control experiment is an identity diagnostic, not a candidate speed win.

`rendered_tokens` binds the same aggregate `hf2q.offline-rendered-inputs.v1` receipt for all four models. The validator joins its declared model, effective template and tokenizer hashes to the frozen files, checks each prompt/token sidecar hash, token count and context bound, and requires clean source provenance. DeepSeek must identify its native encoding; other families use GGUF metadata templates. The receipt records implementation, configuration and capture-binary hashes. Those provenance fields still require independent source/build review. Successful validation returns both the canonical manifest digest and checked file observations; the CLI also returns the exact raw manifest-file digest.

The ignored Rust offline capture test shares HTTP tokenization without adding BOS/EOS postprocessor tokens and selects the effective serving template. It accepts a restricted text/tool request subset and rejects unsupported controls and context overflow. It does not load model weights or prove API admission, tool execution or generation quality. Build it from a clean committed tree with `GIT_COMMIT_SHA` set to that commit; retain the exact capture binary and configuration. The checked-in canonical coding requests are synthetic renderer fixtures, not completed coding tasks.
