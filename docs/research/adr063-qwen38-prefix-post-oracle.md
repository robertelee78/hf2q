# ADR-063 Qwen3.8 N1 exact prefix post-oracle

Status: model-free replay helper; a passing real-run prefix receipt is separate
acceptance evidence. This helper runs after the model has been unloaded. It is
not an additional model prelaunch requirement or a four-model performance gate.

The helper replays seven archived HTTP requests through the production request
schema, reasoning controls, Qwen template kwargs, renderer and HTTP tokenizer.
It constructs no engine/device, reads no model weight payload and sends no HTTP
requests. It uses the exact exported tokenizer and effective template from the
frozen artifact; it never substitutes the bundled template.

## Frozen inputs

Set `HF2Q_PREFIX_POST_ORACLE_CONFIG` to an absolute JSON configuration path. The
configuration denies unknown fields. Every referenced path must be absolute and
paired with its exact raw-file SHA256. Keep configuration, request archives,
metadata exports and receipts outside Git; archived messages may contain private
source or local paths.

```json
{
  "tokenizer": {"path": "/private/campaign/qwen38.tokenizer.json", "sha256": "RAW_FILE_SHA256"},
  "template": {"path": "/private/campaign/qwen38.template.jinja", "sha256": "RAW_FILE_SHA256"},
  "readiness": {"path": "/private/campaign/readiness.json", "sha256": "RAW_FILE_SHA256"},
  "template_source": "GgufEmbedded",
  "turns": [
    {
      "request": {"path": "/private/campaign/client/turn-0/request.json", "sha256": "RAW_FILE_SHA256"},
      "receipt": {"path": "/private/campaign/client/turn-0/receipt.json", "sha256": "RAW_FILE_SHA256"}
    }
  ],
  "output": "/private/campaign/prefix-post-oracle.json"
}
```

Replace every digest placeholder with a real lowercase SHA256 and supply exactly
seven ordered turn entries, 0 through 6. The output must be a new absolute file
outside every Git checkout ancestor. It is created with private permissions only
after all replay assertions pass.

The archived readiness envelope is `{health, runtime, readyz, ...}` from the
owned N1 supervisor. Its `runtime.measurement_snapshot` supplies the actual
loaded model ID, architecture and tokenizer/template semantic fingerprints.
These differ from the raw-file digest domains: the tokenizer fingerprint hashes
canonical JSON from `tokenizer.to_string(false)`; the template fingerprint hashes
canonical JSON containing the effective template text and its source name.
Replay recomputes these production measurement domains and requires equality.

Each archived client receipt must bind its turn, HTTP status, exact request
bytes/digest and actual usage. The helper requires the N1 text-only three-tool
`auto` request shape, output budget 8192, context 262144 and overflow rejection.
Schemas, resolved thinking enablement and template kwargs stay unchanged;
`preserve_thinking` must remain true. Server-default grammar/GCD and vision must
be inactive. Those restrictions define this oracle's supported lane, not new
universal serving policy.

## Exact predicate

For every turn i, let P_i be the actual rendered prompt token IDs with generation
cue enabled, and S_i the same request rendered with the cue disabled. Both use
the production renderer and `encode(..., false)` HTTP tokenization.

- S_i must be nonempty and a strict token prefix of P_i.
- Actual `usage.prompt_tokens` must equal the length of P_i, and the prompt plus
  declared output budget must fit the fixed context without overflow rewriting.
- Initial cached tokens must be zero.
- For every continuation i=1..6, S_(i-1) must be an exact token prefix of P_i and
  `len(S_(i-1)) <= cached_tokens_i <= len(P_i)`.

All six continuations must pass. Positive cached tokens alone are insufficient.
The receipt records each actual excluded generation-cue token count
`len(P_i)-len(S_i)`; no fixed allowance or averaged result replaces exact checks.
Generated assistant bytes are not compared directly because production rendering
reconstructs historical thinking and tool calls.

## Focused validation

Run only after release coordination confirms that the host is unloaded and a
build is permitted. Use an isolated target directory and locked dependencies.
The pure controls call the same `accept_prefix` function as archived replay;
they do not implement a second tokenization or prefix oracle.

```sh
CARGO_TARGET_DIR=/private/build/hf2q-prefix-oracle \
  cargo test --locked --bin hf2q prefix_post_oracle_tests::prefix_acceptance

HF2Q_PREFIX_POST_ORACLE_CONFIG=/private/campaign/prefix-config.json \
CARGO_TARGET_DIR=/private/build/hf2q-prefix-oracle \
  cargo test --locked --bin hf2q \
  prefix_post_oracle_tests::qwen38_n1_exact_prefix_post_oracle -- --ignored
```

Pure negative controls reject partial positive reuse, changed prior/current
prefixes, an oversized prior prefix, empty/equal/longer/divergent stable cues,
missing or mismatched prompt usage, missing/noninteger/negative cached counts,
an initially warm cache and cache counts exceeding the current prompt. Positive
controls accept the exact minimum prior stable length and complete prompt reuse.
Compilation and control results must be recorded against the exact helper source
before accepting any archived-run replay receipt.

## Limits

The helper trusts the hash-bound readiness and client usage receipts. Independent
review must bind them to the single owned run, exact model/binary/launcher/client
archive and raw responses. A synthetic pure-control pass cannot establish actual
model prefix reuse. A passing replay does not score tool correctness, final patch
quality, output distributions, memory admission, N4 concurrency or speed wins.
Keep `full_prefix_reuse=PendingPostOracle` until actual seven-wire replay passes
and receives independent review. Model-driven protocol completion and final
patch acceptance remain separate obligations.

## Validation recorded for this change

Locked test compilation and both new negative-control tests passed on Apple Silicon with published registry mlx-native0.15.4. Four existing Qwen prompt-history and reasoning-replay checks also passed. Only the new helper file was formatted; its focused tests were recompiled and passed afterward. No model, device or HTTP inference ran in these checks. Actual seven-wire replay remains PendingPostOracle.
