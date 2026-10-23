# ADR-063 comparison settings and client capture

Status: source-bound configuration design, not a certified campaign or model-quality result. Baseline hf2q source: `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`; current peer source: `4231b23dc2c46661d01157aac5245bdcf03ce4b1`. Revalidate these settings at each frozen campaign revision.

## Observed client request

An isolated copy of the operator's selected local provider/model was exercised using installed OpenCode **v2.1.1**, a synthetic prompt and a localhost HTTP stub. No inference model ran and the original configuration was not edited. The configured model was the abliterated SFT Qwen3.8 alias; that alias is not a substitute for the required vanilla model. This capture establishes transport/default behavior for that configuration, not capability or task quality for another model.

The client made a separate tool-free title request, followed by the actual agent request. The agent request used `/v1/chat/completions` and contained:

- `model`, `messages`, `stream: true`, `stream_options`;
- 12 tool definitions;
- `max_completion_tokens: 8192`;
- no `temperature`, `top_p`, `top_k`, `min_p`, repetition/frequency/presence penalties or seed.

A second run using the checked-in capture utility reproduced those wire keys, tool count and output budget. Full request bytes vary with generated IDs and session context; do not require their digests to match across independent client invocations. Preserve each invocation's exact request hash and identity. Do not mistake the title request for the agent workload.

At the pinned hf2q source, `src/serve/api/handlers.rs` resolves omitted temperature/top-p/top-k to 0/1/0, frequency/presence/min-p to 0, and leaves seed absent. It gives `max_completion_tokens` precedence over `max_tokens`. Thus the observed configuration effectively uses greedy sampling with an 8192 completion-token budget, not an assumed nonzero temperature or a historical 2048 default. Client/model configurations that actually supply other values need separate captures.

Family defaults come from `src/serve/operator_settings.rs::family_serve_profile` and canonical launchers:

| Model family | Repetition penalty | Fresh-turn thinking budget | Tool-turn thinking budget |
| --- | --- | --- | --- |
| Qwen3.8 / Qwen3.6 | 1.05 | 2048 | 512 |
| Gemma4 | 1.05 | none | none |
| DeepSeek-V4 | 1.0 | none | 512 |

The Gemma row has no thinking budget for either turn type; its template has no thinking channel. Request controls override startup controls, which override family built-ins. Freeze the actual resolved runtime controls rather than relying solely on this table. The capture above did not establish all four model-specific client configurations.

## Controlled and peer-best arms

Freeze separate settings files per model/engine and label their purpose:

1. **Production own-output continuation:** capture the real client's request and resolve all omitted controls for each engine. Preserve the user-facing thinking/tool policies, output budget and task semantics; score actual tool/coding outcomes. Different returned token streams are not token-matched timing evidence.
2. **Matched-settings diagnostic:** explicitly declare temperature/top-p/top-k/min-p, penalties and their history scopes, context, output budget, tool encoding, reasoning policy, cache priming and speculation. Equal scalar values do not prove equal semantics. Reject an unresolved difference rather than call the lane matched.
3. **Latest-peer best:** select the best documented valid configuration under the same task/quality/fairness/memory constraints. Screen available MTP/draft modes; retain MTP tensors and hash separate draft weights. This is the mandatory peer-performance arm. A matched diagnostic cannot silently replace it.

The peer executable's default help output reports temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, repetition penalty 1.0 and a 64-token repetition window. Therefore an unchanged request omitting controls does not imply identical effective settings. An explicitly greedy diagnostic must set the corresponding effective values, not inherit those defaults. Production peer-best settings still require capability/quality qualification; these observations are not a recommendation to disable valid optimizations.

### Repetition history is a material semantic difference

At the pinned hf2q source, Qwen `sample_logits_qwen35`/constrained sampling receives generated-token history; `sampler_pure::apply_repetition_penalty` penalizes each unique token from that supplied history. At the pinned peer source, `tools/server/server-context.cpp::init_sampler` accepts prompt tokens into the sampler, and `common/sampling.cpp::common_sampler_accept` passes them to the sampling chain even when `is_generated` is false. The peer then also uses its configured repetition-history window.

Consequently, setting both penalties to 1.05 is insufficient to declare a distribution-matched lane. A separate explicitly neutral-penalty diagnostic may isolate numerical/transaction behavior, but it changes production settings and cannot replace production-client quality or the required peer-best performance verdict. Do not change hf2q's sampling policy or handicap the peer simply to remove this comparison difficulty. Q1/F2 must record the exact history-scope disposition and use independent task oracles where cross-engine token identity is not a valid expectation.

## Reproduce the wire capture without model inference

Requires OpenCode's v2 `providers` configuration shape and Python 3.11+. Supply a JSON config containing the chosen provider/model. The utility copies only that provider/model, replaces provider settings with a synthetic localhost URL/key, strips other providers/plugins, isolates HOME/XDG roots, and runs a private client server from an isolated scratch directory. Nested provider/model controls are allowlisted; unknown routing or override fields fail closed. This redirects the configured model endpoint but is not an outbound network sandbox. The original config is not edited.

```sh
python3 scripts/four_agent/capture_client.py \
  --client /path/to/opencode \
  --config /path/to/opencode.json \
  --output /path/outside/checkout/new-capture-directory
```

Use `--model provider/model` only for a key actually present in that config. Only inspected routing/credential provider settings are accepted and replaced; additional provider-level controls require an explicit capture design. The checked-in utility targets the inspected local configuration, whose provider settings consisted only of routing and credentials.

Outputs are `summary.json`, raw `requests.json`, `client.log`, and an isolated config/state tree. Permissions default private; raw requests can contain system context and local paths, so keep them outside Git. The summary separates each request's keys, sampling controls and tool count. The output must be a new directory outside every ancestor Git checkout. The summary records separate raw configuration, canonical configuration and effective-profile hashes, request kinds, and owned process-group cleanup. Failure/timeout is an error, not a successful capture; the utility terminates only its own process group and localhost stub.

Validation performed: installed OpenCode v2.1.1 completed the real two-request stub exchange with a synthetic response; a repeated capture reproduced the settings above. An in-checkout output path was rejected with exit 2 before creating output. Eleven focused tests cover profile/override rejection, credential isolation, setup failure and descendant cleanup, output placement, request logging and timeout cleanup. These are client transport controls, not a substitute for a real six-tool-round, four-slot model test.
