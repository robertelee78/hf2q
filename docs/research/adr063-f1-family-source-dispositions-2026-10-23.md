# F1 family source capability dispositions

Read-only source screen, no builds or model loads. Peer basis: `36e36e63a54fc8f85b95e9c499a022b597e478dd`. hf2q stock basis: `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`; observed current main: `4cd1e79c49a836f2eb8f2ccdea5d87bc256d2a20`. A targeted git diff between those hf2q revisions returned no changes for engine.rs, DeepSeek slots/sampling, sampler_pure.rs, DeepSeek agentic policy and standard quantization policy. Findings below apply to both revisions for those exact paths; they are not a blanket equivalence claim for the repositories or binaries.

## Peer tools[].function.strict

**Disposition: source-inspected ignored flag in the Chat Completions tool normalization path; not a reproduced request pass or strict-schema guarantee.**

`tools/server/server-common.cpp:1435-1442` passes tool definitions through `common_chat_tools_parse_oaicompat`. `common/chat.cpp:556-584` validates tool type/function and retains only function name, description and serialized parameters. It does not read `function.strict`. Re-emission in `common/chat.cpp:527-542` reconstructs only those three fields. Therefore `strict:false`, `strict:null`, `strict:true` and absent strict have no distinct representation after this normalization. With otherwise valid tool objects, the ignored member does not itself enter a boolean conversion that would reject null. This source fact does not waive other request/schema validation or imply arbitrary malformed values are supported by the API.

Crucially **ignored strict does not mean unconstrained tool bodies**. `common/chat-auto-parser-generator.cpp:66-80` generates grammar based on tool availability, tool choice, template markers and response format, independent of strict. AUTO can use lazy triggered grammar; REQUIRED enables grammar. Specialized templates can follow another parser/grammar path (`common/chat.cpp:1404-1415`). Thus false/null may still encounter parameter-schema-derived constraints, and true does not independently prove OpenAI strict-schema compliance. Changing strict may leave effective peer grammar unchanged while changing hf2q policy; the manifest must record effective policy, not only identical wire JSON.

The `server-chat.cpp:283-286` code that inserts `strict:true` when absent is a **Responses-to-Chat translation** path, not evidence that the Chat Completions parser enforces strict. It ultimately constructs the same normalized tools representation.

Pending bounded protocol probes: four otherwise identical Chat Completions requests with strict absent/false/null/true, each AUTO and REQUIRED where relevant, unary/SSE, correct name/arguments, supplied tool-result continuation and prefix reuse. Capture effective grammar/parser receipts if exposed and rendered tokens. Test a schema constraint violation as a negative control. Until reproduced, label `strict_member_source_ignored`, `effective_tool_grammar_source_conditional`, `wire_runtime_pending`. Do not claim strict=false necessarily matches hf2q's non-strict execution policy merely because both accept that JSON.

## DeepSeek4 agenticq2 base and peer format compatibility

**Disposition: standard-block mixed-quant recipe and real DeepSeek4 loader paths are source-inspected; exact campaign artifact loader execution remains pending.**

hf2q `src/quantize/ggml_quants/deepseek4_agentic.rs:1-18,36-51` describes and implements an overlay on standard MostlyQ2_K: gate/up expert projections retain the base Q2_K decision, down experts use standard Q3_K promotion, context-discrimination matrices are pinned Q8_0, and scalar/integer gates run first. `:53-84` identifies the pinned output/embedding/hyperconnection/indexer/compressor/attention names. `src/convert/orchestrator.rs:553-555,585-611` handles I32 and F32 preservation and applies the overlay followed by shape fallback. Therefore “agenticq2” is a policy name, not a private new peer block type. The supplied base inventory's Q2_K/Q3_K/Q8_0/F32/I32 set is consistent with that recipe. Source policy alone does not prove an artifact contains only those types: shape fallback and any optional tensors require the actual hashed inventory.

Peer `src/llama-model.cpp:213-214` selects the DeepSeek4 model class; `src/models/deepseek4.cpp:19-65` reads family-specific settings and checks supported MoE scoring; `:82` onward creates explicit family tensors, not approximate Qwen fallback. Missing MTP projection clears nextn-layer metadata (`:19-25`), consistent with excluding embedded MTP for this campaign's base artifact. Peer `ggml/src/ggml.c:745-751,776-791` binds Q8_0/Q2_K/Q3_K type traits to actual conversion functions, implemented in `ggml/src/ggml-quants.c:553,961,1306`. Those implementations are stronger evidence than enum names but still do not prove every operation/layout on Metal. Metal operation code names these K-block types in dispatch applicability (`ggml/src/ggml-metal/ggml-metal-ops.cpp:2647-2651`); a backend-operation gate remains necessary.

Unresolved: exact GGUF metadata keys/values, tensor names/count/ranks/strides, expert axis ordering, grouped projections/hyperconnection layouts, I32 routing table semantics, quant block lengths and row alignment, compressor/indexer topology, source hash and all expected optional tensors. The loader checks expected names/dimensions (`src/llama-model-loader.cpp:881-921,1345`) and tensor count (`:1400-1408`); payload validation is separately conditional (`:1553`). A standard type inventory cannot bypass these checks or prove the model will load, run within Q5, or produce correct logits. Require metadata/tensor-catalog audit before load; after Q5, actual loader/no-fallback receipt, finite forward logits/parity and realistic four-slot multi-turn tool semantics. No `loader_pass` status is supplied by this review.

## Gemma4 q4km and peer compatibility

**Disposition: source-inspected standard mixed-quant recipe and explicit Gemma4 loader; tensor-layout/backend/real-model proof pending.**

Q4_K_M is a whole-model selection policy, not a uniform tensor type. hf2q `src/quantize/ggml_quants/standard_policy.rs:653-665` includes Q6_K promotion under MostlyQ4_K_M. F32/integer keep and shape fallback occur in the orchestrator paths above. Inspect the actual target inventory rather than assuming every tensor is Q4_K or overlooking Q6_K/F16/floating fallbacks.

Peer's Gemma4 class reads explicit family hparams (`src/models/gemma4.cpp:4-35`) and creates explicit embeddings, optional tied output, per-layer embeddings/projections/norms and attention tensors (`:37-90`). Dense/MoE/shared-KV/per-layer embedding variants must match the exact target, not just architecture string. GGML implements actual Q4_K/Q6_K traits/dequantization (`ggml/src/ggml.c:792-814`; `ggml/src/ggml-quants.c:1530` for Q4_K), but no execution was exercised here. Required audit: family metadata/sliding pattern, tied/shared tensor expectations, per-layer embeddings, rotary transforms, quant row/block alignment, tensor names/shapes/count, optional vision/projector and assistant artifacts, and matching tokenizer/template. Exact loader and backend operation receipts, finite logits/parity and four-slot continuation remain runtime gates. Do not label this as a passed loader from a recognized enum or class.

## hf2q repetition-history scope: resolved from source

**DeepSeek4 current serving sampling uses the current response's generated tokens only, not prompt/conversation tokens and not a fixed last-N window.** `src/serve/api/engine_deepseek4/slots.rs:529-537` starts an empty generated vector; `:596-604` passes `&self.generated` to sampling and `:614` appends the selected token. Unary `sampling.rs:231-244` likewise creates an empty generated vector and passes it. `sampling.rs:75-90,126-134` forwards previous unchanged to the common sampler/grammar helper. Thinking-budget forced output tokens follow the same generated ledger after selection; EOG/dead-grammar tokens exit before append in the inspected slot path (`slots.rs:585-614`). No prompt concatenation or last-N truncation appears in this penalty input.

**Gemma4 inspected slot-aware batched, unary and SSE text generation paths likewise use only tokens generated in the current response.** Batched first token passes `&[]` (`src/serve/api/engine.rs:7516-7524`); later tokens pass `&self.generated_tokens` (`:7731-7739`). Serial unary initial and later sampler calls are `:25156-25171` and `:25275-25291`; SSE equivalents are `:26089` and `:26228`, with empty/generated histories respectively. The vector covers generated token IDs, including any reasoning/tool syntax emitted by that request; prior assistant/tool history is merely prompt context for this penalty, not included in its explicit repetition list. This conclusion is scoped to inspected canonical text-serving paths, not an automatic claim for every experimental/draft/vision branch.

The penalty applies once per **unique** in-range token ID, multiplicatively according to logit sign (`src/serve/sampler_pure.rs:557-571`); repeated occurrences do not compound counts, and the helper has no fixed history window. At penalty=1 it is neutral. The grammar-greedy helper receives the same previous-token list and penalty (`engine.rs:583-593`); it does not broaden history to the prompt. This resolves the source-scope placeholder for these family paths, while runtime route proof remains necessary.

For matched settings, record `repetition_scope=current_response_unique_token_ids; window=unbounded_within_response; prompt_included=false`, with the path/source basis above. Do not assume peer `repeat_last_n` or `penalize_prompt` defaults match it. A repetition-neutral greedy diagnostic can use repeat_penalty=1, but production/best settings must preserve the frozen family policy and disclose any unavoidable scope mismatch. Pending runtime checks should prove the actual selected path and logits/outputs with a repeated prompt-only token versus a generated-only token, cached follow-up and concurrent slots; no separate product preference is needed to establish the source scope.

F1 may replace vague source-compatibility placeholders with these precise dispositions and remaining gates. F2/Q5 must still establish admission, actual load/backend execution, numerical quality, matched effective policies and all-four measured performance. This screen supplies no runtime pass or speed verdict.


## Exact-artifact directory inventory

A metadata-only GGUF directory read records the following actual type counts. Numeric IDs are the standard GGML wire values; no tensor payload was read or decoded and no device/model was initialized. Directory identities are joined to the separately full-file-hashed model identities in the private campaign evidence.

| Family | Tensors | Actual encoded types/counts |
| --- | --- | --- |
| Qwen3.8 | 866 | F32 360; Q4_K 439; Q6_K 67 |
| DeepSeek4 | 1328 | F32 535; Q8_0 532; Q2_K 172; Q3_K 86; I32 3 |
| Qwen3.6 | 753 | F32 308; F16 2; Q4_K 379; Q6_K 64 |
| Gemma4 | 658 | F32 392; Q5_0 32; Q8_0 28; Q4_K 192; Q6_K 14 |

The Gemma inventory demonstrates why a whole-model Q4_K_M label is insufficient: Q5_0 and Q8_0 also require source/backend applicability checks. The complete name/shape/type/relative-offset catalogs and reader-source hash are retained; they are inputs to loader/Metal validation, not a claimed loader pass. The catalog's model digest is declared from the conversion receipt; the identity validator independently hashes the complete GGUF and must agree before accepting the joined evidence.
