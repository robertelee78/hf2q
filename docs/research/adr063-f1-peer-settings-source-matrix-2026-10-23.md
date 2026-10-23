# F1 latest-peer settings: finite source-derived candidate matrix

Source revision `36e36e63a54fc8f85b95e9c499a022b597e478dd`. CLI evidence: archived `adr063-peer-36e36e63-help.txt`, bound by the private inventory manifest. This is F1 candidate registration, not F2 qualification or measured “best.” No model loads/builds were performed. Candidates require Q5 admission and runtime correctness before timing. Retain the current-peer primary arm even if it fails a task: report the failure rather than claim a win against an omitted comparator.

## Fixed comparison envelope and commands

Use the identical owned-produced GGUF hash for stock, candidate and peer. Qwen GGUFs retain their MTP tensors even in no-speculation diagnostics. DeepSeek's existing base artifact has no eligible embedded MTP for this campaign; do not route it to Qwen MTP. DSpark is a separate target-specific external-drafter arm, currently unavailable as a pinned eligible artifact, not a synonym for DeepSeek-V4 MTP.

The commands below are exact launch templates. `PEER`, `MODEL`, `SLOT_CTX`, `RESULT_CAP`, `TEMPLATE`, and `TEMPLATE_KWARGS` must be bound to immutable manifest values before execution; they are not tuning dimensions. Their absent binding fails F1 validation. In particular this source review does not invent a smaller context to make a memory projection fit. MODEL is the exact local path/hash for the selected family, TEMPLATE is the frozen model-native Jinja template file, and TEMPLATE_KWARGS is the frozen JSON string used by hf2q. Template replacement must first pass rendered-token parity. RESULT_CAP bounds all emitted output, including the campaign's explicit reasoning policy. Freeze HTTP request fields too; request-level fields can override CLI defaults.

```bash
# zsh/bash array; execute only after required manifest fields and Q5 approval.
: "${PEER:?}" "${MODEL:?}" "${SLOT_CTX:?}" "${RESULT_CAP:?}"
: "${TEMPLATE:?}" "${TEMPLATE_KWARGS:?}"
base=("$PEER" --model "$MODEL" --host 127.0.0.1 --port 18081
  --parallel 4 --kv-unified --ctx-size "$((4*SLOT_CTX))"
  --kv-unified-per-slot "$SLOT_CTX" --fit off --gpu-layers 999
  --cont-batching --no-context-shift --flash-attn on
  --cache-type-k f16 --cache-type-v f16
  --batch-size 2048 --ubatch-size 512
  --jinja --chat-template-file "$TEMPLATE"
  --chat-template-kwargs "$TEMPLATE_KWARGS"
  --cache-prompt --cache-reuse 0 --cache-ram 8192 --cache-idle-slots
  --ctx-checkpoints 32 --checkpoint-min-step 8192
  --predict "$RESULT_CAP" --seed 42
  --temp 0 --top-k 0 --top-p 1 --min-p 0 --repeat-penalty 1
  --metrics --slots)

# G: matched greedy diagnostic; deliberately NOT automatically the primary best arm.
"${base[@]}" --spec-type none
# N: no-extra-weight lookup candidate, every family.
"${base[@]}" --spec-type ngram-simple --spec-draft-n-max 64 \
  --spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 32 \
  --spec-ngram-simple-min-hits 1
# R: shared-slot lookup candidate, every family.
"${base[@]}" --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
# M1/M3/M7: Qwen3.8 and Qwen3.6 only; issue the three launches separately.
for n in 1 3 7; do
  "${base[@]}" --spec-type draft-mtp --spec-draft-n-max "$n" \
    --spec-draft-n-min 0 --spec-draft-p-min 0 \
    --spec-draft-sampling greedy --spec-draft-backend-sampling
  # External supervisor must stop/drain/unload each launch before the next.
done
```

The last loop is a list of candidate commands, not a ready unattended runtime script. Every server is blocking and requires bounded external supervision. No two model engines may co-reside. All launches require their own admission projection including the explicitly bounded 8 GiB CPU prompt-cache allowance, 32 per-slot checkpoint limit, native/draft contexts and transition overlap. A maximum is not proof that peak allocated memory reaches it, nor proof that it excludes other allocations. If this envelope fails memory admission, mark the candidate inadmissible; do not silently change its context, slot count, result cap or speculation.

## Finite candidate selection

Register exactly these starting candidates (before any runtime outcomes):

| Family | Required initial settings IDs | Additional conditional registration |
| --- | --- | --- |
| Qwen3.8 dense | G, N, R, M1, M3, M7 | DFlash2 only after exact target-specific drafter provenance, owned conversion, hash and runtime architecture compatibility are established. |
| Qwen3.6 MoE | G, N, R, M1, M3, M7 | Same conditional DFlash2 requirement; do not assume the dense drafter is compatible. |
| DeepSeek4 | G, N, R | Embedded MTP excluded for this base artifact. A separately pinned DSpark checkpoint must prove target compatibility before a candidate is added. A DSpark demo targeting Qwen3-4B is not such proof. |
| Gemma4 | G, N, R | A distinct Gemma assistant/MTP or EAGLE3 drafter requires exact target/variant compatibility and owned artifact provenance. Gemma target support alone is not proof of an embedded trained MTP head. |

Register both the baseline cache envelope and K0 (`--cache-ram 0 --no-cache-idle-slots`) for every starting speculation candidate before any measurements. K0 retains live-slot prefix reuse and the same context/result/slot bounds. An inadmissible 8 GiB cache envelope therefore does not hide an otherwise valid cache-policy candidate; no candidate is chosen by silently shrinking the workload.

Once correctness-admitted initial candidates identify each family's fastest valid speculation setting on a held-out selection workload, freeze **one** selected setting for confirmation; preserve all failures and timing receipts. Tie-break by median per-agent completion/TTFT plus aggregate throughput under the program's explicit SLO, not highest single-slot tokens/sec. Then compare the registered cache envelopes and run one additional batching alternative for that selected setting: B1024 uses `--batch-size 1024 --ubatch-size 512` with the baseline cache settings. Implement these by constructing replacement arrays, not ambiguous repeated CLI flags. These confirmation envelopes are finite and all retain four slots and the same per-slot context/result bounds. K0 is a labeled cache-policy alternative, not an excuse to disable live prefix reuse.

If a documented conditional drafter becomes available, register its exact candidate before observing timing: DFlash `--spec-type draft-dflash --spec-draft-model "$DRAFTER" --spec-draft-n-max 15`; DSpark `--spec-type draft-dspark --spec-draft-model "$DRAFTER" --spec-draft-n-max 7`; Gemma EAGLE3 `--spec-type draft-eagle3 --spec-draft-model "$DRAFTER" --spec-draft-n-max 3`; Gemma assistant MTP `--spec-type draft-mtp --spec-draft-model "$DRAFTER" --spec-draft-n-max 3 --spec-draft-sampling greedy`. These are conditional named configurations, not current availability claims. Source docs clamp DFlash/DSpark to trained block length. Freeze the drafter's target-specific tensor/vocabulary mapping and precision, charge it to Q5, and rerun held-out selection. Do not declare “peer best” while omitting a known compatible, available and potentially profitable mode without an explicit tested failure or justified exclusion. If flash attention or f16 KV is rejected by a family/backend, record that source/runtime incompatibility and approve a separate exact alternative before testing; never conceal backend fallback.

## Source interpretation and pending probes

- **Context is not one flag divided by assumption.** `tools/server/server.cpp:177-186` auto-sizes pool to parallel × per-slot limit only when context is unset; commands pin both deliberately. `tools/server/server-context.cpp:4954-4962` caps each slot using sequence context, per-slot limit and model training context. Check startup `/slots` and logs report four slots at SLOT_CTX; reject silent shrink or RoPE changes. `--fit off` prevents implicit fit tuning from changing workload.
- **Prefix reuse differs from chunk shifting.** `server-context.cpp:4000-4047` separates normal longest-prefix reuse from optional KV-shift chunk reuse. `--cache-reuse 0` disables that extra shifting feature, not normal unchanged-prefix caching. `server-context.cpp:1398-1408` disables shifting for memory types that cannot shift. Keep chunk reuse zero in these initial exact-prefix scenarios; do not misreport it as cache disabled. Validate cached-token counts on tool-result continuation and isolated interleaved conversations.
- **Cache RAM is additional host retention.** `server-context.cpp:1571-1585` creates the prompt cache from the configured MiB limit. Live KV, checkpoint state, staging and queued resources are separate. Do not call 8192 MiB a whole-process bound or assume cache eviction frees all retained command resources. Checkpoint caps are not byte caps.
- **MTP supported path is concrete.** `common/speculative.cpp:1489-1584` implements per-sequence state, Qwen's single trained head, Gemma shared-memory assistant behavior and chain-head clamping. `common/speculative.cpp:2705-2759` creates an MTP context against the target model if no external drafter is supplied; `cparams.n_ctx` equals target context. Charge the extra context. Qwen loader materializes NextN projection/norm tensors (`src/models/qwen35.cpp:97-126`; corresponding MoE loader must also match actual tensor inventory). Verify every required MTP tensor exists and draft acceptance is nonzero/useful before treating the mode as exercised.
- **Do not infer applicability from auto-detection alone.** `common/speculative.cpp:2465-2478` probes the last block's NextN projection. Explicit mode plus artifact tensor proof is safer than unrecorded auto-selection. DeepSeek loader clears n_layer_nextn when the projection is absent (`src/models/deepseek4.cpp:19-25`). Gemma's target hidden-state exposure (`src/models/gemma4.cpp:418-422`) enables an assistant but does not manufacture its weights.
- **Tool correctness is template-specific.** The help exposes Jinja/model metadata templates and reasoning controls; `common/chat.cpp:327-336` probes thinking support. Preserve native template behavior and explicit program thinking settings; `--reasoning-format` only changes output extraction, not necessarily inference work. Do not impose a generic Qwen tool parser or force thinking off to improve timings. Verify exact rendered token IDs, tool definitions, generated function name/arguments, supplied tool result, SSE finish reason, semantic streamed TTFT, cancellation/rewind and no cross-slot leakage. Missing template/token proof is an unresolved F1 binding.
- **Source defaults are not measured optima.** `docs/speculative.md:55-105` documents DFlash/DSpark; `:126-186` supplies concrete ngram commands; `:370-372` identifies the modes. Help confirms all listed CLI flags. These establish valid candidate syntax and intended semantics, not four-slot performance or numerical parity. Synthetic acceptance options are excluded entirely from real speed claims.

F1 can deliver this bounded settings registry, artifact/source/binary identities and explicit unresolved capability bindings. F2, after Q5 and harness certification, must reproduce load/slot/template/speculation behavior, reject bad candidates, measure held-out selection and repeat matched confirmation. Until then label these `source_candidate`, never `qualified_peer_best`.
