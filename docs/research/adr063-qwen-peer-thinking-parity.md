# Qwen peer reasoning-budget transport parity

This benchmark-client repair maps canonical Qwen HF reasoning defaults to the
existing native peer request field; it does not change inference, prompts,
tools, sampling,8192 output allowance or the six-step coding task. The retained
Qwen3.6 N4 E0282/workflow failure remains failed and unrescored. No model/backend
cause is established by this transport change.

The actual shared build_request now receives explicit engine_kind (HF default).
Only Qwen/peer maps the actual message chain to reasoning_budget_tokens: 2048
for a non-tool continuation and 512 for every tool-result continuation since
the latest user. System messages are skipped for latest-role selection; a new
user turn resets the chain. This mirrors qwen_thinking_policy.rs
qwen_tool_chain_state and adaptive_qwen_default_thinking_budget with canonical
2048/512 defaults. Before 2026-10-30, hf2q halved the continuation budget to a
256 floor from the second tool cycle; that halving forced agent-2 to close its
reasoning mid-plan in the failed Qwen3.6 N4 run, and was removed from hf2q and
this mirror together. The budget is derived from chain semantics, not a guessed
turn ID.

Peer d854d98bae598388ecea1ed84396587e5edc326d server-common.cpp1516–1526 accepts
reasoning_budget_tokens. Request messages/tools are preserved by reference,
HF-only overflow metadata removed only for peer wire, and the canonical
request envelope is still validated before serialization. Explicit reasoning
or thinking-disable overrides refuse rather than being overwritten/dropped.
Unsupported nonQwen peer selection refuses; HF/Gemma/DeepSeek requests retain
their existing controls. No inference fallback or new module/framework.

Seven new focused controls use the actual builder, covering all7 budgets and
only native-field/overflow delta, unchanged HF/schemas/messages, new-user reset,
system/history/parallel-cycle semantics, injected override refusal before wire,
invalid metadata/chains and explicit nonQwen routes. Author performed static
AST/source inspection only; root owns execution. Existing six peer adapter
controls are retained. Real peer quality/context/template/cache and latest-best
selection are separate pending proof; this is not a matched runtime result,
speed win or Q5 qualification.

Root validation: all 13 focused peer controls passed, including the seven actual request-builder turns and both additional override aliases. No peer model was loaded; runtime parity, the full client regression suite and latest-best reference selection remain separate gates.
