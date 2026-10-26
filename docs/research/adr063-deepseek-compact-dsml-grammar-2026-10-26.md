# DeepSeek compact DSML grammar: scoped validation

The compact grammar repair preserves delimiter exclusion and existing argument
bounds while restoring compilation of realistic bounded tool schemas. The
recorded four-agent DeepSeek gate passed coding, real tool execution, unary/SSE
transport, native-prefix reuse and cleanup. This is correctness evidence, not a
speed, physical-memory admission, maximum-context or all-family model claim.

## Source and artifact identities

| Evidence | Exact identity |
| --- | --- |
| Actual server source | `56704f29783e011ab2c2a57c9eaeb80df3d68050` |
| Actual release binary SHA256 | `54a2e90cba23a0dbd7ba400a4ef41836d706c381bbacd60523c0255fc3dcc10e` |
| Follow-up test-only source | `bc909d194726f274eebd3567e0214f0deaac7f72` |
| Combined-stack unit candidate | `b7b92c2df9dac851e1e25a223fccaa8d7f96d3ef` |
| Published native dependency | `mlx-native = 0.15.6`, crates.io checksum `f0927b25a60092797fdb15bf850e60c89269dca07461c929901081ab1209f65d` |
| Model | DeepSeek V4 Flash0731 agentic Q2, 107,431,343,264 bytes |
| Model SHA256 | `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` |
| Actual client source | `20e5ad83c5e86d76617efa0a19a4e63a46c40347` |

The model identity joins retained F1 full-content verification to fresh file
stat identity; it is not a claim that this gate reread the entire model.
The runtime used the published dependency without a local Cargo patch.
`bc909d19` changes only assertions in cfg(test) engine modules: dispatch-table
source witnesses skip preceding startup checks, and scaffold witnesses follow
the checked descriptor/phase planner. It changes no production behavior.
The actual model execution belongs to `56704f29`, not to a newly executed
`bc909d19` or combined-stack binary.

## Grammar contract and retained failures

The preceding delimiter/length DFA multiplied string length by delimiter-prefix
and escape states. Existing regressions then rejected maxLength4096 arguments
and large tool lists at the existing grammar resource limit. The mandatory
matrix includes forty bounded4096 tools and three tools at the65536 ceiling.
Argument bounds and byte/rule/expanded-element caps were not raised or reduced.

The repair reuses the established exact bounded-repetition compiler and adds a
private HF runtime lexical constraint. The reserved underscore-bearing names
`hf2q_dsml_data_raw`, `hf2q_dsml_json_raw` and `hf2q_dsml_raw_end` require exact
approved rule bodies. A data-terminal branch cannot consume the final scalar
of the raw `</｜DSML｜parameter>` delimiter; the real structural closing branch
remains legal. Raw length counts each Unicode scalar, including two scalars in
a backslash pair. Existing JSON quoting/escape-unit behavior is preserved.
Ordinary code angle brackets remain valid.

A branch-local empty raw-end predicate rejects a dangling escape only on the
generic raw branch. Its first global-scope prototype failed an actual compiled
mixed-tool regression: an enum/const literal ending in backslash inherited
another branch's escape restriction. That failed result remains failed. The
revised predicate permits the literal branch and still rejects an incomplete
generic raw value. Collective masking groups matched candidates by their state
after consumption before expanding this predicate; sampled acceptance uses the
same guard and expansion logic.

Another source-review counterexample found that hyphen-only private names could
collide with valid Gemma property-derived rules. Both public property-name
sanitizers replace underscores with hyphens, isolating the final private names.
Function names remain wire literals, not private rule names. Bare and numeric
function/schema-variant namespace chains preserve the constraint; unknown
private renames, changed reserved bodies and mixed token-terminal grammars
refuse runtime creation. Non-DS runtimes keep the no-guard path.

This is private generated HF grammar semantics, not portable standard GBNF
syntax. A compact binary-path grammar experiment was rejected without execution
because speculative intermediate states could multiply pushdown continuations.
It is absent from the landing implementation and supplies no passing evidence.
See [ADR-042](../adr/ADR-042-deepseek-v4-flash-rust-native.md) for the decision.

## Validation and evidence boundaries

The actual retained gate is `deepseek-new-grammar-n4-20261026T222718Z`; its
post-unload replay packet is `deepseek-compact-v6-prefix-replays-20261026T222718Z`.
Independent reviews verified all4,854 gate artifacts and20 replay artifacts
against their manifests, with zero hash mismatches. Raw machine artifacts and
model data remain private and are not checked into this repository.

All four agents completed seven HTTP200 requests and six single real tool
calls:28 requests,24 tool executions and20 verified compiler/test processes.
Alternating unary/SSE responses joined actual request/response hashes and usage;
SSE finished with valid DONE framing. Each final Rust patch passed the actual
ports and full test tools, with checked numeric parsing and zero/overflow
rejection. Server, client and owned tool processes were reaped and their groups
were empty; no pending cases remained.

All four independent native-prefix replays executed exactly one test each,
exit0 with verified cleanup. Across all24 follow-up edges, observed cached usage
reached the prior complete DeepSeek native anchor. That anchor excludes the
existing eight-token native recovery tail; this is not the Qwen/Gemma cue
predicate. The helper binary SHA256 was
`8f56507833d57a0df04f276ccb12dc9eb85926075876fc82a9ef3a95e15231c5`.
Its retained pre-format source/binary evidence is composed with independently
verified executed renderer/cache identities; it is not server-build
self-attestation by the offline helper.

Focused checks covered4096/65536 exact bounds, large all-family tool lists,
actual compiled DSML parallel-call boundaries, mixed literals, raw/JSON escapes,
Unicode byte splits, clone/lazy-trigger state, collective/greedy masks versus
acceptance, private namespace collisions and ordinary token-grammar behavior.
The combined-stack full-bin regression at `b7b92c2d` then reported **5,414 passed,
0 failed,77 ignored**,229.78 seconds. Ignored cases were not executed by that
command. Its complete log SHA256 is
`f3b3c4c6374daf0684c91adbbcd7d3675bfb7e00af8f8990f6564049f5ebf0ee`.
Earlier failing runs remain separate; this result does not relabel them.

Reproduce the model-free checks from the pinned source with locked dependencies:

```sh
cargo test --locked --bin hf2q dsml_compact_
cargo test --locked --bin hf2q deepseek_dsml_boundary_tests
cargo test --locked --bin hf2q tool_bounds_tests
cargo test --locked --bin hf2q grammar::mask
cargo test --locked --bin hf2q grammar::sampler
cargo test --locked --bin hf2q -- --test-threads=1
```

Some broad tests use tiny actual Metal fixtures when available; these commands
are not a declaration that every test is hardware-free. Real-model reproduction
additionally requires the exact artifact, canonical DeepSeek launcher, frozen
four-agent client/approval, sole-runtime memory and listener checks, retained
raw responses/tool outputs, verified unload and independently bound offline
prefix configuration. Do not launch a second large runtime beside this model.
The recorded operating-memory guards are diagnostic policy, not formal fit or
native final-owner accounting. No matched peer, speed study, cancellation/growth
stress rerun or four-family model qualification is supplied by this gate.
