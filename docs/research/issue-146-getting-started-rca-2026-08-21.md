# Issue 146 RCA: broken Qwen3.8 + OpenCode onboarding

Date: 2026-08-21
Issue: [#146](https://github.com/robertelee78/hf2q/issues/146)
Affected public guide source: `docs/getting-started.md` at and after
`d59a7e2df4cfcdc20a8903f3b91bfd21691bebd7`, including the page built from
`1f4087aedbb252523fe5cca1806fd1651f6689e5`
Severity: release-blocking onboarding failure

## Executive summary

The guide promised a working local abliterated Qwen3.8 model inside a working
OpenCode/Agentic Kit harness. It delivered neither.

At the affected release/dependency boundary, the recommended
`qwen38-abliterated-sft-q5_k_m.gguf` had no admitted direct Metal gather for its
Q5_K embedding tensor. hf2q loaded the file and reported ready, but every
generation failed with HTTP 500. The guide never made a real completion request
before opening OpenCode, so its acceptance path could not detect this.

Independently, the guide treated the model card's prompt-sensitivity note as a
configuration prescription. It replaced `agent.assistant`, selected that agent
as the default, set a minimal prompt, disabled every tool, and denied every
permission. That removed Bash, file operations, tasks, skills, and MCP tools.
The guide then described the operation as a non-destructive merge and described
the web stack as usable, neither of which was true for the selected agent.

The incident was not one bad command. It was a failure to define and execute the
end-to-end product contract: exact artifact bytes -> real unary/SSE/vision
generation -> stock OpenCode Build tools -> full Agentic Kit -> live
search/fetch/crawl/extract.

## Post-landing regression discovered during review

The first landing incorrectly split one guide into two. A newer source-first
rewrite of `docs/getting-started.md` had reached `main` while the issue-146
correction was being prepared. During rebase, that file was preserved as the
newer side of the conflict and a second guide file was added for the complete
journey. Neither the split nor the second file was requested. The regression
was then made worse by changing `scripts/test_getting_started_guide.sh` to
accept both journeys.

That alternate document told users to run a fresh source conversion, claimed
the command produced a projector, served only the text model, and reduced
OpenCode to an optional provider snippet. It omitted the exact published
artifact pair, multimodal serve and image proof, full Agentic Kit, preservation
of stock coding tools, the research stack, and cleanup. The README repeated the
source-first path, and the complete guide included the same unqualified
conversion claim as an optional section.

CI passed because the test encoded the regression as required behavior. This
was a validation-oracle failure, not evidence that both paths met the same
contract.

The correction restores the complete journey directly in the one canonical
`docs/getting-started.md`, removes the extra guide file, and points every
repository entry to that file. The guide no longer offers an unproved
conversion substitute. The contract test rejects a second guide, the
source-conversion command, and text-only wording while requiring the published
pair, `--mmproj`, text/SSE/image proof, full AK, stock Build tools, research
tools, and cleanup.

## User-visible failures

1. **Unqualified GGUF at the affected boundary.** The guide downloaded the
   author-hosted Q5_K_M without proving its runtime tensor coverage. At that
   release/dependency boundary, hf2q had no admitted direct gather for the Q5_K
   `token_embd.weight`, so generation returned 500.
2. **Readiness was treated as generation proof.** `/readyz` returned 200 after
   load even though the first embedding/prefill could not execute. Runtime
   admission and native coverage are separate from readiness; the guide must
   still execute a real completion.
3. **OpenCode obscured the original runtime error.** The guide opened a retrying
   coding client before proving a unary and streamed request, turning one clear
   backend failure into repeated harness failures.
4. **The server/client port contract was implicit.** `hf2q serve` omitted
   `--port 8081` while every client example hard-coded 8081. It happened to work
   only when setup state supplied that port.
5. **An existing `agent.assistant` was overwritten.** Object addition at the
   `assistant` key was destructive even though the surrounding provider merge
   preserved unrelated keys.
6. **The replacement assistant was made the default.** Every new OpenCode
   session entered the broken profile unless the user knew to switch agents.
7. **All tools were removed from the request.** `tools: {"*": false}` removed
   Bash, read/write/edit, task, skill, web, and MCP schemas.
8. **All execution was denied.** Agent-level `permission: "deny"` won over
   global and Agentic Kit permissions, so even otherwise present tools could not
   run.
9. **The stock coding prompt was replaced without a coding qualification.** A
   generic one-line assistant prompt was substituted for OpenCode Build's
   operational instructions.
10. **The documentation claimed a non-destructive merge.** Existing unrelated
    keys survived, but the `assistant` key and default-agent selection did not.
    The stated safety property was false at the semantic level.
11. **Prompt sensitivity was conflated with permission policy.** The cited note
    measured behavioral sensitivity to framing. It did not establish that a
    useful coding harness should remove all capabilities.
12. **The advertised web setup could not work in the default session.** MCP
    entries were added for an agent that denied MCP tools. The path also used an
    unpinned container tag and had no actual OpenCode tool-call acceptance.
13. **Agentic Kit was not fully installed for the project.** The later guide
    used `ak setup --minimal --opencode --yes`; `--minimal` explicitly skips
    project setup, contrary to the promised full AK experience.
14. **Multimodal capability was omitted despite the source model and accepted
    pair.** The guide selected a text-only artifact instead of a source-matched
    text/projector pair and never sent an image request.
15. **Cleanup was overbroad.** `pkill -f "hf2q serve"` could terminate unrelated
    servers instead of the exact process launched by the guide.

## Technical root causes

### RC1: artifact provenance was confused with runtime compatibility

The Q5 file was immutable and checksum-pinned, but those properties prove only
identity and integrity. They do not prove that the running hf2q revision
supports every tensor type needed for inference. The guide substituted
“externally produced GGUF validates at serve time” for an execution
compatibility test.

The conversion pipeline already produced a native hf2q Q4_K_M text GGUF, but
the known-good bytes were not published as the guide artifact. This encouraged
the shortcut of reusing an unrelated pre-quantized upload.

### RC2: load/readiness and generation were collapsed into one gate

The guide waited for `/readyz`, then immediately opened OpenCode. Its acceptance
oracle was process health plus model metadata. At the affected boundary, the
missing direct-gather capability was encountered on the first forward pass,
after those checks. No unary
completion, SSE reconstruction, or image request existed in the documented
happy path.

Runtime admission should reject an artifact that cannot perform its first
embedding/prefill operation, but documentation must still test the observable
client contract. Either layer can regress independently.

### RC3: a research observation was promoted directly into product config

The prompt-sensitivity note was evidence that the stock OpenCode frame required
controlled investigation. It was not evidence for a tool-less default agent.
No ablation matrix isolated system-prompt sections from tool schemas, AGENTS
content, permissions, or sampling. No realistic coding or tool-use gate was run
on the proposed minimal profile.

The change optimized a single behavioral concern by deleting the product's core
capabilities. This violated the intended contract rather than trading one
accepted metric for another.

### RC4: configuration correctness was assessed structurally, not semantically

The `jq` operation merged top-level objects, so it looked non-destructive in a
JSON diff. At runtime it replaced a well-known agent key, selected it globally,
and applied agent-level denies after global permissions. Review verified JSON
shape but not OpenCode's resolved config or an actual Bash/file/task/MCP call.

### RC5: the guide was optimized for three short paste blocks before it was
optimized for successful completion

The simplification series retained the most visible commands while removing
or omitting the expensive evidence: self-conversion, real generation, vision,
full project setup, and tool acceptance. “Short” became the design constraint;
“working end to end” was never encoded as a blocking test.

## Five whys

1. Why did users get HTTP 500? Because the guide selected a Q5 embedding tensor
   type that hf2q's direct GPU gather does not support.
2. Why was that artifact selected? Because it was already uploaded, smaller than
   source conversion, and checksum-pinnable; compatibility was assumed from
   GGUF parse/load success.
3. Why was the assumption not caught? Because the guide stopped at readiness
   and had no real generation acceptance.
4. Why did OpenCode become unusable? Because prompt sensitivity was handled by
   replacing the default agent and deleting/denying its tools.
5. Why did that ship? Because review and validation had no end-to-end harness
   contract covering resolved OpenCode configuration, real tool calls, full AK,
   and web research capabilities.

## Corrective actions

### Completed artifact correction

A source-matched hf2q pair was published to
`jenerallee78/Qwen3.8-27B-Abliterated-SFT` at immutable artifact commit
`40d771ee15d826017f297261f5bedcf2c32cf4c2`:

| Role | File | Size | SHA-256 |
|---|---|---:|---|
| Q4_K_M text | `gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf` | 16,810,714,944 | `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a` |
| F16 projector | `gguf/qwen38-abliterated-sft-hf2q-q4_k_m-mmproj.gguf` | 927,606,848 | `463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b` |

The root `GGUFs.md`, checksum file, and machine-readable manifest distinguish
this pair from the existing peer-engine/L40 artifacts. The text model embeds the
required projector digest and mixed pairs fail closed.

### Completed Q5_K_M runtime correction

The author-hosted Q5_K_M file is no longer rejected at first forward. Native
inference now admits direct packed Q5_K and Q6_K embedding gathers through one
typed routing authority. It validates the artifact's packed byte geometry,
storage type, capability, and resolved route, and fails closed for unsupported
representations. It does not dequantize the embedding or output head and
re-encode either tensor into a convenient runtime format.

The exact 19,535,701,568-byte artifact at immutable revision
`0a72776892f98db49381fdf69f4b9982222ec9dc` has SHA-256
`4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e`.
The real Apple-Silicon gate proves its Q5_K embedding, Q5_K MTP projection,
Q6_K output head, native quantized target layers, and exact target/MTP
output-head allocation sharing. It also proves four-position speculative
verification remains decision-equivalent to serial target decoding.

hf2q pins the published registry release `mlx-native 0.11.2`, checksum
`22f4bd6661e77994c6f26a79fdd2c188f3d5252aa7e51616f5feb080b22da8e0`.
That release additionally makes equal-logit GPU argmax choose the lowest token
index, matching the CPU greedy contract. Full matched Q5 throughput remains a
separate acceptance gate and is not inferred from this correctness result.

### Completed guide correction

The replacement guide:

- downloads the exact hf2q Q4/projector pair and verifies both hashes;
- passes `--mmproj`, host, port, scheduler, slots, overflow policy, and KV
  budget explicitly;
- carries the qualified OpenCode performance defaults for repetition control,
  bounded reasoning, tool continuations, TQ KV, encoder sessions, exact
  speculation, and K-quant width routing;
- runs real unary, SSE-to-`[DONE]`, and embedded-PNG vision checks before
  OpenCode;
- runs full `ak setup --yes`, then `ak setup --opencode --yes`;
- merges only the hf2q provider/model selection and preserves agents, tools,
  permissions, plugins, instructions, and MCP configuration;
- uses stock OpenCode Build and requires real Bash/file/tool transcripts;
- installs a persistent local search/fetch/crawl/extract stack with pinned
  dependencies and automatic static -> browser -> stealth fetch escalation;
- keeps all coding, task, skill, and MCP tools; replaces only the redundant
  built-in fetch-only `webfetch` after a measured Qwen3.8 run guessed a URL
  instead of selecting search; and adds lowercase plus Ruflo-compatible
  capitalized research tools; and
- records and stops only the exact server PID.

### Completed prevention automation

`scripts/test_getting_started_guide.sh` binds the guide to the artifact commit,
file names, hashes, multimodal serve command, real API checks, full AK commands,
qualified performance settings, provider-only OpenCode merge, research-stack
installer, and required tool names. It rejects the broken minimal-AK command,
agent/default-agent writes, blanket tool/permission denial, stale README claims,
and broad process killing. It also syntax-checks every guide Bash block and the
installed web-stack assets. `scripts/test_shipping_contract.sh` invokes this
contract so the documentation gate cannot pass without it.

## Exact-host acceptance evidence

The corrected path was exercised on an Apple M5 Max with 128 GiB unified
memory on 2026-08-21. This is current-host evidence, not a substitute for the
remaining clean-user run below.

- Installed hf2q 0.1.7 loaded the exact published Q4/projector pair with four
  slots and a 48 GiB physical KV budget. Server logs proved Q4_K execution,
  projector digest `463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`,
  TQ KV, the 262,144-token logical slot context, and vision GPU warmup.
- A real unary request returned Rust code, SSE returned exactly one `[DONE]`,
  and the embedded red-PNG request answered `Red`.
- Stock OpenCode 1.18.21 Build called Bash and received the exact requested
  output, then called Read on `Cargo.toml` and identified package `hf2q`.
  Follow-ups reused 13,098 and 13,097 cached prompt tokens respectively.
- Full Agentic Kit 4.0.0-alpha.42 reports the native AgentDB writer, project
  setup, OpenCode convergence, three MCP integrations, compact `ak_*` gateway,
  and connected Ruflo/Agentic QE as healthy. Its current-user status also has
  non-blocking stale-projection/statusline and historical learning-row warnings;
  those pre-existing maintenance warnings are not hidden as guide success.
- Real model calls succeeded for `web_search`, `web_fetch`, `web_crawl`, and
  `web_extract`. The final combined AK + stock Build + installed research-plugin
  run called `web_extract` once, extracted `Example Domain`, completed with the
  exact requested answer, and reused 14,132 cached prompt tokens.
- Measured failures changed the implementation before acceptance: leaving
  OpenCode's built-in `webfetch` available made Qwen guess a GitHub API URL
  instead of searching; malformed but unambiguous JSON-CSS schemas caused two
  failed extraction turns; and the first repeat installer run exposed
  SearXNG's missing `msgspec` in an isolated editable-build sandbox. The final
  plugin removes only that redundant built-in fetch tool, normalizes both
  observed schema shorthands, and the installer uses its already-populated
  pinned SearXNG environment for the editable build. All three revised paths
  were rerun successfully.

The guide and shipping contracts, five fetch-routing unit tests, compact-schema
plugin contract, disable/enable/uninstall lifecycle test, Python compilation,
JavaScript syntax check, plist validation, live search/fetch validation, and
`git diff --check` pass on the resulting tree.

## Remaining actions and ownership

- **Runtime admission/readiness:** owned separately. hf2q should reject the
  unsupported Q5 execution path before advertising ready, or perform a minimal
  startup admission forward pass. The guide's real completion gate remains
  required after that lands.
- **Prompt-sensitivity qualification:** run an isolated ablation matrix over the
  stock Build system prompt, tool schemas by family, AGENTS injection, sampling,
  and thinking mode. Score both the sensitive-legitimate panel and realistic
  coding/tool tasks. Do not change the default harness until a candidate keeps
  Bash/file/task/MCP correctness and materially improves the measured behavior.
- **Installed-host acceptance:** execute the exact public commands from a clean
  macOS user configuration, including a fresh model download or hash-verified
  cache, four-slot OpenCode tool use, full AK status, and all four local research
  tools. Record exact versions, artifact hashes, prompts, and logs.

## Prevention rules

1. A model mentioned in a public hf2q guide must be an hf2q-executed artifact,
   not merely a parseable GGUF.
2. Integrity, compatibility, quality, and harness usability are separate gates.
3. `/readyz` is never the only acceptance proof for a generation guide.
4. A configuration “merge” is reviewed against resolved application semantics,
   not only JSON key preservation.
5. Prompt research cannot remove product capabilities without an explicit,
   realistic replacement acceptance matrix.
6. Public OpenCode guidance must prove Bash, file, task/agent, and external tool
   continuations using the selected default model and agent.
7. Full Agentic Kit means machine and project setup; `--minimal` must never be
   presented as equivalent.
8. Published multimodal models ship and validate the exact bound pair.
9. Every background process started by a guide gets an exact, scoped cleanup
   path.
