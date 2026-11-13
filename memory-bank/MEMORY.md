---
schema: "agent-memory/v1"
project-id: "479fefc0-3903-4e30-b7fd-877c69979b81"
router-updated: "2026-11-12T23:02:20.743885+00:00"
canonical-remote: "repo://github.com/robertelee78/hf2q"
---
# Project memory

Router time describes this map, not every file's revision.

## Routes

| Path | Purpose | Scope | Lifecycle | Read when |
| --- | --- | --- | --- | --- |
| [projectbrief.md](<projectbrief.md>) | Project purpose and authority boundaries | repository | active | At session start to identify the project |
| [activeContext.md](<activeContext.md>) | Scoped handoff, unresolved questions and next step | repository | active | When resuming the matching task or investigating its questions |
| [progress.md](<progress.md>) | Delivery evidence and work-source references | repository | active | When checking delivery claims or release scope |
| [topics/8534547f-299d-485c-87b5-ec5793dab93a.md](<topics/8534547f-299d-485c-87b5-ec5793dab93a.md>) | Operator rule: use standard technical terms (whitelist/blacklist), never allowlist/denylist-style renamings | repository | active | Before writing code, docs, issues, PRs, commits or chat in this repository |
| [history/events/1c8097a7-3bf7-4a60-a3eb-e69012d0302c.md](<history/events/1c8097a7-3bf7-4a60-a3eb-e69012d0302c.md>) | Admission history for operator writing rules | repository | active | When auditing how a writing rule was recorded |
| [topics/4e7f609c-41d0-4083-a336-daff25bfe75d.md](<topics/4e7f609c-41d0-4083-a336-daff25bfe75d.md>) | Operator rule: all hf2q GitHub Projects are public; make each new awa Project public | repository | active | When creating or reconciling an hf2q GitHub Project |
| [history/events/2e1d1a80-c640-4e91-bf8f-ff975a0c423c.md](<history/events/2e1d1a80-c640-4e91-bf8f-ff975a0c423c.md>) | Admission history for the Project-visibility rule | repository | active | When auditing how the Project-visibility rule was recorded |
| [topics/d74d2104-0524-4df0-a1d0-b7c5574cbdb5.md](<topics/d74d2104-0524-4df0-a1d0-b7c5574cbdb5.md>) | Operator decision: every qualified fast route is the default; only kill switches are user-facing; model needs are fetched automatically; the release gate is measured on defaults | repository | active | Before adding, promoting or gating a performance route, switch or model download, and before measuring a release gate |
| [history/events/6b992e36-fc51-4fc2-8bd4-028623a83d3b.md](<history/events/6b992e36-fc51-4fc2-8bd4-028623a83d3b.md>) | Admission history for the fastest-route-is-default decision | repository | active | When auditing how the default-route decision was recorded |
| [topics/c8e85df0-c387-4025-aedb-b1571aae8c86.md](<topics/c8e85df0-c387-4025-aedb-b1571aae8c86.md>) | Measurement rule: never export MLX_PROFILE_CB (even empty) in a timing run; it forced synchronous commits in Qwen3.8 runs from 2026-11-09 until the supervisor fix | repository | active | Before launching or interpreting a timing run or campaign, and before relying on Qwen3.8 timings or decisions measured 2026-11-09 or later |
| [history/events/2b0180d6-953f-4fb4-bd24-b792b9895a4a.md](<history/events/2b0180d6-953f-4fb4-bd24-b792b9895a4a.md>) | Admission history for the MLX_PROFILE_CB measurement rule | repository | active | When auditing how the MLX_PROFILE_CB measurement rule was recorded |
| [topics/661decc8-766b-49bf-8bb4-d5c08f65130f.md](<topics/661decc8-766b-49bf-8bb4-d5c08f65130f.md>) | Measurement rule: judge a performance route on the combined candidate with that switch removed (single-switch results are not additive); between arms that emit different text compare stage time per cycle, not makespan | repository | active | When designing or interpreting a performance comparison, ablation or campaign, or deciding whether a route is worth keeping |
| [history/events/65c4fd4c-a1d3-4065-9cbd-517c4fd537f6.md](<history/events/65c4fd4c-a1d3-4065-9cbd-517c4fd537f6.md>) | Admission history for the combined-candidate and different-text measurement rules | repository | active | When auditing how the combined-candidate and different-text measurement rules was recorded |
| [topics/0ab63f2a-e2ef-4564-a7f0-5ef4a130a1d7.md](<topics/0ab63f2a-e2ef-4564-a7f0-5ef4a130a1d7.md>) | Qwen3.6 four-agent repair workload fails the workflow oracle in about one run in twelve with or without new routes (control 18/20, candidate 19/20); few runs prove nothing about quality there | repository | active | When judging Qwen3.6 quality from four-agent runs, reading a workflow-oracle failure, or sizing a Qwen3.6 campaign or release gate |
| [history/events/2b362872-a9d0-457c-8d8b-fe5764f10ac1.md](<history/events/2b362872-a9d0-457c-8d8b-fe5764f10ac1.md>) | Admission history for the Qwen3.6 workflow-oracle failure-rate finding | repository | active | When auditing how the Qwen3.6 workflow-oracle failure-rate finding was recorded |
| [topics/48186f89-b2f4-45cd-a3f7-7974220b91b1.md](<topics/48186f89-b2f4-45cd-a3f7-7974220b91b1.md>) | Workflow fact: a change needing new mlx-native code is reproducible only after the crate is published and pinned; the pin lives in Cargo.toml, Cargo.lock, the provenance script constants and WITNESS_NATIVE_VERSION | repository | active | Before landing or measuring a change that depends on new mlx-native code, and when bumping the mlx-native version |
| [history/events/607b1501-ab73-47e9-aed1-5d718a2c22b1.md](<history/events/607b1501-ab73-47e9-aed1-5d718a2c22b1.md>) | Admission history for the mlx-native pin workflow fact | repository | active | When auditing how the mlx-native pin workflow fact was recorded |

## Omitted contexts

- `productContext.md`: No distinct useful source context was established during initialization
- `systemPatterns.md`: No distinct useful source context was established during initialization
- `techContext.md`: No distinct useful source context was established during initialization
