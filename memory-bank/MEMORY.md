---
schema: "agent-memory/v1"
project-id: "479fefc0-3903-4e30-b7fd-877c69979b81"
router-updated: "2026-11-01T04:03:03.948143+00:00"
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

## Omitted contexts

- `productContext.md`: No distinct useful source context was established during initialization
- `systemPatterns.md`: No distinct useful source context was established during initialization
- `techContext.md`: No distinct useful source context was established during initialization
