# Affine/DWQ retirement planning packet

Start with the [complete RCA and cleanup plan](../mlx-affine-retirement-plan-2026-10-06.md).
**Implementation has not started. This packet is published for planning and
handoff; the reviewed plan and review hashes are unchanged.**

GLM-5.3's [final review](glm53-review-final.md) is **READY WITH NONBLOCKING NOTES**
for later execution. This is a plan-level verdict, not execution, release or
publication approval. The [initial review](glm53-review-initial.md) identified
four blockers; the revised plan closes all four:

| Finding | Resolution in the reviewed plan |
|---|---|
| B1: P10 shell wrapper omitted | Retire `peer_parity_run.sh` and its active links with the obsolete experiment. It directly invokes external peers; it does not invoke the Rust harness. |
| B2: incomplete diagnostic field policy | Preserve both `engine_config.dwq_overlay` and `active_controls.dwq_overlay` as literal-false v1 wire fields, with emitter and endpoint tests specified. Remove all executable overlay state. |
| B3: separate DWQ smoke/threshold scaffold omitted | Retire the Skipped-as-success DWQ route and its architecture-only threshold/report closure; reject unsupported smoke selectors before dry-run/external work; preserve unrelated quality code and real GGUF fixtures. |
| B4: remaining helper ownership unclear | Retire both old peer wrappers and their exclusive metrics/env-lock closure. Preserve independently used serving helpers and corpus/PPL code; recheck new callers on execution main. |

The remaining review note is part of R4's execution checklist: search active
docs for `max_median_kl`, `median KL`, `ppl_ratio_dwq46`, `ppl_ratio_dwq48`, and
the retired ADR-012 threshold claims. The obsolete KL field retires with the
DWQ architecture struct; this is **not** permission to change unrelated current
family quality gates or historical measurement records. No plan blocker remains.

## Evidence and reproducibility

- [Inventory](inventory.json) and [read-only measurement script](measure.py):
  committed source at `cca6425860c1fb770c6223c2bbdfa5d2d769dc70`.
- Original selected affine spans: **2,890 physical lines**; review-added
  architecture definitions: **65**; selected generic loader helpers/tests to
  preserve: **151**. These are partial source footprints, including comments
  and tests, not a total deletion budget or a performance measurement.
- Whole-file context counts include the 1,710-line P10 harness (21 tests,
  eight ignored `NotMeasured` cells), its helper closure and retired scripts.
- [Review receipt](review-receipt.json): exact plan/artifact hashes, provider,
  model, session and verdict. Provider/model identity was checked against
  OpenCode's stored assistant-message metadata: `opencode/glm-5.3` throughout.

The first review attempt encountered worktree-read restrictions and briefly
consulted a different checkout. It was stopped; read access was corrected,
and the reviewer explicitly discarded those findings and re-verified against
the pinned worktree before issuing its initial verdict. The saved initial and
final verdicts are the corrected, source-bound reviews. Machine-specific
checkout paths in the archived prose are normalized; raw response hashes are
recorded in their headers. No model fallback or substitute reviewer was used.

Validation performed during the initial RCA/review pass: repeat inventory generation and byte
comparison, all selected anchors/spans against pinned Git bytes, script syntax,
relative Markdown links, artifact/plan hash checks and `git diff --check`.
Product-source/dependency/test/script/CI diff is empty. That initial pass ran no build, Metal/model test or performance measurement,
and performed no commit, push, merge or GitHub task creation. Subsequent
publication of this packet and creation of its tracking project are planning
actions, not cleanup execution. The earlier local benefit/ADR drafts are separate
unpublished work; do not merge them as an active affine implementation mandate.

Execution starts at R0 on the then-current main. The cache proof-or-epoch,
current caller checks, guard self-test and hardware acceptance remain mandatory.
The repository has unrelated AWA-managed work but no unambiguous affine work
key was found; its phases were left unchanged. Assign the approved cleanup
work its own keys before executing, using the required `robertelee78` account.
