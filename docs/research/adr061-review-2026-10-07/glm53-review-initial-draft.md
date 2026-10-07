<!-- Archived model response from OpenCode opencode/glm-5.3. Raw response SHA256: 3548eec711b1de2e59f5a0c3d33c4c841137d06e1af2b86b890ec156ec9cc3cc -->

I'll review this ADR against the repository source. Let me read the key files.
Now let me verify the ADR's factual claims against git history and the CLI surface.
# ADR-061 Review

## Findings

**1. Blocker — D4 (and G3): the in-run hf2q.us gate is impossible as sequenced.**
D4 puts "fails when … the hf2q.us stable record disagree[s]" in a final job of `release.yml`, but `release.yml` publishes the GitHub Release ("Publish the complete GitHub release", `.github/workflows/release.yml:527`) and never touches hf2q.us; D5 promotes the website only *from* a trusted release *run*, i.e., afterwards. As written, either every legitimate release fails its final job (stable record still names the old version), or the gate is silently skipped. G3's "passes against v0.1.22 when complete" inherits the same ambiguity.
Replace D4's first paragraph with:

> A final job in `release.yml` fails when the GitHub Release is public while crates.io disagrees with it; both are published by the same run, so the check is complete when it runs. The hf2q.us stable record cannot be checked inside that run, because the website is promoted only afterwards (D5); that half of the gate runs in the runbook immediately after promotion dispatch, and daily thereafter, against the latest public release. The daily gate treats a public release younger than the runbook's promotion window (D2) as in-flight, not drifted.

**2. Blocker — D6/J1/J8: the journeys cannot run "before any release is dispatched" through hf2q.us.**
D6 says journeys run before dispatch, but J1 (`curl … https://hf2q.us/install.sh | sh`) installs whatever is published *today* (0.1.21), not the candidate, and J8's `hf2q update` from 0.1.21 necessarily reports "current" (`src/distribution/standalone/update.rs:73-77` returns `Current` when `release.version <= current`) because the stable record still names 0.1.21 pre-promotion. D1 bullet 4 and G7 correctly place the live update probe *after* promotion. As written, J1 and J8 are unexecutable in the position the ADR puts them.
Add to D6 preamble:

> Journeys run against the exact release candidate, not the currently published release. Before dispatch hf2q.us still serves the previous version: J1 installs through the candidate's rendered installer with `HF2Q_RELEASE_BASE_URL` pointed at the staged exact release assets (the same override `release.yml` uses at line 309), and the live hf2q.us install and update legs run after promotion as part of the D1 probe, not as pre-dispatch journeys.

Replace J1's first sentence with:

> Fresh account: run the candidate's rendered `install.sh` with `HF2Q_RELEASE_BASE_URL` pointed at the staged exact release assets, then `hf2q setup --accept-defaults`, `hf2q doctor`, `hf2q --version`. After promotion, repeat once against live `https://hf2q.us/install.sh` and record it as part of the D1 probe.

**3. Blocker — Implementation sequence step 3: "main-built candidate" is the wrong artifact.**
The hard-lock was observed on the *standalone* 0.1.21 binary; a locally built main binary differs in Developer ID signing, hardened runtime, and notarization (`standalone-candidate.yml:207-311` signs/notarizes in a protected job; `update.rs:314-356` verifies those properties at update time). A local build may neither reproduce the hard-lock nor qualify the bytes users receive — contradicting ADR-045 §2 ("tested from the bytes that the channel actually installs") and undermining D7's premise.
Replace step 3 with:

> 3. D6 journeys J1 through J9 on the exact signed standalone candidate produced by `standalone-candidate.yml` from the exact release SHA — the same binary the release will attach — with its SHA-256 recorded in `docs/qualification/<version>.md`. Any commit landing after the journeys requires re-running them on the new candidate.

**4. Major — D1 bullet 3: field-level correctness still lets `hf2q update` fail.**
`update.rs` rejects, at the record URL: redirects/origin changes (`fetch_stable_release`, lines 104-118), any content encoding (lines 262-266), records over 4 KiB (line 27), and any deviation from the byte-exact canonical JSON encoding including trailing newline (lines 148-155); it also rejects a candidate whose Developer ID team/identifier differs from the *installed* binary (lines 331-337). A promotion with correct fields but re-serialized JSON, a gzip'ing or redirecting Apache, or a rotated signing identity produces a "done" release every D1 field check passes while every user's update fails. D1 also never probes a *fresh* install of the new release.
Replace bullet 3 with:

> `https://hf2q.us/releases/stable-aarch64-apple-darwin.json` answers a direct HTTP 200 at that exact URL (no redirect, identity encoding, no-store cache headers, at most 4 KiB), is byte-identical to the release's `stable-aarch64-apple-darwin.json` asset (canonical managed JSON), names `X.Y.Z` with the release binary's size and SHA-256, and the candidate's Developer ID team and identifier match the previous standalone release's; `https://hf2q.us/install.sh` redirects to the `vX.Y.Z` installer asset, and one fresh `curl -fsSL https://hf2q.us/install.sh | sh` install completes and reports `hf2q X.Y.Z`.

**5. Major — D2: the runbook never gates dispatch on the D6 journeys.**
D2's fixed order has no journey step, and D6 is prose with no enforcement. A tired operator can dispatch a release with zero journeys run — exactly the failure mode this ADR exists to prevent. Add to D2:

> The runbook's first step is the qualification gate: it refuses to dispatch unless `docs/qualification/<version>.md` exists, records every journey as pass or an explicit owner waiver, and binds the exact source SHA and candidate binary SHA-256 it qualified; the dispatch's `commit_sha` must equal the qualified SHA.

**6. Major — D7: bisect range contradicts the ADR's own Context, and "J2 is written so that it would have caught it" is unproven.**
Context item 4 says "whether that is a regression from the 0.1.21 changes or an older defect is unknown," yet D7 bisects only "between the v0.1.21 tag and main" — an unfounded window. The "would have caught it" claim is circular until the hard-lock is reproduced *by J2 itself*. Replace D7's second paragraph with:

> The hard-lock is first reproduced by running J2's exact script against the v0.1.21 standalone binary — only a reproduction on 0.1.21 proves the journey catches it; if J2's script does not reproduce it there, J2 is strengthened first. The regression window is not assumed to start at the tag: bisect from the last release on which the J2 script passes (possibly older than v0.1.21) to the candidate, root-cause, and fix. J2 is then run unchanged on the candidate; the release is blocked until it passes.

**7. Major — J2: "the session must never stall" is unobservable, and the model artifact is unpinned.**
No timeout, no inter-token bound, no watchdog — a session that hangs for 40 minutes and then emits one token is arguably "never stalled." G6 ("J2 passing") therefore has no objective meaning. "Local Qwen3.8 abliterated Q4_K_M GGUF" is also ambiguous between the hf2q-converted guide pair (text digest `1ee55c65…`, ADR-045 Slice B) and a community GGUF; D7's reproducibility depends on which. Replace J2 with:

> **J2 Chat.** `hf2q chat` against the exact Qwen3.8 abliterated Q4_K_M GGUF named by path and SHA-256 in the record (state whether it is the guide pair or the artifact that reproduced the hard-lock): a first message under sixteen tokens, a multi-turn conversation of at least ten turns, and one reply of at least 512 generated tokens. Every turn must complete or return an error within 10 minutes; time to first token and inter-token silence are recorded; any silence over 60 seconds is a stall failure; the decode-rate median of the long reply is noted.

**8. Major — J8: misplaced update legs, and `hf2q update --rollback` is never exercised.**
Rollback is the shipped user recovery path for a bad release (`src/cli.rs:277-282`; ADR-045 §5) — precisely the D7 scenario — yet no journey touches it. Meanwhile J8's `hf2q update` legs cannot run pre-dispatch (finding 2). Replace J8 with:

> **J8 Update, rollback, and uninstall.** Pre-dispatch, on a machine holding the previous standalone release: install the candidate over it through the staged channel, confirm the previous binary is retained as `.hf2q-previous`, run `hf2q update --rollback` and confirm the previous version runs with data preserved, re-update to the candidate, then `hf2q uninstall --yes` and confirm configuration and model data are preserved. The live `hf2q update --check`, `hf2q update`, `hf2q --version` legs from the previous published release run after promotion as the D1 probe (G7).

**9. Major — missing journey: vision.**
The canonical guide makes vision a basic onboarding step (`docs/getting-started.md` §5 "Prove vision with one request", with the red-fixture check) and the guide pair is multimodal (Qwen3-VL projector). A release that regresses the vision path would pass J1–J8. Add:

> **J9 Vision.** With the guide's projector pair served, one image turn through `hf2q chat` and one through `/v1/chat/completions` with an `image_url` message, per the guide's vision check; the reply must demonstrate the model saw the image (the guide's red-fixture assertion), not merely return HTTP 200.

Update G5 and implementation-sequence step 3 to "J1 through J9."

**10. Major — J5 has no output-quality gate.**
"Serve and chat with the result" passes on any token stream; a quantization bug that mangles weights still chats. Repo rules require parity or a documented quality threshold before accepting conversion output. Append to J5:

> … then serve and chat with the result, passing the guide's fixed-prompt coherence probes (or a recorded parity result against the F16 reference); emitting any tokens is not a pass — the record must show the converted model answers the probes correctly.

**11. Major — G2 cannot be observed without polluting main.**
`standalone-candidate.yml` refuses any SHA that is not an ancestor of `origin/main` ("Verify immutable main identity", lines 50-59), so a deliberately oversized crate cannot be dispatched on a scratch branch; making main itself oversized to test the gate is unacceptable. Replace G2 with:

> G2: the D3 size check fails an oversized crate before signing, proven by a fixture-driven contract test of the check itself (the repository's `test_*_contract.sh` pattern); a live drill, if run, uses a temporary oversized commit on main reverted immediately, with both SHAs recorded — `standalone-candidate.yml` rejects non-main SHAs, so a scratch branch cannot demonstrate it.

**12. Major — G1/G3: refusal paths are never exercised by a green release.**
"Refuses success until the D1 probe passes" (G1) and "fails against a hand-constructed mismatch" (G3) are negative-path claims; a successful v0.1.22 run demonstrates neither, and G3's "hand-constructed mismatch" is operationally undefined (it implies deliberately publishing a broken public state). Replace G1's tail and G3 with:

> G1: … and its refusal path is proven once by a recorded drill — a run with one channel deliberately withheld (e.g., promotion blocked) must print the partial-release report naming the missing channel.
> G3: the channel consistency gate fails against a fixture-driven mismatch (contract test over recorded responses) and passes against v0.1.22 only after promotion completes; the pre-promotion window is defined as in-flight, not a failure.

**13. Major — Consequences omits the security skew D8 creates.**
crates.io's newest version is 0.1.20, published 2026-08-28 (verified via crates.io API); main carries the rustls fix for RUSTSEC-2026-0285. Every week the new process takes, Cargo-channel users run a crate with a known advisory. Add to Negative consequences:

> - crates.io users remain on 0.1.20 (2026-08-28) for the entire duration of the new process, without the RUSTSEC-2026-0285 rustls fix main already carries. A security-only release may use the fast tier (its diff touches no inference/grammar path) with a reduced mandatory journey set (J1, J2, J4, and the update legs) and the owner's recorded waiver, to bound advisory exposure.

**14. Minor — Consequences omits alarm fatigue and the non-atomic promotion window.**
The daily drift check will see a *true* mismatch during every normal release (public release before the website promotion completes), and the promotion itself writes record, selector, and reload non-atomically. A check that fires on every routine window trains the operator to ignore it. Add to Neutral:

> - The drift check treats a public release younger than the runbook's promotion window (one hour) as in-flight, not drifted; the promote-release deploy stages the stable record and Apache selector before a single reload, and the served record carries no-store cache headers.

**15. Minor — D4/D5: two daily checks, ambiguous ownership.**
D4 says the gate "also runs on a daily schedule"; D5 says hf2q.us has "a daily drift check." Two repos, two overlapping checks, no assignment. Add to D4:

> The hf2q repository's daily gate compares crates.io with the GitHub release; hf2q.us's daily drift check compares its served stable record and installer redirect with the GitHub release. One check per repository, no duplication.

**16. Minor — J3/J4: observability looseness.**
"Prefix reuse observed across turns" names no metric source; "triggers at least two tool calls" does not require the calls to be *correct* (repo rule: "structurally valid JSON with the wrong tool or arguments is a fail"); "valid per the OpenAI contract" for SSE should reuse the repo's proven criterion (ADR-045 Slice B: "reconstructable SSE ending in `[DONE]`"). Replace the J3 tail with:

> … with prefix reuse recorded from the server's cached-token counts per turn (a follow-up turn must not recompute the unchanged prefix), and the final answer must use the correct tools and arguments and content drawn from the returned results — emitting tool calls is not by itself a pass.

Replace the J4 tail with:

> … the unary response is a well-formed completion object with usage; the SSE stream ends with `[DONE]` and reassembles to the same content.

**17. Minor — factual and sequencing nits.**
(a) "v0.1.21 was tagged on 2026-09-10": the tag was created 2026-09-09T19:46:09Z; the Release went public 2026-09-10T06:39:46Z, 1m53s after the failed run concluded 06:37:53Z (the "two minutes" claim is correct). Fix the date. (b) D9's ADR-058 correction (its status falsely says "implemented (v0.1.22)") is a one-paragraph truth fix for a live document; sequence it as step 0, not step 6 — do not leave a false published claim in place while journeys and infrastructure proceed. (c) G6's "recorded root cause" should name where: "recorded in `docs/adr/` or the qualification record." (d) D6's "fresh account" should permit "a fresh macOS account, or an isolated `$HOME`/`--state-root` with equivalent cleanliness (no prior hf2q state, completion files, or model paths), stating how the journey model artifacts are made available to it" — the release workflow itself uses isolated HOMEs (`release.yml:302-362`).

## Verified-correct claims (no finding)

Checked against source: 59 commits past the tag; `2fa74d22` is the post-tag size fix; 17 files under `src/inference/**` changed (full tier is correctly mandatory); `update.rs` reads exactly `https://hf2q.us/releases/stable-aarch64-apple-darwin.json`; `release.yml` never touches hf2q.us and its step names ("Publish exact package", "Publish the complete GitHub release") match; crates.io serves 0.1.20; `hf2q chat`, `hf2q doctor`, `--gcd`, `--gcd-schema`, `--gcd-schema-locked`, `--glp`, `--glp-alpha`, and the GCD example schema (`examples/recon-opportunities.schema.json`) all exist; `scripts/release_runbook.sh` and `docs/qualification/` do not yet exist, consistent with "proposed."

## Verdict

**Accept with the listed edits; findings 1–3 are blocking and must be resolved before the implementation sequence starts.** The architecture — three-channel definition of done, runbook, promotion-from-trusted-run, journeys — is sound. The blockers are paragraph-level sequencing contradictions (in-run website gate, pre-dispatch hf2q.us journeys, wrong candidate artifact), not wrong design; but as written, the ADR's central promises — "refuses to print a success line until the D1 probe has passed" and "journeys before dispatch" — cannot both be executed as specified, and D7's investigation would be invalidated by qualifying an unsigned local build.
