# ADR-061: Release checklist: crate, website, and real use

- **Status**: accepted (owner approval in session, 2026-10-07); not yet implemented
- **Date**: 2026-10-07
- **Deciders**: Robert E. Lee (product owner)
- **Tags**: release, crates.io, hf2q.us, qualification
- **Related**: ADR-045 (distribution and `hf2q update`), ADR-058 (release
  tiers), hf2q.us `docs/release-activation.md`

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as
described in RFC 2119.

## Context

A release from this repository has three places users get hf2q from:
the GitHub Release, crates.io, and hf2q.us (which `hf2q update` and the
`install.sh` one-liner read). v0.1.21 reached only the first. The crate
upload was rejected for size (12.28 MiB against the 10 MiB cap) and
nobody updated hf2q.us, so `hf2q update` could not see the release for 26
days. Separately, the GLP and GCD features in v0.1.21 were never used the
way a user would use them before release, and the owner has since seen
`hf2q chat` hard-lock during decode on the 0.1.21 binary.

Nothing here needs new automation. It needs one checklist that is followed
every time.

## Decision

### D1. `docs/RELEASING.md` is the release checklist

Every release MUST follow it. It has three steps:

1. **Check the crate fits.** Run `cargo package --locked` and confirm the
   `.crate` is under 10 MiB.
2. **Release.** Dispatch `release.yml` as today. It tags, publishes the
   GitHub Release, and publishes the crate. If any step fails, the release
   MUST NOT be un-drafted by hand; fix and re-run.
3. **Update the website.** In hf2q.us, follow `docs/release-activation.md`
   for the new version and deploy.

A release is done when a machine on the previous version runs
`hf2q update` and gets the new one, and `cargo install hf2q` installs it.

QE (D2) is not a step of this checklist. A release never waits on it.

### D2. Ad hoc QE playbook: the product used as a user would

`docs/qe-playbook.md` is a hands-on QE process that agents run ad hoc, when
the owner asks, against whatever build the owner names. It is not a release
step and MUST NOT become a CI gate, a workflow step, or a required check; CI
stays simple and fast. The playbook MUST be written so an agent can follow it
from a single request such as "run the QE pass on this build". Its output is
a short written record: what was run, with which binary and model files,
what happened, and anything that felt wrong even if it did not fail.

The playbook covers at least these journeys, and grows whenever a user-facing
bug escapes:

- **J1 Install.** Fresh `$HOME`: run the candidate's installer, then
  `hf2q setup --accept-defaults`, `hf2q doctor`, `hf2q --version`.
- **J2 Chat.** `hf2q chat` with a real local model: a very short first
  message, ten or more turns, and one reply of 512 or more tokens. Note time
  to first token and decode speed. Any pause over a minute is a failure.
- **J3 Agentic coding.** `hf2q serve` with OpenCode on a small coding task
  that needs tool calls and tool results; the tools and arguments must be
  right and follow-up turns must reuse the cached prefix.
- **J4 API.** One normal and one streaming `/v1/chat/completions` request
  with `curl`; the stream ends with `[DONE]`.
- **J5 Convert.** Convert a Hugging Face model to Q4_K_M, serve it, and check
  that it answers sensibly, not just that it emits tokens.
- **J6 GCD.** `--gcd`, `--gcd-schema` with the example schema, and
  `--gcd-schema-locked`, through `hf2q chat` and `curl`.
- **J7 GLP.** `--glp` with an artifact that matches the model, with and
  without `--glp-alpha`, then back to baseline.
- **J8 Update and uninstall.** From the previous release: update, confirm the
  new version, `hf2q update --rollback`, update again, `hf2q uninstall
  --yes`, and confirm config and models survive.
- **J9 Vision.** One image turn through `hf2q chat` and one through the API
  using the guide's red-image check.

Problems found during QE are filed as issues and MAY lead to research and
ADR corrections.

GLP and GCD shipped in v0.1.21 without ever being used the way a user would
use them, so they get a dedicated deep ad hoc pass now, beyond J6 and J7. It MUST exercise every GLP and GCD path the guide and README document:
`--gcd`, `--gcd-schema`, `--gcd-schema-locked`, `--glp` with a path, `--glp`
bare auto-discovery, `--glp-alpha`, `--gcd --glp` together, `hf2q calibrate`,
and switching back to baseline. Each is run through `hf2q chat`, the API, and
OpenCode where it applies, on every model family the docs claim, checking
that outputs actually change the way the docs say, that ordinary chat and
tool calling still work while the feature is on, that bad inputs fail with a
clear message, and that the documentation matches what really happens.

### D3. Fix the decode hard-lock before the next release

Reproduce it on the 0.1.21 binary, find the cause, fix it on main, and
confirm a long `hf2q chat` session no longer locks.

### D4. 0.1.21 stays off crates.io

Its tag cannot produce a crate under the size cap. v0.1.22 is the next
release on all three channels and is the first run of the checklist.

### D5. ADR-058 status correction

ADR-058 says v0.1.22 shipped; it has not. Correct the status line.

## Consequences

- Every release gets the crate and website steps that were previously
  forgotten.
- Agents can run a thorough hands-on QE pass whenever the owner asks.
- No new workflows, jobs, or CI gates. QE is ad hoc and outside both CI and
  the release checklist by design.

## Links

- Failed v0.1.21 publish run: https://github.com/robertelee78/hf2q/actions/runs/34404507584
- hf2q.us activation steps: `docs/release-activation.md` in the hf2q.us repository
