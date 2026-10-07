<!-- Archived model response from OpenCode opencode/glm-5.3. Raw response SHA256: 013e46788803ca529b1f6573de18c14a554be1a600c2643ae2b6744e2f86a572 -->

`hf2q update` reads `https://hf2q.us/releases/stable-aarch64-apple-darwin.json` (update.rs:21) — the ADR's premise holds. Now the CLI surface and docs:
One more premise check — what `install.sh` reads by default:
ADR-061 re-review against the checkout. Verified sound first: release.yml publishes and byte-verifies crates.io (`Publish exact package`/`Verify crates.io bytes`, release.yml:463-498) *before* un-drafting the GitHub release (release.yml:527-541), so with D1's no-hand-un-draft rule no path reaches public GitHub without crates.io; `hf2q update` does read hf2q.us (update.rs:21); every J1/J8 command surface exists (`setup --accept-defaults` cli.rs:321, `doctor` cli.rs:147, `update --rollback` cli.rs:282, `uninstall --yes` cli.rs:263); `--gcd --glp` together is legal and documented (cli.rs has no conflict; README:960) while `--gcd`+`--gcd-schema` correctly is not listed (conflict, cli.rs:1124/1387); the example schema and the guide's red-image check exist; D5's premise is factually right (latest tag v0.1.21, Cargo.toml `version = "0.1.21"`).

## Findings

**1. major — Context / D1 (release-is-done definition)**
Problem: The done-definition covers only two of the three channels the ADR itself names. `hf2q update` verifies the hf2q.us stable record; `cargo install` verifies crates.io; nothing verifies the `install.sh` one-liner, which is the first-install path (README:36). The Context also misstates the mechanism: install.sh does not "read" hf2q.us — it downloads from GitHub with the version baked in at render time (install.sh.in:9,15), and hf2q.us merely serves a redirect to that asset (README:124-125). A partial activation (stable JSON updated, install.sh redirect still on the previous version) passes the current done-check while new users install the old release — the exact 26-day failure class this ADR exists to close.
Evidence: update.rs:21; install.sh.in:9,15,84; README:36,124-125.
Replacement (Context, line 16-17): `and hf2q.us (which serves the stable record \`hf2q update\` reads and redirects the \`install.sh\` one-liner to the versioned GitHub release asset).`
Replacement (D1, lines 41-42): `A release is done when a machine on the previous version runs \`hf2q update\` and gets the new one, \`cargo install hf2q\` installs it, and a fresh \`$HOME\` install via \`curl -fsSL https://hf2q.us/install.sh | sh\` followed by \`hf2q --version\` reports the new version.`

**2. major — D1, step 1**
Problem: The checklist omits the version bump. release.yml asserts the release SHA's `Cargo.toml` version equals the dispatched version (release.yml:75-76), and no bump script or documented bump step exists. Followed literally, step 1 sizes the previous version's `.crate` (the known-oversize 0.1.21 artifact) and step 2 fails the version gate — fail-closed, but the checklist's size check validates the wrong artifact and the release cannot dispatch without an unlisted action.
Evidence: Cargo.toml:4 (`0.1.21`); release.yml:75-76; no bump tooling in `scripts/`.
Replacement (D1, step 1): `1. **Bump and check the crate fits.** Set the new version in \`Cargo.toml\` and refresh \`Cargo.lock\` on main, then run \`cargo package --locked\` and confirm \`target/package/hf2q-<version>.crate\` is under 10 MiB.`

**3. major — D2, GLP/GCD deep pass (MUST list)**
Problem: The MUST sentence demands exercising "every GLP and GCD path the guide and README document," but the enumerated list omits `--glp` with an explicit Hub repository or `huggingface.co` file URL — a distinct documented resolution form alongside local path and bare discovery. The list contradicts its own governing sentence.
Evidence: README:1009-1012 ("Explicit Hub repositories and file URLs resolve to immutable revisions"); src/inference/glp/discovery.rs:160 ("local file, owner/repo, or https://huggingface.co/owner/repo/resolve/revision/file.gguf").
Replacement (line 86-87): `\`--glp\` with a local path, \`--glp\` with an explicit Hub repository or \`huggingface.co\` file URL, \`--glp\` bare auto-discovery,`

**4. major — D2, deep pass (tool-calling check)**
Problem: "ordinary chat and tool calling still work while the feature is on" directly contradicts the documented `--gcd-schema-locked` contract: locked mode rejects requests that replace or defer the schema before streaming, and requests with tool definitions must use `tool_choice: "none"`. Taken literally, an agent will either file a false bug (correct rejection observed as failure) or miss a real one (tool call passing under locked). Related documented paths also missing from the MUST list: unlocked `--gcd`/`--gcd-schema` defaults deferring to explicit `grammar`, `response_format`, `json_schema`, or `structured_outputs` requests, and a selected tool grammar taking precedence over an unlocked response default.
Evidence: README:977-985.
Replacement (lines 90-92): `Each is run through \`hf2q chat\`, the API, and OpenCode where it applies, on every model family the docs claim, checking that outputs actually change the way the docs say, that ordinary chat and tool calling still work while the feature is on — except under \`--gcd-schema-locked\`, where the documented contract is that requests which replace or defer the schema are rejected before streaming and requests with tool definitions must use \`tool_choice: "none"\` — that unlocked defaults defer to an explicit \`grammar\`, \`response_format\`, \`json_schema\`, or \`structured_outputs\` request, that bad inputs fail with a clear message, and that the documentation matches what really happens.`

**5. major — D2, J7 / deep pass (GLP artifact sourcing)**
Problem: No instruction on where a checkpoint-matching GLP artifact comes from. Bind refuses mismatched checkpoint/site/width by design, bare `--glp` rejects ambiguous discovery by design, and the guide warns the example Qwen checkpoint is not compatible with stock-Qwen vectors. Without sourcing guidance an agent cannot execute J7/deep pass from one request and will read documented typed refusals as QE failures.
Evidence: src/inference/glp/bind.rs:27-34; getting-started.md:305-316; README:1010-1012; calibrate is DeepSeek-V4 only (cli.rs:1138, README:1023).
Replacement (append to the deep-pass paragraph, after line 92): `Matching artifacts come from \`hf2q calibrate --out <file>\` (DeepSeek-V4 only) or a published checkpoint-matching artifact; a bind refusal for a mismatched checkpoint, site, or width and a bare-\`--glp\` ambiguity rejection are documented typed failures, not QE findings.`

**6. minor — D2, deep pass (families unnamed)**
Problem: "every model family the docs claim" forces a cross-repo doc hunt and invites running `hf2q calibrate` on Qwen, which the CLI restricts to DeepSeek-V4.
Evidence: README:1000-1002 (GLP application paths: Qwen 3.5/3.6/3.8 and DeepSeek-V4); cli.rs:1138 ("v1: DeepSeek4 architecture only"); README:1023.
Replacement (line 89): `...where it applies, on every model family the docs claim for it — Qwen 3.5/3.6/3.8 and DeepSeek-V4 for GLP application, DeepSeek-V4 only for \`hf2q calibrate\` — checking that...`

**7. minor — D2, J6 (example schema unnamed)**
Problem: "the example schema" is resolvable but unnamed; name the path so the journey runs from one request.
Evidence: `examples/recon-opportunities.schema.json` exists; README:948,976.
Replacement (line 72): `\`--gcd-schema\` with \`examples/recon-opportunities.schema.json\`,`

**8. minor — D2 (chat flag scope trap)**
Problem: Chat GCD/GLP flags only configure a server chat itself spawns; they do not reconfigure an endpoint selected with `--url`. An agent attaching via `--url` with flags would observe them silently not applying and could file a false bug.
Evidence: README:1031-1032; cli.rs:1093-1095.
Replacement (append after line 92, same paragraph as finding 5's sentence): `GCD and GLP flags on \`hf2q chat\` take effect only when chat starts its own server; on a chat attached with \`--url\` they do not reconfigure the endpoint, and that is documented behavior.`

**9. minor — D5 (no replacement text)**
Problem: D5 mandates correcting ADR-058's status line but does not state the corrected line, and the accurate characterization is "implemented on main, unreleased" (the tier guard and workflow_dispatch CI acceptance are in release.yml:79-106,108-125; `scripts/release_gate_preflight.sh` exists; no v0.1.22 tag).
Evidence: ADR-058:3-5; release.yml:79-125; tag list ends at v0.1.21.
Replacement (lines 105-106): `ADR-058's status line says the tier work is "implemented (v0.1.22)"; v0.1.22 has not shipped. Replace the status line with: "Accepted — implemented on main, first ships in v0.1.22: host preflight in the qualification workflow, fast-path tier guard in publish, exact-SHA CI check relaxed to accept \`workflow_dispatch\` runs."`

**10. minor — D2, deep pass (hidden alias unlisted)**
Problem: README documents the hidden `--uncensor` alias for `--gcd`; under the "every path the README documents" rule it is missing from the MUST list (low impact — same code path as `--gcd`).
Evidence: README:967; cli.rs:1090,1373.
Replacement (line 86): `\`--gcd\` (including its hidden \`--uncensor\` alias), \`--gcd-schema\`, ...`

All findings are text-level replacements inside the owner's constraints (no CI, no workflows, no automation; QE stays ad hoc). The decision structure, the crates.io-before-GitHub ordering argument, and the ad hoc QE framing are consistent with the repository.

VERDICT: SIGN-OFF WITH NOTES
