# Security Policy

## Supported versions

`hf2q` is at `0.1.x`. While the API surface is still pre-1.0, only
the latest published release receives security fixes.

| Version | Supported          |
| ------- | ------------------ |
| `0.1.x` | :white_check_mark: |
| `< 0.1` | :x:                |

## Reporting a vulnerability

Please report security issues privately, **not** through a public
GitHub issue.

- **Email:** `robert@agidreams.us`
- **Subject line:** `hf2q security: <one-line summary>`

Include in the report:

1. A description of the issue and the impact you observe.
2. The exact `hf2q --version`, macOS version, and Apple Silicon
   generation if relevant.
3. A minimal reproduction — the smallest model + command line that
   triggers the issue.
4. Any proof-of-concept code or input file. Attachments are welcome.

## What to expect

- **Acknowledgement** within 5 working days.
- **Initial assessment** (confirmed / not reproducible / out of
  scope) within 14 days.
- **Coordinated disclosure**: by default we aim for a fix released
  within 90 days of acknowledgement. The window may be shortened if
  there is evidence of active exploitation, or extended for issues
  that need an upstream fix in `mlx-native` or another dependency.
- **Credit**: with your permission, the release notes for the fix
  will credit you (handle or name).

## Out of scope

- Issues that require the attacker to already have local code
  execution on the machine running `hf2q`.
- Resource-exhaustion via crafted but legitimate model files (large
  context, large vocabularies) — these are documented operational
  limits, not vulnerabilities.
- Findings against the `_bmad/`, `.claude/`, `.swarm/`,
  `.cfa-archive/` directories. Those are local agent / tooling
  state, are gitignored, and are not part of the published crate.

## Scope clarification

In-scope components:

- The `hf2q` binary and library code under `src/`.
- Default convert / inference / serve paths.
- HTTP request handling in `src/serve/`.

Out-of-scope dependencies (please report upstream):

- `mlx-native` — https://github.com/robertelee78/mlx-native /
  https://crates.io/crates/mlx-native.
- `tokenizers`, `minijinja`, `reqwest`, `axum`, `tokio`, and other
  third-party crates — please follow their projects' security
  policies.
