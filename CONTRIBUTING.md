# Contributing

Thanks for the interest. `hf2q` is a small project; the contribution
loop is intentionally narrow so the codebase stays coherent.

## Before you start

1. Open an issue describing the problem or the change before writing
   code. Most accepted changes start with a written rationale —
   often an Architecture Decision Record (`docs/ADR-NNN-*.md`).
2. Skim the **ADR ledger** in `docs/` for context. Every load-bearing
   choice in the codebase (the mlx-native backend, the
   TurboQuant KV cache, the persistent block-prefix cache,
   per-arch parity gates) has a numbered ADR explaining why.
3. Read `docs/ARCHITECTURE.md` for the source-grounded map of
   `src/`.

## Issues

- **Bug reports**: include the exact `hf2q --version`, macOS version,
  Apple Silicon generation, the model + quant, and the command line
  used. A `hf2q doctor` snippet is usually the fastest way to share
  environment context.
- **Feature requests**: describe the use case before proposing an
  API. If the feature touches a model architecture, expect the
  conversation to route through an arch-onboarding gate (see below).

## Pull requests

- **Branch off `main`.** This repo uses a single mainline; long-lived
  feature branches are rare.
- **Tests before code.** The project is TDD-heavy: every ADR closes
  only when its acceptance tests + smoke prompts pass. New behavior
  needs a failing test first.
- **Match the existing code style.** No formatter beyond `rustfmt`
  defaults. Files over ~500 lines exist in legacy paths; do not
  enlarge them — split into modules instead.
- **Keep commits focused.** Each commit should compile and pass
  `cargo test` for the area it touches.
- **Update the ADR** that governs the area you're changing. If no
  ADR covers it yet, draft one as part of the PR.

### Running checks locally

```bash
cargo build --release
cargo test
cargo run -- doctor
```

`cargo test` exercises the full integration suite in `tests/` plus
the per-module `mod tests` blocks. A few tests need GGUF fixtures
that download on first run.

## Adding a new model architecture

This is the most common request. Follow the checklist in
`docs/arch-onboarding.md`. The short version:

1. Add the registry entry under `src/arch/entries/<arch>.rs`
   (single source of truth: tensor catalog, quality thresholds,
   smoke prompt, MTP / vision flags, disk floor).
2. Add the tensor-rename + MoE-merge logic under
   `src/models/<arch>/`.
3. Stand up the forward graph under
   `src/inference/models/<arch>/`.
4. Wire the smoke prompt into `hf2q smoke` and watch it pass.
5. Only after smoke passes: open the convert + serve paths.

Reviewers will reject forward-pass code that landed before the
registry entry, tensor catalog, and smoke prompt are in place.

## Security disclosures

See [`SECURITY.md`](SECURITY.md). Do **not** open a public issue for
a security report.

## License

By contributing you agree your work is licensed under both the
Apache-2.0 and MIT licenses, matching the project license
(`Apache-2.0 OR MIT`). No CLA is required.
