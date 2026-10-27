# Gemma largest-prompt metadata fixture

This test-only producer prepares four independent requests for the Q5 memory
stress. It uses the production Gemma renderer and Rust tokenizer, not a surrogate
or guessed token count. It starts no engine, device, server or HTTP request.

Each wire has exactly 253,952 input tokens and an 8,192-token completion allowance,
summing to the configured 262,144-token per-slot context. Two are unary and two
request SSE with usage. Independent IDs occur near the start and the final
instruction requests the corresponding `CAPACITY_i` reply. Pairwise common
prefixes are seven tokens, preventing a common large prefix from replacing the
four independent inputs.

The selected tokenizer/template hashes and model identity are fixed in the
producer. Its configuration admits only family, model, bound tokenizer/template
files and an existing private output directory. It rejects padding/truncation,
unknown fields, noncanonical/Git output paths and overwrite. Generated request
bodies remain below the product 8 MiB limit. The canonical coding harness's
256 KiB wire bound is a separate harness policy.

Four pure controls passed. The actual metadata export executed one nonignored
passing test in 46.50 seconds from clean source
`c58bf4fa7ba0423b06743b597cd74233ad896b6d`; helper SHA256
`8765f6e8e68f34775703a01cb938bcd420a4af6ad4e26945536ced9f7040bb4e`.
Independent review checked all ten output files, complete token vectors and
little-endian u32 digests, source units and build receipt. The wire sizes are
508,134 bytes for unary and 508,173 for SSE. No model/admission/quality/prefix or
speed qualification follows from generating these files.

Reproduce this historical fixture from its exact source in an isolated worktree;
keep the primary checkout on main. Supply a private configuration with the
selected local tokenizer/template paths and hashes, and an empty mode-0700
output directory outside Git:

```sh
GIT_COMMIT_SHA=c58bf4fa7ba0423b06743b597cd74233ad896b6d \
HF2Q_GEMMA_LARGEST_PROMPT_CONFIG=/absolute/private/config.json \
cargo test --locked --bin hf2q \
  serve::api::handlers::gemma_largest_prompt_tests::gemma_largest_prompt_exports_four_exact_production_inputs \
  -- --ignored --exact --nocapture --test-threads=1
```

An explicit compiled SHA is metadata, not independent build provenance. The
retained recipe/source/binary receipt supplies that separate binding. Source-unit
checks intentionally refuse a later tree that differs from the frozen producer;
use the archived helper or exact source for historical replay rather than
silently weakening the guard.

The separately reviewed stress client retains the full output allowance even if
EOS occurs early. A successful largest-prompt run can expose full configured
capacity and long-prefill scratch; it cannot by itself prove highest-index writes,
8,192 generated tokens, long decode history, tool-result stress or whole-Q5
acceptance. Its 3,600-second absolute request and 7,200-second whole-run deadlines,
96 GiB operating packet and live memory/cleanup guards are separate from this
metadata producer. The actual model stress is still pending completion.
