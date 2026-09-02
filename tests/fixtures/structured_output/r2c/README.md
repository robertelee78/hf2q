# r2c structured-output conformance fixtures

These fixtures exercise generic JSON Schema behavior. The compiler MUST NOT
branch on their names, `$id` values, property names, or paths.

The Stage 6 schema is derived from r2c's Rust/Serde provider-evidence contract
at r2c commit `f0e0fe2de1f472ed7bf73cb67e9a89a0d421976b`, file
`src/pipeline/stage6_securityreview/candidate_family/review_lens.rs`, lines
14–90. The source defines the enum spellings, tagged assessment union, required
fields, and `deny_unknown_fields`. The JSON Schema is an attributed equivalent
written for hf2q's conformance suite; r2c does not check in a canonical schema
document for that Rust type.

The Stage 9 schema is the hf2q/r2c integration acceptance contract supplied by
the product work. The searched r2c worktrees do not contain a producer for the
`primary_cwe`/`related_cwes` object. r2c's `CweId` parser does independently
support the `CWE-<digits>` identifier family and reject empty/unclassified
values, but the stricter nonzero identifier pattern and the abstention
conditional are intentionally recorded here as integration requirements, not
misattributed as an upstream r2c schema.

Each `invalid_*.json` file falsifies an individual assertion. The test suite
must reject it through the same generic grammar used for ordinary callers.
