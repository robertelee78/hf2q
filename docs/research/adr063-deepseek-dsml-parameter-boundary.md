# DeepSeek single-call grammar/parser boundary repair

Source candidate based on e4164099, separate from the preserved unshippable
router-trigger spike. No tests/builds/models executed by this source author.

The failed N4 run retained at `deepseek390-n4-20261026T144546Z` emitted two
read_file calls in unary turn0 for agents0/3. Native terminal-single-tool logs
showed grammar acceptance. The old raw rule accepted the parameter closing tag
as data, permitting an ambiguous parse through both invokes. Unary lazy token
activation already existed; routing-only activation was an incorrect hypothesis
and its isolated source spike is excluded from this patch.

Root's old-code control compiled the actual three-tool request with default
nonparallel semantics, armed the actual token scanner and fed split-marker token
bytes through accept_grammar_token. One parsed call passed, trigger activated,
and the two-call rejection assertion failed exactly as expected. That is root
observed negative evidence, not author execution or a fixed-code result.

The new focused lexical module excludes the exact native parameter terminator
with prefix-state transitions across all raw characters, including escapes.
Length composition counts Unicode scalars and refuses products beyond existing
4MiB/262144-rule grammar budgets. The old bounded raw atom counted a backslash
pair as one repetition despite the native parser retaining both characters;
new counts reflect actual raw string length. No all-angle-bracket ban is used.

Accepted DS raw alternatives and fixed enum/const literals are delimiter safe.
Nested/untyped JSON is restored through explicit DeepSeek lexical context in
the shared recursive schema traversal; other families retain their previous
context. DS-only JSON-string DFA supports generic and bounded values, recursive
objects/arrays, additional properties and freeform keys. Wildcard declared-key
exclusion reserves raw '<' but retains JSON escape spellings of that key scalar.
Literal enum/const values and declared names containing the sentinel refuse.
Pattern/format bodies use their unchanged compiled grammar when its parsed
terminal alphabet excludes at least one sentinel scalar. Existing anchored
identifier/CWE patterns and date formats therefore remain supported; broader
bodies without this conservative proof explicitly refuse pending safe
intersection. No regex-text substring heuristic or all-schema promise is used. Existing nested
question/todowrite positive tests were neither removed nor weakened.

New CPU-only controls use the actual request compiler and token acceptance,
cover one/two calls, ordinary source code/partial sentinels/backslashes, Unicode
lengths and reserved-marker rejection, safe enum/const values, parallel=true,
and unsafe literals/pattern/resource refusals plus nested JSON/key/length positives
and delimiter negatives. Root validation passed seven actual handler/token controls,
164 registry controls (two existing ignored), 66 JSON-schema controls, and the
locked all-target/all-feature check. The old-code two-call rejection failed
before this change; the same control passes after it. A new exact
combined server/client packet and real N4 tool/prefix/cleanup proof are required
before any repaired runtime is accepted. Neither the negative control nor this
source candidate is a performance or whole-Q5 result.
