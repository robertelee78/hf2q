# ADR-063 Q5 independent design review record

Status: revisions requested, no runtime approval. Design implementer: `q5-design-agent`; integration owner: `codex-adr063-q5`. These are independent design reviews, not a Q5 acceptance verdict or a product approval.

The first source review by `q5-review-agent` checked hf2q `5b385d4ff580cc2c9cc70cdb425809b0c64dff63` and native `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`. It required separate process/host caps, optional-versus-unknown ceilings, independent Q5 memory and Q3/Q4 speed acceptance, atomic reservations, complete native retention lifetimes and bounded pressure handling. The reviewer did not approve runtime execution.

The second read-only review by Opus checked proposed design commit `94550a85` and hf2q source. Its sandbox excluded the native checkout, so it explicitly did not verify native source claims. Its verdict was request changes before concrete design approval. It found the following required corrections:

1. Include the fifth eager DeepSeek boot cache alongside four serving sessions.
2. Close the category schema with Bound / ProvenZero / Unknown states; include projectors, grafts, logits and vision embeddings.
3. Specify each family's anchor overlap and host capacities, rather than use incomplete existing payload-length estimates.
4. Reserve serialized old-plus-new growth and charge plan metadata rebinding.
5. Count pool instances per dispatch thread and reject unsafe active release in production.
6. Bound busy/pending/constructing persistence images and hydration; define lock composition with the model pool.
7. Define an external sampler, bounded drain/termination and post-kill recovery.
8. Verify macOS clean mapped-page accounting; use a weight-specific cross-check.
9. Freeze and enforce a conservative GPU working-set policy separately from host/process caps.
10. Define complete aggregate competition observations, privileges and inaccessible-process rejection.
11. Select native counters/lifetime ticket coverage before implementation.
12. Preserve durable review provenance and a separate approval boundary.

The corrected v2 proposal addresses these as design decisions and keeps numerical bounds, metric calibration, workload packets and measured watchdog guarantees explicitly unproven. A new independent review is still required; recording these revisions does not self-approve them. The pure ledger slice must reject every real packet with unknown scratch or telemetry and cannot substitute for full Q5 qualification.

## Private raw review identities

The original full reviewer outputs are retained in private campaign evidence. Their raw UTF-8 SHA-256 identities are:

- q5-review-agent source review: `38661a14391846c78628e4b23399c2760dd2b22135c14ed74000dad36c5ab080`
- Opus review of design94550a85: `88a164c4fe6a565e30fb80403f21d29ff6b668c9b643eadc41288356eb2c4d80`


## Independent v2 review

`q5-review-agent` reviewed exact candidate `734910d3781e3798c63b76360a6c955acf743998` against published native source. The pure ledger slice was judged concrete enough for product approval, while full qualification remained closed. Three corrections were required before the integration slice: charge completed-but-owner-retained command buffers until final ownership release (including raw exported handles), include external harness memory in competition accounting, and gate post-kill cleanup on external observations rather than a final report from a dead target. The proposal now includes those corrections; no implementation or runtime approval follows from the review.

Raw v2 review SHA-256: `4d43b57a386592ff5edc53a4f680712bb1089da6b79d2664d4174810287a9e09`.
