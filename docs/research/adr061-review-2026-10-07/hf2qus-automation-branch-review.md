# Review of hf2q.us branch fix/release-channel-automation (2026-10-07, read-only, Claude subagent)

Verdict: REWORK (do not discard). Core design sound: immutable-release digests, canonical stable-record copy, template-only Apache, lock+journal+exchange+rollback.

Top fixes:
1. BLOCKER scripts/verify-promotion-source.mjs:41-54,66-68 — gate hard-codes an hf2q contract that does not exist: release.yml has no run-name (display_title is "Release"), one `publish` job does standalone AND cargo publish (no publish-standalone/publish-cargo), no cargo-recovery.yml, no dispatch to hf2q.us. No real run can pass the gate.
2. MAJOR idempotency — deploy-release.mjs:173 and release-deploy-server.py:241 fail if <op> dir exists; re-runs with same operation_id die; incoming/<op> and .release-transactions/<op> never pruned.
3. MAJOR ordering — promote-release.yml:102-121 pushes to main BEFORE deploy; deploy failure leaves main/verify asserting new version while site serves old.
4. MAJOR Apache split-brain — root-owned /etc/hf2q-release/apache.conf.in not in repo; nothing checks ops/apache/hf2q.us.conf derives from it.
5. MAJOR trust — source_run_id bound only by title string; run.head_sha never compared; proof/notary JSON unsigned claims; codesign --verify without -R team requirement; promoted binary never codesign/spctl-assessed.
6. deploy key written to $TMPDIR while an older downloaded binary executes.
7-12. minor: previousRelease() picks newest-older not current site version; vacuous owner check; dist deployed verbatim; drift ignores half-published releases and fails until first automated deploy; deployment.json added after verify; tar mac metadata.
13. Stable record byte-for-byte copy: OK (matches hf2q update.rs Policy::none + canonical JSON).
14. npm run verify conceptually passes; npm test now needs python3.

Missing for the website half of a pipeline dispatched from hf2q release.yml:
- hf2q: run-name release-<version>-<sha>; split publish-standalone/publish-cargo (or re-spec gate on head_sha+inputs); cargo-recovery.yml; GitHub-App-token dispatch of promote-release.yml with operation_id=hf2q-<run_id>-<attempt>; wait; fail hf2q release when site run fails.
- Final independent probe AFTER lock release: promoted binary -> "already current"; prior-site-version binary -> "update available", against public URL; also in drift.
- Commit ops/apache/hf2q.us.conf.in; derive/verify conf from it; provisioning docs; rollback/demotion operation (both sides refuse regression today).
- Idempotent re-run + pruning; push-after-accept or promotion PR; alerting.

Tests: release-promotion.test.mjs real contracts (good); test_release_deployment.py real functions but main()/exchange() untested; deployment.test.mjs SSH state machine untested; release-drift flow untested.
