# Independent Python-TUF fixture

This retained corpus is generated through the public metadata APIs from
`tuf==7.0.0` and `securesystemslib[crypto]==1.4.0`. It is independent of
hf2q's Rust fixture signer and the `sigstore-tuf` verifier under test.

The fixture contains a two-of-two root at version 1, a version-2 root signed
by both the complete old and new two-of-two thresholds, and version-2 lower
roles signed by the new threshold. Snapshot and timestamp descriptors carry
exact SHA-256 and length pins. Targets contains one synthetic stable-channel
target descriptor; target payload bytes are deliberately not part of this
metadata-authentication corpus.

The metadata uses canonical SHA-256 key IDs and consistent-snapshot wire names:
`1.root.json`, `2.root.json`, `timestamp.json`, `2.snapshot.json`, and
`2.targets.json`.

All private-key seeds are public deterministic test data. They are not hf2q
release keys and must never be used in production.

To regenerate or verify the retained bytes in an isolated environment:

```bash
python3.14 -m venv /var/tmp/hf2q-python-tuf-fixture
/var/tmp/hf2q-python-tuf-fixture/bin/pip install --require-hashes \
  -r src/distribution/update_auth/testdata/python-tuf-v1/requirements.lock
/var/tmp/hf2q-python-tuf-fixture/bin/python \
  src/distribution/update_auth/testdata/python-tuf-v1/generate.py
/var/tmp/hf2q-python-tuf-fixture/bin/python \
  src/distribution/update_auth/testdata/python-tuf-v1/generate.py --check
```

`requirements.lock` pins the complete generator dependency closure and package
artifact hashes. `PROVENANCE.json` binds the exact generator and lock,
interpreter, dependency versions, key identities, parameters, and metadata
digests. `SHA256SUMS` covers the metadata, generator, lock, and provenance.
Rust regression tests separately pin each retained metadata digest and
authenticate the full rotation and lower-role chain through hf2q's production
verifier.
