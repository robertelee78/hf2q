# Published mlx-native 0.15.14 adoption

This Q5 component pins the public registry package exactly. Native commit
`2109aa183d61ddd65e7d39a3d24b9bf048c16a57` adds bounded, read-only page observations
for certified NoCopy roots and read-only mappings. The observer rejects invalid,
oversized, stale and unknown results. It does not establish physical ownership,
reclaim credit, model admission or a performance improvement. Existing execution
paths do not invoke it automatically.

The registry archive checksum is
`6126089f48c7a2170008f2da93e72ab139819f77630ef149d8ad4ebbd5abe1b4`.
Independent verification matched the registry index, non-yanked entry, downloaded
archive, GitHub release asset and checksum, tag, clean VCS identity, and all 677
packaged source files including Cargo.toml.orig against the release commit.
[Release workflow](https://github.com/robertelee78/mlx-native/actions/runs/40902148545)
passed. No local Cargo patch is used.

Native validation on Apple Silicon passed locked all-target/all-feature checking,
the release build, full all-feature tests, package hygiene and reference-policy
checks. The full native test log has 215 successful summaries and 77 ignored
entries; its 1,533 summed passing entries include nested child runs and are not
a unique-test count. The separate tiny actual observer control passed with both
aliases observed; that fixture submitted no GPU commands. These are
API and observation checks, not full-model qualification.

Downstream source `7db0c9cebc839de746cf2e208bbf33cc4812323f` passed locked
all-target/all-feature checking (38.86s), the locked release build (50.63s),
and 48 focused release-mode host controls: four native-pin contracts, four
backing-observation controls, and 40 investigation-environment controls. Three
ignored hardware controls were not executed in this adoption run. No model was
loaded. Hosted CI remains a merge gate. The final note-only change corrects the
native tiny fixture claim to no GPU submissions by source, without claiming
an observed counter invariant.

This dependency adoption does not complete Q5; its connected owner accounting,
physical operating policy and required workload qualification remain open.
