# Published native 0.15.15 consumer preparation

The exact normal-registry pin resolves the public `mlx-native 0.15.15` crate,
SHA256 `bb508d36e311a24147e917282a8ccae0d47720f4a12ae678717df5d569c75e3c`,
from native commit `9c67e1fb670303aa5f30da9bbfc52fbe79652a4e` (PR #76).
The publication workflow is run 40937109637. Its exact-source and packed-source
tests passed before publication. Independent registry/index/VCS/source-member
verification supports this reversible consumer preparation; final workflow,
tag and release-asset verification remain pending at this checkpoint.
No local Cargo patch or unpublished dependency is used.

The dependency adds bounded same-file GGUF header parsing and a device-free map
plan using the production coalescer. The cfg(test) DeepSeek diagnostic consumes
that API without model loading, mmap, payload reads or device creation. Geometry
remains an explicit caller-assumed projection. Logical file extents, rounded
registrations, source F32 expansion and cache projections remain separate;
none grants physical memory credit or largest-profile runtime admission.

This checkout prepares hf2q 0.1.28. Local validation and full native release proof
must be recorded before merge/publication; neither is claimed by this checkpoint.
