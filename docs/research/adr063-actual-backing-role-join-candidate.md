# Q5 actual backing-role join candidate

This diagnostic-only source consumes the exact published mlx-native0.15.13
registry pin. Root verified the archive against commit512c023b271666bd1df0e5e4f4b93c47ead7b30f,
all675 packed source files, registry checksum and GitHub release bytes. No local
path patch is supplied. The
module is cfg(test), with no server/loader activation. Existing model-free
serve/memory_ledger remains a declared projection, not an allocation observation.

## Actual source association

ObservedGgufOwner accepts one actual readonly regular File, checks its real
F_GETFL mode, retains that FD, and passes a clone of THE SAME FD to native
GgufFile::from_file. Actual fstat dev/ino/length/mtime/ctime are retained and checked
around header parsing, mappings and snapshots. No pathname reopen, invented FD,
full-content hash, immutability or external receipt qualification is inferred.
The operator must separately bind the actual retained full-content receipt and
immutable artifact policy. File mutation between observations remains a limitation.

ObservedMappedWeights uses the existing native map_tensor_data and load_tensor
path, preserving its actual segment topology. A produced ObservedWeight retains
its buffer and original FD; snapshot joins that actual buffer's NoCopy range to
native tensor_data_offset+TensorInfo.offset using checked subtraction/addition.
Both mmap slice extent and logical tensor extent must fit the actual FD; registered
Metal final-page padding is explicit and is never content or a refund. Parent/root
identities and overlapping file-page ranges can therefore be joined downstream.
Equal contents alone never imply aliasing. These factories must replace/bind the
selected loader seam in a future admitted diagnostic; DO NOT load duplicate model
weights alongside a running model merely to manufacture ownership evidence.

Existing HF production loaders holding only native GgufFile cannot retroactively
recover its private reader FD from pointer metadata. Copied/expanded Qwen weights
must be observed as actual Owned roots; sharing a source GGUF does not make their
CPU staging or Metal owned allocation alias file pages. Generic observe(role,buffer)
records actual native parent/view metadata, dtype and shape, preserving Unknown.
KnownNative means lineage only; it does not attest healthy complete authority.

## Actual cache roles

qwen_cache_roles borrows the actual HybridKvCache and records entire allocated
conventional or packed KV/norm roots, MTP KV, both named physical conv/recurrent
ping-pong roots and both optional capture roots. Parities/cursors are separate:
physical0 is not mislabeled CURRENT for every slot. Cache vector rank is explicit,
not claimed to be original layer index. Full parent extents remain visible even
when cursor-active rows are small. Existing root IDs support alias dedup; this
candidate does not sum logical ranges as physical bytes or mint a new allocator.

Bounded inputs: <=1024 full and1024 recurrent ranks/slots, <=16384 roles,
<=256B role labels and <=16 shape dimensions. No tensor payload copying or GPU
command/synchronization occurs during observation. Device creation/mapping requires
normal separate prospective resource admission. Borrow under owning-worker
quiescence; snapshots do not synchronize commands or keep their source owners alive.
External retained anchors/pending checkpoints/MTP carry/host weights/CPU staging/
cache arenas/scratch/tools/client and OS competing charges are NOT covered by this
cache-only enumeration. Gemma and DeepSeek enumerators are not silently inferred
from Qwen. They require their actual owning fields/source-specific adapters.

## Authored controls and remaining admission gap

Three host controls exercise actual-used join arithmetic including parent padding,
underflow/overflow, actual FD identity versus equal contents, readonly FD refusal,
same-FD header binding and changed-stat refusal. One ignored fresh-process tiny
actual mapped-weight control authors an80B owned GGUF, observes clone-root identity,
checks file offsets/exact four F32 values and retained FD/mmap ownership after the
factory drops. All controls are UNRUN; no model/header/weight/process/GPU command
execution was performed by the author.

This supplies a missing backing association only. Full Q5 still requires early
native authority coverage at the selected real load, source-complete role inventory
including all transient/pending/CPU owners, disjoint same-file/root accounting,
conservative observed OS competing/nonreclaimable partition and positive reaction
margins, then all required largest-case/arm/negative gates. No physical PFNs,
residency/release/eviction credit, old-result rescoring or physical admission follows.

## V2 actual tiny cache traversal control (authored, UNRUN)

A separate ignored fresh-process control now calls qwen_cache_roles on TWO actual
HybridKvCache allocations: F32 and TQ, C8/N2, one full-attention rank, one MTP rank,
one linear rank, D256 KV and Dk=Dv128/Hv1 recurrent layout. It uses the existing
new_with_options and ensure_la_capture(2) constructors; no model, forward or GPU
commands. All24 actual allocation roots are authority-known. Exact inventories
are10 F32-cache roles and14 TQ-cache roles: conventional or packed/norm KV including
MTP, both physical conv/recurrent roots and both optional capture roots. Both
caches coexist, preventing allocator-address-reuse from masquerading as identity.

The source PRE-DEVICE requested ledger is F32KV65536 +TQKV16640 +two LA280576
+two capture280576 =1204480B.24parents×64KiB pessimistic rounding allowance plus
256KiB metadata keeps test userdata below8MiB, not a device/process physical cap.
Root-creation authority freezes exactly24roots/1204480requested bytes. Assertions
check exact role names/counts, dtypes, parent extents, disjoint IDs, zero initial
cursors/parities, then diagnostic-only cursor/parity metadata changes without
any descriptor/root mutation. An actual recurrent-buffer clone preserves origin
without creating a root. Native dispatch/commit/sync counters must remain unchanged.
This is allocation/traversal coverage only, not token-state/forward correctness or
Q5 completeness. Native1513 public source verification and the focused root execution below are
complete. Broader CI and selected full-model ownership/admission remain pending. V1 source freeze is preserved, not reinterpreted as containing this test.


## Root validation with the published dependency

Clean source2f6a4fc675e8e4aeabe60c449779748f2f277fe0 compiled in locked release-test
mode against registry0.15.13 (checksum
66d9f8bb0296ca7497ebe7834bc817b2a0326aa5755fdcc7660760ae2e442270).
All677 files in the actual compiled registry directory matched the verified
archive, including its package metadata. No local path patch was used.

Three focused host controls passed. Both ignored fresh-process controls executed
and passed: the actual same-FD mapped-weight/alias join and the actual F32/TQ/MTP/
recurrent/cache-capture traversal. The latter observed exactly24 roots and
1204480requested bytes. Both owned process groups were empty after execution;
no model or forward was loaded. The locked all-target/all-feature check passed.
These are backing/role observations only, with no physical-fit, residency,
reclamation or selected full-loader coverage claim. Independent source review
covers the actual factories and traversal; additional family/CPU/transient/OS
roles and real largest-case gates remain necessary.
