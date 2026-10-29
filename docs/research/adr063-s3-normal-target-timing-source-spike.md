# S3 same-input normal versus retained shared target diagnostic

This is an ignored, release-only producer diagnostic, not a serving, AUTO-tail,
quality, physical-fit or performance acceptance. The earlier serving results
remain unchanged. The normal target is the actual
`forward_gpu_with_nextn_hidden` call (including its full-logit download), without
`checked_fixture`; the shared target is the existing retained checked graph.
There are no new GPU waits, kernel changes, native APIs or production profiling.

The explicit packet `HF2Q_S3_TARGET_TIMING_PACKET` selects four hash-bound actual
canonical seven-wire prefix-oracle configs/reports. The original Rust oracle
core re-renders and tokenizes every wire and checks every existing receipt,
usage, stable-prefix and token-vector hash. Turn 5 supplies all four prefixes
(about 2k tokens); owner order is [3,0]. Each owner's declared 12-token diagnostic
suffix is the slice of its actual turn-6 prompt at its turn-5 length. This is
teacher forcing, not a claim those IDs were selected by this fixture. Suffixes
are neither padded nor reselected.

There is ONE loaded model and ONE C262144/N4/TQ8 cache. Each of 24 samples uses
real `Qwen35PrefillState::begin(cached_tokens=0)` and repeated `advance`, for all
four owners, outside the target clock. `begin` calls the existing
`HybridKvCache::reset_for_slot`: full-attention and MTP cursors/graft tags reset;
all four per-slot DeltaNet ping-pong roots are zeroed in place, and parity resets.
The reset contains no buffer replacement/allocation. Subsequent full prefill
rebuilds every active row, including actual MTP catch-up. No snapshot, partial
restore, backup, second cache or assumed GPU retirement is used. The 260-root
address/byte multiset (17 TQ slots x4, 48 GDN slots x4) must match before/after
EVERY preparation and producer. Speculative capture allocation is refused.
This is cold rebuilt state within retained capacity, not warm-serving timing.

For logical step s=0..11, selected cold prefixes are base+suffix[..s], other
owners retain the base; the target consumes suffix[s]. Two warm steps per arm
precede five fixed ABBA blocks (10 timed samples/arm). Preparation, cache scans,
nextn CPU copy, comparisons and JSON I/O are outside each target clock. Full
normal-logit download is inside its clock. Shared timing includes the bounded
cfg(test) profiling overhead, so it is not an uninstrumented serving estimate.
All ten samples contribute to each median; there is no favorable selection.

After every healthy producer return and before the next reset, all output
values are copied to CPU, checked finite, and retained until the paired arm is
compared bitwise. Full active primary/MTP/conv/recurrent/carry records are streamed
to hashes, including encoded TQ bytes (not integer IDs). Paired full pre/post
manifests must be exact; untouched owners and all MTP/carry states must remain
exact within the sample. Only bounded hashes/metadata are written, not per-step
raw cache dumps. Failures stop; no replay or controller/token alignment occurs.

Chosen shared CB records are the existing 128 row-copy, 64 packed FFN and 96
retained Delta checks per N2 target. Host timing surrounds the existing check;
`completed_gpu_interval_ns()` is read only AFTER its successful Result. Zero is
Unavailable. No recommit, additional wait, observer substitution, raw escape or
GPU timestamp is invented. Normal helpers already dropped, LayerEncoder and
unhooked stages remain Unknown. Summing these CB durations is not a GPU union,
GPU-running interval, utilization or whole-target GPU measurement.

## Resource contract before Device/model construction

The actual production allocation planner is called with C262144/N4/TQ8/MTP.
For this selected configuration it must return 38,332,792,832 requested bytes:
37,077,647,360 TQ bytes plus 1,255,145,472 GDN bytes. Roots are charged at full
capacity, with no credit for unwritten pages. One 16,810,714,976-byte weight
mapping, twice one OS-page padding per actual tensor, the full 24 GiB pool cap,
1 GiB host allowance, 64 MiB total metadata/I/O, 16 MiB pending CPU outputs,
2 GiB driver allowance and 4 GiB free/reaction margin are all added with checked
arithmetic. At 4096 tensors/16KiB pages this is 88,647,608,160 bytes; the actual
header count/page size supplies the exact attempted upper. This is a named
requested-resource bound and operating allowance, not a physical-memory theorem.

The ignored entry refuses unless an independently reviewed fresh outer admission
supplies sufficient available working bytes and exactly the unchanged 96 GiB
process/4 GiB runtime limits. The outer supervisor still must enforce normal
pressure, swap/compressor/RSS/footprint, all-history I/O deductions and owned
process cleanup. Twenty-four full preparations require a separately declared
finite deadline before execution (a prospective 1800s case is suggested); no
in-attempt deadline or policy tuning is implemented. No runtime approval is
contained in this source or in its packet schema.

Focused authored controls cover fixed schedule counts, positive/refused/overflow
resource admission, suffix extent/vocabulary refusal, full-cold selected-prefix
construction, and inactive/nested profiling scope restoration. All are unrun by
the source author. Root must compile and run host controls, independently review
packet/resource guards, then own any actual model invocation.
