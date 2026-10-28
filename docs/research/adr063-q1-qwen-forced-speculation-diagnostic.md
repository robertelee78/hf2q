# Q1 Qwen forced-proposal diagnostic (source candidate)

This test-only candidate adds a genuine K3 verification/rollback diagnostic for
Qwen dense and MoE. It does not qualify ordinary AUTO proposal selection,
proposal quality, performance, all Q1 trajectories, or another family. Source
review, compilation, hosted controls, model admission and actual execution are
separate gates. No model execution is recorded here.

The root supervisor runs three fresh processes per selected artifact. All use
four distinct independently reset physical-slot-zero lanes, context 262144,
TQ8, coherent MTP, an ungrafted artifact and greedy sampling (temperature 0,
top-p 1, top-k 0, repetition 1, output allowance 8192). Serving defaults do not
change. The first process performs an ordinary 128-token prefill and 16 real
ordinary warmups per lane. It freezes exactly 144 input IDs and the final g16
carry/terminal metadata. There is no seventeenth continuation input, favourable
lineage selection, retry or retrospective approval. Early termination before
that boundary fails preparation. Preparation cannot produce an acquired model
comparison: its prefix-only digest has a distinct domain and its provenance is
`WorkloadPreparationNotComparable`.

After independent review of that frozen file and separate acquisition packets,
the ordinary baseline and forced-proposal process consume identical frozen
lineages. The forced process performs one ordinary warmup, then explicitly
selects the existing real K3 implementation after its actual cost readiness
check. Three real native drafts execute and their speculative cache is actually
discarded. Only then does a test hook replace proposals with frozen g2, one
deterministic valid nonterminal ID different from g3, and the original third
draft. The real verifier, canonical selection, truncation, recurrent rollback,
MTP catch-up and pending-output commit must yield one accepted draft, one
rejection, two valid input rows, carry row one and cursor 131. Otherwise the
case fails without retry. The remaining ordinary inputs end at cursor 144.

Both comparable stores capture all 16 logical input-row logits and complete
active target/MTP cache extents at 129, 131 and 144. The raw store separately
retains all four verifier logits/hidden rows, three original and three forced
proposal IDs, and the two actually visited canonical IDs. Rejected rows are
not relabelled as committed baseline inputs. Existing coordinate-aware streamed
comparison remains responsible for first divergence. A missing approved
alternate-route numerical policy remains explicitly inconclusive; this source
adds no tolerance or tie smoothing.

The actual shared manifest fixes cumulative tensor payload before model work:

| Selected header | Each committed store | Raw verifier store | Both committed + raw |
|---|---:|---:|---:|
| Q38 dense | 2,003,429,888 B | 16,220,288 B | 4,023,080,064 B |
| Q36 MoE | 872,421,120 B | 16,023,680 B | 1,760,865,920 B |

Each store retains the existing separate 128 MiB metadata ceiling. Packet,
input and small output JSON files are at most 1 MiB each. Tensor payload is
streamed, not accumulated as a multi-GiB CPU vector. The temporary complete
four-row CPU logits copy is 4*vocabulary*4 bytes; cache readback uses existing
bounded capture pages. Native four-row recurrent capture storage persists after
capture deactivation: the selected Q38 additional allocation is 627,572,736 B
and Q36 263,454,720 B. These are requested extents, not physical-memory claims.
The root must separately admit actual Shared/CPU weight ownership, full model
cache/scratch, retained capture allocation, driver/OS uncertainty and reaction
reserve. Neither the disk ceiling nor a wrapper drop proves model admission.

The root alone may compile/run the authored hosted filters `q1_t5_` and
`q1_preparation_prefix_digest_`. The actual ignored filter is
`q1_qwen_actual_t5_speculation_phase`, with an absolute JSON packet in
`HF2Q_Q1_QWEN_T5_PACKET`, selected via `--ignored --exact` using the compiled
fully qualified test name. Packet fields are closed: `execution_approved`,
`mode`, `acquisition`, `identity`, `input`, `model`, `output`. Modes are
`Preparation`, `OrdinaryBaseline`, `ForcedProposal`. Input is a hash-bound
four-prefix document for preparation or the complete frozen output thereafter.
Acquisition still requires independent build/executable/FD/content-receipt
composition; these hooks do not self-attest source, artifact or cleanup.
All output roots must be new absolute canonical private paths outside Git.

Hooks are `cfg(test)` and inactive scopes preserve the original path. Errors
end the diagnostic without publishing a successful store or continuing model
work. External timeout, process cleanup and resource qualification remain the
root supervisor's responsibility. The F2 target-verification role guards in a
separate branch must be preserved when integrating the two verifier call sites.
