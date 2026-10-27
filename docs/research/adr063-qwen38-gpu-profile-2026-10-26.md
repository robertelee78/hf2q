# Qwen3.8 N4 profiling: ordinary head-only slice not selected

The retained AUTO diagnostic does **not justify implementing ordinary-target-only
output-head batching**. Most observed target projections follow temporal
multi-row forwards. This is an opportunity assessment, not a speed comparison
or an approved replacement architecture. S3 remains Designing.

## Exact diagnostic

| Identity | Value |
| --- | --- |
| Evidence packet | `qwen38-n4-decode-profile-cb-20261026T233932Z` |
| Server source | `1ce3527d5982f3a9a78d57f4328b1d4643314926` |
| Release binary SHA256 | `7b1a52c575edbb9c8445d1827221c6d138bc42f220cc35aaa7484decf46fffee` |
| Published dependency | mlx-native0.15.5, checksum `be6b17fcc030f03615f66873ba22e59aa29b5ae9ff30d6370cbdedf8649245e9` |
| Retained stderr SHA256 | `04ca986b00de50f0657c49a0d326caa68f6143c800f9e577fdcb98ec507f1607` |

The canonical four-agent workload retained AUTO speculation and enabled
`HF2Q_DECODE_PROFILE=1` and `MLX_PROFILE_CB=1`. The outer receipt records all28
requests complete, no pending cases, final verification and proved cleanup.
Analysis occurred after unload and is not a new independent coding/prefix gate.
The earlier233837 attempt failed before model load because its source-read path
had been cleaned; it supplies no runtime or timing observation. The next attempt
corrected that read path, preserving the failed attempt separately.

All1,247 printed command-buffer tables parsed completely. Each contained3–12
labels, so none lost entries to the printer's top20 limit. Their totals sum
34,518.47ms of **reported labeled GPU work**, not all process/device work:

| Label category | GPU milliseconds | CB count |
| --- | ---: | ---: |
| `layer.dense_ffn` | 10750.73 | 3179 |
| `layer.gdn.ops5-9` | 9716.10 | 58512 |
| `mtp.shared_head_argmax` | 5889.82 | 2994 |
| `output_head.fused_norm_lm_head` | 2799.96 | 1190 |
| `output_head.nextn_norm_last_lm_head` | 108.03 | 56 |

Target-head labels total2,907.99ms,8.4244% of the labeled total. The distinct MTP
shared head adds17.06%; it is not ordinary target work. Norm/projection/selection
labels are inclusive, not isolated vocabulary-weight time. No head label with
last-layer-FFN fusion occurred here. Dense FFN and GDN dominate the labeled
categories; unlabeled work remains unmeasured.

## Correct the print boundary before interpreting widths

In the exact Qwen `forward_gpu.rs` source, `print_and_reset_cb_profile` runs near
line6226 **before** the current output head near6244. A table's head must not be
assigned to its own `seq_len`. The initial table contains no target-head CB;
every later table contains exactly one, with exactly one intervening host head
timer between adjacent headers. Under this explicit single-worker/source
sequence, assigning that head to the preceding forward gives:

- 28 preceding seq1 heads:52.69ms, only an upper envelope for ordinary eligibility;
- 1152 preceding seq4 heads:2722.44ms;
- The final forward is seq4 and its head GPU duration was not subsequently dumped.

The other completed heads follow seq2 or last-row/next-N cases. All1,247 headers
are general-forward, with no greedy-forward header. These observations do not
prove which seq1 calls were ordinary decode rather than prefill, or four
simultaneously ready independent lanes. Temporal MTP rows are not slot rows.

Even counting every seq1 head as ordinary and perfectly amortizing its entire
inclusive cost across four lanes would save at most39.52ms: approximately
**0.1145% of represented labeled GPU service**. The global8.42% head fraction
would suggest a misleading6.32% optimistic four-lane ceiling by including mostly
multi-row temporal work excluded from that proposed slice. This bound is for
the ordinary head-only hypothesis; it is not a theorem about every possible
whole-body or mixed temporal/cross-slot architecture.

The1,247 host head timers sum3573.7ms with median2.9ms. They include host work and
terminal waits that may settle earlier backlog, and are not isolated GPU head
cost. Native `commit_labeled` becomes synchronous under `MLX_PROFILE_CB`, so
pipelining and AUTO cost decisions may be perturbed. Rounded printed durations,
the undumped tail and unlabeled CBs preclude a whole-run physical GPU bound.
Nothing here establishes production throughput, semantic-TTFT or a5% speed win.

## Disposition

Do not select the historical ordinary-only first slice for implementation on
this evidence. In particular, discard its head-only refinement as the next
optimization lane for this observed AUTO workload. The preceding
[pool study](adr063-qwen38-pool-ablation-negative-result.md) also remains below
its5% criterion; neither result authorizes widening a claim to reach a target.

A future S3 selection must first measure the existing temporal/speculative and
independent-slot regimes, their overlap and real costs, while preserving AUTO.
Compare DeepSeek's actual path before choosing a common change. Do not force
speculation off, assume ready4, add an unmeasured replacement architecture, or
introduce another20-run prerequisite merely from these source hypotheses.
Current mlx-native0.15.6 is published, but this run used0.15.5; no output-kernel
change or improvement is attributed to the newer version by this report.
