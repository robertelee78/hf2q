# ADR-063 Q1-QWEN38-RERUN (#664): Qwen3.8 T1/T3 restore-point evidence (attempts b4 and b5)

Date: 2026-11-03. Host: Apple M5 Max, 128 GiB. Evidence root: `$C` = `~/.local/state/hf2q/campaigns/adr063-preflight-20261023/q1-closure-v1`.

## Result

#664 acceptance: Qwen3.8 T1 vs T3 is `ExactOnCommon` with restored-64 extras only. **It is met as written, so the ADR-063 #664 acceptance text needs no rewording.**

- b5: `exact t1 t3` = **`ExactOnCommon`**. Records: 4,800 common, 0 differing, 0 right-only, **1,408 left-only**.
- Every left-only record is a restored-64 record (logical step 64, segment ≥ 1000). Its sha256 equals that of its segment-0 twin, and the twin is in both stores. The 1,408 records hold 636,092,416 B.
- The same check ran in-process: T1 `execution.json` has `qwen38_restore64_exact {"Ok":636092416}`, and the supervisor requires it.
- b4: T0 vs T4 and T0 vs T7 are **`ExactEqual`** (3,392 common, 0 left-only, 0 right-only, 0 differing).

## Why two attempts

- **b4** used test binary `d817dffa` (origin/main on 2026-11-03), built after #561 (`9fc7b31b`) and before ADR-065 P4b (#527) and J1 (#512).
  - On that source, `qwen_checkpoint::captures64` was Qwen3.6-only, so Qwen3.8 T1 recorded no restored-64 records.
  - T1 vs T3 could therefore only be `ExactEqual`, which it was.
- **b5**, by operator decision, added the direct restore-point check for Qwen3.8. The change is in the test harness only:
  - `captures64` covers Qwen3.8 T1 and T3, and the runner observes the restore for both families.
  - `qwen_actual` runs `require_restore64_exact` on Qwen3.8 T1 and writes `qwen38_*` keys.
  - The endpoint payload cap rises from 2 to 4 GiB and the packet read cap from 1 to 4 MiB, because Qwen3.8 T1 plans 3,135,668,224 B.
  - The supervisor requires `qwen38_restore64_exact` on Qwen3.8 T1.
  - Qwen3.6 plans, packets and keys are unchanged.
- The b5 binary was built from `5504f4ee` = `d817dffa` + that one test-only commit. This PR carries the same commit rebased onto main; it has the same patch-id. Main between `d817dffa` and the rebase base changed no `src/` file and no Q1 script.

## Identities

| | b4 | b5 |
|---|---|---|
| source | `d817dffa7371a2db85a43ed1d6502350906c33f5` | `5504f4ee408d16a5244d09577d48be0b19d8af09` |
| test binary sha256 | `8c0b6a054530beb18bec76d7d43b457095cf53b12011f77c39e8596dae26d823` | `21781dec73e6c03eb114c6318002921b6af751557791164e18dfa315badba2af` |
| build receipt sha256 | `ebfce144a72bf57523ff80dc219b378363e2752183f528a55e7eb30d1634b37d` | `8149f6c28939e7cb235c6a001c2cb50ac5153232f64c24a5a67cdb2ddb1fa0c0` |
| native | mlx-native 0.15.16, checksum `baace413…562f` | same |
| controls | adr063_invariance 104/0, q1_ 61/0, full 5743/0, python 35 OK | adr063_invariance 105/0, q1_ 61/0, full 5744/0, python 36 OK |
| supervisor source sha256 | `6656d94e…915d` | `dca5ccba…3eb3` (admitted; only `RESTORE64_FIELD['qwen38']` changed) |
| decision tool `q1_decision_policy.py` | `b152c92d…45e2` | same |

Model: `Qwen3.8-27B-vanilla.gguf`, retained sha256 `cf0eb010…`. Workload: synthetic `2ed4b49c` vectors (4 lanes × 144 tokens). Capacity 262144, 1 slot, QwenTQ8.

b5 setup:
- Composition `qwen38/b5/composition-v1.json` sha256 `7f8e2369…dfcd`.
- Policy `resource-policy-qwen38-b5-v1.json` sha256 `e5584649…6028`.
- Final bindings sha256 `0ad696ba…456a`; supervisor copy sha256 `eca35b47…c38d`.
- Acquisition and operating reviews: `$C/qwen38/reviews/b5/` (independent reviewer).

## Receipts (each `case_complete: true`, exit 0, no failure)

| case | out | elapsed | trace records | receipt sha256 |
|---|---|---|---|---|
| b5 t0 (first-case gate) | `qwen38-t0-b5` | 50.9 s | – | `f10d17aa605f97b89620b2ba48ab4b8eff0f7d470169da299aa96c7a9614f526` |
| b5 t1 | `qwen38-t1-b5` | 81.9 s | 6,208 | `5f8374a1b175b37766781d27bf4d0847968bf48f14925cafcf205249762a3051` |
| b5 t3 | `qwen38-t3-b5` | 65.9 s | 4,800 | `243c514ebbe39a723e75436c558cf26b7e54f97dd8699b43955f46f64f188154` |
| b4 t0 | `qwen38-t0-b4` | | | `2ee97be65f50625745550cf8adabcec02489de7cb27c022fbc736d6979677fb4` |
| b4 t4 | `qwen38-t4-b4` | | | `d0a5a692f535ae3232a19a3a03b192381b6c7b2e019ba70fa39f9db39b84c04b` |
| b4 t7 | `qwen38-t7-b4` | | | `063e71830f175590da7b3b504741cd12c67e5b0a1a0b6f8125a890159a03b98a` |

Before t3, free plus cached memory was 111.19 GiB; the reviewed floor is 95.2 GiB.

## Decisions (`$C/decisions/`)

| pair | classification | rule | output sha256 | record sha256 |
|---|---|---|---|---|
| b5 `exact t1 t3` | `ExactOnCommon` | strict `exact-restored`: PASS | `ebea86a9…42cd` | `44915b47…8416` |
| b4 `exact t0 t4` | `ExactEqual` | `exact-equal`: PASS | `4fcb46f2…f503` | `24d99275…c2c6` |
| b4 `exact t0 t7` | `ExactEqual` | `exact-equal`: PASS | `4fcb46f2…f503` | `a9ae7dc7…0812` |

The b4 verdict is `$C/qwen38-verdict-b4.md` (sha256 `f0754990…7bc7`). The tools that produced these decisions are `run_family_b4.sh` and `run_family_b5.sh` in `$C/decisions/`.
