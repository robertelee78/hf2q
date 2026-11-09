# ADR-065 E5: DeepSeek joint-step expert union

Story: [#508](https://github.com/robertelee78/hf2q/issues/508), [ADR-065 E5](../adr/ADR-065-four-agent-joint-speculative-verify.md#e5-trace-deepseek-expert-union-per-joint-step).

## Diagnostic contract

`HF2Q_ADR065_EXPERT_UNION=1` observes DeepSeek-V4 joint decode widths 2–4. Each FFN layer blit-copies its sanitized `[width, top_k]` routed ids into one per-step buffer in the layer's existing command buffer. The host reads that buffer only after the transaction's existing pipeline drain. There is no added host/GPU synchronization; the default-off path allocates and dispatches nothing.

Each completed joint step emits one closed-schema `hf2q.adr065.deepseek-expert-union.v1` record with every layer's routing class, unique-expert count, per-expert routed-weight bytes, naive routed bytes and unique-expert bytes. The byte counterfactual counts each layer's stored gate, up and down expert matrices once per selected expert instead of once per routed row.

`scripts/four_agent/adr065_deepseek_expert_union.py` rejects unknown fields and inconsistent byte arithmetic, requires requested widths, and reduces the records into per-width/per-layer histograms plus hash/score and overall byte reductions.

## Bound identities

| Item | Value |
| --- | --- |
| Implementation commit | `347afd3970923f20b32f8aa01559fd475d80b533` |
| Candidate binary | `$E5/build-347afd39/hf2q`, sha256 `db0abb3fd02c071dcbf874799a57a54d1fc8d69c65e32d49b0994a7f3dfd43f1` |
| Model | DeepSeek-V4 Flash 0731 `$MODELS/deepseek4/DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf`, sha256 `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6` |
| Q5 case | `$E5/deepseek4-e5.json`, sha256 `cbaf462dc49b4ad1957ae4ad8ba3f8bb40ecd7a6e95f05c00d75879cf9f193a4` |
| Q5 receipt | `$E5/run-r2/receipt.json`, sha256 `3bea3655e7b681afe54d0882a5818b314b1ba2179c9cae861f2fadbbb4f9c643` |
| Raw server log | `$E5/run-r2/server.stderr`, sha256 `ac5e264a7150393da1dcfd7140b79ad0b02e8386abf35270e8547a9db74fa98a` |
| Reduced report | `$E5/expert-union-r2.json`, sha256 `70d9337cb01fa06cd149eb051b46ba4a60cd32a8be5d7659667e60be4466a55e` |

`$E5` is `/Users/robert/.local/state/hf2q/v0130/e5-508`; `$MODELS` is `/Users/robert/.local/share/hf2q/models`. The source-stamped binary contains the implementation commit above. The earlier `$E5/run` is excluded: its binary was built from the same source but carried an incorrect manually supplied commit stamp. The binary was rebuilt with the exact `git rev-parse HEAD` value and the complete hardware run was repeated as `run-r2`; no result from the excluded run is pooled here.

## Retained four-agent result

The canonical DeepSeek-V4 four-agent coding run returned `CalibrationPass` and proved server cleanup. It captured 1,148 joint steps at every required width.

| Joint width | Steps | Hash-layer mean unique experts | Hash range | Score-layer mean unique experts | Score range | Maximum selected rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 330 | 11.338 | 6–12 | 10.642 | 6–12 | 12 |
| 3 | 169 | 17.134 | 6–18 | 15.559 | 6–18 | 18 |
| 4 | 649 | 19.809 | 6–24 | 17.617 | 6–24 | 24 |

The checked-in report retains a histogram for every `(width, layer, routing class)`, preserving the per-step/per-layer counts rather than only these means.

| Routing layers | Naive routed bytes | Unique-expert bytes | Reduction |
| --- | ---: | ---: | ---: |
| Hash | 617,023,143,936 | 532,732,903,424 | 13.661% |
| Score | 8,226,975,252,480 | 6,404,017,750,016 | 22.158% |
| All | 8,843,998,396,416 | 6,936,750,653,440 | 21.565% |

On this workload, a path that reads each selected expert's routed matrices once per layer/step instead of once per routed row would remove 1,907,247,742,976 routed weight bytes under the declared counterfactual. This is workload evidence, not a performance claim; E5 times no candidate implementation.

## Reproduction

From the repository root, after binding the case's prerequisite paths:

```sh
python3 -B scripts/four_agent/q5_calibration.py "$E5/deepseek4-e5.json" \
  --output "$E5/run-r2" --negative-control-receipt "$NEGATIVE_CONTROL"
python3 -B scripts/four_agent/adr065_deepseek_expert_union.py \
  "$E5/run-r2/server.stderr" \
  --require-width 2 --require-width 3 --require-width 4 \
  --output "$E5/expert-union-r2.json"
```
