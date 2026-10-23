# ADR-063 F1 reference parity validation

Scope: source/build/codec and one real-model conversion boundary for the proposed peer-pin replacement. This is not pin adoption, a serving loader pass, runtime qualification or a speed result. Work key: `robertelee78/hf2q:ADR-063:F1`.

## Corrected parity checker

The existing tensor checker could print missing tensors or type mismatches and still exit successfully. It also accepted different shapes with the same payload bytes and accepted an empty comparison. That could turn an unexecuted or structurally different comparison into false proof.

The checker now rejects empty inputs, missing/extra names, different tensor types and different shapes, in addition to different payload bytes. Six real, tiny GGUF controls exercise identical data, missing/extra names, type, shape, payload and empty inputs. The four structural negative cases reproduce false success against the original checker and fail against the corrected checker; identical data still passes and changed payload still fails. Run in the same frozen gguf/numpy reference environment:

```sh
python scripts/test_byte_cmp_gguf.py -v
python scripts/byte_cmp_gguf.py reference.gguf candidate.gguf
```

The original checker and complete negative-control results remain archived. Original control receipt SHA-256: `1001860213513512ee9e6100510ed038e1cab48a986a3ae5e0348d55dbbcfcf9`.

## Declared dependency failure and explicit compatibility variant

Unmodified reference source `36e36e63a54fc8f85b95e9c499a022b597e478dd` was first exercised under its declared Transformers 4.57.6 dependency. Gemma conversion failed because that tokenizer API expects a mapping while the frozen model supplies `extra_special_tokens` as a list. Preserve this as a failed probe. The model config declares `transformers_version: 5.5.0.dev0`; a separate Transformers 5.5.0 environment completed conversion. Torch 2.11.0 and NumPy 2.2.6 remain fixed. Full dependency locks and commands are retained for both environments; the [compatible reference lock](adr063-reference-gemma55-environment.lock) is checked in for reproduction.

No upstream source or original environment was silently patched. The compatibility environment is a named reference experiment; it does not establish that the declared upstream environment supports the model. Production conversion remains owned Rust hf2q.

## Matched precision, not a relaxed threshold

The first F32-reference quantization probe differed on 145 Q4_K tensors. Current `src/convert/orchestrator.rs` explicitly rounds quantized inputs through F16 before calling the codec, matching ADR-033's canonical F16 intermediate. Feeding the reference quantizer F32 skipped that step and did not test the same numerical input contract. That failed probe remains preserved.

A requested-F32 owned diagnostic also contained a Q6_K tied embedding: it was not an all-F32 oracle. The corrected checker rejected that structural difference while reporting 657 equal F32 tensors. No full-floating parity claim is made from it.

The deciding rerun uses the existing canonical F16-to-F32 staging on the reference side and the unchanged owned Q4_K_M production pipeline. There is no accepted tolerance increase or ignored tensor. Symbolic reproduction, with each variable bound by the private receipt:

```sh
"$HF2Q" convert "$SOURCE" --source-repo google/gemma-4-26B-A4B-it \
  --source-revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
  --quant q4_k_m --text-only --state-root "$STATE" -o "$OWNED"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$REFERENCE_PYTHON" \
  "$PEER_CONVERTER" "$SOURCE" --outtype f16 --outfile "$REFERENCE_F16"
"$PEER_QUANTIZER" "$REFERENCE_F16" "$REFERENCE_Q4" Q4_K_M 6
"$REFERENCE_PYTHON" scripts/byte_cmp_gguf.py "$REFERENCE_Q4" "$OWNED"
```

## Observed real-model result

Official Gemma source revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; owned signed hf2q 0.1.27 source `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`; reference source `36e36e63a54fc8f85b95e9c499a022b597e478dd`. Both converters use the same existing source snapshot; no model download was needed.

All **658** tensors have matching names, shapes, types and payload bytes: 392 F32, 192 Q4_K, 32 Q5_0, 14 Q6_K and 28 Q8_0. All three deciding reference/compare commands exited 0. This result is explicitly tensor-level; whole-file digests and their equality are recorded separately. It does not silently equate metadata compatibility, tokenization, loading, quality or execution across architectures.

Together with 48 unchanged codec fixture files and 24 verified owned fixture cells, this validates the proposed replacement at the stated boundaries. The legacy real-model fixture cell that skipped is still a skip and contributes no proof. Keep the product pin unchanged until the governing adoption review and any wider required checks complete. Qwen/DeepSeek/Gemma runtime and all-four speed verdicts remain separate obligations.


The existing frozen campaign Gemma artifact (SHA-256 `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53`) also matches all 658 reference tensor payloads. Its conversion history was preserved; this check does not swap the campaign artifact.

Whole-file bytes differ between fresh owned and reference outputs. The typed metadata audit finds no differing shared model-math or tokenizer values/types. Differences are three descriptive identity fields (`general.name`, `general.finetune`, `general.size_label`), the KV count, and two additional owned provenance keys (`hf2q.producer_version`, `hf2q.source_sha256`). That is documented metadata variation, not a claim of whole-file identity or runtime compatibility. Full parity receipt SHA-256: `82ea2e6aa982d862905c73a1fa615d88b4aeccd06e97115f947e5dbc0c1a7c74`.
