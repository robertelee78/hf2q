//! NomicBert (nomic-embed-text v1 / v1.5 / v2-moe) HF→GGUF tensor-name
//! + metadata mapper.
//!
//! Port of `/opt/llama.cpp/conversion/bert.py::NomicBertModel`'s name
//! mapping (transitively via `BertModel::modify_tensors` →
//! `gguf-py/gguf/tensor_mapping.py`) and `set_gguf_parameters`. Covers
//! BOTH the dense v1.5 path AND the v2-moe path (with caveats called
//! out in the table below).
//!
//! Per ADR-033 §P0 "Per-arch convert-side mapping": this is the
//! convert-side tensor-name + KV mapper for `LLM_ARCH_NOMIC_BERT`.
//!
//! NomicBert highlights (vs plain BERT, verified against
//! `/opt/llama.cpp/gguf-py/gguf/tensor_mapping.py` lines 233/326/345/495/555/624/710
//! and `conversion/bert.py::NomicBertModel.__init__` lines 340-354):
//!
//!   - **Fused QKV.** The three Q/K/V projections live in one packed
//!     tensor `encoder.layers.<N>.attn.Wqkv` (no `query`/`key`/`value`
//!     split). Maps to GGUF `blk.<N>.attn_qkv.weight`.
//!   - **Rotary position embeddings, not absolute.** NomicBert uses RoPE
//!     (`rotary_emb_fraction == 1.0`, `rotary_emb_interleaved is False`),
//!     not learned position embeddings. The HF checkpoint typically
//!     omits `embeddings.position_embeddings` entirely; if it is present
//!     (legacy / mistakenly-trained), it is unused — we still return
//!     `None` so the caller can decide whether to skip-with-warning or
//!     error per [[feedback-no-loop-suppression-2026-05-17]].
//!   - **SwiGLU FFN, not up-then-down.** Non-MoE NomicBert asserts
//!     `activation_function == "swiglu"`. The FFN is two parallel
//!     up-projections `mlp.fc11` (gate side, applied through SiLU) and
//!     `mlp.fc12` (up side), then a single down `mlp.fc2`. Maps to
//!     GGUF `blk.<N>.{ffn_gate, ffn_up, ffn_down}.weight`. Note: this
//!     is the OPPOSITE of plain GELU BERT, which has only `mlp.fc1`
//!     (up) + `mlp.fc2` (down). The MoE variant flips back to plain
//!     `fc1`+`fc2` GELU per-expert — out of scope here.
//!   - **No biases on linear projections.** NomicBertModel asserts
//!     `qkv_proj_bias == is_moe`, `mlp_fc1_bias == is_moe`,
//!     `mlp_fc2_bias == is_moe`. With `is_moe == false` (the v1 path),
//!     attn.Wqkv, attn.out_proj, mlp.fc11/fc12/fc2 are all
//!     weights-only. LayerNorms (`norm1`, `norm2`, `emb_ln`) retain
//!     their `.bias` half (standard LayerNorm has γ + β).
//!   - **Mean pooling.** Nomic embeddings are mean-pooled over tokens;
//!     emitted as `nomic-bert.pooling_type = u32 1` (MEAN). Matches
//!     llama.cpp's `LLAMA_POOLING_TYPE_MEAN`.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: no fallback / no
//! aliasing — every HF name we recognize maps to exactly one GGUF name,
//! and every other name returns `None` (the caller decides how to
//! surface the error). Per [[feedback-no-loop-suppression-2026-05-17]]:
//! callers MUST NOT silently skip a `None` — propagate as a typed
//! error.

use crate::backends::gguf::types::MetaValue;
use crate::convert::arch::bake::BakeOp;
use crate::convert::model_card::{get_model_id_components, split_base_model, ModelCard};

/// Per-arch context plumbed into [`map_tensor_name`] when the tensor
/// transform depends on hparams that aren't recoverable from the
/// safetensors shape alone. For nomic-bert that's just `num_experts`
/// (v2-moe; absent on v1.5). Other hparams either don't affect tensor
/// transforms (rope, layer_norm_epsilon, …) or are derivable from
/// the tensor shape (`n_inner`/`n_embd` are the two axes of the
/// expert tensors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NomicBertCtx {
    /// Some(N) for v2-moe (canonical hparam `num_experts` /
    /// `num_local_experts` per `bert.py:372`); None for v1.5 (no MoE
    /// path). Used to compute the `[E, F, H]` view of
    /// `mlp.experts.mlp.w1` / `.w2`.
    pub num_experts: Option<usize>,
}

/// Output of the NomicBert tensor-name mapper.
///
/// `Direct(s)` is "emit this tensor under GGUF name `s`"; `Drop` is
/// "canonical's `filter_tensors` deliberately discards this — do NOT
/// surface as an `UnmappedTensor` error". `DirectWithBake` is the
/// same as `Direct` but with a post-load data transform (see
/// [`BakeOp`]). The caller turns these into `MapOutcome::Direct` /
/// `MapOutcome::Drop` / `MapOutcome::DirectWithBake` in `cli_driver`.
/// Any HF name not handled here returns `None`, which the caller
/// surfaces as `MapOutcome::Unmapped` (typed error per
/// [[feedback-no-loop-suppression-2026-05-17]]).
#[derive(Debug, Clone, PartialEq)]
pub enum MappedTensor {
    /// Emit under the given GGUF tensor name.
    Direct(String),
    /// Emit under the given GGUF tensor name with a post-load
    /// [`BakeOp`] transform.
    DirectWithBake { gguf_name: String, bake: BakeOp },
    /// Canonical drops this tensor in
    /// `/opt/llama.cpp/conversion/bert.py` (`BertModel.filter_tensors`
    /// at `bert.py:72-92` or `NomicBertModel.filter_tensors` at
    /// `bert.py:362-369`). Do not error; just skip.
    Drop,
}

/// Translate one HuggingFace tensor name (as seen in `model.safetensors`)
/// to a [`MappedTensor`] outcome. Returns `None` for genuinely unknown
/// names so the caller can raise a typed `UnmappedTensor`.
///
/// Accepts an optional `bert.` HF prefix (some checkpoints — typically
/// those exported from a wrapping `BertForMaskedLM`-style head — carry
/// it; nomic-embed-text-v1/v1.5 typically does not). Mirrors the strip
/// in `BertModel.filter_tensors` (`/opt/llama.cpp/conversion/bert.py:72-73`).
///
/// NomicBert weight kinds. v1.5 columns vs v2-moe columns:
///
/// | HF name                                          | GGUF name                            | v1.5 | v2-moe |
/// |--------------------------------------------------|--------------------------------------|:----:|:------:|
/// | `embeddings.word_embeddings.weight`              | `token_embd.weight`                  |  ✓   |   ✓    |
/// | `embeddings.token_type_embeddings.weight`        | `token_types.weight`                 |      |   ✓    |
/// | `embeddings.LayerNorm.weight`                    | `token_embd_norm.weight`             |      |   ✓    |
/// | `embeddings.LayerNorm.bias`                      | `token_embd_norm.bias`               |      |   ✓    |
/// | `emb_ln.weight`                                  | `token_embd_norm.weight`             |  ✓   |        |
/// | `emb_ln.bias`                                    | `token_embd_norm.bias`               |  ✓   |        |
/// | `encoder.layers.<N>.attn.Wqkv.weight`            | `blk.<N>.attn_qkv.weight`            |  ✓   |   ✓    |
/// | `encoder.layers.<N>.attn.Wqkv.bias`              | `blk.<N>.attn_qkv.bias`              |      |   ✓    |
/// | `encoder.layers.<N>.attn.out_proj.weight`        | `blk.<N>.attn_output.weight`         |  ✓   |   ✓    |
/// | `encoder.layers.<N>.attn.out_proj.bias`          | `blk.<N>.attn_output.bias`           |      |   ✓    |
/// | `encoder.layers.<N>.norm1.weight`                | `blk.<N>.attn_output_norm.weight`    |  ✓   |   ✓    |
/// | `encoder.layers.<N>.norm1.bias`                  | `blk.<N>.attn_output_norm.bias`      |  ✓   |   ✓    |
/// | `encoder.layers.<N>.mlp.fc11.weight`             | `blk.<N>.ffn_gate.weight`            |  ✓   |        |
/// | `encoder.layers.<N>.mlp.fc12.weight`             | `blk.<N>.ffn_up.weight`              |  ✓   |        |
/// | `encoder.layers.<N>.mlp.fc2.weight`              | `blk.<N>.ffn_down.weight`            |  ✓   |   ✓    |
/// | `encoder.layers.<N>.mlp.fc2.bias`                | `blk.<N>.ffn_down.bias`              |      |   ✓    |
/// | `encoder.layers.<N>.mlp.fc1.weight`              | `blk.<N>.ffn_up.weight`              |      |  ✓ †   |
/// | `encoder.layers.<N>.mlp.fc1.bias`                | `blk.<N>.ffn_up.bias`                |      |  ✓ †   |
/// | `encoder.layers.<N>.mlp.router.layer.weight`     | `blk.<N>.ffn_gate_inp.weight`        |      |  ✓ ‡   |
/// | `encoder.layers.<N>.norm2.weight`                | `blk.<N>.layer_output_norm.weight`   |  ✓   |   ✓    |
/// | `encoder.layers.<N>.norm2.bias`                  | `blk.<N>.layer_output_norm.bias`     |  ✓   |   ✓    |
///
/// `†` Dense FFN layers only (those where `bid % moe_every_n_layers != 0`
/// in v2-moe; canonical bert.py:340 asserts `activation_function == "gelu"`
/// when `is_moe`, so the v2-moe dense path is GELU and uses fc1/fc2 with
/// biases). MoE layers don't have fc1/fc2.
///
/// `‡` MoE layers only (`mlp.router.layer.weight` is the routing
/// projection; emitted as `ffn_gate_inp.weight` per
/// `tensor_mapping.py:438-442`).
///
/// Returns `Some(MappedTensor::Drop)` for:
///   - `mlp.experts.bias` — canonical `NomicBertModel.filter_tensors`
///     drops this single-vector expert bias at `bert.py:366-369`.
///     Not loaded by llama.cpp's nomic-bert-moe inference path.
///   - `pooler.dense.weight/bias` — canonical `BertModel.filter_tensors`
///     drops the pooler (`bert.py:74-81`).
///   - `embeddings.position_embeddings.weight` — unused on NomicBert
///     (RoPE replaces absolute positions); canonical drops it via
///     `BertModel.filter_tensors` at `bert.py:82-90`.
///
/// Returns `None` (`Unmapped` at caller) for:
///   - `encoder.layers.<N>.mlp.experts.mlp.w1` /
///     `encoder.layers.<N>.mlp.experts.mlp.w2` when `ctx.num_experts`
///     is `None` (i.e. v1.5 checkpoints don't carry expert tensors;
///     a v2-moe checkpoint that fails to populate `num_experts` is a
///     hard error). When `ctx.num_experts` is `Some(E)`, both surface
///     as `MappedTensor::DirectWithBake` carrying
///     [`BakeOp::MoeExpertReshape`] (w1) or
///     [`BakeOp::MoeExpertTranspose`] (w2).
///   - Any structurally-rejected name (wrong prefix, malformed layer
///     index, unknown per-block suffix).
///
/// `hf_shape` is the PyTorch-order safetensors shape; used only for
/// the MoE expert tensors to derive `n_inner` / `n_embd` from `E *
/// n_inner = shape[0]`.
pub fn map_tensor_name(
    hf_name: &str,
    hf_shape: &[usize],
    ctx: &NomicBertCtx,
) -> Option<MappedTensor> {
    // Strip the optional `bert.` prefix — see BertModel.filter_tensors.
    let name = hf_name.strip_prefix("bert.").unwrap_or(hf_name);

    // ---- Globals (embedding table + embedding LayerNorm) -----------------
    match name {
        "embeddings.word_embeddings.weight" => {
            return Some(MappedTensor::Direct("token_embd.weight".to_string()))
        }
        // Present in nomic-bert v2-moe (XLM-RoBERTa-style, segment IDs
        // re-introduced). Maps to BERT-family `token_types.weight`
        // (`MODEL_TENSOR.TOKEN_TYPES` in GGUF). Canonical handles via
        // BertModel inheritance at `bert.py:88-90`. NomicBert v1.5
        // didn't have this tensor — see prior assumption at function
        // doc-comment.
        "embeddings.token_type_embeddings.weight" => {
            return Some(MappedTensor::Direct("token_types.weight".to_string()))
        }
        // Present in nomic-bert v2-moe (XLM-RoBERTa LayerNorm at the
        // embedding output). Maps to `token_embd_norm.{weight,bias}`.
        "embeddings.LayerNorm.weight" => {
            return Some(MappedTensor::Direct("token_embd_norm.weight".to_string()))
        }
        "embeddings.LayerNorm.bias" => {
            return Some(MappedTensor::Direct("token_embd_norm.bias".to_string()))
        }
        "emb_ln.weight" => {
            return Some(MappedTensor::Direct("token_embd_norm.weight".to_string()))
        }
        "emb_ln.bias" => {
            return Some(MappedTensor::Direct("token_embd_norm.bias".to_string()))
        }
        // BertModel.filter_tensors drops the pooler at `bert.py:74-81`.
        "pooler.dense.weight" | "pooler.dense.bias" => return Some(MappedTensor::Drop),
        // BertModel.filter_tensors drops position embeddings at
        // `bert.py:82-90`. NomicBert uses RoPE — these are absent on
        // v1.5/v2-moe checkpoints anyway, but if a stray export carries
        // them, Drop matches canonical.
        "embeddings.position_embeddings.weight" => return Some(MappedTensor::Drop),
        _ => {}
    }

    // ---- Per-block: `encoder.layers.<N>.<rest>` --------------------------
    let stripped = name.strip_prefix("encoder.layers.")?;
    let dot = stripped.find('.')?;
    let (layer_str, rest_with_dot) = stripped.split_at(dot);
    // Parse layer index (must be a bare non-negative integer; reject
    // leading zeros / signs to keep the mapper strict — mirrors the
    // llama3 sibling).
    let layer: usize = layer_str.parse().ok()?;
    if layer.to_string() != layer_str {
        return None;
    }
    let rest = &rest_with_dot[1..]; // skip the dot

    // `mlp.experts.bias` is a single (n_embd,) shared expert bias that
    // NomicBertModel.filter_tensors at `bert.py:366-369` drops. NOT a
    // per-expert bias — llama.cpp's nomic-bert-moe inference doesn't
    // consume it.
    if rest == "mlp.experts.bias" {
        return Some(MappedTensor::Drop);
    }

    // v2-moe expert weight tensors. Shape-aware bakes per canonical
    // `bert.py:373-380`:
    //   - `mlp.experts.mlp.w1`: view to `[E, F, H]`, no data move →
    //     [`BakeOp::MoeExpertReshape`].
    //   - `mlp.experts.mlp.w2`: view to `[E, F, H]` + transpose(1,2) →
    //     [`BakeOp::MoeExpertTranspose`].
    //
    // Both safetensors are 2-D `[E * F, H]`; deriving `F` and `H`
    // requires `ctx.num_experts`. If the caller couldn't recover that
    // hparam (e.g. v1.5 checkpoint, or v2-moe checkpoint with malformed
    // config), the tensor is `Unmapped` so the convert errors with a
    // typed `UnmappedTensor`.
    if rest == "mlp.experts.mlp.w1" || rest == "mlp.experts.mlp.w2" {
        let Some(n_experts) = ctx.num_experts else {
            return None;
        };
        // safetensors stores w1/w2 as 2-D `[n_experts * n_inner, n_embd]`.
        // Sanity-check the rank and derive `n_inner`/`n_embd`.
        if hf_shape.len() != 2 {
            return None;
        }
        let outer = hf_shape[0];
        let n_embd = hf_shape[1];
        if n_experts == 0 || outer % n_experts != 0 {
            return None;
        }
        let n_inner = outer / n_experts;
        return Some(if rest == "mlp.experts.mlp.w1" {
            MappedTensor::DirectWithBake {
                gguf_name: format!("blk.{layer}.ffn_up_exps.weight"),
                bake: BakeOp::MoeExpertReshape {
                    n_experts,
                    n_inner,
                    n_embd,
                },
            }
        } else {
            MappedTensor::DirectWithBake {
                gguf_name: format!("blk.{layer}.ffn_down_exps.weight"),
                bake: BakeOp::MoeExpertTranspose {
                    n_experts,
                    n_inner,
                    n_embd,
                },
            }
        });
    }

    let suffix = match rest {
        // Fused QKV.
        // v1.5 has weight only; v2-moe has weight + bias.
        "attn.Wqkv.weight" => "attn_qkv.weight",
        "attn.Wqkv.bias" => "attn_qkv.bias",
        // Attention output projection.
        // v1.5 has weight only; v2-moe has weight + bias.
        "attn.out_proj.weight" => "attn_output.weight",
        "attn.out_proj.bias" => "attn_output.bias",
        // Post-attention LayerNorm — weight + bias on both v1.5 and v2-moe.
        "norm1.weight" => "attn_output_norm.weight",
        "norm1.bias" => "attn_output_norm.bias",
        // v1.5 SwiGLU FFN — fc11 is the gate, fc12 is the up,
        // fc2 is the down. All weight-only on v1.5.
        "mlp.fc11.weight" => "ffn_gate.weight",
        "mlp.fc12.weight" => "ffn_up.weight",
        // v2-moe dense FFN (layers where `bid % moe_every_n_layers != 0`)
        // is plain GELU: fc1 is the up, fc2 is the down. Both weight + bias.
        "mlp.fc1.weight" => "ffn_up.weight",
        "mlp.fc1.bias" => "ffn_up.bias",
        // `fc2` is shared between v1.5 (weight only) and v2-moe dense
        // (weight + bias).
        "mlp.fc2.weight" => "ffn_down.weight",
        "mlp.fc2.bias" => "ffn_down.bias",
        // v2-moe routing projection — `tensor_mapping.py:442` maps
        // `encoder.layers.{bid}.mlp.router.layer` to FFN_GATE_INP.
        "mlp.router.layer.weight" => "ffn_gate_inp.weight",
        // End-of-layer LayerNorm — weight + bias.
        "norm2.weight" => "layer_output_norm.weight",
        "norm2.bias" => "layer_output_norm.bias",
        // v2-moe expert weights (`mlp.experts.mlp.w1` /
        // `mlp.experts.mlp.w2`) need a shape-aware bake — view to
        // `[E, F, H]` + (for w2) transpose-last-two-dims. The bake
        // infrastructure for this lands in a follow-up commit; for
        // now they surface as `Unmapped` so the convert pipeline
        // hard-errors rather than silently emitting wrong-shape data.
        // Per [[feedback-no-loop-suppression-2026-05-17]].
        _ => return None,
    };

    Some(MappedTensor::Direct(format!("blk.{layer}.{suffix}")))
}

/// Build the GGUF metadata KV pairs for a NomicBert model from its HF
/// `config.json`. Port of `conversion/bert.py::NomicBertModel::set_gguf_parameters`
/// + the `BertModel.set_gguf_parameters` + `TextModel.set_gguf_parameters`
/// base it transitively `super()`s into.
///
/// NomicBert HF configs are typically GPT-Neo / Mosaic-style and use
/// `n_embd` / `n_layer` / `n_head` / `n_inner` (not the
/// `hidden_size` / `num_hidden_layers` family). We accept BOTH name
/// conventions and prefer the GPT-style key when present, to match
/// what `find_hparam(["n_embd", "hidden_size"])` does in the Python
/// base.
///
/// Required HF keys (at least one of each pair must be present;
/// missing-both → caller-side panic from the `.expect`):
///   - `n_embd`           OR `hidden_size`
///   - `n_layer`          OR `num_hidden_layers`
///   - `n_head`           OR `num_attention_heads`
///   - `max_position_embeddings` (mandatory)
///
/// Optional HF keys (defaulted):
///   - `n_inner` — defaults to `4 * embedding_length` per the
///     GPT-Neo `Config.__init__` default (NomicBert v1.5 supplies it
///     explicitly = 2048 for the 137M model).
///   - `layer_norm_epsilon` — defaults to `1.0e-12` per
///     `transformers.BertConfig` (NomicBert v1/v1.5 supply it).
///   - `rotary_emb_base` — defaults to `10000.0` per the
///     `NomicBertConfig` factory default.
///   - `_name_or_path` — defaults to `"model"`.
///
/// `file_type` is the chosen ggml file-type as a `u32` (matches
/// `gguf_writer.add_file_type(self.ftype)` at base.py).
pub fn build_metadata(
    config: &serde_json::Value,
    file_type: u32,
    model_card: Option<&ModelCard>,
) -> Vec<(String, MetaValue)> {
    let raw_name = config
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("model");
    // Parse the HF model id into canonical `general.*` components.
    // For nomic v2-moe (`"nomic-ai/nomic-xlm-2048"`) this gives:
    //   name = "Nomic Xlm 2048", organization = "Nomic Ai",
    //   basename = "nomic-xlm", version = "2048".
    // Used only on the v2-moe branch — v1.5 keeps the legacy raw
    // `_name_or_path` for `general.name` (no byte-cmp gate established
    // for the v1.5 nomic path yet, and changing it would break the
    // existing v1.5 metadata-builder tests).
    let id_components = get_model_id_components(raw_name);
    let raw_name_string = raw_name.to_string();

    // Pull a required `u64` from either of two HF keys.
    let pick_u64 = |k_gpt: &str, k_hf: &str| -> u32 {
        let v = config
            .get(k_gpt)
            .and_then(|v| v.as_u64())
            .or_else(|| config.get(k_hf).and_then(|v| v.as_u64()))
            .unwrap_or_else(|| {
                panic!(
                    "config.json missing required key (`{k_gpt}` or `{k_hf}`)"
                )
            });
        v as u32
    };

    let hidden_size = pick_u64("n_embd", "hidden_size");
    let n_layers = pick_u64("n_layer", "num_hidden_layers");
    let n_head = pick_u64("n_head", "num_attention_heads");

    // NomicBert has no GQA — head_count_kv == head_count.
    let n_head_kv = n_head;

    // Detect MoE via presence of `moe_every_n_layers` (canonical
    // `bert.py:323`: `self.is_moe = bool(hparams.get("moe_every_n_layers"))`).
    // On the v2-moe code path: switch the architecture name to
    // `nomic-bert-moe` (canonical `MODEL_ARCH.NOMIC_BERT_MOE` at
    // `bert.py:324`) and emit the three MoE-specific KV pairs that
    // llama.cpp's nomic-bert-moe loader requires:
    //   - `<arch>.expert_count` (`num_experts` or `num_local_experts`,
    //     base TextModel.set_gguf_parameters at `base.py:1194-1196`)
    //   - `<arch>.expert_used_count` (`moe_top_k`, NomicBertModel at
    //     `bert.py:388`)
    //   - `<arch>.moe_every_n_layers` (NomicBertModel at `bert.py:387`)
    let moe_every_n = config
        .get("moe_every_n_layers")
        .and_then(|v| v.as_u64())
        .filter(|&n| n > 0)
        .map(|n| n as u32);
    let is_moe = moe_every_n.is_some();
    let arch_name = if is_moe { "nomic-bert-moe" } else { "nomic-bert" };

    // Hparam selection mirrors what canonical sees AFTER
    // `transformers.AutoConfig.from_pretrained` injects defaults.
    // For nomic v2-moe (transformers v5.5+), the injected fields that
    // override the explicit config.json values are:
    //   - `layer_norm_eps = 1e-12` (BERT-family default, takes priority
    //     over config's `layer_norm_epsilon = 1e-5` per
    //     `base.py:1191` find_hparam priority).
    //   - `max_position_embeddings = 2048` (BERT-family default; then
    //     `_xlmroberta_tokenizer_init` at `bert.py:104-111` subtracts
    //     `1 + pad_token_id` if max_position_embeddings is in hparams).
    //   - `head_dim = hidden_size / num_attention_heads` (transformers
    //     fills in when omitted from config).
    //   - `rope_parameters = {rope_theta: 1000.0, rope_type: 'default'}`
    //     (nomic_bert-specific transformers default; overrides config's
    //     `rotary_emb_base = 10000`).
    //
    // Verified by `python3 -c "from transformers import AutoConfig;
    // print(AutoConfig.from_pretrained(<v2-moe-dir>).to_dict())"` at
    // commit 28525977. The injected values reproduce in canonical's
    // GGUF dump as: `context_length=2046`, `rope.freq_base=1000.0`,
    // `attention.layer_norm_epsilon=1e-12`, `attention.key_length=64`,
    // `attention.value_length=64`.
    let head_dim = hidden_size / n_head;
    let pad_token_id = config
        .get("pad_token_id")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let n_positions_raw = config
        .get("max_position_embeddings")
        .or_else(|| config.get("n_positions"))
        .and_then(|v| v.as_u64())
        .expect("config.json missing required key `max_position_embeddings` (or `n_positions`)")
        as u32;
    let ctx_len = if is_moe {
        // v2-moe uses Unigram (XLM-RoBERTa); canonical subtracts
        // `1 + pad_token_id` from max_position_embeddings via
        // `_xlmroberta_tokenizer_init`. For nomic v2-moe with
        // `pad_token_id=1` and AutoConfig-injected
        // `max_position_embeddings=2048`: `2048 - 2 = 2046`.
        let offset = 1 + pad_token_id.expect(
            "v2-moe config.json: pad_token_id required for context_length offset",
        );
        n_positions_raw.saturating_sub(offset)
    } else {
        n_positions_raw
    };

    let ffn_len = config
        .get("n_inner")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32)
        .unwrap_or(4 * hidden_size);

    // layer_norm_epsilon: for v2-moe canonical sees AutoConfig-injected
    // `layer_norm_eps = 1e-12` (BERT default; priority over config's
    // 1e-5 per `find_hparam` order). For v1.5 the config-explicit
    // value (typically 1e-12) is used directly.
    let ln_eps = if is_moe {
        1.0e-12_f32
    } else {
        config
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0e-12) as f32
    };

    // rope_freq_base: for v2-moe canonical sees AutoConfig-injected
    // `rope_parameters.rope_theta = 1000.0` (nomic_bert-specific
    // transformers default; priority over config's
    // `rotary_emb_base = 10000`). For v1.5 the config value applies.
    let rope_theta = if is_moe {
        1000.0_f32
    } else {
        config
            .get("rotary_emb_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32
    };

    if is_moe {
        // v2-moe path — emit the canonical key order observed in
        // gguf-dump of the canonical reference convert (verified at
        // commit 28525977). Canonical's order is:
        //   architecture, name, block_count, context_length,
        //   embedding_length, feed_forward_length, head_count,
        //   rope.freq_base, attention.layer_norm_epsilon,
        //   expert_count, attention.key_length, attention.value_length,
        //   attention.causal, pooling_type, moe_every_n_layers,
        //   expert_used_count, file_type
        // Mirroring this exact order helps the KV-section bytes
        // line up. (Tensor data bytes are the other independent
        // axis of byte-identity.)
        let n_experts = config
            .get("num_experts")
            .or_else(|| config.get("num_local_experts"))
            .and_then(|v| v.as_u64())
            .expect(
                "v2-moe config.json: `moe_every_n_layers` present but \
                 `num_experts` / `num_local_experts` missing",
            ) as u32;
        let moe_top_k = config
            .get("moe_top_k")
            .or_else(|| config.get("num_experts_per_tok"))
            .and_then(|v| v.as_u64())
            .expect(
                "v2-moe config.json: `moe_every_n_layers` present but \
                 `moe_top_k` / `num_experts_per_tok` missing",
            ) as u32;
        let n_layers_per_moe = moe_every_n.unwrap();
        let _ = n_head_kv; // canonical doesn't emit head_count_kv for nomic
        let mut kv_v2moe: Vec<(String, MetaValue)> = Vec::with_capacity(64);
        kv_v2moe.push((
            "general.architecture".into(),
            MetaValue::String(arch_name.into()),
        ));
        // `general.type` is hardcoded `"model"` for all model GGUFs
        // canonical emits (vs `"adapter"` for LoRA). See canonical's
        // implicit default at `metadata.py`.
        kv_v2moe.push(("general.type".into(), MetaValue::String("model".into())));
        // v2-moe path uses the title-cased name from
        // `get_model_id_components` to match canonical's observed
        // `general.name = "Nomic Xlm 2048"`. Falls back to the raw
        // `_name_or_path` for "human sentence" / unparseable inputs.
        let v2moe_name = id_components
            .name
            .clone()
            .unwrap_or_else(|| raw_name_string.clone());
        kv_v2moe.push(("general.name".into(), MetaValue::String(v2moe_name)));
        // `general.{version, organization, basename}` come from the
        // name-parser heuristic (`get_model_id_components` port of
        // `metadata.py:240-362`). Only emitted when the parser
        // produced a value — canonical's same gating.
        if let Some(v) = &id_components.version {
            kv_v2moe.push(("general.version".into(), MetaValue::String(v.clone())));
        }
        if let Some(o) = &id_components.organization {
            kv_v2moe.push(("general.organization".into(), MetaValue::String(o.clone())));
        }
        if let Some(b) = &id_components.basename {
            kv_v2moe.push(("general.basename".into(), MetaValue::String(b.clone())));
        }
        // TODO follow-up iteration: `general.size_label` from the
        // tensor walk (canonical's `gguf.size_label(total_params,
        // shared_params, expert_params, expert_count)`).
        kv_v2moe.extend([
            (format!("{arch_name}.block_count"), MetaValue::U32(n_layers)),
            (format!("{arch_name}.context_length"), MetaValue::U32(ctx_len)),
            (
                format!("{arch_name}.embedding_length"),
                MetaValue::U32(hidden_size),
            ),
            (
                format!("{arch_name}.feed_forward_length"),
                MetaValue::U32(ffn_len),
            ),
            (
                format!("{arch_name}.attention.head_count"),
                MetaValue::U32(n_head),
            ),
            (
                format!("{arch_name}.rope.freq_base"),
                MetaValue::F32(rope_theta),
            ),
            (
                format!("{arch_name}.attention.layer_norm_epsilon"),
                MetaValue::F32(ln_eps),
            ),
            (
                format!("{arch_name}.expert_count"),
                MetaValue::U32(n_experts),
            ),
            (
                format!("{arch_name}.attention.key_length"),
                MetaValue::U32(head_dim),
            ),
            (
                format!("{arch_name}.attention.value_length"),
                MetaValue::U32(head_dim),
            ),
            (
                format!("{arch_name}.attention.causal"),
                MetaValue::Bool(false),
            ),
            (format!("{arch_name}.pooling_type"), MetaValue::U32(1)),
            (
                format!("{arch_name}.moe_every_n_layers"),
                MetaValue::U32(n_layers_per_moe),
            ),
            (
                format!("{arch_name}.expert_used_count"),
                MetaValue::U32(moe_top_k),
            ),
        ]);
        // HF model-card metadata (from README.md YAML frontmatter).
        // Canonical emits these in a fixed order via
        // `gguf-py/gguf/metadata.py::Metadata.set_gguf_meta_model`.
        // For nomic v2-moe the observed subset is:
        //   general.license, general.base_model.{count, 0.*},
        //   general.tags, general.languages.
        // The basename / version / organization / size_label fields
        // require the name-parsing heuristic at `metadata.py:240-355`
        // — separate iteration (size_label also needs param-counting).
        if let Some(card) = model_card {
            if let Some(license) = &card.license {
                kv_v2moe.push(("general.license".into(), MetaValue::String(license.clone())));
            }
            if !card.base_models.is_empty() {
                kv_v2moe.push((
                    "general.base_model.count".into(),
                    MetaValue::U32(card.base_models.len() as u32),
                ));
                for (i, entry) in card.base_models.iter().enumerate() {
                    let (name, org, url) = split_base_model(&entry.raw);
                    if let Some(name) = name {
                        kv_v2moe.push((
                            format!("general.base_model.{i}.name"),
                            MetaValue::String(name),
                        ));
                    }
                    if let Some(org) = org {
                        kv_v2moe.push((
                            format!("general.base_model.{i}.organization"),
                            MetaValue::String(org),
                        ));
                    }
                    if let Some(url) = url {
                        kv_v2moe.push((
                            format!("general.base_model.{i}.repo_url"),
                            MetaValue::String(url),
                        ));
                    }
                }
            }
            if !card.tags.is_empty() {
                kv_v2moe.push((
                    "general.tags".into(),
                    MetaValue::ArrayString(card.tags.clone()),
                ));
            }
            if !card.languages.is_empty() {
                kv_v2moe.push((
                    "general.languages".into(),
                    MetaValue::ArrayString(card.languages.clone()),
                ));
            }
        }
        // Canonical emits these last (positions 50-51 in the
        // Q8_0 GGUF dump): `general.quantization_version=2` is
        // added by `llama-quantize` (matches GGUF spec), and
        // `general.file_type` is added by the convert step's
        // `add_file_type(ftype)` at `base.py:1220`. We emit both
        // here since hf2q's convert+quantize is a single pipeline.
        kv_v2moe.push((
            "general.quantization_version".into(),
            MetaValue::U32(2),
        ));
        kv_v2moe.push(("general.file_type".into(), MetaValue::U32(file_type)));
        kv_v2moe
    } else {
        // v1.5 path (unchanged): preserve historical key set + order
        // for the working bge-style WordPiece + non-MoE convert.
        vec![
            (
                "general.architecture".into(),
                MetaValue::String(arch_name.into()),
            ),
            ("general.name".into(), MetaValue::String(raw_name_string.clone())),
            (format!("{arch_name}.context_length"), MetaValue::U32(ctx_len)),
            (
                format!("{arch_name}.embedding_length"),
                MetaValue::U32(hidden_size),
            ),
            (format!("{arch_name}.block_count"), MetaValue::U32(n_layers)),
            (
                format!("{arch_name}.feed_forward_length"),
                MetaValue::U32(ffn_len),
            ),
            (
                format!("{arch_name}.attention.head_count"),
                MetaValue::U32(n_head),
            ),
            (
                format!("{arch_name}.attention.head_count_kv"),
                MetaValue::U32(n_head_kv),
            ),
            (
                format!("{arch_name}.attention.layer_norm_epsilon"),
                MetaValue::F32(ln_eps),
            ),
            (
                format!("{arch_name}.rope.freq_base"),
                MetaValue::F32(rope_theta),
            ),
            (format!("{arch_name}.pooling_type"), MetaValue::U32(1)),
            ("general.file_type".into(), MetaValue::U32(file_type)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Ctx with `num_experts = None` (v1.5 layout). Most tests don't
    /// touch expert-tensor mappings so this is the right default.
    fn ctx_v15() -> NomicBertCtx {
        NomicBertCtx { num_experts: None }
    }

    /// Ctx for v2-moe checkpoints (`num_experts = Some(8)`, matching
    /// `/opt/hf2q/models/nomic-ai-nomic-embed-text-v2-moe/config.json`).
    fn ctx_v2_moe() -> NomicBertCtx {
        NomicBertCtx {
            num_experts: Some(8),
        }
    }

    /// Empty shape — used when the tensor mapping doesn't depend on
    /// the safetensors shape (every non-expert tensor).
    const NO_SHAPE: &[usize] = &[];

    /// Small helper — assert that `map_tensor_name(hf, shape, ctx)`
    /// returns `Some(MappedTensor::Direct(gguf))`. Keeps the per-case
    /// row noise low in the table tests.
    #[track_caller]
    fn assert_direct(hf: &str, expected_gguf: &str) {
        let got = map_tensor_name(hf, NO_SHAPE, &ctx_v15());
        match got.as_ref() {
            Some(MappedTensor::Direct(s)) => assert_eq!(
                s.as_str(),
                expected_gguf,
                "map_tensor_name({hf:?}) = Direct({s:?}), want Direct({expected_gguf:?})"
            ),
            other => panic!(
                "map_tensor_name({hf:?}) = {other:?}, want Some(Direct({expected_gguf:?}))"
            ),
        }
    }

    /// Acceptance test 1 — round-trip the canonical table for every HF
    /// name kind. Asserts `map_tensor_name(hf) → Some(Direct(gguf))`
    /// with the exact pair from the rustdoc table above.
    #[test]
    fn nomic_bert_tensor_name_round_trip() {
        let cases: &[(&str, &str)] = &[
            // Globals (v1.5 layout)
            ("embeddings.word_embeddings.weight", "token_embd.weight"),
            ("emb_ln.weight", "token_embd_norm.weight"),
            ("emb_ln.bias", "token_embd_norm.bias"),
            // Globals (v2-moe layout: XLM-RoBERTa-style)
            ("embeddings.token_type_embeddings.weight", "token_types.weight"),
            ("embeddings.LayerNorm.weight", "token_embd_norm.weight"),
            ("embeddings.LayerNorm.bias", "token_embd_norm.bias"),
            // Per-block — sample at L=0, L=5, L=11 (nomic-embed v1.5
            // has 12 layers so 0/5/11 covers edge/mid/depth).
            ("encoder.layers.0.attn.Wqkv.weight", "blk.0.attn_qkv.weight"),
            ("encoder.layers.5.attn.Wqkv.weight", "blk.5.attn_qkv.weight"),
            (
                "encoder.layers.11.attn.Wqkv.weight",
                "blk.11.attn_qkv.weight",
            ),
            (
                "encoder.layers.0.attn.out_proj.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "encoder.layers.3.norm1.weight",
                "blk.3.attn_output_norm.weight",
            ),
            (
                "encoder.layers.3.norm1.bias",
                "blk.3.attn_output_norm.bias",
            ),
            // v1.5 SwiGLU FFN tensors.
            ("encoder.layers.7.mlp.fc11.weight", "blk.7.ffn_gate.weight"),
            ("encoder.layers.7.mlp.fc12.weight", "blk.7.ffn_up.weight"),
            ("encoder.layers.7.mlp.fc2.weight", "blk.7.ffn_down.weight"),
            // End-of-layer LayerNorm.
            (
                "encoder.layers.9.norm2.weight",
                "blk.9.layer_output_norm.weight",
            ),
            (
                "encoder.layers.9.norm2.bias",
                "blk.9.layer_output_norm.bias",
            ),
        ];

        for &(hf, expected_gguf) in cases {
            assert_direct(hf, expected_gguf);
        }
    }

    /// Acceptance test 1b — same canonical mappings work when the HF
    /// checkpoint carries the optional `bert.` prefix (some
    /// `BertForMaskedLM`-style exports do; nomic-embed-text-v1.5
    /// typically does not but the strip is harmless and mirrors
    /// `BertModel.filter_tensors`).
    #[test]
    fn nomic_bert_strips_optional_bert_prefix() {
        assert_direct("bert.embeddings.word_embeddings.weight", "token_embd.weight");
        assert_direct(
            "bert.encoder.layers.0.attn.Wqkv.weight",
            "blk.0.attn_qkv.weight",
        );
        assert_direct(
            "bert.encoder.layers.4.mlp.fc11.weight",
            "blk.4.ffn_gate.weight",
        );
        assert_direct("bert.emb_ln.bias", "token_embd_norm.bias");
    }

    /// Acceptance test 1c — v2-moe block-level patterns: biased
    /// attention, GELU dense FFN (fc1/fc2), router, and `mlp.experts.bias`
    /// drop. Mirrors `/opt/llama.cpp/conversion/bert.py::NomicBertModel`
    /// at the v2-moe `is_moe=True` branch.
    #[test]
    fn nomic_bert_v2_moe_block_patterns_map() {
        // Biased attention (Wqkv + out_proj) — v2-moe only.
        assert_direct("encoder.layers.0.attn.Wqkv.bias", "blk.0.attn_qkv.bias");
        assert_direct("encoder.layers.5.attn.Wqkv.bias", "blk.5.attn_qkv.bias");
        assert_direct(
            "encoder.layers.0.attn.out_proj.bias",
            "blk.0.attn_output.bias",
        );
        assert_direct(
            "encoder.layers.11.attn.out_proj.bias",
            "blk.11.attn_output.bias",
        );

        // GELU dense FFN — v2-moe layers where `bid % moe_every_n_layers != 0`
        // use fc1/fc2 with biases.
        assert_direct("encoder.layers.0.mlp.fc1.weight", "blk.0.ffn_up.weight");
        assert_direct("encoder.layers.0.mlp.fc1.bias", "blk.0.ffn_up.bias");
        assert_direct("encoder.layers.0.mlp.fc2.bias", "blk.0.ffn_down.bias");
        assert_direct("encoder.layers.10.mlp.fc1.weight", "blk.10.ffn_up.weight");

        // MoE routing projection.
        assert_direct(
            "encoder.layers.1.mlp.router.layer.weight",
            "blk.1.ffn_gate_inp.weight",
        );
        assert_direct(
            "encoder.layers.11.mlp.router.layer.weight",
            "blk.11.ffn_gate_inp.weight",
        );
    }

    /// Acceptance test 1d — canonical filter_tensors drops. Mirrors
    /// `bert.py:74-90` (pooler + position_embeddings) and `bert.py:366-369`
    /// (`mlp.experts.bias`).
    #[test]
    fn nomic_bert_filter_tensors_drops_are_drop_outcome() {
        let ctx = ctx_v2_moe();
        assert_eq!(
            map_tensor_name("pooler.dense.weight", NO_SHAPE, &ctx),
            Some(MappedTensor::Drop)
        );
        assert_eq!(
            map_tensor_name("pooler.dense.bias", NO_SHAPE, &ctx),
            Some(MappedTensor::Drop)
        );
        assert_eq!(
            map_tensor_name("embeddings.position_embeddings.weight", NO_SHAPE, &ctx),
            Some(MappedTensor::Drop)
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.experts.bias", NO_SHAPE, &ctx),
            Some(MappedTensor::Drop)
        );
        assert_eq!(
            map_tensor_name("encoder.layers.11.mlp.experts.bias", NO_SHAPE, &ctx),
            Some(MappedTensor::Drop)
        );
    }

    /// Acceptance test 1e — v2-moe expert weight tensors map to
    /// `DirectWithBake` carrying the canonical reshape / transpose.
    /// Mirrors `/opt/llama.cpp/conversion/bert.py:373-380`.
    ///
    /// Real safetensors shape for nomic-embed-text-v2-moe is
    /// `[n_experts=8, n_inner=3072, n_embd=768]` (verified at
    /// `/opt/hf2q/models/nomic-ai-nomic-embed-text-v2-moe/model.safetensors`):
    /// `mlp.experts.mlp.w1` and `.w2` both stored as 2-D
    /// `[24576, 768]` = `[8 * 3072, 768]`.
    #[test]
    fn nomic_bert_v2_moe_expert_weights_map_to_direct_with_bake() {
        let ctx = ctx_v2_moe();
        let hf_shape: &[usize] = &[24576, 768];

        match map_tensor_name("encoder.layers.1.mlp.experts.mlp.w1", hf_shape, &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.1.ffn_up_exps.weight");
                assert_eq!(
                    bake,
                    BakeOp::MoeExpertReshape {
                        n_experts: 8,
                        n_inner: 3072,
                        n_embd: 768,
                    }
                );
            }
            other => panic!(
                "expected DirectWithBake(MoeExpertReshape) for w1, got {other:?}"
            ),
        }

        match map_tensor_name("encoder.layers.7.mlp.experts.mlp.w2", hf_shape, &ctx) {
            Some(MappedTensor::DirectWithBake { gguf_name, bake }) => {
                assert_eq!(gguf_name, "blk.7.ffn_down_exps.weight");
                assert_eq!(
                    bake,
                    BakeOp::MoeExpertTranspose {
                        n_experts: 8,
                        n_inner: 3072,
                        n_embd: 768,
                    }
                );
            }
            other => panic!(
                "expected DirectWithBake(MoeExpertTranspose) for w2, got {other:?}"
            ),
        }
    }

    /// Without `num_experts` (v1.5 ctx), the expert tensors must be
    /// `Unmapped` so the convert errors typed. Mirrors the no-fallback
    /// rule: a v2-moe checkpoint with malformed config (missing
    /// `num_experts`) is a hard error, not a silent skip.
    #[test]
    fn nomic_bert_expert_weights_unmapped_when_ctx_lacks_num_experts() {
        let ctx = ctx_v15();
        let hf_shape: &[usize] = &[24576, 768];
        assert_eq!(
            map_tensor_name("encoder.layers.1.mlp.experts.mlp.w1", hf_shape, &ctx),
            None,
            "without num_experts the expert tensors must surface as \
             Unmapped (typed error) to match no-fallback rule"
        );
        assert_eq!(
            map_tensor_name("encoder.layers.1.mlp.experts.mlp.w2", hf_shape, &ctx),
            None
        );
    }

    /// Expert tensor with non-divisible `outer / n_experts` is a
    /// hard error — surfaces as `Unmapped`.
    #[test]
    fn nomic_bert_expert_weights_unmapped_when_shape_does_not_divide() {
        let ctx = ctx_v2_moe();
        // outer = 24573 is NOT divisible by 8 → reject.
        let hf_shape: &[usize] = &[24573, 768];
        assert_eq!(
            map_tensor_name("encoder.layers.1.mlp.experts.mlp.w1", hf_shape, &ctx),
            None
        );
    }

    /// Sibling — genuinely-unknown names must surface as `None` so the
    /// caller raises `UnmappedTensor`. Per
    /// [[feedback-no-loop-suppression-2026-05-17]]: never silently skip.
    #[test]
    fn nomic_bert_tensor_name_rejects_unknown_kinds() {
        let ctx = ctx_v2_moe();
        // Wrong prefix (singular `layer` is BERT, not nomic-bert).
        assert_eq!(
            map_tensor_name(
                "encoder.layer.0.attention.self.Wqkv.weight",
                NO_SHAPE,
                &ctx
            ),
            None
        );
        // BERT-style split QKV is NOT nomic — fused only.
        assert_eq!(
            map_tensor_name(
                "encoder.layers.0.attention.self.query.weight",
                NO_SHAPE,
                &ctx
            ),
            None
        );
        // v1.5 SwiGLU has no biases on fc11/fc12 — these names should
        // never appear on either v1.5 or v2-moe checkpoints, so they're
        // structural rejects (not Drop).
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc11.bias", NO_SHAPE, &ctx),
            None
        );
        assert_eq!(
            map_tensor_name("encoder.layers.0.mlp.fc12.bias", NO_SHAPE, &ctx),
            None
        );
        // Malformed layer index (leading zero).
        assert_eq!(
            map_tensor_name("encoder.layers.01.attn.Wqkv.weight", NO_SHAPE, &ctx),
            None
        );
        // Empty layer index.
        assert_eq!(
            map_tensor_name("encoder.layers..attn.Wqkv.weight", NO_SHAPE, &ctx),
            None
        );
        // Unknown per-block suffix.
        assert_eq!(
            map_tensor_name("encoder.layers.0.unknown.weight", NO_SHAPE, &ctx),
            None
        );
    }

    /// Acceptance test 2 — feed a minimal hand-written config.json with
    /// the canonical GPT-Neo-style keys (`n_embd`, `n_layer`, `n_head`,
    /// `n_inner`) used by nomic-embed-text-v1.5, and verify all 12 KV
    /// pairs come back with the right types + values.
    #[test]
    fn nomic_bert_metadata_built_from_gpt_style_config() {
        let cfg = json!({
            "_name_or_path": "nomic-ai/nomic-embed-text-v1.5",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "max_position_embeddings": 2048,
            "layer_norm_epsilon": 1.0e-12,
            "rotary_emb_base": 1000.0,
        });

        let kv = build_metadata(&cfg, 17 /* MostlyQ5_K_M */, None);

        // Check count + keyset (don't depend on insertion order).
        assert_eq!(kv.len(), 12, "NomicBert emits 12 KV pairs at v1");
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("nomic-bert".into())
        );
        // v1.5 path keeps the raw `_name_or_path` for backward
        // compat (no byte-cmp gate established for v1.5 nomic yet).
        // The v2-moe path uses the title-cased
        // `get_model_id_components` name to match canonical's
        // observed `general.name = "Nomic Xlm 2048"`.
        assert_eq!(
            by_key["general.name"],
            MetaValue::String("nomic-ai/nomic-embed-text-v1.5".into())
        );
        assert_eq!(by_key["nomic-bert.context_length"], MetaValue::U32(2048));
        assert_eq!(by_key["nomic-bert.embedding_length"], MetaValue::U32(768));
        assert_eq!(by_key["nomic-bert.block_count"], MetaValue::U32(12));
        assert_eq!(
            by_key["nomic-bert.feed_forward_length"],
            MetaValue::U32(3072)
        );
        assert_eq!(
            by_key["nomic-bert.attention.head_count"],
            MetaValue::U32(12)
        );
        assert_eq!(
            by_key["nomic-bert.attention.head_count_kv"],
            MetaValue::U32(12),
            "NomicBert has no GQA — head_count_kv == head_count"
        );
        assert_eq!(
            by_key["nomic-bert.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12)
        );
        assert_eq!(
            by_key["nomic-bert.rope.freq_base"],
            MetaValue::F32(1000.0)
        );
        assert_eq!(
            by_key["nomic-bert.pooling_type"],
            MetaValue::U32(1),
            "Nomic embeddings are mean-pooled (LLAMA_POOLING_TYPE_MEAN = 1)"
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Acceptance test 2b — v2-moe metadata: arch flips to
    /// `nomic-bert-moe`, all KV keys use the new prefix, plus
    /// `expert_count` / `expert_used_count` / `moe_every_n_layers`
    /// are emitted. Mirrors `bert.py:324` (arch) + `bert.py:384-388`
    /// (`set_gguf_parameters` MoE branch).
    #[test]
    fn nomic_bert_v2_moe_metadata_uses_nomic_bert_moe_arch() {
        // Config fixture mirrors the real nomic-embed-text-v2-moe
        // hparams that canonical sees via `AutoConfig.from_pretrained`:
        // the `n_positions=2048` + `pad_token_id=1` combination drives
        // canonical's `_xlmroberta_tokenizer_init` to compute
        // `ctx_len = 2048 - (1 + 1) = 2046`.
        let cfg = json!({
            "_name_or_path": "nomic-ai/nomic-embed-text-v2-moe",
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "n_positions": 2048,
            "pad_token_id": 1,
            "layer_norm_epsilon": 1.0e-5, // config says 1e-5; AutoConfig override → 1e-12
            "rotary_emb_base": 10000.0,   // config says 10000; AutoConfig override → 1000
            "moe_every_n_layers": 2,
            "num_experts": 8,
            "moe_top_k": 2,
        });
        let kv = build_metadata(&cfg, 17, None);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.architecture"],
            MetaValue::String("nomic-bert-moe".into()),
        );
        // Per-arch keys: canonical-equivalent values + key set.
        assert_eq!(
            by_key["nomic-bert-moe.context_length"],
            MetaValue::U32(2046),
            "v2-moe: ctx_len = n_positions - (1 + pad_token_id) = 2046",
        );
        assert_eq!(
            by_key["nomic-bert-moe.embedding_length"],
            MetaValue::U32(768)
        );
        assert_eq!(by_key["nomic-bert-moe.block_count"], MetaValue::U32(12));
        assert_eq!(
            by_key["nomic-bert-moe.feed_forward_length"],
            MetaValue::U32(3072)
        );
        assert_eq!(
            by_key["nomic-bert-moe.attention.head_count"],
            MetaValue::U32(12)
        );
        assert_eq!(
            by_key["nomic-bert-moe.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12),
            "v2-moe: AutoConfig overrides config 1e-5 with BERT-default 1e-12"
        );
        assert_eq!(
            by_key["nomic-bert-moe.rope.freq_base"],
            MetaValue::F32(1000.0),
            "v2-moe: AutoConfig injects rope_parameters.rope_theta=1000.0"
        );
        // Head-dim derived keys (canonical TextModel emits when head_dim
        // is in hparams — AutoConfig fills it from hidden_size/n_head).
        assert_eq!(
            by_key["nomic-bert-moe.attention.key_length"],
            MetaValue::U32(64)
        );
        assert_eq!(
            by_key["nomic-bert-moe.attention.value_length"],
            MetaValue::U32(64)
        );
        // attention.causal=false (BertModel.set_gguf_parameters at bert.py:33).
        assert_eq!(
            by_key["nomic-bert-moe.attention.causal"],
            MetaValue::Bool(false)
        );
        // MoE-specific keys.
        assert_eq!(by_key["nomic-bert-moe.expert_count"], MetaValue::U32(8));
        assert_eq!(
            by_key["nomic-bert-moe.expert_used_count"],
            MetaValue::U32(2)
        );
        assert_eq!(
            by_key["nomic-bert-moe.moe_every_n_layers"],
            MetaValue::U32(2)
        );
        // Canonical does NOT emit head_count_kv for nomic (config has
        // no `num_key_value_heads`).
        assert!(
            !by_key.contains_key("nomic-bert-moe.attention.head_count_kv"),
            "v2-moe must NOT emit head_count_kv (canonical doesn't)"
        );
        // No legacy `nomic-bert.*` keys leaked through.
        assert!(
            !by_key.contains_key("nomic-bert.context_length"),
            "v2-moe must NOT emit nomic-bert.* prefix"
        );
        // general.quantization_version + file_type are emitted at the
        // end of the v2-moe metadata block — canonical's
        // llama-quantize step writes them at GGUF positions 50-51.
        assert_eq!(
            by_key["general.quantization_version"],
            MetaValue::U32(2)
        );
        assert_eq!(by_key["general.file_type"], MetaValue::U32(17));
    }

    /// Sibling — verify the BERT-style key fallback (`hidden_size`,
    /// `num_hidden_layers`, `num_attention_heads`) and the
    /// optional-key defaults all trigger when the HF config omits the
    /// GPT-Neo keys + the defaulted keys.
    #[test]
    fn nomic_bert_metadata_bert_style_keys_and_defaults() {
        let cfg = json!({
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            // n_inner omitted → defaults to 4 * hidden_size = 256
            "max_position_embeddings": 512,
            // layer_norm_epsilon omitted → defaults to 1.0e-12
            // rotary_emb_base omitted → defaults to 10000.0
            // _name_or_path omitted → defaults to "model"
        });
        let kv = build_metadata(&cfg, 0, None);
        let by_key: std::collections::HashMap<_, _> =
            kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();

        assert_eq!(
            by_key["general.name"],
            MetaValue::String("model".into()),
            "name defaults to 'model' when _name_or_path absent"
        );
        assert_eq!(by_key["nomic-bert.embedding_length"], MetaValue::U32(64));
        assert_eq!(by_key["nomic-bert.block_count"], MetaValue::U32(2));
        assert_eq!(by_key["nomic-bert.attention.head_count"], MetaValue::U32(4));
        assert_eq!(
            by_key["nomic-bert.feed_forward_length"],
            MetaValue::U32(256),
            "n_inner defaults to 4 * embedding_length"
        );
        assert_eq!(
            by_key["nomic-bert.attention.layer_norm_epsilon"],
            MetaValue::F32(1.0e-12),
            "layer_norm_epsilon defaults to 1.0e-12 per BertConfig"
        );
        assert_eq!(
            by_key["nomic-bert.rope.freq_base"],
            MetaValue::F32(10000.0),
            "rotary_emb_base defaults to 10000.0"
        );
        assert_eq!(by_key["nomic-bert.pooling_type"], MetaValue::U32(1));
    }
}
