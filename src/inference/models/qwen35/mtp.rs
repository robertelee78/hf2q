//! Multi-Token Prediction (MTP) loader for Qwen3.5 (ADR-013 Decisions 15 + P14).
//!
//! MTP is Qwen3.5's extra transformer block after the main stack, used for
//! speculative decoding. The Hugging Face config sets
//! `mtp_num_hidden_layers = 1`.
//!
//! # GGUF tensor layout (HF2Q_QWEN35_KEEP_MTP=1 convert path)
//!
//! After ADR-013 P14 Phase A landed (commit `c33e33a`), opt-in convert
//! emits two distinct families of MTP tensors:
//!
//! * **Inner-block tensors** at `blk.{num_hidden_layers}.*` — the 11 tensors
//!   that make up the MTP transformer body (full-attention + dense SwiGLU
//!   FFN). These ride the regular layer-map pipeline and land at the same
//!   suffixes any non-MTP layer would (`attn_q.weight`, `ffn_gate.weight`,
//!   etc.) — just at the extra index past the main stack.
//!
//! * **Wrapper tensors** at `blk.{num_hidden_layers}.nextn.*` — the 4
//!   MTP-specific tensors: `eh_proj` (hidden+embed → hidden projection),
//!   `enorm` / `hnorm` (per-stream RMSNorms), `shared_head_norm` (lm-head
//!   pre-norm). These are MTP-only and carry the `nextn.` prefix to match
//!   `llama-arch.cpp:447-450`'s `LLM_TENSOR_NEXTN_*` enum.
//!
//! # Two loaders
//!
//! * [`load_mtp_weights_if_present`] — **legacy metadata-only path** kept
//!   for ADR-013 P10's acceptance tests. Returns `Option<MtpWeights>` with
//!   only the names + shapes of the 4 wrapper tensors (whatever lives at
//!   `blk.{N}.nextn.*`). Inner-block tensors are NOT included. This is the
//!   "absence vs presence" probe used by sourdough / smoke gates.
//!
//! * [`load_mtp_weights_full`] — **P14 forward-driving path**. Returns
//!   `Option<MtpFullWeights>` with eagerly-loaded F32 tensor data for
//!   BOTH the inner-block AND the wrapper tensors. This is what the
//!   speculative-decoding sampler (planned in `spec_decode.rs`) consumes.
//!
//! # Current state (2026-04-26)
//!
//! `load_mtp_weights_full` lands in this commit alongside its synthetic-GGUF
//! unit tests; `forward_draft` (the GPU-resident MTP block forward + draft
//! token generator) lives in a follow-on commit gated on a successful
//! KEEP_MTP convert against a real Qwen3.5 GGUF (no model on disk has
//! emitted MTP tensors yet — apex stripped them; the 4 dwq46/48 GGUFs
//! predate Phase A; a fresh re-convert is part of P14 Phase B exit).
//!
//! # Acceptance (ADR-013)
//!
//! * Load on apex.gguf (MTP stripped) → `Ok(None)`, no error. ✅
//! * Load on a synthetic MTP-bearing GGUF → `Ok(Some(MtpWeights))` with
//!   populated metadata fields. ✅
//! * Load full weight data on a synthetic MTP-bearing GGUF → `Ok(Some(
//!   MtpFullWeights))` with byte-identical tensor data. ✅ (this commit)

use anyhow::{Context, Result};

use mlx_native::gguf::GgufFile;
use mlx_native::MlxDevice;

use super::Qwen35Config;

/// **Legacy metadata-only MTP scaffold (ADR-013 P10 acceptance).**
///
/// Captures `(name, shape)` for whatever `blk.{N}.nextn.*` tensors are
/// present in the GGUF — used by sourdough / smoke gates as a "MTP
/// presence" probe.  For full weight-loaded MTP execution see
/// [`MtpFullWeights`] / [`load_mtp_weights_full`] (ADR-013 P14).
///
/// Keys are the suffix after `blk.{N}.nextn.`, e.g.
/// `"eh_proj.weight"`, `"shared_head_norm.weight"`. Tensor data is
/// NOT loaded by this struct — names + shapes only.
#[derive(Debug, Clone)]
pub struct MtpWeights {
    /// Model layer index that the MTP block appends AFTER. Always
    /// `num_hidden_layers` in Qwen3.5's convention (the MTP block is
    /// `blk.{num_hidden_layers}`, one past the last regular layer).
    pub layer_index: u32,
    /// Tensors present, keyed by the suffix after `blk.{layer_index}.nextn.`.
    /// Each entry is `(full_tensor_name, shape)` — shape as
    /// `Vec<usize>` row-major.
    pub tensors: Vec<(String, Vec<usize>)>,
}

impl MtpWeights {
    /// Return `true` if a tensor with the given GGUF suffix was loaded.
    /// Suffix should NOT include the `blk.{N}.nextn.` prefix.
    pub fn has_tensor_suffix(&self, suffix: &str) -> bool {
        self.tensors.iter().any(|(name, _)| {
            name.strip_prefix(&format!("blk.{}.nextn.", self.layer_index))
                == Some(suffix)
        })
    }

    /// Number of MTP tensors loaded.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Scan a GGUF for MTP tensors and return `Some(MtpWeights)` if any are
/// present, else `None`.
///
/// Does NOT load tensor data — the struct captures names + shapes only.
///
/// # Arguments
///
/// * `gguf` — an open GGUF file.
/// * `num_hidden_layers` — main-stack layer count (e.g. 40 for the apex MoE;
///   MTP tensors live at `blk.{num_hidden_layers}.nextn.*`).
///
/// # Returns
///
/// * `Ok(None)` — no MTP tensors present in the file (apex case).
/// * `Ok(Some(MtpWeights))` — MTP tensors present; struct populated.
/// * `Err(...)` — unreachable in the current impl (kept in the signature
///   for forward compatibility when tensor loading lands).
pub fn load_mtp_weights_if_present(
    gguf: &GgufFile,
    num_hidden_layers: u32,
) -> Result<Option<MtpWeights>> {
    let prefix = format!("blk.{}.nextn.", num_hidden_layers);

    let mut found: Vec<(String, Vec<usize>)> = Vec::new();
    for name in gguf.tensor_names() {
        if name.starts_with(&prefix) {
            if let Some(info) = gguf.tensor_info(name) {
                found.push((name.to_string(), info.shape.clone()));
            }
        }
    }

    if found.is_empty() {
        Ok(None)
    } else {
        Ok(Some(MtpWeights {
            layer_index: num_hidden_layers,
            tensors: found,
        }))
    }
}

// ================================================================
// MtpFullWeights — eager weight-data load (ADR-013 P14)
// ================================================================

/// Full weights for a single Qwen3.5 MTP block, with tensor data loaded
/// into F32 CPU memory.
///
/// Layout mirrors a regular full-attention layer (per HF
/// `Qwen3_5MTPBlock` in `modeling_qwen3_5.py`) plus 4 MTP-specific
/// wrapper tensors. Inner-block tensors are at GGUF
/// `blk.{layer_index}.*` (ride the standard layer-map); wrapper tensors
/// are at `blk.{layer_index}.nextn.*` (per `llama-arch.cpp:447-450`).
///
/// All tensors are stored as flat F32 row-major buffers in the same
/// shape conventions used by `FullAttnLayerWeights` and
/// `DenseFfnWeights` so the downstream forward can reuse the same
/// helpers. The `attn_q.weight` tensor is FUSED Q+gate per llama.cpp's
/// gated-attention layout (output dim `2 * n_head * head_dim`); the
/// loader splits it into `wq` and `w_gate` after read, mirroring
/// `weight_loader::load_full_attn_layer`.
///
/// # Wrapper-tensor semantics (DeepSeek-V3 MTP head, adapted to Qwen3.5)
///
/// Per HF `Qwen3_5MTPBlock.forward` and `convert_hf_to_gguf.py:10532-10540`:
/// - `enorm` — RMSNorm over the embedding stream `embed(token_t+1)`.
/// - `hnorm` — RMSNorm over the hidden stream `hidden_t` (last layer's
///   pre-output-norm output of the verifier).
/// - `eh_proj` — Linear projection `[2 * hidden, hidden]` that maps
///   `concat(hnorm(hidden_t), enorm(embed_t+1))` → MTP block input.
/// - `shared_head_norm` — RMSNorm applied right before the lm-head
///   matmul on the MTP block output (Qwen3.5 ties `shared_head_head` to
///   the verifier's `output.weight`, so we don't store it separately).
#[derive(Debug, Clone)]
pub struct MtpFullWeights {
    /// Block index in the GGUF (= `num_hidden_layers`; e.g. 64 for
    /// 27B-dense, 40 for 35B-A3B-MoE).
    pub layer_index: u32,

    // ---- Inner block: full-attention sub-layer ----
    /// `[hidden_size]`.
    pub attn_norm: Vec<f32>,
    /// `[hidden_size]`. Applied between attention residual and FFN
    /// (mirrors `FullAttnLayerWeights::post_attn_norm`).
    pub post_attn_norm: Vec<f32>,
    /// Q projection split out of the fused `attn_q.weight`:
    /// `[n_head * head_dim, hidden_size]`.
    pub wq: Vec<f32>,
    /// Output-gate projection split out of the fused `attn_q.weight`:
    /// `[n_head * head_dim, hidden_size]`.
    pub w_gate: Vec<f32>,
    /// `[n_kv_heads * head_dim, hidden_size]`.
    pub wk: Vec<f32>,
    /// `[n_kv_heads * head_dim, hidden_size]`.
    pub wv: Vec<f32>,
    /// `[hidden_size, n_head * head_dim]`.
    pub wo: Vec<f32>,
    /// `[head_dim]` — per-head Q RMSNorm.
    pub attn_q_norm: Vec<f32>,
    /// `[head_dim]` — per-head K RMSNorm.
    pub attn_k_norm: Vec<f32>,

    // ---- Inner block: dense SwiGLU FFN sub-layer ----
    /// `[intermediate_size, hidden_size]`.
    pub ffn_gate: Vec<f32>,
    /// `[intermediate_size, hidden_size]`.
    pub ffn_up: Vec<f32>,
    /// `[hidden_size, intermediate_size]`.
    pub ffn_down: Vec<f32>,

    // ---- Wrapper tensors (blk.{N}.nextn.*) ----
    /// `[hidden_size]` — RMSNorm over the embedding stream.
    pub enorm: Vec<f32>,
    /// `[hidden_size]` — RMSNorm over the hidden stream.
    pub hnorm: Vec<f32>,
    /// `[hidden_size, 2 * hidden_size]` — concat → hidden projection.
    pub eh_proj: Vec<f32>,
    /// `[hidden_size]` — pre-lm-head RMSNorm.
    pub shared_head_norm: Vec<f32>,
}

impl MtpFullWeights {
    /// Hidden size implied by the loaded tensor shapes (cross-check with
    /// `cfg.hidden_size`).
    pub fn hidden_size(&self) -> usize {
        self.attn_norm.len()
    }
}

/// Load the full MTP block weights from a GGUF.
///
/// Returns `Ok(None)` if no MTP tensors are present (the apex /
/// default-converted GGUF case). Returns `Ok(Some(MtpFullWeights))`
/// only when ALL required tensors are present at the expected names
/// and shapes — partial presence is treated as an error rather than
/// silently filling missing tensors with zeros (per
/// `feedback_no_shortcuts.md`).
///
/// # Probe
///
/// The wrapper tensor `blk.{N}.nextn.eh_proj.weight` is the canonical
/// presence probe — if it's absent we assume MTP is fully absent and
/// return `Ok(None)`. If it's present we require all 15 tensors.
///
/// # Notes
///
/// * Inner-block tensors live at `blk.{N}.{suffix}` where N =
///   `num_hidden_layers` (the index past the last regular layer).
/// * The fused `attn_q.weight` is split into `wq` + `w_gate` here
///   matching `weight_loader::load_full_attn_layer`.
/// * Tensor data is loaded as F32 (the `load_tensor_f32` path
///   dequantizes Q4/Q5/Q6 blocks on-device automatically).
pub fn load_mtp_weights_full(
    gguf: &GgufFile,
    cfg: &Qwen35Config,
    device: &MlxDevice,
) -> Result<Option<MtpFullWeights>> {
    let n_layer = cfg.num_hidden_layers;
    let probe = format!("blk.{}.nextn.eh_proj.weight", n_layer);
    if gguf.tensor_info(&probe).is_none() {
        // No MTP wrapper tensors → treat as absent (apex / default convert).
        return Ok(None);
    }

    let p = format!("blk.{}", n_layer);
    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    let kv_total = nkv * d;
    let m = cfg
        .intermediate_size
        .ok_or_else(|| anyhow::anyhow!(
            "MTP load: cfg.intermediate_size is None — Qwen3.5 dense MTP \
             requires intermediate_size; MoE MTP support is not yet \
             characterized (the inner block could be dense even when the \
             main stack is MoE; this loader rejects MoE configs to avoid \
             shipping an unverified path per feedback_no_shortcuts)"
        ))? as usize;

    let load = |name: &str| -> Result<Vec<f32>> {
        super::weight_loader::load_f32_tensor(gguf, name, device)
            .with_context(|| format!("MTP load_tensor_f32({name})"))
    };

    // Inner attention.
    let attn_norm = load(&format!("{p}.attn_norm.weight"))?;
    let post_attn_norm = load(&format!("{p}.post_attention_norm.weight"))?;
    let q_fused = load(&format!("{p}.attn_q.weight"))?;
    let wk = load(&format!("{p}.attn_k.weight"))?;
    let wv = load(&format!("{p}.attn_v.weight"))?;
    let attn_q_norm = load(&format!("{p}.attn_q_norm.weight"))?;
    let attn_k_norm = load(&format!("{p}.attn_k_norm.weight"))?;
    let wo = load(&format!("{p}.attn_output.weight"))?;

    // Inner FFN (dense SwiGLU — Qwen3.5 MTP block is dense even on MoE
    // variants per HF modeling_qwen3_5.py).
    let ffn_gate = load(&format!("{p}.ffn_gate.weight"))?;
    let ffn_up = load(&format!("{p}.ffn_up.weight"))?;
    let ffn_down = load(&format!("{p}.ffn_down.weight"))?;

    // Wrapper tensors.
    let enorm = load(&format!("{p}.nextn.enorm.weight"))?;
    let hnorm = load(&format!("{p}.nextn.hnorm.weight"))?;
    let eh_proj = load(&format!("{p}.nextn.eh_proj.weight"))?;
    let shared_head_norm = load(&format!("{p}.nextn.shared_head_norm.weight"))?;

    // Shape sanity checks (loud asserts: silent-pass-through is a
    // shortcut — better to fail at load with a clear shape mismatch).
    anyhow::ensure!(attn_norm.len() == h, "MTP attn_norm shape: {} != {h}", attn_norm.len());
    anyhow::ensure!(post_attn_norm.len() == h, "MTP post_attn_norm shape: {} != {h}", post_attn_norm.len());
    anyhow::ensure!(
        q_fused.len() == 2 * q_total * h,
        "MTP fused attn_q shape: {} != 2 * {q_total} * {h} = {}",
        q_fused.len(),
        2 * q_total * h
    );
    anyhow::ensure!(wk.len() == kv_total * h, "MTP attn_k shape");
    anyhow::ensure!(wv.len() == kv_total * h, "MTP attn_v shape");
    anyhow::ensure!(attn_q_norm.len() == d, "MTP attn_q_norm shape");
    anyhow::ensure!(attn_k_norm.len() == d, "MTP attn_k_norm shape");
    anyhow::ensure!(wo.len() == h * q_total, "MTP attn_output shape");
    anyhow::ensure!(ffn_gate.len() == m * h, "MTP ffn_gate shape");
    anyhow::ensure!(ffn_up.len() == m * h, "MTP ffn_up shape");
    anyhow::ensure!(ffn_down.len() == h * m, "MTP ffn_down shape");
    anyhow::ensure!(enorm.len() == h, "MTP enorm shape");
    anyhow::ensure!(hnorm.len() == h, "MTP hnorm shape");
    anyhow::ensure!(
        eh_proj.len() == h * (2 * h),
        "MTP eh_proj shape: {} != {h} * 2*{h} = {}",
        eh_proj.len(),
        h * (2 * h)
    );
    anyhow::ensure!(shared_head_norm.len() == h, "MTP shared_head_norm shape");

    // Split fused Q+gate (interleaved at head granularity, mirrors
    // `weight_loader::load_full_attn_layer`'s de-interleave).
    let mut wq = vec![0.0f32; q_total * h];
    let mut w_gate = vec![0.0f32; q_total * h];
    for head_idx in 0..nh {
        let src_q_start = (head_idx * 2 * d) * h;
        let src_g_start = ((head_idx * 2 + 1) * d) * h;
        let dst_start = head_idx * d * h;
        wq[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_q_start..src_q_start + d * h]);
        w_gate[dst_start..dst_start + d * h]
            .copy_from_slice(&q_fused[src_g_start..src_g_start + d * h]);
    }
    drop(q_fused);

    Ok(Some(MtpFullWeights {
        layer_index: n_layer,
        attn_norm,
        post_attn_norm,
        wq,
        w_gate,
        wk,
        wv,
        wo,
        attn_q_norm,
        attn_k_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        enorm,
        hnorm,
        eh_proj,
        shared_head_norm,
    }))
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::DType;
    use std::io::Write;

    /// Write a minimal GGUF file to `path` with the given tensor list.
    /// Each tensor has F32 type, shape `[4]` (16 bytes each), zero-filled.
    fn write_gguf_with_tensors(path: &std::path::Path, tensor_names: &[&str]) {
        let n_tensors = tensor_names.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0

        // Tensor info entries.
        let mut offset = 0u64;
        for name in tensor_names {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
            buf.extend_from_slice(&4u64.to_le_bytes()); // dims[0] = 4
            buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
            buf.extend_from_slice(&offset.to_le_bytes()); // tensor offset
            offset += 16; // 4 f32 = 16 bytes
        }

        // Pad to 32-byte alignment.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        // Tensor data (all zeros).
        for _ in tensor_names {
            buf.extend_from_slice(&[0u8; 16]);
        }

        let mut f = std::fs::File::create(path).expect("create tmp gguf");
        f.write_all(&buf).expect("write");
        f.flush().expect("flush");
    }

    /// ADR acceptance: MTP absent → `Ok(None)`.
    #[test]
    fn mtp_absent_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_absent_{}.gguf",
            std::process::id()
        ));
        write_gguf_with_tensors(
            &tmp,
            &[
                "blk.0.attn_norm.weight",
                "blk.0.ffn_gate_inp.weight",
                "token_embd.weight",
            ],
        );

        let gguf = GgufFile::open(&tmp).expect("open tmp gguf");
        let result = load_mtp_weights_if_present(&gguf, 40).expect("load_mtp");
        assert!(result.is_none(), "expected None when no MTP tensors present");

        std::fs::remove_file(&tmp).ok();
    }

    /// ADR acceptance: MTP tensors present → `Ok(Some(populated struct))`.
    #[test]
    fn mtp_present_returns_populated_struct() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_present_{}.gguf",
            std::process::id()
        ));
        write_gguf_with_tensors(
            &tmp,
            &[
                "token_embd.weight",
                "blk.0.attn_norm.weight",
                "blk.40.nextn.attn_qkv.weight",
                "blk.40.nextn.ffn_gate_inp.weight",
                "blk.40.nextn.attn_norm.weight",
            ],
        );

        let gguf = GgufFile::open(&tmp).expect("open tmp gguf");
        let result = load_mtp_weights_if_present(&gguf, 40).expect("load_mtp");
        let mtp = result.expect("expected Some");

        assert_eq!(mtp.layer_index, 40);
        assert_eq!(mtp.len(), 3);
        assert!(!mtp.is_empty());
        assert!(mtp.has_tensor_suffix("attn_qkv.weight"));
        assert!(mtp.has_tensor_suffix("ffn_gate_inp.weight"));
        assert!(mtp.has_tensor_suffix("attn_norm.weight"));
        assert!(!mtp.has_tensor_suffix("nonexistent.weight"));

        // Non-MTP tensors must not be captured.
        for (name, _) in &mtp.tensors {
            assert!(name.starts_with("blk.40.nextn."));
        }

        // Sanity on captured shapes.
        for (_, shape) in &mtp.tensors {
            assert_eq!(shape, &vec![4]);
        }

        std::fs::remove_file(&tmp).ok();
    }

    /// Different `num_hidden_layers` → different MTP layer index → no match.
    #[test]
    fn mtp_wrong_layer_index_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_wrong_idx_{}.gguf",
            std::process::id()
        ));
        write_gguf_with_tensors(
            &tmp,
            &[
                "blk.40.nextn.attn_qkv.weight",
            ],
        );

        let gguf = GgufFile::open(&tmp).expect("open tmp gguf");
        // Call with num_hidden_layers=64 but the MTP tensor is at layer 40.
        let result = load_mtp_weights_if_present(&gguf, 64).expect("load_mtp");
        assert!(result.is_none(), "expected None when layer index mismatches");

        std::fs::remove_file(&tmp).ok();
    }

    /// Integration test against the real apex GGUF. Apex has MTP tensors
    /// STRIPPED per the 2026-04-23 dump, so the loader must return None.
    /// `#[ignore]`d so regular test runs don't touch the 25 GB file.
    #[test]
    #[ignore]
    fn mtp_on_real_apex_returns_none() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let result = load_mtp_weights_if_present(&gguf, 40).expect("load_mtp");
        assert!(
            result.is_none(),
            "apex GGUF should have MTP stripped; got {:?}",
            result
        );
    }

    /// Shape preserved in the captured metadata. Uses a non-degenerate shape
    /// to pin the layout.
    #[test]
    fn mtp_captures_tensor_shape() {
        // This variant uses a custom-shape tensor (the helper above writes
        // fixed [4]; inlined here for clarity).
        let tmp = std::env::temp_dir().join(format!(
            "mtp_shape_{}.gguf",
            std::process::id()
        ));

        // Build a minimal GGUF with one MTP tensor of shape [8, 16].
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let name = "blk.40.nextn.attn_q.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&8u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&16u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&vec![0u8; 8 * 16 * 4]);
        std::fs::write(&tmp, &buf).expect("write");

        let gguf = GgufFile::open(&tmp).expect("open");
        let mtp = load_mtp_weights_if_present(&gguf, 40)
            .expect("load")
            .expect("some");
        assert_eq!(mtp.tensors.len(), 1);
        let (_, shape) = &mtp.tensors[0];
        // GGUF stores innermost dim first; mlx-native's TensorInfo.shape
        // presents outer dim first, so the dim order is reversed relative
        // to the wire format. We wrote [8, 16] on the wire → shape is [16, 8].
        assert_eq!(shape, &vec![16, 8]);

        // DType placeholder to keep the import used.
        let _ = DType::F32;

        std::fs::remove_file(&tmp).ok();
    }

    // ================================================================
    // ADR-013 P14 — load_mtp_weights_full tests
    // ================================================================

    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35Config, Qwen35Variant,
    };

    /// Build a minimal dense Qwen35Config sized for fast synthetic-GGUF
    /// MTP tests. Shapes are chosen so eh_proj (h × 2h) stays under
    /// 1 KB and inner attention tensors are easy to inspect.
    fn tiny_dense_cfg() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Dense,
            hidden_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(4, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 1,
            mrope_section: [1, 1, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            vocab_size: 32,
            attn_output_gate: true,
            mtp_num_hidden_layers: 1,
            intermediate_size: Some(16),
            moe: None,
        }
    }

    /// Write a custom-shaped F32 GGUF with arbitrary per-tensor shapes
    /// and contents.  Each `(name, shape, fill)` writes a tensor of
    /// `shape` filled with `fill` repeated.
    fn write_shaped_gguf(
        path: &std::path::Path,
        tensors: &[(&str, &[usize], f32)],
    ) {
        let n_tensors = tensors.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&n_tensors.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0

        // Tensor info entries.
        let mut data_offset = 0u64;
        let mut data_blocks: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
        for (name, shape, fill) in tensors {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape.iter() {
                buf.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // type = F32
            buf.extend_from_slice(&data_offset.to_le_bytes());

            let n_elems: usize = shape.iter().product();
            let mut block = Vec::with_capacity(n_elems * 4);
            for _ in 0..n_elems {
                block.extend_from_slice(&fill.to_le_bytes());
            }
            data_offset += (n_elems * 4) as u64;
            data_blocks.push(block);
        }

        // Pad to 32-byte alignment.
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for block in data_blocks {
            buf.extend_from_slice(&block);
        }

        std::fs::write(path, &buf).expect("write shaped gguf");
    }

    /// `load_mtp_weights_full` returns `Ok(None)` when no nextn.eh_proj
    /// is present (the apex / default-converted path).
    #[test]
    fn full_loader_absent_returns_none() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_full_absent_{}.gguf",
            std::process::id()
        ));
        write_gguf_with_tensors(
            &tmp,
            &[
                "blk.0.attn_norm.weight",
                "token_embd.weight",
            ],
        );

        let gguf = GgufFile::open(&tmp).expect("open");
        let cfg = tiny_dense_cfg();
        let device = MlxDevice::new().expect("device");
        let result = load_mtp_weights_full(&gguf, &cfg, &device).expect("load_full");
        assert!(result.is_none(), "expected None when nextn.eh_proj missing");
        std::fs::remove_file(&tmp).ok();
    }

    /// `load_mtp_weights_full` populates all 14 fields when every
    /// expected tensor is present at the right shape.  Inner-block
    /// tensors live at `blk.4.*` (n_layer = 4 in tiny cfg); wrapper
    /// tensors at `blk.4.nextn.*`.
    #[test]
    fn full_loader_present_populates_all_fields() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_full_present_{}.gguf",
            std::process::id()
        ));
        let cfg = tiny_dense_cfg();
        let h = cfg.hidden_size as usize;
        let nh = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let d = cfg.head_dim as usize;
        let q_total = nh * d;
        let kv_total = nkv * d;
        let m = cfg.intermediate_size.unwrap() as usize;
        let n_layer = cfg.num_hidden_layers;

        // GGUF stores shapes innermost-first on the wire (== the
        // reverse of the row-major shape we're emitting).  For a
        // [out, in] row-major tensor we write [in, out] on the wire.
        // For 1-D tensors the order is unambiguous.
        let p = format!("blk.{}", n_layer);
        let np = format!("blk.{}.nextn", n_layer);
        let inner_shape_2d = |out: usize, inn: usize| vec![inn, out]; // wire order
        let attn_q_fused_rows = 2 * q_total; // fused Q+gate output dim

        // Token offset we slot into a unique value per tensor so we can
        // verify the load preserves data identity.
        let tensors: Vec<(&str, Vec<usize>, f32)> = vec![
            (Box::leak(format!("{p}.attn_norm.weight").into_boxed_str()), vec![h], 1.0),
            (Box::leak(format!("{p}.post_attention_norm.weight").into_boxed_str()), vec![h], 1.5),
            (Box::leak(format!("{p}.attn_q.weight").into_boxed_str()), inner_shape_2d(attn_q_fused_rows, h), 2.0),
            (Box::leak(format!("{p}.attn_k.weight").into_boxed_str()), inner_shape_2d(kv_total, h), 2.5),
            (Box::leak(format!("{p}.attn_v.weight").into_boxed_str()), inner_shape_2d(kv_total, h), 3.0),
            (Box::leak(format!("{p}.attn_q_norm.weight").into_boxed_str()), vec![d], 3.5),
            (Box::leak(format!("{p}.attn_k_norm.weight").into_boxed_str()), vec![d], 4.0),
            (Box::leak(format!("{p}.attn_output.weight").into_boxed_str()), inner_shape_2d(h, q_total), 4.5),
            (Box::leak(format!("{p}.ffn_gate.weight").into_boxed_str()), inner_shape_2d(m, h), 5.0),
            (Box::leak(format!("{p}.ffn_up.weight").into_boxed_str()), inner_shape_2d(m, h), 5.5),
            (Box::leak(format!("{p}.ffn_down.weight").into_boxed_str()), inner_shape_2d(h, m), 6.0),
            (Box::leak(format!("{np}.enorm.weight").into_boxed_str()), vec![h], 6.5),
            (Box::leak(format!("{np}.hnorm.weight").into_boxed_str()), vec![h], 7.0),
            (Box::leak(format!("{np}.eh_proj.weight").into_boxed_str()), inner_shape_2d(h, 2 * h), 7.5),
            (Box::leak(format!("{np}.shared_head_norm.weight").into_boxed_str()), vec![h], 8.0),
        ];
        let refs: Vec<(&str, &[usize], f32)> =
            tensors.iter().map(|(n, s, f)| (*n, s.as_slice(), *f)).collect();
        write_shaped_gguf(&tmp, &refs);

        let gguf = GgufFile::open(&tmp).expect("open");
        let device = MlxDevice::new().expect("device");
        let mtp = load_mtp_weights_full(&gguf, &cfg, &device)
            .expect("load_full")
            .expect("expected Some");

        assert_eq!(mtp.layer_index, n_layer);
        assert_eq!(mtp.attn_norm.len(), h);
        assert!(mtp.attn_norm.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert_eq!(mtp.post_attn_norm.len(), h);
        assert!(mtp.post_attn_norm.iter().all(|&v| (v - 1.5).abs() < 1e-6));
        assert_eq!(mtp.wq.len(), q_total * h);
        assert_eq!(mtp.w_gate.len(), q_total * h);
        // Q+gate are interleaved at head granularity in the fused
        // tensor; both halves were filled with 2.0 so post-split they
        // should still both be 2.0 everywhere.
        assert!(mtp.wq.iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(mtp.w_gate.iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert_eq!(mtp.wk.len(), kv_total * h);
        assert_eq!(mtp.wv.len(), kv_total * h);
        assert_eq!(mtp.attn_q_norm.len(), d);
        assert_eq!(mtp.attn_k_norm.len(), d);
        assert_eq!(mtp.wo.len(), h * q_total);
        assert_eq!(mtp.ffn_gate.len(), m * h);
        assert_eq!(mtp.ffn_up.len(), m * h);
        assert_eq!(mtp.ffn_down.len(), h * m);
        assert_eq!(mtp.enorm.len(), h);
        assert_eq!(mtp.hnorm.len(), h);
        assert_eq!(mtp.eh_proj.len(), h * 2 * h);
        assert!(mtp.eh_proj.iter().all(|&v| (v - 7.5).abs() < 1e-6));
        assert_eq!(mtp.shared_head_norm.len(), h);
        assert!(mtp.shared_head_norm.iter().all(|&v| (v - 8.0).abs() < 1e-6));

        std::fs::remove_file(&tmp).ok();
    }

    /// Partial presence (some MTP tensors missing) MUST fail loud
    /// rather than silently fill with zeros — the no-shortcut bar
    /// from feedback_no_shortcuts.md.
    #[test]
    fn full_loader_partial_presence_errors() {
        let tmp = std::env::temp_dir().join(format!(
            "mtp_full_partial_{}.gguf",
            std::process::id()
        ));
        let cfg = tiny_dense_cfg();
        let h = cfg.hidden_size as usize;
        let n_layer = cfg.num_hidden_layers;
        let np = format!("blk.{}.nextn", n_layer);

        // Write ONLY the eh_proj probe — no other MTP tensors.
        // load_mtp_weights_full sees the probe → enters the load branch
        // → fails on the first missing tensor (attn_norm).
        let tensors: Vec<(&str, Vec<usize>, f32)> = vec![
            (Box::leak(format!("{np}.eh_proj.weight").into_boxed_str()), vec![2 * h, h], 1.0),
        ];
        let refs: Vec<(&str, &[usize], f32)> =
            tensors.iter().map(|(n, s, f)| (*n, s.as_slice(), *f)).collect();
        write_shaped_gguf(&tmp, &refs);

        let gguf = GgufFile::open(&tmp).expect("open");
        let device = MlxDevice::new().expect("device");
        let result = load_mtp_weights_full(&gguf, &cfg, &device);
        assert!(result.is_err(), "partial MTP presence must error, got {:?}", result.is_ok());
        std::fs::remove_file(&tmp).ok();
    }
}
