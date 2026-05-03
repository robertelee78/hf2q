//! Qwen3-VL **text-side** LM module (ADR-005 Wedge-4 / iter-228a).
//!
//! # Scope
//!
//! Qwen3-VL is a vision-language model: a ViT + DeepStack image encoder
//! produces image-token embeddings (Wedge-4c LANDED in `vit_gpu_qwen3vl.rs`)
//! that are spliced into the LM's input embedding stream. **This module
//! owns only the dense LM transformer** that consumes the spliced token
//! stream and emits next-token logits. The ViT side is intentionally
//! orthogonal (and already shipped).
//!
//! # Module layout
//!
//! - `mod.rs` (this file): [`Qwen3VlTextConfig`] parser, arch detection
//!   helpers, [`Qwen3VlTextLayerKind`] enum, public re-exports.
//! - `weights.rs`: [`Qwen3VlTextWeights`] — per-layer + global tensor
//!   buffers loaded from the GGUF (q4_0 dense FFN, q4_0 attention,
//!   per-head Q/K RMSNorm, layer norms, embed, final norm).
//! - `model.rs`: [`Qwen3VlTextModel`] — the loaded-model bundle (config +
//!   weights + GPU context).
//! - `forward.rs`: forward-path scaffolding + the iter-228b sentinel
//!   surface. The actual transformer-forward implementation is
//!   iter-228b scope (see [`Qwen3VlTextModel::forward_not_yet_wired`]).
//!
//! # Why a new module instead of reusing `qwen35`?
//!
//! Qwen3-VL text is a **plain dense transformer with biased GQA
//! attention + per-head Q/K RMSNorm + 3D-mRoPE**. It has:
//!
//! - **Zero** SSM / DeltaNet structure (Qwen3.5 hybrid mandates it).
//! - **No** `ssm.{state_size,group_count,inner_size,conv_kernel}`
//!   metadata keys, **no** `full_attention_interval` key.
//! - Optional attn-q/k/v **biases** (peer `qwen3vl.cpp` references
//!   `wo_b`; in practice the q4_0 GGUFs hf2q's converter emits do NOT
//!   include biases).
//! - Per-head Q-norm + K-norm (one-per-head, shape [head_dim]).
//! - DeepStack residual injection on the first `n_deepstack_layers`
//!   layers (ground-truth at `qwen3vl.cpp` lines 96-100; hf2q's
//!   `image_token_residual_add_gpu` already implements the per-position
//!   add).
//!
//! Routing it through `Qwen35Model::load_from_gguf` is structurally
//! impossible — that loader requires SSM keys absent from Qwen3-VL
//! GGUFs. iter-227 LANDED an actionable-error dispatch shim; iter-228a
//! (this module) lands the **load** path so the GGUF actually opens
//! through hf2q's engine without falling back to the iter-227 bail.
//!
//! # iter-228a vs iter-228b split
//!
//! - **iter-228a (this iter)**: module skeleton, [`Qwen3VlTextConfig`]
//!   parser, [`weights::Qwen3VlTextWeights`] full loader, engine seam
//!   ([`crate::serve::api::engine::LoadedModel::Qwen3VlText`]),
//!   operator-gated load harness against the real
//!   `Qwen/Qwen3-VL-2B-Instruct` GGUF.
//! - **iter-228b**: dense transformer forward (per-layer attn + GQA
//!   flash-attn + 3D-mRoPE + SiLU FFN + DeepStack residual) wired through
//!   [`crate::serve::api::engine::worker_run`]'s Generate / GenerateStream /
//!   GenerateWithSoftTokens arms, replacing the [`forward::QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL`]
//!   bail with the live forward chain. This mirrors the iter-215 →
//!   Wedge-3 split that landed Qwen3.5/3.6 (see
//!   `engine_qwen35.rs::QWEN35_NOT_IMPLEMENTED_SENTINEL`).
//!
//! Splitting the work this way means iter-228a unblocks operator-script
//! Step 5 (model load + readyz) without faking forward, and lets
//! iter-228b focus on the LM-forward surgery against a known-good load
//! pipeline (no compounding load-bug-vs-forward-bug ambiguity).

use anyhow::{anyhow, Context, Result};
use mlx_native::gguf::GgufFile;

pub mod forward;
pub mod model;
pub mod weights;

pub use model::Qwen3VlTextModel;
pub use weights::Qwen3VlTextWeights;

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------
//
// The arch-detection predicates [`is_qwen3_vl_arch`] /
// [`is_qwen3_vl_moe_arch`] live in the sibling `qwen35` module
// (`crate::inference::models::qwen35::is_qwen3_vl_arch`) so iter-227's
// dispatch shim can call them without first deciding which family
// owns the dispatch. They're re-exported here for callers that already
// know they're on a Qwen3-VL load path.

pub use crate::inference::models::qwen35::{
    is_qwen3_vl_arch, is_qwen3_vl_moe_arch, ARCH_QWEN3VLMOE_UPSTREAM,
    ARCH_QWEN3VL_UPSTREAM, ARCH_QWEN3_VL,
};

// ---------------------------------------------------------------------------
// Layer-kind enum
// ---------------------------------------------------------------------------

/// Per-layer kind for the Qwen3-VL text transformer.
///
/// **Intentionally trivial** — every Qwen3-VL text-LM layer is a
/// [`Self::Dense`] block with the same shape (biased GQA + per-head
/// Q/K RMSNorm + SiLU-gated FFN). Unlike Qwen3.5's hybrid stack, there
/// are no DeltaNet / sliding-window / per-layer variants. The enum
/// exists for forward-compatibility (a future Qwen3-VL-MoE convert
/// pipeline could add a `Moe` variant) and to keep the shape parallel
/// to [`crate::inference::models::qwen35::Qwen35LayerKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3VlTextLayerKind {
    /// Dense GQA + SiLU-gated FFN (every Qwen3-VL-2B/4B text layer).
    Dense,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Default RoPE base frequency for Qwen3-VL text (matches HF
/// `text_config.rope_theta` for `Qwen/Qwen3-VL-2B-Instruct` /
/// `Qwen/Qwen3-VL-4B-Instruct`).
///
/// The real-model GGUF at
/// `/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf` does NOT
/// emit `qwen3_vl.rope.freq_base`, so this default applies. iter-228b's
/// 3D-mRoPE wiring consumes this value.
pub const DEFAULT_ROPE_THETA: f32 = 5_000_000.0;

/// Default RMSNorm epsilon for Qwen3-VL text (matches HF
/// `text_config.rms_norm_eps`).
///
/// The real-model GGUF does NOT emit
/// `qwen3_vl.attention.layer_norm_rms_epsilon`, so this default applies.
/// iter-228b's per-layer norm dispatch consumes this value.
pub const DEFAULT_RMS_NORM_EPS: f32 = 1.0e-6;

/// Default 3D-mRoPE section sizes `[t, h, w]` for Qwen3-VL text.
///
/// Matches HF `text_config.rope_scaling.mrope_section = [3, 3, 2]` for
/// the 2B / 4B Instruct variants. Sum = head_dim/2 = 64/2 = 8 for
/// head_dim=128 (the rotary slot count is half of head_dim under the
/// IMROPE convention `mrope_interleaved=true`).
///
/// iter-228b's 3D-mRoPE wiring consumes this value to thread the
/// (t, h, w) axes built by `build_qwen3vl_positions`
/// (`src/serve/forward_prefill.rs:220`) into mlx-native's
/// `apply_imrope` kernel.
pub const DEFAULT_MROPE_SECTION: [u32; 3] = [3, 3, 2];

/// Default number of DeepStack residual injection layers for
/// Qwen3-VL-2B/4B text.
///
/// Matches HF `text_config.deepstack_visual_indexes` length = 3. The
/// peer source `qwen3vl.cpp:96-100` injects DeepStack residuals into
/// the first `n_deepstack_layers` LM layers; hf2q's
/// `image_token_residual_add_gpu` (Wedge-4c.5) implements the
/// per-position add, and the per-LM-layer dispatch is iter-228b
/// scope.
pub const DEFAULT_N_DEEPSTACK_LAYERS: usize = 3;

/// Vision/IMROPE rotary mode index for Qwen3-VL.
///
/// Matches mlx-native's `RopeMultiMode::Vision = 24` (NOT IMROPE = 40,
/// which is the Qwen3.5 mode). Per the canonical peer
/// `qwen3vl.cpp:45-58` the kernel is `ggml_rope_multi(...)` with
/// `rope_type = MROPE` (24). Iter-228b's RoPE wiring routes through
/// this mode.
pub const QWEN3VL_ROPE_MODE: u32 = 24;

/// Architecture config for a Qwen3-VL **text** LM, parsed from GGUF.
///
/// Source of truth is the GGUF metadata at the `qwen3_vl.*` prefix,
/// validated against the real `Qwen/Qwen3-VL-2B-Instruct` q4_0 dump:
///
/// ```text
///   qwen3_vl.block_count           = 28
///   qwen3_vl.embedding_length      = 2048
///   qwen3_vl.attention.head_count  = 16
///   qwen3_vl.attention.head_count_kv = 8
///   qwen3_vl.feed_forward_length   = 6144
///   qwen3_vl.context_length        = 262144
/// ```
///
/// Defaults filled by [`Qwen3VlTextConfig::from_gguf`] when keys are
/// absent: `rope_theta = 5e6`, `rms_norm_eps = 1e-6`, `head_dim`
/// derived from `hidden_size / num_attention_heads`,
/// `mrope_section = [3, 3, 2]`, `n_deepstack_layers = 3`.
///
/// `vocab_size` and `tied_word_embeddings` are derived from the GGUF's
/// tensor table (the converter emits 310 tensors for tied; 311 for
/// untied — the presence/absence of `output.weight` is authoritative).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlTextConfig {
    /// Number of transformer layers (28 for 2B, 36 for 4B).
    pub num_hidden_layers: u32,
    /// Hidden / model dimension (2048 for 2B, 2560 for 4B).
    pub hidden_size: u32,
    /// Multi-head attention head count (16 for 2B).
    pub num_attention_heads: u32,
    /// KV-head count (GQA: 8 for 2B → 2:1 GQA ratio).
    pub num_key_value_heads: u32,
    /// Per-head dimension (128 for 2B; derived as
    /// `hidden_size / num_attention_heads` when GGUF doesn't emit it).
    pub head_dim: u32,
    /// FFN intermediate (gate/up/down hidden) dim (6144 for 2B).
    pub intermediate_size: u32,
    /// Vocabulary size (151936 for Qwen3-VL family).
    pub vocab_size: u32,
    /// Maximum context length declared by the GGUF (262144 for 2B).
    pub max_position_embeddings: u32,
    /// RoPE base frequency. Defaults to [`DEFAULT_ROPE_THETA`] when
    /// `qwen3_vl.rope.freq_base` is absent.
    pub rope_theta: f32,
    /// RMSNorm epsilon. Defaults to [`DEFAULT_RMS_NORM_EPS`] when
    /// `qwen3_vl.attention.layer_norm_rms_epsilon` is absent.
    pub rms_norm_eps: f32,
    /// 3D-mRoPE section sizes `[t, h, w]`. Defaults to
    /// [`DEFAULT_MROPE_SECTION`].
    pub mrope_section: [u32; 3],
    /// Number of LM layers that consume DeepStack residual injections
    /// (the first `n_deepstack_layers` layers; peer `qwen3vl.cpp:96`).
    pub n_deepstack_layers: usize,
    /// `true` iff `output.weight` is absent from the GGUF tensor table
    /// → the LM head reuses `token_embd.weight`. Derived at parse
    /// time from the tensor table; cannot be overridden.
    pub tied_word_embeddings: bool,
    /// Per-layer kind table. Authoritative — every entry is
    /// [`Qwen3VlTextLayerKind::Dense`] for the 2B/4B variants, but
    /// kept as a `Vec` so a future hybrid Qwen3-VL variant could
    /// override on a per-layer basis without breaking callers.
    pub layer_types: Vec<Qwen3VlTextLayerKind>,
}

impl Qwen3VlTextConfig {
    /// Parse the config from an open Qwen3-VL GGUF.
    ///
    /// Reads all `qwen3_vl.*` metadata keys; falls back to the defaults
    /// declared in this module's constants for keys the converter omits.
    /// Detects tied embeddings by checking for `output.weight` in the
    /// tensor table. `vocab_size` is read from the embedding tensor's
    /// shape (more reliable than the metadata key, which the converter
    /// may omit).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // The arch string is recognized as Qwen3-VL, so the metadata
        // prefix is one of the three known spellings — pick the one
        // that has tensor-side keys actually present. iter-228a's real
        // GGUF uses the underscored `qwen3_vl` form; the upstream
        // `qwen3vl` form is forward-compat for a future llama.cpp-aligned
        // converter.
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow!("Qwen3VlTextConfig: missing general.architecture"))?;
        if !is_qwen3_vl_arch(arch) {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: arch {arch:?} is not a Qwen3-VL family arch \
                 (expected one of {:?}, {:?}, {:?})",
                ARCH_QWEN3_VL,
                ARCH_QWEN3VL_UPSTREAM,
                ARCH_QWEN3VLMOE_UPSTREAM,
            ));
        }
        if is_qwen3_vl_moe_arch(arch) {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: MoE Qwen3-VL ({arch:?}) is not yet supported \
                 by hf2q (no convert pipeline emits it; iter-228 closes only the dense path)"
            ));
        }
        // Try the underscored prefix first (what hf2q's converter
        // actually emits today; matches the real-model GGUF). Fall
        // back to the upstream prefix for forward-compat with a future
        // llama.cpp-aligned converter.
        let prefix = if gguf
            .metadata_u32(&format!("{}.block_count", ARCH_QWEN3_VL))
            .is_some()
        {
            ARCH_QWEN3_VL
        } else if gguf
            .metadata_u32(&format!("{}.block_count", ARCH_QWEN3VL_UPSTREAM))
            .is_some()
        {
            ARCH_QWEN3VL_UPSTREAM
        } else {
            return Err(anyhow!(
                "Qwen3VlTextConfig::from_gguf: neither {:?}.block_count nor {:?}.block_count \
                 is present in metadata — GGUF appears Qwen3-VL by arch but is missing core \
                 architecture facts.",
                ARCH_QWEN3_VL,
                ARCH_QWEN3VL_UPSTREAM,
            ));
        };
        let req_u32 = |key: &str| -> Result<u32> {
            gguf.metadata_u32(&format!("{prefix}.{key}"))
                .ok_or_else(|| anyhow!("Qwen3VlTextConfig: missing {prefix}.{key}"))
        };

        let num_hidden_layers = req_u32("block_count")
            .context("Qwen3VlTextConfig: block_count")?;
        let hidden_size = req_u32("embedding_length")
            .context("Qwen3VlTextConfig: embedding_length")?;
        let num_attention_heads = req_u32("attention.head_count")
            .context("Qwen3VlTextConfig: attention.head_count")?;
        let num_key_value_heads = req_u32("attention.head_count_kv")
            .context("Qwen3VlTextConfig: attention.head_count_kv")?;
        let intermediate_size = req_u32("feed_forward_length")
            .context("Qwen3VlTextConfig: feed_forward_length")?;
        let max_position_embeddings = gguf
            .metadata_u32(&format!("{prefix}.context_length"))
            .unwrap_or(0);

        // head_dim: prefer explicit `attention.key_length` (some
        // converters emit it); else derive from hidden / heads. For the
        // real-model GGUF the explicit key is absent → derived value
        // is 2048 / 16 = 128 ✓.
        let head_dim = gguf
            .metadata_u32(&format!("{prefix}.attention.key_length"))
            .unwrap_or_else(|| {
                if num_attention_heads == 0 {
                    0
                } else {
                    hidden_size / num_attention_heads
                }
            });
        if head_dim == 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: derived head_dim is zero (hidden={hidden_size}, \
                 heads={num_attention_heads}); refusing to load"
            ));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(anyhow!(
                "Qwen3VlTextConfig: invalid GQA shape — num_attention_heads={num_attention_heads} \
                 must be divisible by num_key_value_heads={num_key_value_heads}"
            ));
        }

        let rope_theta = gguf
            .metadata_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(DEFAULT_ROPE_THETA);
        let rms_norm_eps = gguf
            .metadata_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(DEFAULT_RMS_NORM_EPS);

        // vocab_size: derive from the embedding tensor's shape.
        //
        // GGUF stores dimensions innermost-first; the mlx-native loader
        // REVERSES them at parse time
        // (`mlx_native::gguf::mod::rs:858-861`) to match the
        // `[rows, cols]` convention. So `token_embd.weight` is shape
        // `[vocab_size, hidden_size]` after the reverse: gguf-dump
        // shows `2048, 151936` on disk → `[151936, 2048]` in memory →
        // `shape[0] = vocab_size`.
        let vocab_size = match gguf.tensor_info("token_embd.weight") {
            Some(info) if info.shape.len() == 2 => info.shape[0] as u32,
            Some(info) => {
                return Err(anyhow!(
                    "Qwen3VlTextConfig: token_embd.weight has unexpected rank {} (shape={:?})",
                    info.shape.len(),
                    info.shape
                ));
            }
            None => {
                return Err(anyhow!(
                    "Qwen3VlTextConfig: missing token_embd.weight tensor (cannot derive vocab_size)"
                ));
            }
        };

        // Tied embeddings: presence of `output.weight` in the tensor
        // table is the authoritative signal. The real-model GGUF emits
        // 310 tensors total → tied; an untied variant would emit 311
        // (one extra `output.weight`). HF text_config.tie_word_embeddings
        // is already baked into this signal at convert time.
        let tied_word_embeddings = gguf.tensor_info("output.weight").is_none();

        let layer_types =
            vec![Qwen3VlTextLayerKind::Dense; num_hidden_layers as usize];

        Ok(Self {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            mrope_section: DEFAULT_MROPE_SECTION,
            n_deepstack_layers: DEFAULT_N_DEEPSTACK_LAYERS,
            tied_word_embeddings,
            layer_types,
        })
    }

    /// GQA group ratio (`num_attention_heads / num_key_value_heads`).
    /// Always divisible — invariant checked at parse time.
    pub fn gqa_group_ratio(&self) -> u32 {
        self.num_attention_heads / self.num_key_value_heads
    }
}

// ---------------------------------------------------------------------------
// Tests — config parser
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Synthetic-GGUF helpers used by [`tests`] below + by integration tests
/// in `tests/qwen3vl_text_lm_forward.rs`. These produce a header-only +
/// metadata-only + tensor-info-only GGUF (no tensor data payload), which
/// is sufficient for callers that consume metadata + tensor-shape
/// information but never invoke `load_tensor` / `load_tensor_f32`.
///
/// Public-but-cfg(test)-or-feature-gated would be cleaner; this is
/// `pub(crate)` so the integration test crate (which is a separate
/// compilation unit) can't reach it. The integration test re-implements
/// the helper inline against the GGUF binary format — that's intentional:
/// the integration test against the real 1.32 GB GGUF needs no synthetic
/// fixture, and the synthetic-fixture surface is unit-test-only.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use std::io::Write;
    use std::path::PathBuf;

    /// Minimal GGUF metadata key descriptor for [`write_minimal_qwen3vl_gguf`].
    /// `Str` is currently unused by in-tree tests but kept for forward-
    /// compat with future tests that need string-typed metadata keys.
    pub(crate) enum Kv {
        #[allow(dead_code)]
        Str(String, String),
        U32(String, u32),
    }

    /// Tensor descriptor: name + shape (innermost-first per GGUF
    /// convention). All synthetic tensors are emitted as F32 with byte_len
    /// = product(shape) * 4, but the tensor data section is left
    /// uninitialized — callers must NOT call `load_tensor*` on these
    /// tensors. [`Qwen3VlTextConfig::from_gguf`] only consumes
    /// [`mlx_native::gguf::GgufFile::tensor_info`] which is satisfied by
    /// the on-disk tensor-info section.
    pub(crate) struct TensorDesc<'a> {
        pub name: &'a str,
        pub shape: &'a [usize],
    }

    /// Write a minimal valid GGUF (header + metadata + N tensor infos +
    /// zero-padded tensor-data section) and return its on-disk path.
    /// Caller is responsible for `std::fs::remove_file` (the file lives
    /// in the OS tmp dir; cleanup is best-effort).
    pub(crate) fn write_minimal_qwen3vl_gguf(
        arch_str: &str,
        extra_kvs: &[Kv],
        tensors: &[TensorDesc<'_>],
    ) -> PathBuf {
        const GGUF_TYPE_UINT32: u32 = 4;
        const GGUF_TYPE_STRING: u32 = 8;
        const GGML_TYPE_F32: u32 = 0;
        const ALIGNMENT: u64 = 32;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
        // metadata_kv_count = 1 (general.architecture) + extras
        let kv_count = 1 + extra_kvs.len() as u64;
        buf.extend_from_slice(&kv_count.to_le_bytes());
        // KV: general.architecture
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        let val = arch_str.as_bytes();
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        // Extra KVs.
        for kv in extra_kvs {
            match kv {
                Kv::Str(k, v) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
                    buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    buf.extend_from_slice(v.as_bytes());
                }
                Kv::U32(k, v) => {
                    buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
                    buf.extend_from_slice(k.as_bytes());
                    buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        // Tensor info section.
        let mut data_offset: u64 = 0;
        for t in tensors {
            buf.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
            buf.extend_from_slice(t.name.as_bytes());
            buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in t.shape {
                buf.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            buf.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            buf.extend_from_slice(&data_offset.to_le_bytes());
            let elem_count: usize = t.shape.iter().product();
            data_offset += (elem_count * 4) as u64;
        }
        // Pad to ALIGNMENT and add zero-tensor-data section.
        let header_len = buf.len() as u64;
        let pad = ((header_len + ALIGNMENT - 1) & !(ALIGNMENT - 1)) - header_len;
        buf.extend(std::iter::repeat(0u8).take(pad as usize));
        // Tensor data: zero-padded.
        buf.extend(std::iter::repeat(0u8).take(data_offset as usize));

        let path = std::env::temp_dir().join(format!(
            "hf2q-qwen3vl-test-{}.gguf",
            std::process::id() as u64 ^ rand_u64()
        ));
        let mut f = std::fs::File::create(&path).expect("create temp gguf");
        f.write_all(&buf).expect("write gguf");
        f.flush().expect("flush gguf");
        path
    }

    /// Cheap process-local random u64 to avoid temp-file name collisions
    /// when multiple test threads write fixtures concurrently. Uses
    /// nanos-of-day XOR'd with a thread-local counter; collision-free
    /// enough for a unit-test temp-path namespace.
    fn rand_u64() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        nanos ^ (COUNTER.fetch_add(1, Ordering::Relaxed)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

// ---------------------------------------------------------------------------
// Tests — config parser
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::test_fixtures::{write_minimal_qwen3vl_gguf, Kv, TensorDesc};
    use super::*;

    fn standard_2b_kvs(prefix: &str) -> Vec<Kv> {
        vec![
            Kv::U32(format!("{prefix}.block_count"), 28),
            Kv::U32(format!("{prefix}.embedding_length"), 2048),
            Kv::U32(format!("{prefix}.attention.head_count"), 16),
            Kv::U32(format!("{prefix}.attention.head_count_kv"), 8),
            Kv::U32(format!("{prefix}.feed_forward_length"), 6144),
            Kv::U32(format!("{prefix}.context_length"), 262144),
        ]
    }

    #[test]
    fn from_gguf_parses_underscored_arch_with_real_2b_shape() {
        // The real-model GGUF arch.
        let kvs = standard_2b_kvs("qwen3_vl");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from synthetic 2B GGUF");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.head_dim, 128, "derived head_dim 2048/16");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert!(
            (cfg.rope_theta - DEFAULT_ROPE_THETA).abs() < 1.0,
            "rope_theta default {}",
            cfg.rope_theta
        );
        assert!(
            (cfg.rms_norm_eps - DEFAULT_RMS_NORM_EPS).abs() < 1e-12,
            "rms_norm_eps default {}",
            cfg.rms_norm_eps
        );
        assert_eq!(cfg.mrope_section, DEFAULT_MROPE_SECTION);
        assert_eq!(cfg.n_deepstack_layers, DEFAULT_N_DEEPSTACK_LAYERS);
        assert!(
            cfg.tied_word_embeddings,
            "tied detection — output.weight is NOT emitted in this fixture"
        );
        assert_eq!(cfg.layer_types.len(), 28);
        assert!(
            cfg.layer_types.iter().all(|k| *k == Qwen3VlTextLayerKind::Dense),
            "every layer is Dense for Qwen3-VL-2B/4B"
        );
        assert_eq!(cfg.gqa_group_ratio(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_parses_upstream_no_underscore_arch() {
        // Forward-compat: a future converter that aligns with llama.cpp
        // upstream emits `qwen3vl` (no underscore). Both prefix + arch
        // string forms must work.
        let kvs = standard_2b_kvs("qwen3vl");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from upstream-arch GGUF");
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_non_qwen3vl_arch() {
        let kvs = standard_2b_kvs("gemma4");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("gemma4", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("non-Qwen3-VL arch must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not a Qwen3-VL family arch"),
            "expected 'not a Qwen3-VL family arch' in error message; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_moe_variant_with_specific_message() {
        // MoE Qwen3-VL is recognized at the dispatch level (so we route
        // to a clear error rather than a Gemma-path crash), but the
        // dense config parser must refuse it explicitly.
        let kvs = standard_2b_kvs("qwen3vlmoe");
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3vlmoe", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("MoE Qwen3-VL must be rejected by the dense parser");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("MoE Qwen3-VL"),
            "expected MoE-specific message; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_detects_untied_embeddings_when_output_present() {
        // The real Qwen3-VL-2B converter emits 310 tensors (tied). A
        // future converter (or a hand-untied fork) would emit
        // `output.weight` → tied_word_embeddings = false.
        let kvs = standard_2b_kvs("qwen3_vl");
        let tensors = [
            TensorDesc {
                name: "token_embd.weight",
                shape: &[2048, 151936],
            },
            TensorDesc {
                name: "output.weight",
                shape: &[2048, 151936],
            },
        ];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig with output.weight present");
        assert!(
            !cfg.tied_word_embeddings,
            "output.weight present → must NOT report tied"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_gguf_rejects_invalid_gqa_shape() {
        // num_attention_heads not divisible by num_key_value_heads —
        // structurally impossible for valid GQA.
        let kvs = vec![
            Kv::U32("qwen3_vl.block_count".to_string(), 28),
            Kv::U32("qwen3_vl.embedding_length".to_string(), 2048),
            Kv::U32("qwen3_vl.attention.head_count".to_string(), 16),
            Kv::U32("qwen3_vl.attention.head_count_kv".to_string(), 5),
            Kv::U32("qwen3_vl.feed_forward_length".to_string(), 6144),
            Kv::U32("qwen3_vl.context_length".to_string(), 262144),
        ];
        let tensors = [TensorDesc {
            name: "token_embd.weight",
            shape: &[2048, 151936],
        }];
        let path = write_minimal_qwen3vl_gguf("qwen3_vl", &kvs, &tensors);
        let gguf = GgufFile::open(&path).expect("open synthetic GGUF");
        let err = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect_err("indivisible GQA shape must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("invalid GQA shape"),
            "expected 'invalid GQA shape' in error; got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn arch_constants_match_qwen35_module_re_export() {
        // Sanity: the re-exported constants here are byte-equal to
        // the Qwen35 module's authoritative copies.
        assert_eq!(ARCH_QWEN3_VL, "qwen3_vl");
        assert_eq!(ARCH_QWEN3VL_UPSTREAM, "qwen3vl");
        assert_eq!(ARCH_QWEN3VLMOE_UPSTREAM, "qwen3vlmoe");
    }

    #[test]
    fn defaults_match_hf_text_config_for_qwen3vl_2b() {
        // Pin the defaults to the HF text_config values — if these
        // ever drift, this test fails loudly and forces a docstring
        // update.
        assert!((DEFAULT_ROPE_THETA - 5_000_000.0).abs() < 1.0);
        assert!((DEFAULT_RMS_NORM_EPS - 1.0e-6).abs() < 1e-12);
        assert_eq!(DEFAULT_MROPE_SECTION, [3, 3, 2]);
        assert_eq!(DEFAULT_N_DEEPSTACK_LAYERS, 3);
        assert_eq!(QWEN3VL_ROPE_MODE, 24);
    }

    /// Real-model load gate: when
    /// `HF2Q_QWEN3VL_LM_LOAD=1` and the canonical fixture
    /// `/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf` is
    /// present, parse the config and pin against text_config.
    #[test]
    fn from_gguf_parses_real_qwen3vl_2b_when_operator_gated() {
        if std::env::var("HF2Q_QWEN3VL_LM_LOAD").ok().as_deref() != Some("1") {
            eprintln!("skip: HF2Q_QWEN3VL_LM_LOAD!=1");
            return;
        }
        let p = std::path::PathBuf::from(
            "/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf",
        );
        if !p.exists() {
            eprintln!("skip: real GGUF fixture not present at {}", p.display());
            return;
        }
        let gguf = GgufFile::open(&p).expect("open real Qwen3-VL-2B GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf)
            .expect("parse Qwen3VlTextConfig from real Qwen3-VL-2B GGUF");
        // Pin against HF text_config.
        assert_eq!(cfg.num_hidden_layers, 28, "Qwen3-VL-2B layers");
        assert_eq!(cfg.hidden_size, 2048, "Qwen3-VL-2B hidden");
        assert_eq!(cfg.num_attention_heads, 16, "Qwen3-VL-2B heads");
        assert_eq!(cfg.num_key_value_heads, 8, "Qwen3-VL-2B kv heads");
        assert_eq!(cfg.intermediate_size, 6144, "Qwen3-VL-2B ffn");
        assert_eq!(cfg.head_dim, 128, "Qwen3-VL-2B head_dim");
        assert_eq!(cfg.vocab_size, 151936, "Qwen3-VL family vocab");
        assert_eq!(cfg.max_position_embeddings, 262144, "Qwen3-VL-2B ctx");
        assert!(cfg.tied_word_embeddings, "Qwen3-VL-2B tied embeddings");
    }
}
