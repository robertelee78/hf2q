//! Multi-Token Prediction (MTP) load-only scaffold (ADR-013 Decision 15).
//!
//! MTP is Qwen3.5's extra transformer block after the main stack, used for
//! speculative decoding. The Hugging Face config sets
//! `mtp_num_hidden_layers = 1`. GGUF-emitted tensor names follow the
//! convention `blk.{N}.nextn.*` where `N = num_hidden_layers` (ADR-012
//! Decision 11).
//!
//! **This module is LOAD-ONLY.** The production forward ignores MTP weights.
//! Executing MTP (the speculative-decoding verification loop) is a future
//! ADR.
//!
//! # Acceptance (ADR-013)
//!
//! * Load on apex.gguf (MTP stripped) → `Ok(None)`, no error.
//! * Load on a hypothetical MTP-bearing GGUF → `Ok(Some(MtpWeights))`
//!   with populated fields, no crash.
//!
//! # Current state
//!
//! The apex GGUF available at test time has MTP tensors stripped. Our
//! loader is therefore validated against the "none present" path via
//! direct GGUF inspection (no MTP tensor → empty list → `None`); the
//! "present" path is exercised via a synthetic in-memory GGUF.

use anyhow::Result;

use mlx_native::gguf::GgufFile;

/// Weights for a single MTP block.
///
/// Intentionally structured as a flexible bag-of-tensors keyed by suffix:
/// the exact set of tensors an MTP block carries depends on whether it's
/// a full-attention or linear-attention block, which is not pinned down
/// in the apex GGUF (the reference ships with MTP stripped). The loader
/// captures whatever `blk.{N}.nextn.*` tensors are present so a future
/// speculative-decoding implementation can ship without a loader change.
///
/// Keys are the suffix after `blk.{N}.nextn.`, e.g. `"attn_qkv.weight"`,
/// `"ssm_conv1d.weight"`, `"ffn_gate_inp.weight"`. Tensor data is NOT
/// loaded yet — only names + shapes + types are captured. The full
/// weight load happens when MTP execution lands in a follow-up ADR.
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
}
