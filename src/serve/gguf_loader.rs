//! GGUF file loader — lazy tensor dequantization for inference.
//!
//! Loads tensors from GGUF on demand, keeping the file open and dequantizing
//! each weight only when requested.  This keeps startup fast and memory bounded.

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A loaded GGUF model — holds the file open for lazy tensor access.
pub struct GgufModel {
    content: gguf_file::Content,
    reader: Mutex<BufReader<std::fs::File>>,
    device: Device,
    /// Cache of already-loaded QTensors (quantized form, on device).
    #[allow(dead_code)]
    qtensor_cache: Mutex<HashMap<String, Arc<QTensor>>>,
}

impl GgufModel {
    /// Open a GGUF file and parse its header.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Cannot open GGUF: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        let content = gguf_file::Content::read(&mut reader)
            .with_context(|| format!("Failed to parse GGUF header: {}", path.display()))?;

        tracing::info!(
            "GGUF loaded: {} tensors, {} metadata keys",
            content.tensor_infos.len(),
            content.metadata.len(),
        );

        Ok(Self {
            content,
            reader: Mutex::new(reader),
            device: device.clone(),
            qtensor_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Load a tensor by name, dequantizing to the target dtype.
    pub fn get_tensor(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        let qt = self.content.tensor(&mut *reader, name, &self.device)
            .map_err(|e| anyhow::anyhow!("GGUF tensor '{}': {}", name, e))?;
        qt.dequantize(&self.device)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }

    /// Load a tensor, returning None if not found.
    pub fn try_get_tensor(&self, name: &str, dtype: DType) -> Result<Option<Tensor>> {
        if !self.content.tensor_infos.contains_key(name) {
            return Ok(None);
        }
        self.get_tensor(name, dtype).map(Some)
    }

    /// Load a QTensor (quantized form, stays compressed on device).
    pub fn get_qtensor(&self, name: &str) -> Result<Arc<QTensor>> {
        // Check cache first
        {
            let cache = self.qtensor_cache.lock().expect("cache lock poisoned");
            if let Some(qt) = cache.get(name) {
                return Ok(qt.clone());
            }
        }

        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        let qt = self.content.tensor(&mut *reader, name, &self.device)
            .map_err(|e| anyhow::anyhow!("GGUF qtensor '{}': {}", name, e))?;
        let qt = Arc::new(qt);

        let mut cache = self.qtensor_cache.lock().expect("cache lock poisoned");
        cache.insert(name.to_string(), qt.clone());
        Ok(qt)
    }

    /// Check if a tensor exists.
    #[allow(dead_code)]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// Get the device.
    #[allow(dead_code)]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Read a string value from GGUF metadata, returning `None` if the key is
    /// missing or not a string.
    ///
    /// Mirrors the reader side of the writer at `backends/gguf.rs:1846-1858`,
    /// which stores `tokenizer.chat_template` from `chat_template.jinja` or
    /// `tokenizer_config.json`. Per ADR-005 Phase 1 (lines 44, 691), the
    /// serve/load path must read this key so inference matches llama.cpp.
    pub fn get_metadata_string(&self, key: &str) -> Option<String> {
        self.content
            .metadata
            .get(key)
            .and_then(|v| v.to_string().ok())
            .cloned()
    }
}
