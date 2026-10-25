//! Checked metadata planning and the single production codec writer.
use super::{
    row_bytes, CacheError, CacheKind, Deepseek4Cache, Deepseek4CachePlan,
    Deepseek4CacheSnapshot, ANCHOR_IMAGE_CHECKSUM_LEN, ANCHOR_IMAGE_CODEC_VERSION,
    ANCHOR_IMAGE_HEADER_LEN, ANCHOR_IMAGE_MAGIC,
};
use mlx_native::{DType, MlxBuffer};
use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "serialize_tests.rs"]
mod tests;

const KINDS: [CacheKind; 7] = [
    CacheKind::WindowKv, CacheKind::CompressedKv, CacheKind::IndexerKv,
    CacheKind::MainKvState, CacheKind::MainScoreState,
    CacheKind::IndexerKvState, CacheKind::IndexerScoreState,
];
#[derive(Clone, Copy)]
struct BufferMetadata<'a> {
    dtype: DType,
    shape: &'a [usize],
    bytes: usize,
}
impl<'a> From<&'a MlxBuffer> for BufferMetadata<'a> {
    fn from(buffer: &'a MlxBuffer) -> Self {
        Self { dtype: buffer.dtype(), shape: buffer.shape(), bytes: buffer.data_byte_len() }
    }
}
struct LayerSections {
    index: u32,
    ratio: u32,
    lengths: [usize; 7],
}
struct ImagePlan {
    position: u32,
    token_count: u32,
    context: u32,
    layers: Vec<LayerSections>,
    total: usize,
}
fn capacity(reason: &'static str, requested: usize) -> CacheError {
    CacheError::AnchorImageCapacity { reason, requested }
}
fn checked_add(a: usize, b: usize) -> Result<usize, CacheError> {
    a.checked_add(b).ok_or_else(|| capacity("image length overflow", a))
}
fn field(value: usize) -> Result<u32, CacheError> {
    u32::try_from(value).map_err(|_| capacity("header field exceeds u32", value))
}
impl ImagePlan {
    /// Metadata only: no payload access or image-sized allocation.
    fn new<'a>(
        plan: &Deepseek4CachePlan,
        position: usize,
        token_count: usize,
        graft_slots: usize,
        live_layers: usize,
        anchor_layers: usize,
        mut metadata: impl FnMut(usize, usize) -> Option<BufferMetadata<'a>>,
    ) -> Result<Self, CacheError> {
        if position == 0 || position != token_count || position > plan.context_length || graft_slots != 0 || plan.layers.is_empty() {
            return Err(CacheError::AnchorImageIncompatible { reason: "invalid anchor position, tokens or graft" });
        }
        if live_layers != plan.layers.len() {
            return Err(CacheError::AnchorImageIncompatible { reason: "live layer count differs from plan" });
        }
        if anchor_layers != plan.layers.len() {
            return Err(CacheError::SnapshotLayerCount { expected: plan.layers.len(), actual: anchor_layers });
        }
        let (position, token_count, context) = (field(position)?, field(token_count)?, field(plan.context_length)?);
        field(plan.layers.len())?;
        let ledger = (token_count as usize).checked_mul(4)
            .ok_or_else(|| capacity("token ledger overflow", token_count as usize))?;
        let mut total = checked_add(ANCHOR_IMAGE_HEADER_LEN, ledger)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(plan.layers.len())
            .map_err(|_| capacity("layer metadata reservation failed", plan.layers.len()))?;
        for (layer, p) in plan.layers.iter().enumerate() {
            let index = field(p.layer_index)?;
            let plans = [Some(&p.window_kv), p.compressed_kv.as_ref(), p.indexer_kv.as_ref(),
                p.main_kv_state.as_ref(), p.main_score_state.as_ref(),
                p.indexer_kv_state.as_ref(), p.indexer_score_state.as_ref()];
            let mut lengths = [0; 7];
            for (role, expected) in plans.into_iter().enumerate() {
                let actual = metadata(layer, role);
                match (expected, actual) {
                    (None, None) => continue,
                    (Some(expected), Some(actual)) => {
                        let bytes = expected.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
                            .and_then(|n| n.checked_mul(expected.dtype.size_of()))
                            .ok_or_else(|| capacity("buffer metadata overflow", layer))?;
                        if actual.dtype != expected.dtype || actual.shape != expected.shape.as_slice()
                            || actual.bytes != bytes || u64::try_from(bytes).ok() != Some(expected.bytes)
                        {
                            return Err(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] });
                        }
                        lengths[role] = if role < 3 {
                            if expected.dtype != DType::BF16 || expected.shape.len() != 2 {
                                return Err(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] });
                            }
                            let rows = match role {
                                0 => (position as usize).min(expected.shape[0]),
                                1 if p.compress_ratio != 0 => position as usize / p.compress_ratio as usize,
                                2 if p.compress_ratio == 4 => position as usize / 4,
                                _ => return Err(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] }),
                            };
                            let len = row_bytes(expected.shape[1], rows)?;
                            if rows > expected.shape[0] || len > bytes {
                                return Err(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] });
                            }
                            len
                        } else { bytes };
                    }
                    _ => return Err(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] }),
                }
            }
            total = checked_add(total, 64)?;
            for len in lengths { total = checked_add(total, len)?; }
            layers.push(LayerSections { index, ratio: p.compress_ratio, lengths });
        }
        total = checked_add(total, ANCHOR_IMAGE_CHECKSUM_LEN)?;
        if total > isize::MAX as usize || u64::try_from(total).is_err() {
            return Err(capacity("image exceeds Vec/u64 layout", total));
        }
        Ok(Self { position, token_count, context, layers, total })
    }
}

struct ExactWriter { bytes: Vec<u8>, limit: usize }
impl ExactWriter {
    fn new(bytes: Vec<u8>, limit: usize) -> Result<Self, CacheError> {
        if !bytes.is_empty() || bytes.capacity() != limit {
            return Err(capacity("reservation capacity differs from image plan", limit));
        }
        Ok(Self { bytes, limit })
    }
    fn append(&mut self, value: &[u8]) -> Result<(), CacheError> {
        let end = checked_add(self.bytes.len(), value.len())?;
        if end > self.limit || self.bytes.capacity() != self.limit || end > self.bytes.capacity() {
            return Err(capacity("append exceeds reserved image", end));
        }
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    fn patch(&mut self, start: usize, value: &[u8]) -> Result<(), CacheError> {
        let end = checked_add(start, value.len())?;
        let destination = self.bytes.get_mut(start..end)
            .ok_or_else(|| capacity("patch exceeds initialized image", end))?;
        destination.copy_from_slice(value);
        Ok(())
    }
    fn finish(self) -> Result<Vec<u8>, CacheError> {
        if self.bytes.len() != self.limit || self.bytes.capacity() != self.limit {
            return Err(capacity("incomplete image", self.bytes.len()));
        }
        Ok(self.bytes)
    }
}
fn reserve(total: usize) -> Result<Vec<u8>, CacheError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(total).map_err(|_| capacity("image reservation failed", total))?;
    Ok(bytes)
}
fn emit<'a>(
    plan: &ImagePlan,
    tokens: &[u32],
    allocate: impl FnOnce(usize) -> Result<Vec<u8>, CacheError>,
    mut read: impl FnMut(usize, usize, usize) -> Result<&'a [u8], CacheError>,
) -> Result<Vec<u8>, CacheError> {
    if tokens.len() != plan.token_count as usize {
        return Err(capacity("token ledger changed after planning", tokens.len()));
    }
    let mut out = ExactWriter::new(allocate(plan.total)?, plan.total)?;
    out.append(&ANCHOR_IMAGE_MAGIC)?;
    for value in [ANCHOR_IMAGE_CODEC_VERSION, plan.position, plan.token_count,
        field(plan.layers.len())?, plan.context, 0] {
        out.append(&value.to_le_bytes())?;
    }
    out.append(&0u64.to_le_bytes())?;
    for token in tokens { out.append(&token.to_le_bytes())?; }
    for (layer, section) in plan.layers.iter().enumerate() {
        out.append(&section.index.to_le_bytes())?;
        out.append(&section.ratio.to_le_bytes())?;
        for len in section.lengths { out.append(&(len as u64).to_le_bytes())?; }
        for (role, len) in section.lengths.into_iter().enumerate() {
            if len == 0 { continue; }
            let source = read(layer, role, len)?;
            if source.len() != len { return Err(capacity("payload length differs from image plan", source.len())); }
            out.append(source)?;
        }
    }
    if out.bytes.len() != plan.total - ANCHOR_IMAGE_CHECKSUM_LEN {
        return Err(capacity("image body differs from plan", out.bytes.len()));
    }
    out.patch(32, &(plan.total as u64).to_le_bytes())?;
    let checksum: [u8; 32] = Sha256::digest(&out.bytes).into();
    out.append(&checksum)?;
    out.finish()
}
fn buffer<'a>(cache: &'a Deepseek4Cache, anchor: &'a Deepseek4CacheSnapshot, layer: usize, role: usize) -> Option<&'a MlxBuffer> {
    let live = &cache.layers[layer];
    let saved = &anchor.layers[layer];
    match role {
        0 => Some(&saved.window_kv), 1 => live.compressed_kv.as_ref(), 2 => live.indexer_kv.as_ref(),
        3 => saved.main_kv_state.as_ref(), 4 => saved.main_score_state.as_ref(),
        5 => saved.indexer_kv_state.as_ref(), 6 => saved.indexer_score_state.as_ref(),
        _ => None,
    }
}
#[allow(clippy::too_many_arguments)]
fn encode_from_metadata<'a>(
    cache_plan: &Deepseek4CachePlan,
    position: usize,
    tokens: &[u32],
    graft_slots: usize,
    counts: (usize, usize),
    metadata: impl FnMut(usize, usize) -> Option<BufferMetadata<'a>>,
    allocate: impl FnOnce(usize) -> Result<Vec<u8>, CacheError>,
    read: impl FnMut(usize, usize, usize) -> Result<&'a [u8], CacheError>,
) -> Result<Vec<u8>, CacheError> {
    let plan = ImagePlan::new(cache_plan, position, tokens.len(), graft_slots, counts.0, counts.1, metadata)?;
    emit(&plan, tokens, allocate, read)
}
pub(super) fn serialize(cache: &Deepseek4Cache, anchor: &Deepseek4CacheSnapshot, tokens: &[u32]) -> Result<Vec<u8>, CacheError> {
    encode_from_metadata(&cache.plan, anchor.position(), tokens, cache.graft_slots,
        (cache.layers.len(), anchor.layers.len()),
        |layer, role| buffer(cache, anchor, layer, role).map(Into::into), reserve,
        |layer, role, len| {
            let source = buffer(cache, anchor, layer, role)
                .ok_or(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] })?
                .as_slice::<u8>()
                .map_err(|source| CacheError::SnapshotCopy { layer, kind: KINDS[role], source })?;
            source.get(..len).ok_or(CacheError::SnapshotBufferMismatch { layer, kind: KINDS[role] })
        })
}
