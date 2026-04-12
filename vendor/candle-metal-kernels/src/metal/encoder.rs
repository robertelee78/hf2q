use crate::metal::{instrument, Buffer, CommandSemaphore, CommandStatus, ComputePipeline, MetalResource};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage, MTLSize,
};
use std::{ffi::c_void, ptr, sync::Arc, sync::Mutex};

/// hf2q ADR-006 Phase 0 vendor patch (2026-04-11): per-encoder Phase 0
/// instrumentation state. `slot_pair` is the (start, end) slot indices
/// allocated from the shared counter sample buffer when the encoder was
/// created with a sample-buffer-bearing `MTLComputePassDescriptor`.
/// `current_kernel` is updated on every `set_compute_pipeline_state`
/// and flushed into the global pending list on `end_encoding`.
/// `ended` guards against double-record on drop + explicit end.
///
/// If `instrument::is_enabled()` is false at encoder-creation time,
/// this entire field is `None` and all downstream code paths in this
/// module short-circuit on the `Option::is_none()` check. That is what
/// preserves byte-identical behavior under the observer-effect gate.
struct InstrumentSlot {
    start_slot: usize,
    end_slot: usize,
    current_kernel: Mutex<Option<String>>,
    ended: std::sync::atomic::AtomicBool,
}

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
    instrument: Option<InstrumentSlot>,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}
impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            instrument: None,
        }
    }

    /// hf2q ADR-006 Phase 0 vendor patch (2026-04-11): construct an
    /// encoder whose enclosing compute pass descriptor already has the
    /// shared counter sample buffer attached at attachment index 0 with
    /// `startOfEncoderSampleIndex=start_slot` and
    /// `endOfEncoderSampleIndex=end_slot`. The kernel name is captured
    /// later via `set_compute_pipeline_state`, and the (name, start,
    /// end) triple is pushed to the pending list on `end_encoding`.
    pub(crate) fn new_instrumented(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
        start_slot: usize,
        end_slot: usize,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            instrument: Some(InstrumentSlot {
                start_slot,
                end_slot,
                current_kernel: Mutex::new(None),
                ended: std::sync::atomic::AtomicBool::new(false),
            }),
        }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup)
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        )
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
        // Phase 0 instrumentation: remember the Metal function name so
        // `end_encoding` can attribute this encoder's pass-level
        // start/end GPU timestamps to a kernel identity. Cold path is
        // a single Option::is_none() check; only instrumented encoders
        // pay the objc msg-send to `computeFunction().name()`.
        if let Some(slot) = self.instrument.as_ref() {
            if let Some(name) = instrument::current_kernel_name(pipeline.as_ref()) {
                if let Ok(mut g) = slot.current_kernel.lock() {
                    *g = Some(name);
                }
            }
        }
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        // Phase 0 instrumentation: push the kernel-identity + slot-pair
        // onto the global pending list so the next `flush_and_wait`
        // call can resolve the GPU timestamps. Guarded by the per-encoder
        // `ended` flag so the Drop impl doesn't double-record an encoder
        // that was already ended explicitly.
        if let Some(slot) = self.instrument.as_ref() {
            use std::sync::atomic::Ordering;
            if !slot.ended.swap(true, Ordering::AcqRel) {
                let kernel = slot
                    .current_kernel
                    .lock()
                    .ok()
                    .and_then(|g| g.clone())
                    .unwrap_or_else(|| "<no_pipeline_set>".to_string());
                instrument::record_pending(kernel, slot.start_slot, slot.end_slot);
            }
        }
        self.signal_encoding_ended();
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        use MTLComputeCommandEncoder as _;
        self.raw.setComputePipelineState(pipeline.as_ref());
        // Phase 0 instrumentation parity with `set_compute_pipeline_state`.
        if let Some(slot) = self.instrument.as_ref() {
            if let Some(name) = instrument::current_kernel_name(pipeline.as_ref()) {
                if let Ok(mut g) = slot.current_kernel.lock() {
                    *g = Some(name);
                }
            }
        }
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder { raw, semaphore }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn end_encoding(&self) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn set_label(&self, label: &str) {
        use objc2_metal::MTLCommandEncoder as _;
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}
