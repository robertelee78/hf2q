//! One immutable-capacity execution root per model GPU cache. Weight planes
//! never own partials. The active scope transports only a cache-owned Rc;
//! leaving it (including error/nesting) restores the previous context.

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

use anyhow::{ensure, Context, Result};
use mlx_native::ops::quantized_matmul_ggml::GgmlQuantizedMatmulParams;
use mlx_native::ops::repacked_gguf_mm::{
    dispatch_repacked_gguf_mm_with_scratch, PreparedGgufWeightPlanes, RepackedGgufMmExecution,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

pub(crate) struct ExecutionScratch {
    device: MlxDevice,
    capacity: usize,
    buffer: OnceCell<MlxBuffer>,
}

impl ExecutionScratch {
    pub(crate) fn new(device: &MlxDevice, capacity: usize) -> Rc<Self> {
        Rc::new(Self { device: device.clone(), capacity, buffer: OnceCell::new() })
    }

    fn check_device(&self, device: &MlxDevice) -> Result<()> {
        ensure!(
            std::ptr::eq(self.device.metal_device(), device.metal_device())
                && std::ptr::eq(&**self.device.metal_queue(), &**device.metal_queue()),
            "prepared MM scratch belongs to a different model device/queue"
        );
        Ok(())
    }

    pub(crate) fn enter(self: &Rc<Self>, device: &MlxDevice) -> Result<Scope> {
        self.check_device(device)?;
        Ok(Scope { previous: ACTIVE.with(|active| active.replace(Some(self.clone()))) })
    }

    fn buffer(&self, device: &MlxDevice, required: usize) -> Result<&MlxBuffer> {
        self.check_device(device)?;
        ensure!(required <= self.capacity,
            "prepared MM requires {required} scratch bytes, model cache reserved {}", self.capacity);
        if self.buffer.get().is_none() {
            let buffer = device.alloc_buffer(self.capacity, DType::F32, vec![self.capacity / 4])
                .context("allocate model prepared MM execution scratch")?;
            // Shared roots are hazard tracked: Metal orders conflicting accesses
            // between separate compute encoders/command buffers of this queue.
            ensure!(buffer.metal_buffer().hazard_tracking_mode() == mlx_native::metal::MTLHazardTrackingMode::Tracked,
                "prepared MM execution scratch requires tracked storage");
            let _ = self.buffer.set(buffer);
        }
        self.buffer.get().context("model prepared MM scratch was not retained")
    }

    /// Standalone consumer fixtures have no ForwardGpuCache. This fixed bound
    /// covers their admitted shapes; production never installs a test owner.
    #[cfg(test)]
    pub(crate) fn test_scope(device: &MlxDevice) -> Scope {
        Self::new(device, 32 * 1024 * 1024).enter(device).expect("test scratch scope")
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<ExecutionScratch>>> = const { RefCell::new(None) };
}

pub(crate) struct Scope {
    previous: Option<Rc<ExecutionScratch>>,
}

impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|active| { active.replace(self.previous.take()); });
    }
}

/// The encoder must originate from `device` and be committed in queue order,
/// as all cache consumers already require. No root is resized, retired, or
/// CPU-written here; checked submission drains remain with the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    planes: &PreparedGgufWeightPlanes,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    execution: RepackedGgufMmExecution,
) -> Result<()> {
    let required = planes.execution_scratch_bytes(params.m, execution)?;
    ACTIVE.with(|active| {
        let active = active.borrow();
        if let Some(owner) = active.as_ref() { owner.check_device(device)?; }
        let scratch = if required == 0 { None } else {
            Some(active.as_ref().context("prepared MM needs the model GPU-cache scratch scope")?
                .buffer(device, required)?)
        };
        dispatch_repacked_gguf_mm_with_scratch(
            encoder, registry, device, input, planes, output, scratch, params, execution,
        )?;
        if required != 0 {
            // The reducer is the last reader. Order its read before the next
            // split product overwrites this root, even for an unequal weight.
            // Serial/OnePass products retain their independent dispatch wave.
            encoder.memory_barrier();
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "prepared_scratch_tests.rs"]
mod tests;
