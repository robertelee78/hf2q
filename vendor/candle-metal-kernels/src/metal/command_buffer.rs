use crate::metal::instrument;
use crate::{BlitCommandEncoder, ComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::borrow::Cow;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone, Debug, PartialEq)]
pub enum CommandStatus {
    Available,
    Encoding,
    Done,
}

#[derive(Debug)]
pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> CommandSemaphore {
        CommandSemaphore {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    pub fn wait_until<F: FnMut(&mut CommandStatus) -> bool>(
        &self,
        mut f: F,
    ) -> MutexGuard<'_, CommandStatus> {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        // We notify the condvar that the value has changed.
        self.cond.notify_one();
    }

    pub fn when<T, B: FnMut(&mut CommandStatus) -> bool, F: FnMut() -> T>(
        &self,
        b: B,
        mut f: F,
        next: Option<CommandStatus>,
    ) -> T {
        let mut guard = self.wait_until(b);
        let v = f();
        if let Some(status) = next {
            *guard = status;
            self.cond.notify_one();
        }
        v
    }
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    semaphore: Arc<CommandSemaphore>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        // hf2q ADR-006 Phase 0 vendor patch (2026-04-11): when
        // instrumentation is enabled, try to create the encoder via a
        // `MTLComputePassDescriptor` carrying the shared counter sample
        // buffer so the pass emits start/end GPU timestamps at stage
        // boundary (the only sampling point M5 Max supports). If the
        // instrumented path fails for any reason — slot pool exhausted,
        // descriptor alloc failure, newCommand encoder returns nil —
        // fall back to the byte-identical pre-patch path.
        //
        // The env probe in `instrument::is_enabled()` short-circuits to
        // an atomic bool load on the hot path, so the uninstrumented
        // build pays for one Acquire load per encoder creation and
        // nothing else. This keeps the observer-effect gate achievable:
        // the phase0 bench script runs 5 runs with the env var UNSET
        // and asserts median tok/s is within ±2% of the 84.9 baseline.
        if instrument::is_enabled() {
            if let Some(inst) = instrument::make_instrumented_encoder(self) {
                return ComputeCommandEncoder::new_instrumented(
                    inst.raw,
                    Arc::clone(&self.semaphore),
                    inst.start_slot,
                    inst.end_slot,
                );
            }
        }
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
