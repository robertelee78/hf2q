//! Optional process-once requested-bucket history cap. Direct allocations and
//! escaped handle/driver metadata are outside this domain; never admission.
use mlx_native::{
    DType, GrowthBucket, GrowthLimitedMlxBufferPool, GrowthPoolKey, GrowthQuota,
    LegacyGrowthLimits, LegacyPoolGrowthAuthority, MlxBuffer, MlxDevice, MlxError,
};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;

const FRESH: u8 = 0;
const LEGACY: u8 = 1;
const INSTALLING: u8 = 2;
const SELECTED: u8 = 3;
const UNHEALTHY: u8 = 4;
static MODE: AtomicU8 = AtomicU8::new(FRESH);
static WORKER: AtomicU8 = AtomicU8::new(0);
const KEY: GrowthPoolKey = GrowthPoolKey {
    kind: 0x5157454e,
    worker_generation: 1,
};
pub const BYTES: usize = 24 * 1024 * 1024 * 1024;
pub const ROOTS: usize = 16_384;
pub const ATTEMPTS: usize = 32_768;
type Result<T> = std::result::Result<T, MlxError>;
fn error(message: &str) -> MlxError {
    MlxError::InvalidArgument(format!("Qwen pool adr063-24g-v1: {message}"))
}
fn intent() -> Result<bool> {
    crate::debug::INVESTIGATION_ENV
        .qwen_pool_profile
        .validate()
        .map_err(error)
}
fn direct_spawn_precondition(requested: bool) {
    assert!(!requested,
        "Qwen pool profile requires fallible Engine::spawn_with_mode SlotAware; direct Engine::spawn is invalid");
}
/// The infallible serial API cannot represent a refused profile startup.
pub fn assert_direct_spawn_legacy() {
    direct_spawn_precondition(selected());
}
pub fn selected() -> bool {
    intent().unwrap_or(true)
}
pub fn unhealthy() -> bool {
    MODE.load(Ordering::Acquire) == UNHEALTHY
}
pub fn quarantine() {
    if selected() {
        MODE.store(UNHEALTHY, Ordering::Release);
        // Do not quarantine an unrelated authority if our installation was
        // refused because another caller already owned the native singleton.
        if POLICY.get().is_some() {
            if let Some(a) = LegacyPoolGrowthAuthority::get() {
                let _ = a.quarantine_generation(KEY.worker_generation);
            }
        }
    }
}
fn record_result<T>(mode: &AtomicU8, result: Result<T>) -> Result<T> {
    if result.is_err() {
        mode.store(UNHEALTHY, Ordering::Release);
    }
    result
}
fn checked<T>(result: Result<T>) -> Result<T> {
    let result = record_result(&MODE, result);
    if result.is_err() {
        quarantine();
    }
    if result.is_ok() && MODE.load(Ordering::Acquire) == UNHEALTHY {
        quarantine();
        return Err(error("operation raced sticky failure; refusing success"));
    }
    result
}
fn require_health(mode: &AtomicU8, registered: bool) -> Result<()> {
    if mode.load(Ordering::Acquire) != SELECTED || !registered {
        return Err(error(
            "sticky profile failure or unregistered thread; process recovery required",
        ));
    }
    Ok(())
}

/// Same atomic boundary used by install and the first legacy operation.
fn mark_legacy(mode: &AtomicU8) -> Result<()> {
    if mode.load(Ordering::Acquire) == LEGACY {
        return Ok(());
    }
    match mode.compare_exchange(FRESH, LEGACY, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) | Err(LEGACY) => Ok(()),
        _ => Err(error("legacy operation after profile installation")),
    }
}
fn start_install(mode: &AtomicU8) -> Result<()> {
    mode.compare_exchange(FRESH, INSTALLING, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| ())
        .map_err(|_| error("prior legacy use or another installation/generation"))
}
fn publish_install(mode: &AtomicU8) -> Result<()> {
    mode.compare_exchange(INSTALLING, SELECTED, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| ())
        .map_err(|_| error("installation raced sticky failure; refusing to heal"))
}
pub fn quota() -> GrowthQuota {
    GrowthQuota {
        bytes: BYTES,
        roots: ROOTS,
        attempts: ATTEMPTS,
    }
}
pub fn buckets() -> Result<Vec<GrowthBucket>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(35)
        .map_err(|_| error("bucket metadata reservation"))?;
    for shift in 0..=34 {
        let bytes = 1usize
            .checked_shl(shift)
            .ok_or_else(|| error("bucket overflow"))?;
        let roots = ROOTS.min(BYTES / bytes);
        result.push(GrowthBucket {
            bytes,
            quota: GrowthQuota {
                bytes: bytes
                    .checked_mul(roots)
                    .ok_or_else(|| error("bucket quota overflow"))?,
                roots,
                attempts: ATTEMPTS,
            },
        });
    }
    Ok(result)
}
struct Policy {
    quota: GrowthQuota,
    buckets: Vec<GrowthBucket>,
}
static POLICY: OnceLock<Policy> = OnceLock::new();
fn install_policy(policy: Policy) -> Result<()> {
    start_install(&MODE)?;
    LegacyPoolGrowthAuthority::install(LegacyGrowthLimits {
        quota: policy.quota,
        max_pools: 1,
        max_buckets_per_pool: policy.buckets.len(),
        max_rank: 8,
        max_batch: 1,
    })?;
    POLICY
        .set(policy)
        .map_err(|_| error("policy already installed"))?;
    publish_install(&MODE)
}
fn eligibility(qwen: bool, slots: Option<u32>, native: bool) -> Result<()> {
    if !qwen || !slots.is_some_and(|n| (1..=4).contains(&n)) || !native {
        return Err(error(
            "profile supports only native DenseQ/MoeQ Qwen SlotAware text/tools with 1..4 slots",
        ));
    }
    Ok(())
}
/// Called by actual spawn before provisioning/worker creation. No Metal here.
pub fn prepare(qwen: bool, slots: Option<u32>, native: bool) -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    checked((|| {
        eligibility(qwen, slots, native)?;
        if crate::debug::INVESTIGATION_ENV.qwen_pool_diagnostic_conflict {
            return Err(error(
                "diagnostic capture/MTP_PROFILE is outside the selected no-future-use contract",
            ));
        }
        install_policy(Policy {
            quota: quota(),
            buckets: buckets()?,
        })
    })())
}
struct Worker {
    model: usize,
    device: Option<MlxDevice>,
    pool: Option<GrowthLimitedMlxBufferPool>,
    issued: usize,
}
thread_local! { static LOCAL: RefCell<Option<Worker>> = const { RefCell::new(None) }; }
/// Register the actual inference worker before it accepts Warmup or generation.
pub fn register_worker(model: usize) -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    checked((|| {
        if MODE.load(Ordering::Acquire) != SELECTED
            || WORKER
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(error(
                "worker installation absent, unhealthy or already registered",
            ));
        }
        LOCAL.with(|cell| {
            *cell.borrow_mut() = Some(Worker {
                model,
                device: None,
                pool: None,
                issued: 0,
            });
        });
        Ok(())
    })())
}
pub fn entry(model: usize) -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    checked((|| {
        if MODE.load(Ordering::Acquire) != SELECTED {
            return Err(error("profile unhealthy or not installed"));
        }
        LOCAL.with(|cell| {
            if cell.borrow().as_ref().is_some_and(|w| w.model == model) {
                Ok(())
            } else {
                Err(error("unregistered thread or model generation changed"))
            }
        })
    })())
}
fn creation_permitted(has_pool: bool, has_device: bool) -> Result<()> {
    if has_pool || has_device {
        Err(error(
            "GPU cache generation replacement forbidden before device creation",
        ))
    } else {
        Ok(())
    }
}
/// Reject a cache replacement BEFORE creating another device/queue/registry.
pub fn before_device_creation(model: usize) -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    entry(model)?;
    checked(LOCAL.with(|cell| {
        let local = cell.borrow();
        let w = local.as_ref().ok_or_else(|| error("worker absent"))?;
        creation_permitted(w.pool.is_some(), w.device.is_some())
    }))
}
/// Bind ONLY the actual cached verifier device, never the separate KV device.
pub fn bind(model: usize, device: &MlxDevice) -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    entry(model)?;
    checked(LOCAL.with(|cell| {
        let mut local = cell.borrow_mut();
        let w = local.as_mut().ok_or_else(|| error("worker absent"))?;
        if w.pool.is_some() {
            return Err(error("cached device replacement/rebind forbidden"));
        }
        let authority =
            LegacyPoolGrowthAuthority::get().ok_or_else(|| error("authority absent"))?;
        let policy = POLICY.get().ok_or_else(|| error("policy absent"))?;
        w.pool = Some(authority.bind(device, KEY, policy.quota, &policy.buckets)?);
        w.device = Some(device.clone());
        Ok(())
    }))
}
pub fn healthy() -> Result<()> {
    if !intent()? {
        return Ok(());
    }
    LOCAL.with(|cell| require_health(&MODE, cell.borrow().is_some()))
}
/// None is the default legacy route; selected errors NEVER return None.
pub(super) fn alloc(
    device: &MlxDevice,
    bytes: usize,
    dtype: DType,
    shape: &[usize],
) -> Option<Result<MlxBuffer>> {
    match intent() {
        Ok(false) => {
            return match mark_legacy(&MODE) {
                Ok(()) => None,
                Err(e) => Some(Err(e)),
            }
        }
        Err(e) => return Some(Err(e)),
        Ok(true) => {}
    }
    Some(checked((|| {
        healthy()?;
        LOCAL.with(|cell| {
            let mut local = cell.borrow_mut();
            let w = local.as_mut().ok_or_else(|| error("worker absent"))?;
            let result = w
                .pool
                .as_mut()
                .ok_or_else(|| error("pool not bound"))?
                .alloc(device, bytes, dtype, shape.to_vec());
            if result.is_ok() {
                w.issued = w
                    .issued
                    .checked_add(1)
                    .ok_or_else(|| error("issued overflow"))?;
            }
            result
        })
    })()))
}
/// Result propagation is mandatory at production forward boundaries.
pub(super) fn reset() -> Option<Result<()>> {
    match intent() {
        Ok(false) => {
            return match mark_legacy(&MODE) {
                Ok(()) => None,
                Err(e) => Some(Err(e)),
            }
        }
        Err(e) => return Some(Err(e)),
        Ok(true) => {}
    }
    Some(checked((|| {
        healthy()?;
        LOCAL.with(|cell| {
            let mut local = cell.borrow_mut();
            let w = local.as_mut().ok_or_else(|| error("worker absent"))?;
            let device = w
                .device
                .as_ref()
                .ok_or_else(|| error("reset before actual device binding"))?;
            // SAFETY: production call sites separately establish BOTH last use
            // of issued scratch and successful drain. Direct outer arenas,
            // residual rings and durable returned parents are not issued here.
            unsafe {
                w.pool
                    .as_mut()
                    .ok_or_else(|| error("pool absent"))?
                    .reset(device)?;
            }
            w.issued = 0;
            Ok(())
        })
    })()))
}
pub(super) fn counts() -> Option<(usize, usize)> {
    if !selected() || unhealthy() {
        return None;
    }
    LOCAL.with(|cell| {
        let local = cell.borrow();
        let w = local.as_ref()?;
        let roots = w.pool.as_ref()?.snapshot().ok()?.pool.roots;
        Some((w.issued, roots.saturating_sub(w.issued)))
    })
}
/// Read-only telemetry: never pass census failures through checked()/quarantine.
pub(super) fn memory_snapshot_if_selected() -> Option<std::result::Result<mlx_native::PoolMemorySnapshot, String>> {
    if !selected() { return None; }
    Some(LOCAL.try_with(|cell| {
        let local = cell.try_borrow().map_err(|_| "selected pool borrow busy".to_string())?;
        let worker = local.as_ref().ok_or_else(|| "selected worker absent".to_string())?;
        let pool = worker.pool.as_ref().ok_or_else(|| "selected pool unbound".to_string())?;
        pool.memory_snapshot().map_err(|error| error.to_string())
    }).map_err(|_| "selected worker TLS unavailable".to_string()).and_then(|r| r))
}

/// Unknown/early failure poisons selected operation, even after stringification.
pub struct Operation {
    active: bool,
    complete: bool,
}
impl Operation {
    pub fn enter(model: usize) -> Result<Self> {
        entry(model)?;
        Ok(Self {
            active: selected(),
            complete: false,
        })
    }
    pub fn complete(mut self) -> Result<()> {
        healthy()?;
        self.complete = true;
        Ok(())
    }
}
impl Drop for Operation {
    fn drop(&mut self) {
        if self.active && !self.complete {
            quarantine();
        }
    }
}
/// Shared by the actual AUTO catch/replay boundary and pure failure controls.
pub fn permit_replay() -> Result<()> {
    healthy()
}
pub fn reject_unsupported(requested: bool) -> Result<()> {
    if selected() && requested {
        return Err(error("selected pool profile supports text/tool requests only; unsupported diagnostic/vision/embedding route"));
    }
    healthy()
}
