//! ADR-065 E1 diagnostic stage attribution.
//!
//! Exact-value opt-in (`HF2Q_ADR065_STAGE_PROFILE=1`). The normal path reads
//! one cached boolean and otherwise does not allocate, timestamp, reset native
//! profile state, or change command-buffer topology. Profiled runs must also
//! set `MLX_PROFILE_CB=1`; category boundaries deliberately commit and wait,
//! so paired unprofiled runs quantify their distortion.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use serde_json::json;

pub const ENV: &str = "HF2Q_ADR065_STAGE_PROFILE";
pub const SCHEMA: &str = "hf2q.adr065.stage-profile-round.v1";

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Off,
    Wall,
    Profile,
}

impl Mode {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("1") => Self::Profile,
            Some("wall") => Self::Wall,
            _ => Self::Off,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Wall => "wall",
            Self::Profile => "profile",
        }
    }
}

static MODE: LazyLock<Mode> = LazyLock::new(|| Mode::parse(std::env::var(ENV).ok().as_deref()));
static NATIVE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("MLX_PROFILE_CB").as_deref() == Ok("1"));
static SCRATCH_RELEASES: AtomicU64 = AtomicU64::new(0);
static SCRATCH_REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static PENDING_SCRATCH_RELEASES: AtomicU64 = AtomicU64::new(0);
static PENDING_SCRATCH_REALLOCATIONS: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct CountTime {
    count: u64,
    ns: u64,
}

struct State {
    family: &'static str,
    route: &'static str,
    width: usize,
    mode: Mode,
    started: Instant,
    cpu: BTreeMap<&'static str, CountTime>,
}

struct WallRecord {
    family: &'static str,
    route: &'static str,
    width: usize,
    wall_ns: u64,
    scratch_releases: u64,
    scratch_reallocations: u64,
    scratch_releases_total: u64,
    scratch_reallocations_total: u64,
}

static WALL_RECORDS: Mutex<Vec<WallRecord>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: RefCell<Option<State>> = const { RefCell::new(None) };
}

#[inline]
pub fn enabled() -> bool {
    *MODE == Mode::Profile && *NATIVE_ENABLED
}

#[inline]
fn observes_rounds() -> bool {
    *MODE != Mode::Off
}

/// One complete decode round. Only one round may be active on the owning model
/// worker; the server already serializes that worker's Metal transactions.
pub struct Round {
    active: bool,
}

impl Round {
    pub fn begin(family: &'static str, route: &'static str, width: usize) -> Option<Self> {
        if !observes_rounds() {
            return None;
        }
        if *MODE == Mode::Profile && !*NATIVE_ENABLED {
            eprintln!("[ADR065_STAGE_PROFILE_ERROR] {ENV}=1 requires MLX_PROFILE_CB=1");
            return None;
        }
        let installed = CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            if current.is_some() {
                return false;
            }
            if enabled() {
                mlx_native::kernel_profile::reset();
            }
            *current = Some(State {
                family,
                route,
                width,
                mode: *MODE,
                started: Instant::now(),
                cpu: BTreeMap::new(),
            });
            true
        });
        installed.then_some(Self { active: true })
    }

    fn emit(&self) {
        let Some(state) = CURRENT.with(|current| current.borrow_mut().take()) else {
            return;
        };
        let wall_ns = u64::try_from(state.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if state.mode == Mode::Wall {
            let scratch_releases_total = SCRATCH_RELEASES.load(Ordering::Relaxed);
            let scratch_reallocations_total = SCRATCH_REALLOCATIONS.load(Ordering::Relaxed);
            WALL_RECORDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(WallRecord {
                    family: state.family,
                    route: state.route,
                    width: state.width,
                    wall_ns,
                    scratch_releases: PENDING_SCRATCH_RELEASES.swap(0, Ordering::Relaxed),
                    scratch_reallocations: PENDING_SCRATCH_REALLOCATIONS.swap(0, Ordering::Relaxed),
                    scratch_releases_total,
                    scratch_reallocations_total,
                });
            return;
        }
        let mut gpu: BTreeMap<&'static str, CountTime> = BTreeMap::new();
        let mut raw_unknown = BTreeMap::new();
        if state.mode == Mode::Profile {
            for (label, entry) in mlx_native::kernel_profile::dump() {
                let class = classify_label(&label);
                let bucket = gpu.entry(class).or_default();
                bucket.count = bucket.count.saturating_add(entry.count);
                bucket.ns = bucket.ns.saturating_add(entry.total_ns);
                if class == "unknown" {
                    raw_unknown.insert(
                        label,
                        json!({"count": entry.count, "gpu_ns": entry.total_ns}),
                    );
                }
            }
        }
        let gpu_ns = gpu.values().map(|entry| entry.ns).sum::<u64>();
        let cpu_ns = state.cpu.values().map(|entry| entry.ns).sum::<u64>();
        let host_gap_ns = wall_ns.saturating_sub(gpu_ns.saturating_add(cpu_ns));
        let gpu = gpu
            .into_iter()
            .map(|(name, entry)| {
                (
                    name.to_string(),
                    json!({"count": entry.count, "gpu_ns": entry.ns}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let mut cpu = state
            .cpu
            .into_iter()
            .map(|(name, entry)| {
                (
                    name.to_string(),
                    json!({"count": entry.count, "cpu_ns": entry.ns}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        if state.mode == Mode::Profile {
            cpu.insert(
                "host_gap".into(),
                json!({"count": 1, "cpu_ns": host_gap_ns}),
            );
        }
        let record = json!({
            "schema": SCHEMA,
            "mode": state.mode.label(),
            "family": state.family,
            "route": state.route,
            "width": state.width,
            "wall_ns": wall_ns,
            "gpu": gpu,
            "cpu": cpu,
            "unknown_labels": raw_unknown,
            "deepseek_decode_scratch": {
                "release_events": PENDING_SCRATCH_RELEASES.swap(0, Ordering::Relaxed),
                "reallocation_events": PENDING_SCRATCH_REALLOCATIONS
                    .swap(0, Ordering::Relaxed),
                "release_events_total": SCRATCH_RELEASES.load(Ordering::Relaxed),
                "reallocation_events_total": SCRATCH_REALLOCATIONS.load(Ordering::Relaxed),
            },
        });
        eprintln!("[ADR065_STAGE_PROFILE_JSON] {record}");
        if state.mode == Mode::Profile {
            mlx_native::kernel_profile::reset();
        }
    }
}

impl Drop for Round {
    fn drop(&mut self) {
        if self.active {
            self.emit();
        }
    }
}

/// Emit wall-only records after HTTP work has drained, so receipt collection
/// cannot perturb cohort formation or request decisions.
pub fn flush_wall() {
    let records = WALL_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .drain(..)
        .collect::<Vec<_>>();
    for record in records {
        let value = json!({
            "schema": SCHEMA,
            "mode": "wall",
            "family": record.family,
            "route": record.route,
            "width": record.width,
            "wall_ns": record.wall_ns,
            "gpu": {},
            "cpu": {},
            "unknown_labels": {},
            "deepseek_decode_scratch": {
                "release_events": record.scratch_releases,
                "reallocation_events": record.scratch_reallocations,
                "release_events_total": record.scratch_releases_total,
                "reallocation_events_total": record.scratch_reallocations_total,
            },
        });
        eprintln!("[ADR065_STAGE_PROFILE_JSON] {value}");
    }
}
/// True while E1 owns the native profile table for a complete round.
pub fn owns_native_profile_tables() -> bool {
    enabled()
}

/// CPU stage timer. It is inert and timestamp-free when E1 is disabled or no
/// complete-round guard owns the current worker thread.
pub struct CpuSpan {
    class: Option<&'static str>,
    started: Option<Instant>,
}

impl CpuSpan {
    pub fn begin(class: &'static str) -> Self {
        if !enabled() || !CURRENT.with(|current| current.borrow().is_some()) {
            return Self {
                class: None,
                started: None,
            };
        }
        Self {
            class: Some(class),
            started: Some(Instant::now()),
        }
    }
}

impl Drop for CpuSpan {
    fn drop(&mut self) {
        let (Some(class), Some(started)) = (self.class, self.started) else {
            return;
        };
        let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        CURRENT.with(|current| {
            if let Some(state) = current.borrow_mut().as_mut() {
                let bucket = state.cpu.entry(class).or_default();
                bucket.count = bucket.count.saturating_add(1);
                bucket.ns = bucket.ns.saturating_add(ns);
            }
        });
    }
}

/// Split one diagnostic GPU category. The boundary is a complete synchronization
/// by design and is therefore never called on the default path.
#[inline]
pub fn boundary(
    encoder: &mut mlx_native::CommandEncoder,
    class: &'static str,
) -> Result<(), mlx_native::MlxError> {
    if enabled() {
        encoder.profile_stage_boundary(class)?;
    }
    Ok(())
}

pub fn scratch_release() {
    if enabled() {
        SCRATCH_RELEASES.fetch_add(1, Ordering::Relaxed);
        PENDING_SCRATCH_RELEASES.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn scratch_reallocation() {
    if enabled() {
        SCRATCH_REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        PENDING_SCRATCH_REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

fn classify_label(label: &str) -> &'static str {
    if label.starts_with("adr067.dflash2_draft") {
        "dflash2_draft"
    } else if label.starts_with("adr065.mtp_catchup") || label.contains("process_target_batch") {
        "mtp_catchup"
    } else if label.starts_with("adr065.mtp_draft") || label.starts_with("mtp.") {
        "mtp_draft"
    } else if label.starts_with("adr065.projections") || label.contains("ops1-4") {
        "projections"
    } else if label.starts_with("adr065.delta_conv") {
        "delta_conv"
    } else if label.starts_with("adr065.delta_gdn")
        || label.contains("delta_net.ops")
        || label.contains("gdn.ops5-9")
    {
        "delta_gdn"
    } else if label.starts_with("adr065.kv_write") || label.contains("kv_cache_write") {
        "kv_write"
    } else if label.starts_with("adr065.attention")
        || label.contains("sdpa")
        || label.contains("attn")
    {
        "attention"
    } else if label.starts_with("adr065.ffn_moe")
        || label.contains("ffn")
        || label.contains("FFN")
        || label.contains("expert")
        || label.contains("MoE")
    {
        "ffn_moe"
    } else if label.starts_with("adr065.head")
        || label.contains("output_head")
        || label.contains("lm_head")
    {
        "head"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_label, Mode};

    #[test]
    fn classifies_closed_stage_labels_and_fails_unknown() {
        for (label, expected) in [
            ("adr065.projections", "projections"),
            ("adr067.dflash2_draft", "dflash2_draft"),
            ("adr065.delta_conv", "delta_conv"),
            ("adr065.delta_gdn", "delta_gdn"),
            ("layer.gdn.ops5-9", "delta_gdn"),
            ("adr065.kv_write", "kv_write"),
            ("adr065.attention", "attention"),
            ("adr065.ffn_moe", "ffn_moe"),
            ("DeepSeek-V4 FFN gate/up projections", "ffn_moe"),
            ("adr065.head", "head"),
            ("mtp.shared_head_argmax", "mtp_draft"),
            ("mtp.process_target_batch", "mtp_catchup"),
            ("new.unclassified.command", "unknown"),
        ] {
            assert_eq!(classify_label(label), expected, "{label}");
        }
    }

    #[test]
    fn parses_only_exact_profile_modes() {
        assert!(Mode::parse(None) == Mode::Off);
        assert!(Mode::parse(Some("wall")) == Mode::Wall);
        assert!(Mode::parse(Some("1")) == Mode::Profile);
        assert!(Mode::parse(Some("true")) == Mode::Off);
    }
}
