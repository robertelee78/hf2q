//! Paired host clock samples for GPU-timestamp traces (ADR-063 Q4-TRACE-GPU-TIMING).
//!
//! Metal reports command-buffer GPU times in mach host seconds, while the phase trace uses
//! CLOCK_MONOTONIC. No mapping between them is assumed. One sample reads CLOCK_MONOTONIC,
//! CLOCK_UPTIME_RAW and mach_absolute_time back to back and records how long the reads took,
//! so any cross-clock figure can carry a measured bound.
use serde::Serialize;

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ClockPair {
    pub monotonic_ns: u64,
    pub uptime_raw_ns: u64,
    pub mach_absolute_ticks: u64,
    pub mach_timebase: [u32; 2],
    /// CLOCK_MONOTONIC from before the first read to after the last.
    pub read_span_ns: u64,
}

fn clock_ns(clock: libc::clockid_t) -> Result<u64, &'static str> {
    let mut t = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: valid writable timespec; no payload or device access.
    if unsafe { libc::clock_gettime(clock, t.as_mut_ptr()) } != 0 {
        return Err("host clock unavailable");
    }
    // SAFETY: clock_gettime succeeded and initialized the timespec.
    let t = unsafe { t.assume_init() };
    u64::try_from(t.tv_sec)
        .ok()
        .and_then(|s| s.checked_mul(1_000_000_000))
        .and_then(|s| u64::try_from(t.tv_nsec).ok().and_then(|n| s.checked_add(n)))
        .ok_or("host clock overflow")
}

pub(crate) fn sample() -> Result<ClockPair, &'static str> {
    let mut timebase = MachTimebaseInfo::default();
    // SAFETY: valid writable struct of the documented layout.
    if unsafe { mach_timebase_info(&mut timebase) } != 0 || timebase.denom == 0 {
        return Err("mach timebase unavailable");
    }
    let before = clock_ns(libc::CLOCK_MONOTONIC)?;
    let monotonic_ns = clock_ns(libc::CLOCK_MONOTONIC)?;
    let uptime_raw_ns = clock_ns(libc::CLOCK_UPTIME_RAW)?;
    // SAFETY: no arguments; reads the host tick counter.
    let mach_absolute_ticks = unsafe { mach_absolute_time() };
    let after = clock_ns(libc::CLOCK_MONOTONIC)?;
    Ok(ClockPair {
        monotonic_ns,
        uptime_raw_ns,
        mach_absolute_ticks,
        mach_timebase: [timebase.numer, timebase.denom],
        read_span_ns: after
            .checked_sub(before)
            .ok_or("monotonic clock regressed")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_ordered_and_mach_ticks_track_uptime_raw() {
        let first = sample().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let second = sample().unwrap();
        assert!(second.monotonic_ns > first.monotonic_ns);
        assert!(second.uptime_raw_ns > first.uptime_raw_ns);
        assert!(second.mach_absolute_ticks > first.mach_absolute_ticks);
        // mach ticks in ns advance with CLOCK_UPTIME_RAW (same host clock on macOS),
        // within the two samples' read spans: no fixed, load-dependent bound.
        let [numer, denom] = first.mach_timebase;
        let ticks_ns = (second.mach_absolute_ticks - first.mach_absolute_ticks) as u128
            * u128::from(numer)
            / u128::from(denom);
        let uptime_ns = u128::from(second.uptime_raw_ns - first.uptime_raw_ns);
        let slack = u128::from(first.read_span_ns + second.read_span_ns) + 1_000;
        assert!(
            ticks_ns.abs_diff(uptime_ns) <= slack,
            "{ticks_ns} vs {uptime_ns} (slack {slack})"
        );
    }
}
