//! Typed current OS working estimate; never a unique physical-page census.
use anyhow::Result;
#[cfg(target_os = "macos")]
use anyhow::{ensure, Context};
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct Sample {
    pub total: u64,
    pub free: u64,
    pub file_cache: u64,
    pub swap: u64,
    pub page: u64,
    pub rss: u64,
    pub footprint: u64,
}
#[cfg(target_os = "macos")]
pub(super) fn sample() -> Result<Sample> {
    fn sys<T: Copy>(name: &std::ffi::CStr) -> Result<T> {
        let mut value = std::mem::MaybeUninit::<T>::uninit();
        let mut bytes = std::mem::size_of::<T>();
        // SAFETY: fixed kernel names and an exactly sized typed output slot.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                value.as_mut_ptr().cast(),
                &mut bytes,
                std::ptr::null_mut(),
                0,
            )
        };
        ensure!(
            rc == 0 && bytes == std::mem::size_of::<T>(),
            "Q5 typed OS query failed"
        );
        Ok(unsafe { value.assume_init() })
    }
    let total = sys::<u64>(c"hw.memsize")?;
    ensure!(
        sys::<i32>(c"kern.memorystatus_vm_pressure_level")? == 1,
        "Q5 abnormal OS pressure"
    );
    let swap = sys::<libc::xsw_usage>(c"vm.swapusage")?.xsu_used;
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })?;
    let mut v = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut n = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: owned host send right, kernel fills fixed vm_statistics64 layout.
    unsafe extern "C" {
        #[link_name = "mach_host_self"]
        fn current_host() -> libc::mach_port_t;
        #[link_name = "mach_task_self_"]
        static CURRENT_TASK: libc::mach_port_t;
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let host = unsafe { current_host() };
    let rc = unsafe {
        libc::host_statistics64(host, libc::HOST_VM_INFO64, v.as_mut_ptr().cast(), &mut n)
    };
    // SAFETY: release only the send right obtained above; no process/resource mutation.
    let released = unsafe { mach_port_deallocate(CURRENT_TASK, host) };
    ensure!(
        released == 0,
        "Q5 host observation send-right release failed"
    );
    ensure!(
        rc == 0 && n == libc::HOST_VM_INFO64_COUNT,
        "Q5 VM statistics unavailable"
    );
    let v = unsafe { v.assume_init() };
    let free = u64::from(v.free_count)
        .checked_mul(page)
        .context("Q5 Free overflow")?;
    let file_cache = u64::from(v.external_page_count)
        .checked_mul(page)
        .context("Q5 file cache overflow")?;
    ensure!(
        page.is_power_of_two()
            && (4096..=65536).contains(&page)
            && free.checked_add(file_cache).is_some_and(|x| x <= total),
        "Q5 invalid OS partition"
    );
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: exactly sized actual-self proc_pid_rusage output, no foreign PID.
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    ensure!(rc == 0, "Q5 actual-self process usage unavailable");
    let usage = unsafe { usage.assume_init() };
    Ok(Sample {
        total,
        free,
        file_cache,
        swap,
        page,
        rss: usage.ri_resident_size,
        footprint: usage.ri_phys_footprint,
    })
}
#[cfg(not(target_os = "macos"))]
pub(super) fn sample() -> Result<Sample> {
    anyhow::bail!("Q5 selected operating OS proof requires Darwin")
}
