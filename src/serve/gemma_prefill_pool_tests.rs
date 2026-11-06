use super::*;
use crate::serve::gpu::GpuContext;

fn no_drain() -> Result<()> {
    panic!("a quiescent previous cycle must not drain the queue")
}

fn fill(buffer: &mut MlxBuffer, byte: u8) {
    buffer.as_mut_slice::<u8>().unwrap().fill(byte);
}

#[test]
fn cycle_hands_out_distinct_buffers_and_reuses_them_across_forwards() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    let (first_ptrs, first_lens) = {
        let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
        let a = cycle
            .alloc(&device, DType::F32, 257, Init::Overwritten, "a")
            .unwrap();
        let b = cycle
            .alloc(&device, DType::F32, 257, Init::Overwritten, "b")
            .unwrap();
        let c = cycle
            .alloc(&device, DType::U32, 33, Init::Zeroed, "c")
            .unwrap();
        assert_ne!(a.contents_ptr(), b.contents_ptr(), "live handouts alias");
        assert_eq!((a.dtype(), a.shape()), (DType::F32, &[257][..]));
        assert_eq!((c.dtype(), c.shape()), (DType::U32, &[33][..]));
        assert_eq!(a.as_slice::<f32>().unwrap().len(), 257);
        assert_eq!(c.as_slice::<u32>().unwrap().len(), 33);
        cycle.gpu_quiescent();
        let ptrs = [a.contents_ptr(), b.contents_ptr(), c.contents_ptr()];
        let lens = [a.data_byte_len(), b.data_byte_len(), c.data_byte_len()];
        drop((a, b, c));
        (ptrs, lens)
    };

    let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
    let a = cycle
        .alloc(&device, DType::F32, 257, Init::Overwritten, "a")
        .unwrap();
    let b = cycle
        .alloc(&device, DType::F32, 257, Init::Overwritten, "b")
        .unwrap();
    let c = cycle
        .alloc(&device, DType::U32, 33, Init::Zeroed, "c")
        .unwrap();
    let mut second: Vec<_> = [a.contents_ptr(), b.contents_ptr()].into_iter().collect();
    let mut first: Vec<_> = first_ptrs[..2].to_vec();
    second.sort();
    first.sort();
    assert_eq!(second, first, "same-shape activations are recycled");
    assert_eq!(c.contents_ptr(), first_ptrs[2]);
    assert_eq!(
        [a.data_byte_len(), b.data_byte_len(), c.data_byte_len()],
        first_lens,
        "a recycled bucket keeps the requested logical length"
    );
    cycle.gpu_quiescent();
}

#[test]
fn zeroed_handouts_are_zero_and_overwritten_handouts_skip_the_fill() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    let ptrs = {
        let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
        let mut zeroed = cycle
            .alloc(&device, DType::F32, 1000, Init::Zeroed, "z")
            .unwrap();
        let mut overwritten = cycle
            .alloc(&device, DType::F32, 64, Init::Overwritten, "o")
            .unwrap();
        assert!(zeroed.as_slice::<u8>().unwrap().iter().all(|&b| b == 0));
        fill(&mut zeroed, 0xA5);
        fill(&mut overwritten, 0x5A);
        cycle.gpu_quiescent();
        (zeroed.contents_ptr(), overwritten.contents_ptr())
    };

    let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
    let zeroed = cycle
        .alloc(&device, DType::F32, 1000, Init::Zeroed, "z")
        .unwrap();
    let overwritten = cycle
        .alloc(&device, DType::F32, 64, Init::Overwritten, "o")
        .unwrap();
    assert_eq!((zeroed.contents_ptr(), overwritten.contents_ptr()), ptrs);
    assert!(
        zeroed.as_slice::<u8>().unwrap().iter().all(|&b| b == 0),
        "a read-before-write buffer must not see the previous forward"
    );
    assert!(
        overwritten
            .as_slice::<u8>()
            .unwrap()
            .iter()
            .all(|&b| b == 0x5A),
        "an overwritten buffer is handed out without a memset"
    );
    cycle.gpu_quiescent();
}

#[test]
fn zero_all_cycle_zeroes_overwritten_handouts() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    {
        let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
        let mut buffer = cycle
            .alloc(&device, DType::U32, 96, Init::Overwritten, "o")
            .unwrap();
        fill(&mut buffer, 0xFF);
        cycle.gpu_quiescent();
    }
    let cycle = PrefillPoolCycle::begin(true, no_drain).unwrap();
    let buffer = cycle
        .alloc(&device, DType::U32, 96, Init::Overwritten, "o")
        .unwrap();
    assert!(buffer.as_slice::<u8>().unwrap().iter().all(|&b| b == 0));
    cycle.gpu_quiescent();
}

#[test]
fn unwound_cycle_drains_before_recycling_and_nested_cycles_are_refused() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    let first = {
        let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
        assert!(PrefillPoolCycle::begin(false, no_drain).is_err());
        let buffer = cycle
            .alloc(&device, DType::F32, 128, Init::Overwritten, "o")
            .unwrap();
        buffer.contents_ptr()
        // Dropped without `gpu_quiescent`: an error between async commits.
    };

    let (executor, _registry) = ctx.split();
    let mut drained = false;
    let cycle = PrefillPoolCycle::begin(false, || {
        executor.begin().unwrap().finish().unwrap();
        drained = true;
        Ok(())
    })
    .unwrap();
    assert!(
        drained,
        "an unwound cycle's buffers are reused only after a drain"
    );
    let buffer = cycle
        .alloc(&device, DType::F32, 128, Init::Overwritten, "o")
        .unwrap();
    assert_eq!(buffer.contents_ptr(), first);
    cycle.gpu_quiescent();
}

#[test]
fn requests_past_the_limit_fall_back_to_zeroed_device_buffers() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    set_limit_for_test(4096);

    let pooled_ptr = {
        let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
        let pooled = cycle
            .alloc(&device, DType::F32, 1024, Init::Overwritten, "p")
            .unwrap();
        let mut direct = cycle
            .alloc(&device, DType::F32, 1, Init::Overwritten, "d")
            .unwrap();
        assert!(direct.as_slice::<u8>().unwrap().iter().all(|&b| b == 0));
        assert_eq!(direct.byte_len(), 4, "a direct buffer is not bucketed");
        fill(&mut direct, 0x11);
        cycle.gpu_quiescent();
        pooled.contents_ptr()
    };

    let cycle = PrefillPoolCycle::begin(false, no_drain).unwrap();
    let pooled = cycle
        .alloc(&device, DType::F32, 1024, Init::Overwritten, "p")
        .unwrap();
    assert_eq!(pooled.contents_ptr(), pooled_ptr);
    let direct = cycle
        .alloc(&device, DType::F32, 1, Init::Overwritten, "d")
        .unwrap();
    assert_eq!(direct.byte_len(), 4, "direct buffers never enter the arena");
    assert!(direct.as_slice::<u8>().unwrap().iter().all(|&b| b == 0));
    cycle.gpu_quiescent();
}
