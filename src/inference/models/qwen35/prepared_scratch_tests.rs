use super::*;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::ops::repacked_gguf_mm::{
    prepare_repacked_gguf_weight_planes, repacked_gguf_mm_scratch_bytes,
};

struct Product {
    params: GgmlQuantizedMatmulParams,
    planes: PreparedGgufWeightPlanes,
    input: MlxBuffer,
    output: MlxBuffer,
    weight_value: f32,
}

impl Product {
    // Real GGML constant blocks, independently decoded: Q4_K has d=value,
    // scales=1, minima=0, quants=1. Q6_K has d=value, scales=1,
    // low nibbles=1 and high pairs=2, hence signed quants=33-32=1.
    fn new(device: &MlxDevice, registry: &mut KernelRegistry,
        ggml_type: GgmlType, n: u32, k: u32, rows: u32, value: f32) -> Self {
        let mut block = vec![0_u8; ggml_type.block_bytes() as usize];
        let d = half::f16::from_f32(value).to_le_bytes();
        match ggml_type {
            GgmlType::Q4_K => {
                block[..2].copy_from_slice(&d);
                block[4..8].fill(1);
                block[12..16].fill(1);
                block[16..].fill(0x11);
            }
            GgmlType::Q6_K => {
                block[..128].fill(0x11);
                block[128..192].fill(0xAA);
                block[192..208].fill(1);
                block[208..210].copy_from_slice(&d);
            }
            _ => unreachable!(),
        }
        let raw = block.repeat(n as usize * k as usize / 256);
        let mut weight = device.alloc_buffer(raw.len(), DType::U8, vec![raw.len()]).unwrap();
        weight.as_mut_slice::<u8>().unwrap().copy_from_slice(&raw);
        let params = GgmlQuantizedMatmulParams { m: rows, n, k, ggml_type };
        let mut prepare = device.command_encoder().unwrap();
        let planes = prepare_repacked_gguf_weight_planes(
            &mut prepare, registry, device, &weight, &params,
        ).unwrap();
        prepare.commit_and_wait().unwrap();
        let mut input = device.alloc_buffer(rows as usize * k as usize * 2,
            DType::BF16, vec![rows as usize, k as usize]).unwrap();
        for (r, row) in input.as_mut_slice::<half::bf16>().unwrap().chunks_mut(k as usize).enumerate() {
            row.fill(half::bf16::from_f32((r + 1) as f32 / 8.0));
        }
        let output = device.alloc_buffer(rows as usize * n as usize * 4,
            DType::F32, vec![rows as usize, n as usize]).unwrap();
        Self { params, planes, input, output, weight_value: value }
    }

    fn encode(&self, encoder: &mut CommandEncoder, registry: &mut KernelRegistry,
        device: &MlxDevice, execution: RepackedGgufMmExecution) -> Result<()> {
        dispatch(encoder, registry, device, &self.input, &self.planes,
            &self.output, &self.params, execution)
    }

    fn assert_reference(&self) {
        for (r, row) in self.output.as_slice::<f32>().unwrap().chunks(self.params.n as usize).enumerate() {
            // F64 dot product over the independent constant block decoding.
            let expected = (0..self.params.k).map(|_| {
                f64::from(self.weight_value) * (r + 1) as f64 / 8.0
            }).sum::<f64>();
            for &actual in row {
                assert_eq!(f64::from(actual), expected, "row {r}, N={}, K={}", self.params.n, self.params.k);
            }
        }
    }
}

#[test]
fn queued_unequal_products_one_pass_and_error_drain_reuse_one_owner() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let a = Product::new(&device, &mut registry, GgmlType::Q4_K, 256, 4096, 5, 1.0);
    let b = Product::new(&device, &mut registry, GgmlType::Q6_K, 128, 8192, 32, 2.0);
    let c = Product::new(&device, &mut registry, GgmlType::Q4_K, 192, 4096, 33, 3.0);
    let capacity = [&a, &b, &c].into_iter().map(|p| {
        p.planes.execution_scratch_bytes(p.params.m, RepackedGgufMmExecution::Auto).unwrap()
    }).max().unwrap();
    assert!(a.planes.execution_scratch_bytes(5, RepackedGgufMmExecution::Auto).unwrap() > 0);
    assert!(b.planes.execution_scratch_bytes(32, RepackedGgufMmExecution::Auto).unwrap() > 0);
    let owner = ExecutionScratch::new(&device, capacity);
    let scope = owner.enter(&device).unwrap();
    let mut first = device.command_encoder().unwrap();
    a.encode(&mut first, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    b.encode(&mut first, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    c.encode(&mut first, &mut registry, &device, RepackedGgufMmExecution::OnePass).unwrap();
    first.commit(); // The next encoder is queued before any host wait.
    let root = owner.buffer.get().unwrap().contents_ptr();
    let mut second = device.command_encoder().unwrap();
    let too_small = device.alloc_buffer(4, DType::F32, vec![1]).unwrap();
    let error = dispatch(&mut second, &mut registry, &device, &a.input, &a.planes,
        &too_small, &a.params, RepackedGgufMmExecution::Auto).unwrap_err();
    assert!(format!("{error:#}").contains("output"), "{error:#}");
    // An error does not cancel the first command. Reuse only by ordered GPU
    // work; CPU reads wait for BOTH checked command statuses below.
    c.encode(&mut second, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    a.encode(&mut second, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    second.commit();
    let first_result = first.wait_until_completed();
    let second_result = second.wait_until_completed();
    first_result.unwrap();
    second_result.unwrap();
    assert_eq!(root, owner.buffer.get().unwrap().contents_ptr());
    for p in [&a, &b, &c] { p.assert_reference(); }
    drop(scope);
    // A later checked operation on the cache reuses the same root, not a
    // per-command or per-weight replacement after the failed dispatch.
    let _scope = owner.enter(&device).unwrap();
    let mut third = device.command_encoder().unwrap();
    b.encode(&mut third, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    third.commit_and_wait().unwrap();
    b.assert_reference();
    assert_eq!(root, owner.buffer.get().unwrap().contents_ptr());
}

#[test]
fn nested_owners_restore_and_other_queues_cannot_borrow_the_root() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let other_queue = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let a = Product::new(&device, &mut registry, GgmlType::Q4_K, 128, 4096, 1, 1.0);
    let required = a.planes.execution_scratch_bytes(1, RepackedGgufMmExecution::Auto).unwrap();
    assert!(required > 0);
    let first = ExecutionScratch::new(&device, required);
    let second = ExecutionScratch::new(&device, required);
    let _outer = first.enter(&device).unwrap();
    let mut alien_encoder = other_queue.command_encoder().unwrap();
    assert!(a.encode(&mut alien_encoder, &mut registry, &other_queue,
        RepackedGgufMmExecution::Auto).is_err());
    assert!(first.enter(&other_queue).is_err());
    let mut encoder = device.command_encoder().unwrap();
    a.encode(&mut encoder, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    let nested: Result<()> = (|| {
        let _inner = second.enter(&device)?;
        a.encode(&mut encoder, &mut registry, &device, RepackedGgufMmExecution::Auto)?;
        anyhow::bail!("end nested operation with an error")
    })();
    assert!(nested.is_err());
    ACTIVE.with(|active| assert!(Rc::ptr_eq(active.borrow().as_ref().unwrap(), &first)));
    assert_ne!(first.buffer.get().unwrap().contents_ptr(), second.buffer.get().unwrap().contents_ptr());
    a.encode(&mut encoder, &mut registry, &device, RepackedGgufMmExecution::Auto).unwrap();
    encoder.commit_and_wait().unwrap();
    a.assert_reference();
}

#[test]
fn one_pass_needs_no_root_and_native_row_bound_covers_all_tile_schedules() {
    for ty in [GgmlType::Q4_K, GgmlType::Q6_K] {
        for (n, k) in [(64, 256), (128, 512), (256, 4096), (17408, 5120), (5120, 17408), (248320, 2048)] {
            let bound = repacked_gguf_mm_scratch_bytes(n, k, ty, 1, RepackedGgufMmExecution::Auto).unwrap();
            assert_eq!(bound, repacked_gguf_mm_scratch_bytes(n, k, ty, 32, RepackedGgufMmExecution::Auto).unwrap());
            for rows in [1, 4, 8, 24, 32, 33, 64, 128, 257, 1024, 4096] {
                assert!(repacked_gguf_mm_scratch_bytes(n, k, ty, rows, RepackedGgufMmExecution::Auto).unwrap() <= bound);
                assert_eq!(repacked_gguf_mm_scratch_bytes(n, k, ty, rows, RepackedGgufMmExecution::OnePass).unwrap(), 0);
            }
        }
    }
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let a = Product::new(&device, &mut registry, GgmlType::Q6_K, 128, 4096, 5, 2.0);
    let owner = ExecutionScratch::new(&device, 0);
    let _scope = owner.enter(&device).unwrap();
    let mut encoder = device.command_encoder().unwrap();
    a.encode(&mut encoder, &mut registry, &device, RepackedGgufMmExecution::OnePass).unwrap();
    encoder.commit_and_wait().unwrap();
    assert!(owner.buffer.get().is_none());
    a.assert_reference();
}

#[test]
fn a_different_model_cannot_enter_the_active_cache_owner() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let first = super::super::prefill_prepared_tests::observable_synthetic_model(&device);
    let second = super::super::prefill_prepared_tests::observable_synthetic_model(&device);
    first.ensure_gpu_cache_primed().unwrap();
    let owner = first.with_gpu_cache_mut(|_, _| {
        Ok(ACTIVE.with(|active| active.borrow().as_ref().unwrap().clone()))
    }).unwrap();
    let called = std::cell::Cell::new(false);
    let error = second.with_gpu_cache_mut(|_, _| {
        called.set(true);
        Ok(())
    }).unwrap_err();
    assert!(!called.get(), "a different model entered the cached device scope");
    assert!(error.to_string().contains("different Qwen35Model"), "{error}");
    first.with_gpu_cache_mut(|_, _| {
        ACTIVE.with(|active| assert!(Rc::ptr_eq(active.borrow().as_ref().unwrap(), &owner)));
        Ok(())
    }).unwrap();
    drop(owner);
    first.release_gpu_cache().unwrap();
}
