//! Scope of the Gemma 4 Q5_0 tensor-matmul default (ADR-063 Q4-GEMMA-ATTN).
//!
//! `GpuContext::new_for_gemma4()` freezes the family policy into its own
//! registries. These tests dispatch real products: only Q5_0 matrix-matrix
//! widths change bits, only in a Gemma 4 context, in either load order, and
//! no process-wide state moves. Model-free.

use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, quantized_matmul_ggml_with_policy, GgmlQuantizedMatmulParams,
};
use mlx_native::{DType, GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxDevice};

use crate::serve::gpu::GpuContext;

const N: usize = 64;
const K: usize = 2816;
const WIDTHS: [usize; 4] = [1, 8, 32, 64];
/// Marks the per-value child process of the kill-switch test.
const CHILD: &str = "HF2Q_Q5_0_TENSOR_MM_TEST_EXPECT";
/// Routing variables that would make the expected route set ambiguous.
const ROUTING_ENV: [&str; 12] = [
    "HF2Q_Q5_0_TENSOR_MM",
    "HF2Q_DISABLE_TENSOR_MM",
    "HF2Q_DISABLE_TENSOR_MM_ID",
    "HF2Q_LARGE_TILE_MM",
    "HF2Q_DECODE_MV_EXT",
    "HF2Q_DECODE_MVN",
    "HF2Q_Q8_0_MV_NR2",
    "HF2Q_Q6K_MV_NR2",
    "HF2Q_Q5K_CANONICAL_Q4X4",
    "HF2Q_MM_ID_SCHEDULE",
    "HF2Q_MM_ID_ROUTING_THRESHOLD",
    "HF2Q_PARALLEL_ENCODE",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random blocks of `block_bytes` bytes whose leading `scales` F16 fields
/// are small finite scales.
fn blocks(values: usize, block_bytes: usize, scales: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut bytes = vec![0u8; N * (K / values) * block_bytes];
    for block in bytes.chunks_exact_mut(block_bytes) {
        for byte in block.iter_mut() {
            *byte = (rng.next() >> 32) as u8;
        }
        for scale in 0..scales {
            let value = half::f16::from_f32(0.01 + 0.05 * rng.unit().abs());
            block[scale * 2..scale * 2 + 2].copy_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    bytes
}

fn weight_bytes(ggml_type: GgmlType) -> Vec<u8> {
    match ggml_type {
        GgmlType::Q5_0 => blocks(32, 22, 1, 0x0832_0005),
        GgmlType::Q8_0 => blocks(32, 34, 1, 0x0832_0008),
        GgmlType::Q4_K => blocks(256, 144, 2, 0x0832_0004),
        other => panic!("no fixture for {other:?}"),
    }
}

fn activations() -> Vec<f32> {
    let mut rng = Rng(0x0832_aaaa);
    (0..WIDTHS[WIDTHS.len() - 1] * K)
        .map(|_| rng.unit())
        .collect()
}

/// One dense product on `device`; `policy` selects the explicit-policy
/// entry point, `None` the registry-reading one production dispatch uses.
fn product(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    policy: Option<&GgmlRoutingPolicy>,
    ggml_type: GgmlType,
    weight: &[u8],
    x: &[f32],
    m: usize,
) -> Vec<u32> {
    let mut w = device
        .alloc_buffer(weight.len(), DType::U8, vec![weight.len()])
        .unwrap();
    w.as_mut_slice::<u8>().unwrap().copy_from_slice(weight);
    let mut input = device
        .alloc_buffer(m * K * 4, DType::F32, vec![m, K])
        .unwrap();
    input
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&x[..m * K]);
    let output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .unwrap();
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: N as u32,
        k: K as u32,
        ggml_type,
    };
    let mut encoder = device.command_encoder().unwrap();
    match policy {
        Some(policy) => quantized_matmul_ggml_with_policy(
            &mut encoder,
            registry,
            device,
            &input,
            &w,
            &output,
            &params,
            policy,
        ),
        None => quantized_matmul_ggml(&mut encoder, registry, device, &input, &w, &output, &params),
    }
    .unwrap();
    encoder.commit_and_wait().unwrap();
    output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn through(
    ctx: &mut GpuContext,
    ggml_type: GgmlType,
    weight: &[u8],
    x: &[f32],
    m: usize,
) -> Vec<u32> {
    let (executor, registry) = ctx.split();
    product(executor.device(), registry, None, ggml_type, weight, x, m)
}

/// The Gemma 4 context carries the Q5_0 tensor policy; a context of any
/// other family built before or after it does not; a registry nobody bound
/// keeps the process default; and only Q5_0 matrix-matrix products
/// (m = 32 and 64, never the few-row decode widths) change bits.
#[test]
fn gemma4_context_freezes_the_q5_0_policy_and_reroutes_only_q5_0_matrix_products() {
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    for name in ROUTING_ENV {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for this test"
        );
    }
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let environment = mlx_native::ggml_routing_policy_from_environment();
    assert!(!environment.q5_0_tensor_mm);
    let on = GgmlRoutingPolicy {
        q5_0_tensor_mm: true,
        ..environment
    };

    // Load order: other family, Gemma 4, other family, Gemma 4.
    let mut other_before = GpuContext::new().unwrap();
    let mut gemma_first = GpuContext::new_for_gemma4().unwrap();
    let mut other_after = GpuContext::new().unwrap();
    let mut gemma_second = GpuContext::new_for_gemma4().unwrap();
    for ctx in [&other_before, &other_after] {
        assert_eq!(ctx.ggml_routing_policy(), &environment);
        assert_eq!(ctx.registry.ggml_routing_policy(), Some(&environment));
    }
    for ctx in [&gemma_first, &gemma_second] {
        assert_eq!(ctx.ggml_routing_policy(), &on);
        assert_eq!(ctx.registry.ggml_routing_policy(), Some(&on));
        assert!(ctx.worker_registry.is_none());
    }
    // The process resolver did not move.
    assert_eq!(
        mlx_native::ggml_routing_policy_from_environment(),
        environment
    );

    let x = activations();
    let reference_device = MlxDevice::new().unwrap();
    let mut rerouted = Vec::new();
    for ggml_type in [GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::Q4_K] {
        let weight = weight_bytes(ggml_type);
        for m in WIDTHS {
            let reference = |policy: &GgmlRoutingPolicy| {
                product(
                    &reference_device,
                    &mut KernelRegistry::new(),
                    Some(policy),
                    ggml_type,
                    &weight,
                    &x,
                    m,
                )
            };
            let (want_off, want_on) = (reference(&environment), reference(&on));
            let label = format!("{ggml_type:?} m={m}");
            // Interleaved: Gemma, other, Gemma, other, then the first two again.
            let g1 = through(&mut gemma_first, ggml_type, &weight, &x, m);
            let o1 = through(&mut other_before, ggml_type, &weight, &x, m);
            let g2 = through(&mut gemma_second, ggml_type, &weight, &x, m);
            let o2 = through(&mut other_after, ggml_type, &weight, &x, m);
            let o1_again = through(&mut other_before, ggml_type, &weight, &x, m);
            let g1_again = through(&mut gemma_first, ggml_type, &weight, &x, m);
            // A registry nobody bound, created after the Gemma contexts.
            let unbound = product(
                &reference_device,
                &mut KernelRegistry::new(),
                None,
                ggml_type,
                &weight,
                &x,
                m,
            );
            assert!(
                g1 == want_on && g2 == want_on && g1_again == want_on,
                "{label}: Gemma context"
            );
            assert!(
                o1 == want_off && o2 == want_off && o1_again == want_off,
                "{label}: other context"
            );
            assert!(unbound == want_off, "{label}: unbound registry");
            if want_on != want_off {
                rerouted.push(label);
            }
        }
    }
    let tensor_api = {
        let (executor, registry) = other_before.split();
        mlx_native::ops::flash_attn_prefill_tensor::flash_attn_prefill_tensor_available(
            registry,
            executor.device(),
            256,
        )
        .unwrap()
    };
    if tensor_api {
        assert_eq!(rerouted, ["Q5_0 m=32", "Q5_0 m=64"]);
    } else {
        eprintln!("RECORDED SKIP: no Metal tensor API on this device");
        assert!(rerouted.is_empty());
    }
}

/// `HF2Q_Q5_0_TENSOR_MM` per value, in a fresh process each: what the two
/// kinds of context freeze. An off spelling gives the Gemma 4 context
/// exactly the policy of every other context (the route set of v0.1.29); a
/// value that is not a route refuses the Gemma 4 context and nothing else.
#[test]
fn q5_0_kill_switch_restores_the_environment_policy_in_the_gemma4_context() {
    if let Some(expected) = std::env::var_os(CHILD) {
        let expected = expected.to_str().unwrap().as_bytes().to_vec();
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let environment = mlx_native::ggml_routing_policy_from_environment();
        let other = GpuContext::new().unwrap();
        assert_eq!(other.registry.ggml_routing_policy(), Some(&environment));
        assert_eq!(environment.q5_0_tensor_mm, expected[1] == b'1');
        if expected[0] == b'x' {
            let error = GpuContext::new_for_gemma4().err().expect("refused");
            assert!(format!("{error:#}").contains("HF2Q_Q5_0_TENSOR_MM"));
            return;
        }
        let gemma = GpuContext::new_for_gemma4().unwrap();
        let frozen = *gemma.registry.ggml_routing_policy().unwrap();
        assert_eq!(frozen.q5_0_tensor_mm, expected[0] == b'1');
        assert_eq!(
            frozen,
            GgmlRoutingPolicy {
                q5_0_tensor_mm: frozen.q5_0_tensor_mm,
                ..environment
            }
        );
        if expected[0] == expected[1] {
            assert_eq!(frozen, environment);
        }
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for (value, expected) in [
        (None, "10"),
        (Some(""), "10"),
        (Some("1"), "11"),
        (Some("0"), "00"),
        (Some("off"), "00"),
        (Some("false"), "00"),
        (Some("stock"), "00"),
        (Some("true"), "x0"),
        (Some("of"), "x0"),
    ] {
        let mut child = std::process::Command::new(&exe);
        child
            .arg("serve::gemma_routing_scope_tests::q5_0_kill_switch_restores_the_environment_policy_in_the_gemma4_context")
            .arg("--exact")
            .env(CHILD, expected);
        match value {
            Some(value) => child.env("HF2Q_Q5_0_TENSOR_MM", value),
            None => child.env_remove("HF2Q_Q5_0_TENSOR_MM"),
        };
        let output = child.output().unwrap();
        assert!(
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "HF2Q_Q5_0_TENSOR_MM={value:?}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
