//! ADR-063 Q4-DS4-RACE §6.2: expert-kernel parity on ds4's real weights
//! (design `docs/research/adr066-iq2xxs-subset-design.md` §6.2 with the §3.7
//! oracles and bounds).
//!
//! For blk.3 and blk.21 `ffn_{gate,up,down}_exps` (IQ2_XXS gate/up, Q2_K
//! down) the whole tensor is loaded from the file and run through
//! - the forced `mv_id` route at 6 and 24 routed rows (1 and 4 tokens × top-6),
//! - the simdgroup `mm_id` route and the tensor-API `mm_id` route at 512
//!   tokens × top-6, plus a 512-token route where every token shares one
//!   expert,
//!
//! with seeded routes and inputs in the §3.7 range (uniform [−4, 4], values
//! below 2⁻¹⁴ set to 0). The oracle is mlx-native's CPU dequant (bit-exact
//! against the peer, checked on these files by §6.1 T3) followed by an F64
//! dot. Bounds: `mv_id` `|got − want| ≤ 2.5e-4 · Σ|wᵢxᵢ|`, `mm_id`
//! `≤ 2e-3 · Σ|wᵢxᵢ|`. About 2 GB of buffers; no model runtime.
//!
//! Run (GPU lane):
//! ```text
//! HF2Q_DS4_FILE=<ds4 gguf> HF2Q_DS4_KERNEL_RECEIPT=<out json> \
//! cargo test --release --bin hf2q ds4_file_expert_kernels_match_cpu_oracle -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_native::gguf::{test_only_dequantize, GgufFile};
use mlx_native::{
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_with_policy, DType,
    GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType,
    KernelRegistry, MlxBuffer, MlxDevice,
};
use serde_json::{json, Value};

const LAYERS: [usize; 2] = [3, 21];
const TOP_K: usize = 6;
const MV_BOUND: f64 = 2.5e-4;
const MM_BOUND: f64 = 2e-3;
const MM_TOKENS: usize = 512;

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Design §3.7 input range.
    fn input_value(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        let value = unit * 8.0 - 4.0;
        if value.abs() < 1.0 / 16384.0 {
            0.0
        } else {
            value
        }
    }
}

#[derive(Clone, Copy)]
struct Shape {
    ggml_type: GgmlType,
    experts: usize,
    n: usize,
    k: usize,
}

impl Shape {
    fn row_bytes(self) -> usize {
        self.k / self.ggml_type.block_values() as usize * self.ggml_type.block_bytes() as usize
    }

    fn expert_bytes(self) -> usize {
        self.n * self.row_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
enum Route {
    ForcedMv,
    Simdgroup,
    Tensor,
}

fn inputs(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * k).map(|_| rng.input_value()).collect()
}

/// Seeded top-k, distinct per token; `shared` puts one expert in every token.
fn routes(tokens: usize, experts: usize, seed: u64, shared: Option<u32>) -> Vec<u32> {
    let mut rng = Rng(seed);
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for token in 0..tokens {
        let start = ids.len();
        if let Some(expert) = shared {
            ids.push(expert);
        }
        while ids.len() - start < TOP_K {
            let expert = rng.below(experts) as u32;
            if !ids[start..].contains(&expert) {
                ids.push(expert);
            }
        }
        // Rotate so the shared expert does not always sit in slot 0.
        ids[start..].rotate_right(token % TOP_K);
    }
    ids
}

fn buffer<T: bytemuck::Pod>(device: &MlxDevice, values: &[T], dtype: DType) -> MlxBuffer {
    let bytes = std::mem::size_of_val(values);
    let mut buffer = device
        .alloc_buffer(bytes, dtype, vec![values.len()])
        .unwrap();
    buffer.as_mut_slice::<T>().unwrap().copy_from_slice(values);
    buffer
}

fn policy(route: Route) -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_tensor_mm: match route {
            Route::Tensor => GgmlTensorMmPreference::AutoProbe,
            Route::ForcedMv | Route::Simdgroup => GgmlTensorMmPreference::ForceSimd,
        },
        ..GgmlRoutingPolicy::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MlxBuffer,
    shape: Shape,
    input: &[f32],
    ids: &[u32],
    tokens: usize,
    route: Route,
) -> Vec<f32> {
    let input = buffer(device, input, DType::F32);
    let ids = buffer(device, ids, DType::U32);
    let output = buffer(device, &vec![0.0_f32; tokens * TOP_K * shape.n], DType::F32);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: tokens as u32,
        top_k: TOP_K as u32,
        n: shape.n as u32,
        k: shape.k as u32,
        n_experts: shape.experts as u32,
        expert_stride: shape.expert_bytes() as u64,
        ggml_type: shape.ggml_type,
    };
    let mut encoder = device.command_encoder().unwrap();
    let routing = policy(route);
    match route {
        Route::ForcedMv => quantized_matmul_id_ggml_mv_with_policy(
            &mut encoder,
            registry,
            device,
            &input,
            weights,
            &ids,
            &output,
            &params,
            &routing,
        ),
        Route::Simdgroup | Route::Tensor => quantized_matmul_id_ggml_with_policy(
            &mut encoder,
            registry,
            device,
            &input,
            weights,
            &ids,
            &output,
            &params,
            &routing,
        ),
    }
    .unwrap_or_else(|error| panic!("{route:?} {:?} dispatch: {error}", shape.ggml_type));
    encoder.commit_and_wait().unwrap();
    output.as_slice::<f32>().unwrap().to_vec()
}

fn sample(total: usize, cap: usize, seed: u64) -> Vec<usize> {
    if total <= cap {
        return (0..total).collect();
    }
    let mut rng = Rng(seed);
    let mut rows = vec![0, total - 1];
    while rows.len() < cap {
        rows.push(rng.below(total));
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Odd-stride columns cover every residue of the 8-row mv and 64-row mm
/// tiles, plus the last row.
fn columns(n: usize, count: usize) -> Vec<usize> {
    let mut columns: Vec<usize> = (0..count.min(n)).map(|i| (i * 1031 + 7) % n).collect();
    columns.push(n - 1);
    columns.sort_unstable();
    columns.dedup();
    columns
}

#[allow(clippy::too_many_arguments)]
fn check(
    label: &str,
    weights: &[u8],
    shape: Shape,
    input: &[f32],
    ids: &[u32],
    got: &[f32],
    rows: &[usize],
    columns: &[usize],
    bound: f64,
) -> Value {
    let row_bytes = shape.row_bytes();
    let mut worst = 0.0_f64;
    let mut failures = Vec::new();
    let mut w = vec![0.0_f32; shape.k];
    for &routed in rows {
        let expert = ids[routed] as usize;
        let token = routed / TOP_K;
        let x = &input[token * shape.k..(token + 1) * shape.k];
        for &column in columns {
            let start = expert * shape.expert_bytes() + column * row_bytes;
            test_only_dequantize(&weights[start..start + row_bytes], shape.ggml_type, &mut w)
                .expect("CPU dequant");
            let (mut want, mut magnitude) = (0.0_f64, 0.0_f64);
            for (wi, xi) in w.iter().zip(x) {
                let product = *wi as f64 * *xi as f64;
                want += product;
                magnitude += product.abs();
            }
            let value = got[routed * shape.n + column];
            let error = (value as f64 - want).abs();
            if !value.is_finite() || error > bound * magnitude {
                failures.push(format!(
                    "routed {routed} expert {expert} column {column}: got {value} want {want} |err| {error:e} > {bound:e}·{magnitude:e}"
                ));
            }
            if magnitude > 0.0 {
                worst = worst.max(error / magnitude);
            }
        }
    }
    eprintln!(
        "{label}: {} rows × {} columns, worst {worst:e} (bound {bound:e}), {} failures",
        rows.len(),
        columns.len(),
        failures.len()
    );
    json!({"case": label, "rows": rows.len(), "columns": columns.len(), "worst_relative": worst,
           "bound": bound, "pass": failures.is_empty(), "failures": failures.iter().take(8).collect::<Vec<_>>()})
}

#[test]
#[ignore = "loads ds4's real expert tensors onto Metal and runs mv_id/mm_id (GPU lane, ~2 GB)"]
fn ds4_file_expert_kernels_match_cpu_oracle() {
    let path = PathBuf::from(std::env::var("HF2Q_DS4_FILE").expect("set HF2Q_DS4_FILE"));
    let receipt_path = PathBuf::from(
        std::env::var("HF2Q_DS4_KERNEL_RECEIPT").expect("set HF2Q_DS4_KERNEL_RECEIPT"),
    );
    let gguf = GgufFile::open(&path).expect("open ds4 file");
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let tensor_available = registry
        .get_pipeline("kernel_mul_mm_id_q4_0_tensor_f32", device.metal_device())
        .is_ok();
    let mut cases = Vec::new();
    for layer in LAYERS {
        for kind in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let name = format!("blk.{layer}.{kind}.weight");
            let info = gguf.tensor_info(&name).unwrap();
            let [experts, n, k] = info.shape[..] else {
                panic!("{name}: shape {:?}", info.shape);
            };
            let shape = Shape {
                ggml_type: info.ggml_type,
                experts,
                n,
                k,
            };
            let expected = if kind == "ffn_down_exps" {
                GgmlType::Q2_K
            } else {
                GgmlType::IQ2_XXS
            };
            assert_eq!(shape.ggml_type, expected, "{name}");
            let weights = gguf
                .load_tensor(&name, &device)
                .expect("load expert tensor");
            assert_eq!(weights.dtype(), DType::U8);
            let bytes = weights.as_slice::<u8>().unwrap().to_vec();
            assert_eq!(bytes.len(), experts * shape.expert_bytes());
            let seed = ((layer as u64) << 32) ^ (k as u64) ^ ((kind.len() as u64) << 16);
            let mut run = |label: String,
                           tokens: usize,
                           ids: Vec<u32>,
                           route: Route,
                           bound: f64,
                           row_cap: usize,
                           column_count: usize| {
                let input = inputs(tokens, k, seed ^ 0x5eed ^ tokens as u64);
                let got = dispatch(
                    &device,
                    &mut registry,
                    &weights,
                    shape,
                    &input,
                    &ids,
                    tokens,
                    route,
                );
                let rows = sample(tokens * TOP_K, row_cap, seed ^ 0x0bad ^ tokens as u64);
                cases.push(check(
                    &label,
                    &bytes,
                    shape,
                    &input,
                    &ids,
                    &got,
                    &rows,
                    &columns(n, column_count),
                    bound,
                ));
            };
            for tokens in [1_usize, 4] {
                let ids = routes(tokens, experts, seed ^ 0x1d5 ^ tokens as u64, None);
                run(
                    format!("{name} mv_id {}", tokens * TOP_K),
                    tokens,
                    ids,
                    Route::ForcedMv,
                    MV_BOUND,
                    usize::MAX,
                    256,
                );
            }
            let mut mm_routes = vec![Route::Simdgroup];
            if tensor_available {
                mm_routes.push(Route::Tensor);
            }
            for route in mm_routes {
                let ids = routes(MM_TOKENS, experts, seed ^ 0x3300, None);
                run(
                    format!("{name} {route:?} mm_id {MM_TOKENS}x{TOP_K}"),
                    MM_TOKENS,
                    ids,
                    route,
                    MM_BOUND,
                    48,
                    64,
                );
                let ids = routes(MM_TOKENS, experts, seed ^ 0x5151, Some(201));
                run(
                    format!("{name} {route:?} mm_id shared-expert {MM_TOKENS}x{TOP_K}"),
                    MM_TOKENS,
                    ids,
                    route,
                    MM_BOUND,
                    48,
                    64,
                );
            }
        }
    }
    let pass = cases
        .iter()
        .all(|case| case["pass"].as_bool() == Some(true));
    let receipt = json!({
        "schema": "hf2q.q4-ds4-race.kernel-parity.v1",
        "ds4_file": path.display().to_string(),
        "mlx_native": env!("HF2Q_MLX_NATIVE_VERSION"),
        "tensor_mm_id_available": tensor_available,
        "layers": LAYERS,
        "bounds": {"mv_id": MV_BOUND, "mm_id": MM_BOUND},
        "pass": pass,
        "cases": cases,
    });
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap())
        .unwrap_or_else(|error| panic!("write receipt {}: {error}", receipt_path.display()));
    eprintln!("receipt: {}", receipt_path.display());
    assert!(
        pass,
        "kernel parity failures; see {}",
        receipt_path.display()
    );
}
