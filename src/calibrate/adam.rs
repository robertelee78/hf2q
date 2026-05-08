//! Adam optimizer for hf2q's ADR-020 Track 2 DWQ-proper training (iter 13).
//!
//! Wraps mlx-native's `dispatch_adam_update_f32` with per-parameter
//! state (m, v buffers) + step counter + bias-correction
//! pre-computation.  Each parameter is registered once by name; the
//! optimizer tracks its state across steps.
//!
//! Designed for the dwq_quantize-style flow where only a small set
//! of `scales` + `biases` per quantizable Linear is trainable —
//! Adam state is allocated once at register time and reused.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use mlx_native::ops::adam_update::dispatch_adam_update_f32;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// Adam hyper-parameters.  Matches mlx-lm + PyTorch defaults at the
/// Adam-as-classic level (β1=0.9, β2=0.999, ε=1e-8).
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamConfig {
    pub fn default_dwq() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err(anyhow!("AdamConfig: lr must be finite and > 0; got {}", self.lr));
        }
        if !(self.beta1.is_finite() && self.beta1 > 0.0 && self.beta1 < 1.0) {
            return Err(anyhow!(
                "AdamConfig: beta1 must be in (0, 1); got {}",
                self.beta1
            ));
        }
        if !(self.beta2.is_finite() && self.beta2 > 0.0 && self.beta2 < 1.0) {
            return Err(anyhow!(
                "AdamConfig: beta2 must be in (0, 1); got {}",
                self.beta2
            ));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err(anyhow!(
                "AdamConfig: eps must be finite and > 0; got {}",
                self.eps
            ));
        }
        Ok(())
    }
}

/// Per-parameter Adam state (m, v moments).
struct AdamParamState {
    /// Trainable parameter buffer (in-place updated).
    param: MlxBuffer,
    /// First-moment EMA.
    m: MlxBuffer,
    /// Second-moment EMA.
    v: MlxBuffer,
    /// Element count (cached for kernel dispatch).
    numel: usize,
}

/// Adam optimizer with per-parameter state.
///
/// Workflow:
///   let mut opt = AdamOptimizer::new(device, cfg);
///   opt.register_param("W_q.scales", scales_buf)?;
///   ... per training step ...
///   opt.step(&grads_map)?;  // grads_map: name → MlxBuffer (same shape as param)
pub struct AdamOptimizer {
    device: MlxDevice,
    cfg: AdamConfig,
    params: BTreeMap<String, AdamParamState>,
    step_count: u32,
    registry: KernelRegistry,
}

impl AdamOptimizer {
    pub fn new(device: MlxDevice, cfg: AdamConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            device,
            cfg,
            params: BTreeMap::new(),
            step_count: 0,
            registry: KernelRegistry::new(),
        })
    }

    /// Register a trainable parameter.  Allocates fresh zero-initialized
    /// `m` and `v` state buffers of the same shape.  The optimizer
    /// retains an Arc-clone of `param` for in-place updates; the
    /// caller's parameter handle continues to alias the same GPU
    /// memory (so subsequent reads observe the updates).
    pub fn register_param(&mut self, name: impl Into<String>, param: MlxBuffer) -> Result<()> {
        let name = name.into();
        if param.dtype() != DType::F32 {
            return Err(anyhow!(
                "AdamOptimizer: param '{name}' dtype {} not f32",
                param.dtype()
            ));
        }
        let numel = param.element_count();
        if numel == 0 {
            return Err(anyhow!(
                "AdamOptimizer: param '{name}' has zero elements"
            ));
        }
        if self.params.contains_key(&name) {
            return Err(anyhow!(
                "AdamOptimizer: param '{name}' already registered"
            ));
        }
        // alloc_buffer is zero-fill (ADR-015 iter61a).
        let m = self
            .device
            .alloc_buffer(numel * 4, DType::F32, param.shape().to_vec())
            .map_err(|e| anyhow!("Adam: alloc m for '{name}': {e}"))?;
        let v = self
            .device
            .alloc_buffer(numel * 4, DType::F32, param.shape().to_vec())
            .map_err(|e| anyhow!("Adam: alloc v for '{name}': {e}"))?;
        self.params.insert(
            name,
            AdamParamState {
                param,
                m,
                v,
                numel,
            },
        );
        Ok(())
    }

    /// Number of registered parameters.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Total step count (1-indexed after first call to step()).
    pub fn step_count(&self) -> u32 {
        self.step_count
    }

    /// Apply one Adam step using the gradient buffers in `grads`.
    /// Each registered parameter must have a corresponding entry in
    /// `grads` keyed by the same name; entries in `grads` for
    /// unregistered names are silently ignored (allows the caller to
    /// pass a superset of names).
    ///
    /// Returns `Err` if any registered parameter is missing from
    /// `grads`, or if a grad's shape doesn't match the param.
    pub fn step(&mut self, grads: &BTreeMap<String, MlxBuffer>) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count;
        let omb1_t = 1.0 - self.cfg.beta1.powi(t as i32);
        let omb2_t = 1.0 - self.cfg.beta2.powi(t as i32);

        // One encoder per step — all parameters update in parallel
        // dispatches that don't depend on each other.
        let mut encoder = self
            .device
            .command_encoder()
            .map_err(|e| anyhow!("Adam step: encoder: {e}"))?;

        for (name, state) in &self.params {
            let g = grads
                .get(name)
                .ok_or_else(|| anyhow!("Adam step: grad missing for '{name}'"))?;
            if g.element_count() != state.numel {
                return Err(anyhow!(
                    "Adam step: grad '{name}' element count {} != param numel {}",
                    g.element_count(),
                    state.numel
                ));
            }
            if g.dtype() != DType::F32 {
                return Err(anyhow!(
                    "Adam step: grad '{name}' dtype {} not f32",
                    g.dtype()
                ));
            }
            // Build per-call params + meta buffers.
            let mut params_buf = self
                .device
                .alloc_buffer(24, DType::F32, vec![6])
                .map_err(|e| anyhow!("Adam step '{name}': alloc params_buf: {e}"))?;
            params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("Adam step '{name}': params write: {e}"))?
                .copy_from_slice(&[
                    self.cfg.lr,
                    self.cfg.beta1,
                    self.cfg.beta2,
                    self.cfg.eps,
                    omb1_t,
                    omb2_t,
                ]);
            let mut meta_buf = self
                .device
                .alloc_buffer(4, DType::F32, vec![1])
                .map_err(|e| anyhow!("Adam step '{name}': alloc meta_buf: {e}"))?;
            meta_buf
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow!("Adam step '{name}': meta write: {e}"))?[0] =
                state.numel as u32;

            dispatch_adam_update_f32(
                &mut encoder,
                &mut self.registry,
                self.device.metal_device(),
                &state.param,
                g,
                &state.m,
                &state.v,
                &params_buf,
                &meta_buf,
            )
            .map_err(|e| anyhow!("Adam step '{name}': dispatch: {e}"))?;
        }
        encoder
            .commit_and_wait()
            .map_err(|e| anyhow!("Adam step: commit_and_wait: {e}"))?;
        Ok(())
    }

    /// Read back a registered parameter's current values to a host Vec.
    /// O(numel) memcpy — primarily for validation/debugging.
    pub fn read_param(&self, name: &str) -> Result<Vec<f32>> {
        let state = self
            .params
            .get(name)
            .ok_or_else(|| anyhow!("AdamOptimizer: no param named '{name}'"))?;
        Ok(state
            .param
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read_param '{name}': as_slice: {e}"))?
            .to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc_init(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
        let mut buf = device
            .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
            .expect("alloc");
        buf.as_mut_slice::<f32>().unwrap().copy_from_slice(data);
        buf
    }

    #[test]
    fn adam_optimizer_optimizes_quadratic_convergence() {
        // Register a single 1-element parameter at x=0; gradient
        // = 2(x-5).  After many steps x should converge to 5.
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(
            device,
            AdamConfig {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .unwrap();
        let device2 = MlxDevice::new().expect("device2");
        let p = alloc_init(&device2, &[0.0]);
        opt.register_param("x", p).unwrap();
        for _ in 0..200 {
            let x = opt.read_param("x").unwrap()[0];
            let g = 2.0 * (x - 5.0);
            let mut grads = BTreeMap::new();
            grads.insert("x".to_string(), alloc_init(&device2, &[g]));
            opt.step(&grads).unwrap();
        }
        let final_x = opt.read_param("x").unwrap()[0];
        assert!(
            (final_x - 5.0).abs() < 0.05,
            "expected x ≈ 5; got {final_x}"
        );
        assert_eq!(opt.step_count(), 200);
    }

    #[test]
    fn adam_optimizer_handles_multiple_params_independently() {
        // Two parameters with different targets; verify each
        // converges independently.
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(
            device,
            AdamConfig {
                lr: 0.1,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
        )
        .unwrap();
        let device2 = MlxDevice::new().expect("device2");
        opt.register_param("a", alloc_init(&device2, &[0.0])).unwrap();
        opt.register_param("b", alloc_init(&device2, &[0.0])).unwrap();
        assert_eq!(opt.n_params(), 2);
        // a target: 5; b target: -3.
        for _ in 0..200 {
            let a = opt.read_param("a").unwrap()[0];
            let b = opt.read_param("b").unwrap()[0];
            let mut grads = BTreeMap::new();
            grads.insert("a".to_string(), alloc_init(&device2, &[2.0 * (a - 5.0)]));
            grads.insert("b".to_string(), alloc_init(&device2, &[2.0 * (b + 3.0)]));
            opt.step(&grads).unwrap();
        }
        let a = opt.read_param("a").unwrap()[0];
        let b = opt.read_param("b").unwrap()[0];
        assert!((a - 5.0).abs() < 0.05, "a={a}");
        assert!((b + 3.0).abs() < 0.05, "b={b}");
    }

    #[test]
    fn adam_optimizer_step_count_increments() {
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(device, AdamConfig::default_dwq()).unwrap();
        assert_eq!(opt.step_count(), 0);
        let device2 = MlxDevice::new().expect("device2");
        opt.register_param("x", alloc_init(&device2, &[1.0])).unwrap();
        let mut grads = BTreeMap::new();
        grads.insert("x".to_string(), alloc_init(&device2, &[0.5]));
        opt.step(&grads).unwrap();
        assert_eq!(opt.step_count(), 1);
        opt.step(&grads).unwrap();
        assert_eq!(opt.step_count(), 2);
    }

    #[test]
    fn adam_optimizer_rejects_missing_grad() {
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(device, AdamConfig::default_dwq()).unwrap();
        let device2 = MlxDevice::new().expect("device2");
        opt.register_param("x", alloc_init(&device2, &[1.0])).unwrap();
        let grads: BTreeMap<String, MlxBuffer> = BTreeMap::new();
        match opt.step(&grads) {
            Err(e) => assert!(format!("{e}").contains("grad missing for 'x'")),
            Ok(_) => panic!("expected missing-grad error"),
        }
    }

    #[test]
    fn adam_optimizer_rejects_shape_mismatch() {
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(device, AdamConfig::default_dwq()).unwrap();
        let device2 = MlxDevice::new().expect("device2");
        opt.register_param("x", alloc_init(&device2, &[1.0, 2.0])).unwrap();
        let mut grads = BTreeMap::new();
        grads.insert("x".to_string(), alloc_init(&device2, &[1.0])); // wrong size
        match opt.step(&grads) {
            Err(e) => assert!(format!("{e}").contains("element count")),
            Ok(_) => panic!("expected shape mismatch error"),
        }
    }

    #[test]
    fn adam_optimizer_rejects_duplicate_register() {
        let device = MlxDevice::new().expect("device");
        let mut opt = AdamOptimizer::new(device, AdamConfig::default_dwq()).unwrap();
        let device2 = MlxDevice::new().expect("device2");
        opt.register_param("x", alloc_init(&device2, &[1.0])).unwrap();
        match opt.register_param("x", alloc_init(&device2, &[2.0])) {
            Err(e) => assert!(format!("{e}").contains("already registered")),
            Ok(_) => panic!("expected duplicate-register error"),
        }
    }

    #[test]
    fn adam_optimizer_config_validation() {
        let device = MlxDevice::new().expect("device");
        // lr < 0
        let cfg = AdamConfig {
            lr: -0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        };
        match AdamOptimizer::new(device, cfg) {
            Err(e) => assert!(format!("{e}").contains("lr must be finite and > 0")),
            Ok(_) => panic!("expected lr error"),
        }

        let device = MlxDevice::new().expect("device");
        // beta1 ≥ 1
        let cfg = AdamConfig {
            lr: 1e-3,
            beta1: 1.0,
            beta2: 0.999,
            eps: 1e-8,
        };
        match AdamOptimizer::new(device, cfg) {
            Err(e) => assert!(format!("{e}").contains("beta1")),
            Ok(_) => panic!("expected beta1 error"),
        }
    }
}
