//! ADR-063 Q4-GEMMA-PREFILL-MOE (P1): route of the Gemma 4 prefill MoE
//! down projection (design `docs/research/adr063-q4-gemma-prefill-moe-design.md`
//! §4.1).
//!
//! The down call either flattens its rows to `seq_len·top_k` tokens with
//! `top_k = 1` (`Flat`, the previous route) or passes the `[seq_len, top_k,
//! K]` activations to the slotted expert entry point (`Slotted`). Both read
//! the same activation and weight bytes, write the same output offsets and
//! list each expert's rows in the same order, so they are bitwise equal
//! (Q1 Class A). Slotted launches `top_k`× fewer expert-matmul threadgroups.
//!
//! `HF2Q_GEMMA_PREFILL_MOE_DOWN=flat|slotted` exists only for the
//! same-binary A/B; it is removed at promotion (design §4.5, §7.4).

use mlx_native::MM_ID_ROUTING_THRESHOLD;

use crate::inference::models::route_default::{is_kill_spelling, Resolved};

pub(crate) const PREFILL_MOE_DOWN_ENV: &str = "HF2Q_GEMMA_PREFILL_MOE_DOWN";

/// Shape of the Gemma prefill MoE down call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaPrefillMoeDown {
    /// `n_tokens = seq_len·top_k`, `top_k = 1` (the previous route).
    Flat,
    /// `n_tokens = seq_len`, `top_k = top_k` on the slotted entry point.
    Slotted,
}

const PREFILL_MOE_DOWN_DEFAULT: GemmaPrefillMoeDown = GemmaPrefillMoeDown::Slotted;

impl GemmaPrefillMoeDown {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Slotted => "slotted",
        }
    }
}

fn parse_switch(raw: Option<&str>) -> std::result::Result<GemmaPrefillMoeDown, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(PREFILL_MOE_DOWN_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "flat" => Ok(GemmaPrefillMoeDown::Flat),
        "slotted" => Ok(GemmaPrefillMoeDown::Slotted),
        _ => Err(format!(
            "{PREFILL_MOE_DOWN_ENV} must be flat or slotted, got {raw:?}"
        )),
    }
}

fn switch_value(raw: Option<&str>) -> GemmaPrefillMoeDown {
    parse_switch(raw).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = PREFILL_MOE_DOWN_DEFAULT.as_str(),
            "invalid Gemma prefill MoE down switch"
        );
        PREFILL_MOE_DOWN_DEFAULT
    })
}

/// `HF2Q_GEMMA_PREFILL_MOE_DOWN`, resolved once at model load.
pub(crate) fn prefill_moe_down_from_env() -> GemmaPrefillMoeDown {
    let raw = std::env::var(PREFILL_MOE_DOWN_ENV).ok();
    let down = switch_value(raw.as_deref());
    tracing::info!(route = down.as_str(), "Gemma prefill MoE down route");
    down
}

/// The value the runtime-controls surface publishes (the environment is
/// fixed for the process, so it equals the value every load resolved).
pub(crate) fn published_prefill_moe_down() -> &'static str {
    static VALUE: std::sync::LazyLock<GemmaPrefillMoeDown> = std::sync::LazyLock::new(|| {
        switch_value(std::env::var(PREFILL_MOE_DOWN_ENV).ok().as_deref())
    });
    VALUE.as_str()
}

/// Expert-ID matmul parameters of one down call:
/// `(n_tokens, top_k, slotted)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DownCall {
    pub(crate) n_tokens: u32,
    pub(crate) top_k: u32,
    pub(crate) slotted: bool,
}

/// Route rule of the down call (DeepSeek-V4's paired-prefill rule,
/// `deepseek4::ffn_forward::use_paired_expert_prefill`):
/// slotted only when the switch asks for it, the stack is block-quantized or
/// scalar GGUF (not affine), the rows take `mm_id`
/// (`seq_len > MM_ID_ROUTING_THRESHOLD`, where the slotted contract accepts
/// them) and no `HF2Q_MM_ID_ROUTING_THRESHOLD` override moves that boundary.
/// Every other case keeps the flattened call, so the `mm_id`/`mv_id` choice
/// is today's in every case (design §3.1 R3).
pub(crate) fn down_call(
    switch: GemmaPrefillMoeDown,
    seq_len: usize,
    top_k: usize,
    affine: bool,
    threshold_override_present: bool,
) -> DownCall {
    let slotted = switch == GemmaPrefillMoeDown::Slotted
        && !affine
        && !threshold_override_present
        && seq_len > MM_ID_ROUTING_THRESHOLD as usize
        // The mm_id route (and its map0 instantiations) exists only for
        // these top_k; any other top_k keeps today's flattened call.
        && matches!(top_k, 1 | 6 | 8);
    if slotted {
        DownCall {
            n_tokens: seq_len as u32,
            top_k: top_k as u32,
            slotted: true,
        }
    } else {
        DownCall {
            n_tokens: (seq_len * top_k) as u32,
            top_k: 1,
            slotted: false,
        }
    }
}

/// `HF2Q_MM_ID_ROUTING_THRESHOLD` presence, the check DeepSeek-V4 makes.
pub(crate) fn mm_id_threshold_override_present() -> bool {
    std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some()
}

/// Expert entry point of one down call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DownEntry {
    /// `dispatch_expert_matmul_id_pooled` (flattened rows).
    Flat,
    /// `dispatch_expert_matmul_id_pooled_slotted`: builds its own schedule.
    Slotted,
    /// `dispatch_expert_matmul_id_pooled_slotted_reusing_schedule`: reuses
    /// the schedule the gate/up `map0` left in the scratch (design §4.3).
    SlottedReusingSchedule,
}

/// Entry point of a down call (design §4.3, P2b). The gate/up call built a
/// reusable schedule only when both expert stacks are block-quantized (a
/// scalar stack never writes the scratch) and the call is slotted (the same
/// `mm_id` rows, ids, `n_tokens` and `top_k` as gate/up). A session the
/// body rotated between gate/up and down (profiling pre-finishes, the
/// imatrix intercept) takes the self-scheduling slotted call, so a valid
/// run never meets a key mismatch; without rotation a reuse error is a bug
/// and propagates.
pub(crate) fn down_entry(
    call: DownCall,
    both_stacks_block_quantized: bool,
    session_rotated: bool,
) -> DownEntry {
    if !call.slotted {
        DownEntry::Flat
    } else if both_stacks_block_quantized && !session_rotated {
        DownEntry::SlottedReusingSchedule
    } else {
        DownEntry::Slotted
    }
}

/// The native expert-matmul tile schedule the process resolves
/// (`HF2Q_MM_ID_SCHEDULE`), for the runtime-controls surface.
pub(crate) fn published_mm_id_schedule() -> serde_json::Value {
    serde_json::to_value(mlx_native::ggml_routing_policy_from_environment().expert_mm_schedule)
        .unwrap_or(serde_json::Value::Null)
}

pub(crate) const Q5_0_TENSOR_MM_ENV: &str = "HF2Q_Q5_0_TENSOR_MM";

/// Gemma 4 resolution of `HF2Q_Q5_0_TENSOR_MM`. Unset (or empty) selects the
/// tensor pipeline for Q5_0 dense and expert matrix multiplications, the
/// configuration of the Gemma 4 release gate (ADR-063 Q4-GEMMA-ATTN,
/// campaign `q4-gemma-release-gate-max40-v3`). `1` selects it explicitly
/// (mlx-native's own spelling, which also turns it on for every other
/// family); `0`, `off`, `false` or `stock` is the kill switch. Any other
/// value is an error: a Gemma 4 load refuses it rather than guess.
fn resolve_q5_0_tensor_mm(raw: Option<&str>) -> std::result::Result<Resolved<bool>, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(Resolved::default_value(true));
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" => Ok(Resolved::environment(true)),
        value if is_kill_spelling(value) => Ok(Resolved::environment(false)),
        _ => Err(format!(
            "{Q5_0_TENSOR_MM_ENV} must be 1 or 0/off/false/stock for a Gemma 4 model, got {raw:?}"
        )),
    }
}

fn q5_0_tensor_mm_from_env() -> std::result::Result<Resolved<bool>, String> {
    resolve_q5_0_tensor_mm(std::env::var(Q5_0_TENSOR_MM_ENV).ok().as_deref())
}

/// `environment` with the Gemma 4 Q5_0 tensor-MM resolution applied; no other
/// field changes.
fn gemma4_policy_over(
    environment: mlx_native::GgmlRoutingPolicy,
    q5_0_tensor_mm: Resolved<bool>,
) -> mlx_native::GgmlRoutingPolicy {
    mlx_native::GgmlRoutingPolicy {
        q5_0_tensor_mm: q5_0_tensor_mm.value,
        ..environment
    }
}

/// The GGML routing policy a Gemma 4 model freezes into its kernel
/// registries: the process-environment policy with `q5_0_tensor_mm` resolved
/// for this family. mlx-native's own default stays off, so no other family's
/// registry sees the Gemma default. The native device probe still decides
/// whether a tensor pipeline exists; without one the simdgroup route serves,
/// as for every other tensor-routed type. An invalid switch value is an
/// error (the load refuses).
pub(crate) fn gemma4_routing_policy() -> anyhow::Result<mlx_native::GgmlRoutingPolicy> {
    Ok(gemma4_policy_over(
        mlx_native::ggml_routing_policy_from_environment(),
        q5_0_tensor_mm_from_env().map_err(anyhow::Error::msg)?,
    ))
}

/// Audit spelling of where the Q5_0 tensor-MM value frozen in a Gemma 4
/// model's `policy` came from: the family resolution (`default` or `env`),
/// or `caller` when the model was given a context that does not carry the
/// family policy (a harness that built its own).
pub(crate) fn q5_0_tensor_mm_audit_source(policy: &mlx_native::GgmlRoutingPolicy) -> &'static str {
    audit_source(policy.q5_0_tensor_mm, q5_0_tensor_mm_from_env().ok())
}

fn audit_source(frozen: bool, family: Option<Resolved<bool>>) -> &'static str {
    match family {
        Some(family) if frozen == family.value => family.source.as_str(),
        _ => "caller",
    }
}

/// The Q5_0 tensor-MM routing value the loaded engine's family resolves, for
/// the runtime-controls surface: the Gemma 4 policy for a Gemma 4 engine,
/// mlx-native's environment policy for every other family. (A process whose
/// switch value a Gemma 4 load refuses has no Gemma 4 engine.)
pub(crate) fn published_q5_0_tensor_mm(gemma4: bool) -> bool {
    let environment = mlx_native::ggml_routing_policy_from_environment();
    if gemma4 {
        gemma4_routing_policy().map_or(environment.q5_0_tensor_mm, |policy| policy.q5_0_tensor_mm)
    } else {
        environment.q5_0_tensor_mm
    }
}

/// Logs the native routing switches a Gemma model froze, once per process at
/// the first Gemma model load: `Gemma4 native routing:
/// q5_0_tensor_mm=<true|false> mm_id_schedule=<grid|compact>` (the
/// `HF2Q_MM_ID_SCHEDULE` spellings). The environment is fixed for the
/// process, so later loads resolve the same values. Campaign checkers match
/// this line; keep its text stable.
pub(crate) fn log_native_routing(policy: &mlx_native::GgmlRoutingPolicy) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        let mm_id_schedule = match policy.expert_mm_schedule {
            mlx_native::ExpertMmSchedule::UniformGrid => "grid",
            mlx_native::ExpertMmSchedule::CompactTiles => "compact",
        };
        tracing::info!(
            "Gemma4 native routing: q5_0_tensor_mm={} mm_id_schedule={mm_id_schedule}",
            policy.q5_0_tensor_mm
        );
    });
}

/// Test-only record of every down entry point the prefill body chose on the
/// calling thread, in dispatch order (real-artifact gates assert reuse vs
/// fallback with it).
#[cfg(test)]
pub(crate) mod entry_probe {
    use super::DownEntry;
    use std::cell::RefCell;

    thread_local! {
        static ENTRIES: RefCell<Vec<DownEntry>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn record(entry: DownEntry) {
        ENTRIES.with(|entries| entries.borrow_mut().push(entry));
    }

    /// Every entry since the previous `take`.
    pub(crate) fn take() -> Vec<DownEntry> {
        ENTRIES.with(|entries| std::mem::take(&mut *entries.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use GemmaPrefillMoeDown::{Flat, Slotted};

    #[test]
    fn gemma_prefill_moe_down_switch() {
        assert_eq!(parse_switch(None), Ok(Slotted));
        assert_eq!(parse_switch(Some("")), Ok(Slotted));
        assert_eq!(parse_switch(Some(" flat ")), Ok(Flat));
        assert_eq!(parse_switch(Some("FLAT")), Ok(Flat));
        assert_eq!(parse_switch(Some("slotted")), Ok(Slotted));
        assert!(parse_switch(Some("grid")).is_err());
        assert!(parse_switch(Some("0")).is_err());
        for invalid in ["grid", "0", "1", "on"] {
            assert_eq!(switch_value(Some(invalid)), PREFILL_MOE_DOWN_DEFAULT);
        }
        assert_eq!(Flat.as_str(), "flat");
        assert_eq!(Slotted.as_str(), "slotted");
        assert!(["flat", "slotted"].contains(&published_prefill_moe_down()));
    }

    #[test]
    fn gemma_prefill_moe_down_route_selection() {
        let threshold = MM_ID_ROUTING_THRESHOLD as usize;
        assert_eq!(threshold, 32, "design R3 pins the mm_id boundary");
        let flat = |seq_len: usize| DownCall {
            n_tokens: (seq_len * 8) as u32,
            top_k: 1,
            slotted: false,
        };
        for seq_len in [1, 5, 32] {
            assert_eq!(down_call(Slotted, seq_len, 8, false, false), flat(seq_len));
        }
        for seq_len in [33, 1420] {
            assert_eq!(
                down_call(Slotted, seq_len, 8, false, false),
                DownCall {
                    n_tokens: seq_len as u32,
                    top_k: 8,
                    slotted: true
                }
            );
        }
        for seq_len in [5, 32, 33, 1420] {
            // A threshold override can send gate/up to mv_id, where the
            // slotted contract refuses: every size keeps the flattened call.
            assert_eq!(down_call(Slotted, seq_len, 8, false, true), flat(seq_len));
            // Affine stacks and the A arm keep the flattened call.
            assert_eq!(down_call(Slotted, seq_len, 8, true, false), flat(seq_len));
            assert_eq!(down_call(Flat, seq_len, 8, false, false), flat(seq_len));
        }
        for top_k in [1usize, 6, 8] {
            assert!(down_call(Slotted, 232, top_k, false, false).slotted, "top_k {top_k}");
        }
        for top_k in [2usize, 4, 7, 16] {
            assert_eq!(
                down_call(Slotted, 232, top_k, false, false),
                DownCall {
                    n_tokens: (232 * top_k) as u32,
                    top_k: 1,
                    slotted: false
                },
                "top_k {top_k} has no mm_id route: flattened call"
            );
        }
    }

    #[test]
    fn gemma_prefill_moe_down_schedule_reuse_falls_back_on_rotation() {
        let slotted = down_call(Slotted, 232, 8, false, false);
        let flat = down_call(Flat, 232, 8, false, false);
        // Not rotated, block-quantized stacks: reuse the gate/up schedule.
        assert_eq!(
            down_entry(slotted, true, false),
            DownEntry::SlottedReusingSchedule
        );
        // Rotated (profile buckets, imatrix): self-scheduling slotted call.
        assert_eq!(down_entry(slotted, true, true), DownEntry::Slotted);
        // A scalar stack wrote no schedule.
        assert_eq!(down_entry(slotted, false, false), DownEntry::Slotted);
        for rotated in [false, true] {
            assert_eq!(down_entry(flat, true, rotated), DownEntry::Flat);
        }
        assert!(matches!(
            published_mm_id_schedule().as_str(),
            Some("uniform_grid" | "compact_tiles")
        ));
    }

    /// `HF2Q_Q5_0_TENSOR_MM` for Gemma 4: unset is on, `1` is on, the four
    /// shared off spellings are the kill switch, anything else is refused.
    #[test]
    fn gemma_q5_0_tensor_mm_defaults_on_and_zero_is_the_kill_switch() {
        for raw in [None, Some(""), Some("  ")] {
            assert_eq!(
                resolve_q5_0_tensor_mm(raw),
                Ok(Resolved::default_value(true)),
                "{raw:?}"
            );
        }
        assert_eq!(
            resolve_q5_0_tensor_mm(Some("1")),
            Ok(Resolved::environment(true))
        );
        for raw in ["0", "off", "false", "stock", "OFF", " 0 "] {
            assert_eq!(
                resolve_q5_0_tensor_mm(Some(raw)),
                Ok(Resolved::environment(false)),
                "{raw:?}"
            );
        }
        for raw in ["true", "on", "2", "yes", "of"] {
            let error = resolve_q5_0_tensor_mm(Some(raw)).unwrap_err();
            assert!(error.contains(Q5_0_TENSOR_MM_ENV), "{raw:?}: {error}");
        }
    }

    /// The Gemma 4 policy differs from the environment policy in
    /// `q5_0_tensor_mm` only, and the kill switch restores the environment
    /// policy exactly (mlx-native's default, which every other family keeps).
    #[test]
    fn gemma4_routing_policy_changes_only_q5_0_tensor_mm() {
        let native = mlx_native::GgmlRoutingPolicy::default();
        assert!(!native.q5_0_tensor_mm, "mlx-native's own default stays off");
        let default_on = gemma4_policy_over(native, Resolved::default_value(true));
        assert_eq!(
            default_on,
            mlx_native::GgmlRoutingPolicy {
                q5_0_tensor_mm: true,
                ..native
            }
        );
        let killed = gemma4_policy_over(native, resolve_q5_0_tensor_mm(Some("0")).unwrap());
        assert_eq!(killed, native);
    }

    /// Admission of the Q5_0 default across quantizations: mlx-native's
    /// capability resolver gives every non-Q5_0 weight the same route with
    /// the field on or off, at every dense width, so an artifact without
    /// Q5_0 tensors (or with DWQ affine overlays, which do not use the GGML
    /// block dispatch) keeps its bits. A Q5_0 weight changes route only on
    /// the matrix-matrix widths; its few-row decode products do not move.
    #[test]
    fn gemma_q5_0_default_reroutes_only_q5_0_matrix_products() {
        use mlx_native::{
            GgmlCapabilityRequest, GgmlInvocation, GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass,
            GGML_CAPABILITY_SCHEMA_VERSION,
        };
        let off = GgmlRoutingPolicy::default();
        let on = gemma4_policy_over(off, Resolved::default_value(true));
        let route = |ggml_type, m, workload, routing| {
            let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
                schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                invocation: GgmlInvocation::DenseAuto {
                    m,
                    n: 2816,
                    k: 2816,
                },
                ggml_type,
                workload,
                routing,
            });
            assert!(
                capability.executable,
                "{ggml_type:?} m={m}: {}",
                capability.diagnostic
            );
            (capability.route, capability.requires_device_probe)
        };
        let widths = [
            (1, GgmlWorkloadClass::DecodeSingle),
            (4, GgmlWorkloadClass::ContinuousWidth),
            (8, GgmlWorkloadClass::ContinuousWidth),
            (32, GgmlWorkloadClass::Prompt),
            (512, GgmlWorkloadClass::Prompt),
        ];
        for ggml_type in [
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            for (m, workload) in widths {
                assert_eq!(
                    route(ggml_type, m, workload, on),
                    route(ggml_type, m, workload, off),
                    "{ggml_type:?} m={m}"
                );
            }
        }
        let mut rerouted = Vec::new();
        for (m, workload) in widths {
            if route(GgmlType::Q5_0, m, workload, on) != route(GgmlType::Q5_0, m, workload, off) {
                rerouted.push(m);
            }
        }
        assert_eq!(rerouted, [32, 512], "only Q5_0 matrix-matrix widths move");
    }

    /// The expert routes of the gate artifact (128 experts, top-8): the
    /// Q5_0 `ffn_down_exps` shape 2816 x 704 and the gate/up shape
    /// 704 x 2816. With the Gemma policy the Q5_0 matrix-matrix widths move
    /// from the simdgroup to the device-selected (tensor) expert kernel and
    /// the per-token decode widths keep their matvec; every other block type
    /// keeps its route at every width.
    #[test]
    fn gemma_q5_0_default_reroutes_only_q5_0_expert_matrix_products() {
        use mlx_native::{
            GgmlCapabilityRequest, GgmlExpertShape, GgmlInvocation, GgmlKernelRoute,
            GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
        };
        const DOWN: (u32, u32) = (2816, 704);
        const GATE_UP: (u32, u32) = (704, 2816);
        let off = GgmlRoutingPolicy::default();
        let on = gemma4_policy_over(off, Resolved::default_value(true));
        let route = |ggml_type: GgmlType, (n, k): (u32, u32), n_tokens: u32, routing| {
            let stride = mlx_native::ggml_matrix_bytes(ggml_type, n, k).unwrap();
            let capability = mlx_native::ggml_capability(GgmlCapabilityRequest {
                schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                invocation: GgmlInvocation::ExpertAutoAllocated {
                    shape: GgmlExpertShape {
                        n_tokens,
                        n,
                        k,
                        top_k: 8,
                        n_experts: 128,
                        expert_stride_bytes: stride,
                        ids_are_distinct_per_token: true,
                        ids_within_expert_range: true,
                    },
                },
                ggml_type,
                workload: if n_tokens == 1 {
                    GgmlWorkloadClass::DecodeSingle
                } else {
                    GgmlWorkloadClass::Prompt
                },
                routing,
            });
            assert!(
                capability.executable,
                "{ggml_type:?} {n}x{k} tokens={n_tokens}: {}",
                capability.diagnostic
            );
            capability.route
        };
        let widths = [1u32, 4, 8, 32, 64, 512];
        // K-quants need K % 256 == 0: the gate/up shape only.
        for (ggml_type, shapes) in [
            (GgmlType::Q4_0, &[DOWN, GATE_UP][..]),
            (GgmlType::Q8_0, &[DOWN, GATE_UP][..]),
            (GgmlType::Q4_K, &[GATE_UP][..]),
            (GgmlType::Q5_K, &[GATE_UP][..]),
            (GgmlType::Q6_K, &[GATE_UP][..]),
        ] {
            for &shape in shapes {
                for tokens in widths {
                    assert_eq!(
                        route(ggml_type, shape, tokens, on),
                        route(ggml_type, shape, tokens, off),
                        "{ggml_type:?} {shape:?} tokens={tokens}"
                    );
                }
            }
        }
        for shape in [DOWN, GATE_UP] {
            let mut rerouted = Vec::new();
            for tokens in widths {
                let (with, without) = (
                    route(GgmlType::Q5_0, shape, tokens, on),
                    route(GgmlType::Q5_0, shape, tokens, off),
                );
                if with != without {
                    assert_eq!(with, Some(GgmlKernelRoute::ExpertMmDeviceSelected));
                    assert_eq!(without, Some(GgmlKernelRoute::ExpertMmSimdgroup));
                    rerouted.push(tokens);
                }
            }
            assert!(
                !rerouted.is_empty()
                    && rerouted
                        .iter()
                        .all(|&tokens| tokens > off.expert_mm_threshold),
                "{shape:?}: only expert matrix-matrix widths move: {rerouted:?}"
            );
            assert!(!rerouted.contains(&1), "{shape:?}");
        }
    }

    #[test]
    fn q5_0_tensor_mm_audit_names_the_source_of_the_frozen_value() {
        let family = |value, set| {
            Some(if set {
                Resolved::environment(value)
            } else {
                Resolved::default_value(value)
            })
        };
        assert_eq!(audit_source(true, family(true, false)), "default");
        // A set variable is reported as `env`, whatever its value.
        assert_eq!(audit_source(false, family(false, true)), "env");
        assert_eq!(audit_source(true, family(true, true)), "env");
        // A context built without the family policy is reported as such.
        assert_eq!(audit_source(false, family(true, false)), "caller");
        assert_eq!(audit_source(false, None), "caller");
    }

    /// `active_controls.q5_0_tensor_mm` per family and environment. Each
    /// value runs in a fresh process so the test never mutates this
    /// process's environment. The expectation is `<gemma4><other>`.
    #[test]
    fn published_q5_0_tensor_mm_is_scoped_to_gemma4() {
        const EXPECT: &str = "HF2Q_Q5_0_TENSOR_MM_TEST_EXPECT";
        if let Some(expected) = std::env::var_os(EXPECT) {
            let expected = expected.to_str().unwrap().as_bytes().to_vec();
            assert_eq!(published_q5_0_tensor_mm(false), expected[1] == b'1');
            match expected[0] {
                // Refused: no Gemma 4 policy exists.
                b'x' => assert!(gemma4_routing_policy().is_err()),
                _ => {
                    assert_eq!(published_q5_0_tensor_mm(true), expected[0] == b'1');
                    assert_eq!(
                        gemma4_routing_policy().unwrap().q5_0_tensor_mm,
                        expected[0] == b'1'
                    );
                }
            }
            return;
        }
        let exe = std::env::current_exe().unwrap();
        for (value, expected) in [
            // Unset: on for Gemma 4 only.
            (None, "10"),
            (Some("1"), "11"),
            // Kill switch: off everywhere.
            (Some("0"), "00"),
            (Some("off"), "00"),
            // Not a value: a Gemma 4 load refuses; mlx-native reads off.
            (Some("true"), "x0"),
        ] {
            let mut child = std::process::Command::new(&exe);
            child
                .arg(
                    "serve::gemma_prefill_moe::tests::published_q5_0_tensor_mm_is_scoped_to_gemma4",
                )
                .arg("--exact")
                .env(EXPECT, expected);
            match value {
                Some(value) => child.env("HF2Q_Q5_0_TENSOR_MM", value),
                None => child.env_remove("HF2Q_Q5_0_TENSOR_MM"),
            };
            let output = child.output().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "HF2Q_Q5_0_TENSOR_MM={value:?}: {stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
