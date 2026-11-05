//! ADR-063 Q4-GEMMA-DECODE switch (`HF2Q_GEMMA_DECODE_KERNELS`, design §4.8).
//!
//! Read once at Gemma 4 model load:
//! - unset or `0`: today's decode path, unchanged (the kill switch);
//! - `1`: the adopted exact width-4 routes on the `[N,hidden]` batched body;
//! - anything else: a startup error.
//!
//! With `1`, the process routing policy is captured once as `base`
//! (`mlx_native::ggml_routing_policy_from_environment()`), and load refuses,
//! naming the flag, unless `base` and the attention overrides hold the §3.1
//! reference. Three policies derive from `base` plus the adopted levers of
//! [`ADOPTED_LEVERS`]: the body (q/k/v/o, gate/up, down), the lm head and the
//! routed experts. Weight types outside §3.1 keep their stock route. Other
//! families and the serial decode path never see these policies.
use std::sync::OnceLock;

use anyhow::{bail, Result};
use mlx_native::{GgmlRoutingPolicy, GgmlType};

pub(crate) const ENV: &str = "HF2Q_GEMMA_DECODE_KERNELS";

/// Attention-reference overrides (§3.1): the batched flash must stay on and
/// the per-slot `(nwg, nsg)` overrides unset.
const BATCHED_FLASH_ENV: &str = "HF2Q_BATCHED_FLASH";
const HYBRID_NWG_ENV: &str = "HF2Q_HYBRID_NWG";
const TQ_NSG_ENV: &str = "HF2Q_TQ_NSG";
/// KV regime selector (`INVESTIGATION_ENV.hybrid_kv`, default on); the
/// GQA mixed-bucket lever only covers the hybrid F16-K regime.
const HYBRID_KV_ENV: &str = "HF2Q_HYBRID_KV";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Switch {
    /// Unset or `0`: stock decode kernels.
    Stock,
    /// `1`: the adopted routes.
    Adopted,
}

pub(crate) fn parse_switch(raw: Option<&str>) -> Result<Switch> {
    match raw {
        None | Some("0") => Ok(Switch::Stock),
        Some("1") => Ok(Switch::Adopted),
        Some(other) => bail!("{ENV} must be 0, 1 or unset, got {other:?}"),
    }
}

/// Decode attention route under the adopted set (design §4.5), consumed by
/// the hybrid-regime flash gate of the batched body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionLever {
    /// `flash_attn_vec_hybrid_batched` plus the per-slot fallback (stock).
    VecHybrid,
    /// Lever A (A1 GQA-grouped + A2 mixed-bucket):
    /// `flash_attn_vec_hybrid_batched_gqa` on every step, each slot at its own
    /// stock `(nwg, nsg)` bucket; no per-slot fallback for mixed-bucket steps.
    /// The group size per layer type is the largest legal G not above its cap
    /// (G = 1 keeps per-head loads but still runs A2). Caps are frozen by M0;
    /// geometry never changes bits.
    GqaMixedBucket { sliding_group: u32, global_group: u32 },
}

impl AttentionLever {
    pub(crate) fn name(self) -> &'static str {
        match self {
            AttentionLever::VecHybrid => "vec_hybrid",
            AttentionLever::GqaMixedBucket { .. } => "gqa_mixed_bucket",
        }
    }

    /// Group-size cap for a layer of head dimension `head_dim` (256 sliding,
    /// 512 global); `None` for the stock lever.
    pub(crate) fn group_cap(self, head_dim: u32) -> Option<u32> {
        match self {
            AttentionLever::VecHybrid => None,
            AttentionLever::GqaMixedBucket { sliding_group, global_group } => {
                Some(if head_dim > 256 { global_group } else { sliding_group })
            }
        }
    }
}

/// The adopted lever set; each field maps to one `GgmlRoutingPolicy` lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Levers {
    pub dense_row_tiled_q4k: bool,
    pub dense_row_tiled_q6k: bool,
    pub dense_mn_q5_0: bool,
    pub dense_mn_q8_0: bool,
    /// `dense_row_tiled_q6k` on the Q6_K lm head.
    pub lm_head_row_tiled_q6k: bool,
    pub expert_grouped_q4k: bool,
    pub expert_grouped_q5_0: bool,
    pub expert_grouped_q8_0: bool,
    pub attention: AttentionLever,
}

/// THE adopted lever set (the one place it is decided). All dense and expert
/// levers are on until M0 (design §7.5) settles adoption.
pub(crate) const ADOPTED_LEVERS: Levers = Levers {
    dense_row_tiled_q4k: true,
    dense_row_tiled_q6k: true,
    dense_mn_q5_0: true,
    dense_mn_q8_0: true,
    lm_head_row_tiled_q6k: true,
    expert_grouped_q4k: true,
    expert_grouped_q5_0: true,
    expert_grouped_q8_0: true,
    // Group caps are provisional until the V1 bench freezes them (design
    // §7.5): warm S0-b measured G = 8 on global layers at NSG 1 2.3-2.4x
    // slower than the vec kernel, so the cap starts at 2 for both types.
    attention: AttentionLever::GqaMixedBucket { sliding_group: 2, global_group: 2 },
};

impl Levers {
    /// Names of the enabled levers, in policy-field order.
    pub(crate) fn names(&self) -> Vec<&'static str> {
        [
            (self.dense_row_tiled_q4k, "dense_row_tiled_q4k"),
            (self.dense_row_tiled_q6k, "dense_row_tiled_q6k"),
            (self.dense_mn_q5_0, "dense_mn_q5_0"),
            (self.dense_mn_q8_0, "dense_mn_q8_0"),
            (self.lm_head_row_tiled_q6k, "lm_head_row_tiled_q6k"),
            (self.expert_grouped_q4k, "expert_grouped_q4k"),
            (self.expert_grouped_q5_0, "expert_grouped_q5_0"),
            (self.expert_grouped_q8_0, "expert_grouped_q8_0"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect()
    }
}

/// Raw values of the attention-reference overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AttentionEnv {
    pub batched_flash: Option<String>,
    pub hybrid_nwg: Option<String>,
    pub tq_nsg: Option<String>,
    /// `HF2Q_HYBRID_KV` turned the hybrid regime off (full-TQ decode), as
    /// engine.rs selects the batched decode `BatchedKvRegime`.
    pub full_tq: bool,
}

impl AttentionEnv {
    pub(crate) fn from_environment() -> Self {
        Self {
            batched_flash: std::env::var(BATCHED_FLASH_ENV).ok(),
            hybrid_nwg: std::env::var(HYBRID_NWG_ENV).ok(),
            tq_nsg: std::env::var(TQ_NSG_ENV).ok(),
            full_tq: !crate::debug::INVESTIGATION_ENV.hybrid_kv,
        }
    }
}

/// Adopted routes: the captured base and the three derived policies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GemmaDecodeKernels {
    base: GgmlRoutingPolicy,
    levers: Levers,
    body_policy: GgmlRoutingPolicy,
    lm_head_policy: GgmlRoutingPolicy,
    expert_policy: GgmlRoutingPolicy,
}

impl GemmaDecodeKernels {
    /// Refuse unless `base` and `env` hold the §3.1 reference, then derive the
    /// body, lm-head and expert policies from `base` plus `levers`.
    pub(crate) fn adopt(base: GgmlRoutingPolicy, env: &AttentionEnv, levers: Levers) -> Result<Self> {
        let mut refused: Vec<String> = Vec::new();
        let mut need = |ok: bool, what: String| {
            if !ok {
                refused.push(what);
            }
        };
        need(
            base.dense_q6k_mv_nr2,
            "dense_q6k_mv_nr2 must be true (HF2Q_Q6K_MV_NR2 not 0)".into(),
        );
        need(
            base.dense_q8_0_mv_nr2,
            "dense_q8_0_mv_nr2 must be true (HF2Q_Q8_0_MV_NR2 not 0)".into(),
        );
        need(
            !base.expert_q8_0_mv_nr2,
            "expert_q8_0_mv_nr2 must be false (HF2Q_Q8_0_ID_MV_NR2 not 1)".into(),
        );
        need(
            !base.dense_decode_mv_ext,
            "dense_decode_mv_ext must be false (HF2Q_DECODE_MV_EXT not 1)".into(),
        );
        need(
            env.batched_flash.as_deref() != Some("0"),
            format!("{BATCHED_FLASH_ENV} must not be 0"),
        );
        need(
            env.hybrid_nwg.is_none(),
            format!("{HYBRID_NWG_ENV} must be unset, got {:?}", env.hybrid_nwg),
        );
        need(
            env.tq_nsg.is_none(),
            format!("{TQ_NSG_ENV} must be unset, got {:?}", env.tq_nsg),
        );
        need(
            !(env.full_tq && matches!(levers.attention, AttentionLever::GqaMixedBucket { .. })),
            format!(
                "{HYBRID_KV_ENV} must keep the hybrid KV regime on: attention lever {} has no full-TQ route",
                levers.attention.name()
            ),
        );
        if !refused.is_empty() {
            bail!(
                "{ENV}=1 refused: the exact decode routes need the stock reference kernels: {}",
                refused.join("; ")
            );
        }
        let body_policy = GgmlRoutingPolicy {
            dense_row_tiled_q4k: levers.dense_row_tiled_q4k,
            dense_row_tiled_q6k: levers.dense_row_tiled_q6k,
            dense_mn_q5_0: levers.dense_mn_q5_0,
            dense_mn_q8_0: levers.dense_mn_q8_0,
            ..base
        };
        let lm_head_policy = GgmlRoutingPolicy {
            dense_row_tiled_q6k: levers.lm_head_row_tiled_q6k,
            ..base
        };
        let expert_policy = GgmlRoutingPolicy {
            expert_grouped_q4k: levers.expert_grouped_q4k,
            expert_grouped_q5_0: levers.expert_grouped_q5_0,
            expert_grouped_q8_0: levers.expert_grouped_q8_0,
            ..base
        };
        Ok(Self {
            base,
            levers,
            body_policy,
            lm_head_policy,
            expert_policy,
        })
    }

    /// Body policy for a dense q/k/v/o, gate/up or down weight; `None` (stock
    /// route) outside Q4_K, Q6_K, Q5_0 and Q8_0.
    pub(crate) fn body_policy_for(&self, ty: GgmlType) -> Option<&GgmlRoutingPolicy> {
        matches!(ty, GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q5_0 | GgmlType::Q8_0)
            .then_some(&self.body_policy)
    }

    /// Lm-head policy; `None` (stock route) unless the head is Q6_K and its
    /// lever is adopted.
    pub(crate) fn lm_head_policy_for(&self, ty: GgmlType) -> Option<&GgmlRoutingPolicy> {
        (ty == GgmlType::Q6_K && self.levers.lm_head_row_tiled_q6k).then_some(&self.lm_head_policy)
    }

    /// Routed-expert policy; `None` (stock route) outside Q4_K, Q5_0 and Q8_0.
    pub(crate) fn expert_policy_for(&self, ty: GgmlType) -> Option<&GgmlRoutingPolicy> {
        matches!(ty, GgmlType::Q4_K | GgmlType::Q5_0 | GgmlType::Q8_0)
            .then_some(&self.expert_policy)
    }

    /// Attention route of the adopted set.
    pub(crate) fn attention(&self) -> AttentionLever {
        self.levers.attention
    }
}

/// Resolve the switch; `base` and `env` are read only when it is `1`.
pub(crate) fn resolve(
    raw: Option<&str>,
    base: impl FnOnce() -> GgmlRoutingPolicy,
    env: impl FnOnce() -> AttentionEnv,
) -> Result<Option<GemmaDecodeKernels>> {
    match parse_switch(raw)? {
        Switch::Stock => Ok(None),
        Switch::Adopted => GemmaDecodeKernels::adopt(base(), &env(), ADOPTED_LEVERS).map(Some),
    }
}

static RUNTIME_REPORT: OnceLock<serde_json::Value> = OnceLock::new();

/// Read the switch from the process environment (once per Gemma model load),
/// log the banner and publish the runtime-control report.
pub(crate) fn resolve_at_load() -> Result<Option<GemmaDecodeKernels>> {
    let raw = std::env::var(ENV).ok();
    let kernels = resolve(
        raw.as_deref(),
        mlx_native::ggml_routing_policy_from_environment,
        AttentionEnv::from_environment,
    )?;
    tracing::info!("{}", banner(kernels.as_ref()));
    let _ = RUNTIME_REPORT.set(report(kernels.as_ref()));
    Ok(kernels)
}

/// The runtime-control report of the loaded Gemma model; `null` before a
/// Gemma 4 load.
pub(crate) fn runtime_report() -> serde_json::Value {
    RUNTIME_REPORT.get().cloned().unwrap_or(serde_json::Value::Null)
}

fn reference_flags(base: &GgmlRoutingPolicy) -> String {
    format!(
        "dense_q6k_mv_nr2={} dense_q8_0_mv_nr2={} expert_q8_0_mv_nr2={} dense_decode_mv_ext={}",
        base.dense_q6k_mv_nr2,
        base.dense_q8_0_mv_nr2,
        base.expert_q8_0_mv_nr2,
        base.dense_decode_mv_ext
    )
}

/// The attention lever with its group caps, for the banner and report.
pub(crate) fn banner_attention(lever: AttentionLever) -> String {
    match lever {
        AttentionLever::VecHybrid => lever.name().to_string(),
        AttentionLever::GqaMixedBucket { sliding_group, global_group } => format!(
            "{}(sliding_g<={sliding_group},global_g<={global_group})",
            lever.name()
        ),
    }
}

/// One-line load banner: switch, adopted lever set and base reference flags.
pub(crate) fn banner(kernels: Option<&GemmaDecodeKernels>) -> String {
    match kernels {
        None => format!("gemma decode kernels: {ENV}=0 (stock decode kernels)"),
        Some(k) => format!(
            "gemma decode kernels: {ENV}=1 levers=[{}] attention={} base: {}",
            k.levers.names().join(","),
            banner_attention(k.attention()),
            reference_flags(&k.base)
        ),
    }
}

/// Machine-readable runtime-control record.
pub(crate) fn report(kernels: Option<&GemmaDecodeKernels>) -> serde_json::Value {
    match kernels {
        None => serde_json::json!({"switch": "0"}),
        Some(k) => serde_json::json!({
            "switch": "1",
            "levers": k.levers.names(),
            "attention": banner_attention(k.attention()),
            "base": {
                "dense_q6k_mv_nr2": k.base.dense_q6k_mv_nr2,
                "dense_q8_0_mv_nr2": k.base.dense_q8_0_mv_nr2,
                "expert_q8_0_mv_nr2": k.base.expert_q8_0_mv_nr2,
                "dense_decode_mv_ext": k.base.dense_decode_mv_ext,
            },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adopted() -> GemmaDecodeKernels {
        GemmaDecodeKernels::adopt(GgmlRoutingPolicy::default(), &AttentionEnv::default(), ADOPTED_LEVERS)
            .expect("stock defaults hold the reference")
    }

    #[test]
    fn decode_kernels_switch_parsing() {
        assert_eq!(parse_switch(None).unwrap(), Switch::Stock);
        assert_eq!(parse_switch(Some("0")).unwrap(), Switch::Stock);
        assert_eq!(parse_switch(Some("1")).unwrap(), Switch::Adopted);
        for bad in ["", "2", "true", "on", " 1", "01"] {
            let e = parse_switch(Some(bad)).unwrap_err().to_string();
            assert!(e.contains(ENV), "{bad:?}: {e}");
        }
    }

    #[test]
    fn decode_kernels_stock_reads_nothing() {
        let k = resolve(Some("0"), || panic!("base read"), || panic!("env read")).unwrap();
        assert!(k.is_none());
        assert!(resolve(None, || panic!("base read"), || panic!("env read")).unwrap().is_none());
        assert!(resolve(Some("x"), || panic!("base read"), || panic!("env read")).is_err());
        let k = resolve(Some("1"), GgmlRoutingPolicy::default, AttentionEnv::default).unwrap();
        assert_eq!(k, Some(adopted()));
    }

    #[test]
    fn decode_kernels_refusal_names_each_reference_flag() {
        let ok = GgmlRoutingPolicy::default();
        let cases: Vec<(GgmlRoutingPolicy, AttentionEnv, &str)> = vec![
            (GgmlRoutingPolicy { dense_q6k_mv_nr2: false, ..ok }, AttentionEnv::default(), "dense_q6k_mv_nr2"),
            (GgmlRoutingPolicy { dense_q8_0_mv_nr2: false, ..ok }, AttentionEnv::default(), "dense_q8_0_mv_nr2"),
            (GgmlRoutingPolicy { expert_q8_0_mv_nr2: true, ..ok }, AttentionEnv::default(), "expert_q8_0_mv_nr2"),
            (GgmlRoutingPolicy { dense_decode_mv_ext: true, ..ok }, AttentionEnv::default(), "dense_decode_mv_ext"),
            (ok, AttentionEnv { batched_flash: Some("0".into()), ..Default::default() }, BATCHED_FLASH_ENV),
            (ok, AttentionEnv { hybrid_nwg: Some("16".into()), ..Default::default() }, HYBRID_NWG_ENV),
            (ok, AttentionEnv { tq_nsg: Some("2".into()), ..Default::default() }, TQ_NSG_ENV),
        ];
        for (base, env, flag) in cases {
            let e = GemmaDecodeKernels::adopt(base, &env, ADOPTED_LEVERS).unwrap_err().to_string();
            assert!(e.contains(ENV) && e.contains(flag), "{flag}: {e}");
        }
        // Explicitly on batched flash is the reference.
        let env = AttentionEnv { batched_flash: Some("1".into()), ..Default::default() };
        assert!(GemmaDecodeKernels::adopt(ok, &env, ADOPTED_LEVERS).is_ok());
    }

    #[test]
    fn decode_kernels_refuses_gqa_mixed_bucket_under_full_tq() {
        let full_tq = AttentionEnv { full_tq: true, ..Default::default() };
        let e = GemmaDecodeKernels::adopt(GgmlRoutingPolicy::default(), &full_tq, ADOPTED_LEVERS)
            .unwrap_err()
            .to_string();
        assert!(e.contains(ENV) && e.contains(HYBRID_KV_ENV) && e.contains("gqa_mixed_bucket"), "{e}");
        // The stock attention lever has no full-TQ restriction.
        let stock_attn = Levers { attention: AttentionLever::VecHybrid, ..ADOPTED_LEVERS };
        assert!(GemmaDecodeKernels::adopt(GgmlRoutingPolicy::default(), &full_tq, stock_attn).is_ok());
    }

    #[test]
    fn decode_kernels_derives_three_policies_from_base() {
        // A non-default but reference-holding base must carry through.
        let base = GgmlRoutingPolicy {
            dense_decode_mvn: !GgmlRoutingPolicy::default().dense_decode_mvn,
            expert_mm_threshold: 7,
            ..GgmlRoutingPolicy::default()
        };
        let k = GemmaDecodeKernels::adopt(base, &AttentionEnv::default(), ADOPTED_LEVERS).unwrap();
        let body = *k.body_policy_for(GgmlType::Q4_K).unwrap();
        assert_eq!(
            body,
            GgmlRoutingPolicy {
                dense_row_tiled_q4k: true,
                dense_row_tiled_q6k: true,
                dense_mn_q5_0: true,
                dense_mn_q8_0: true,
                ..base
            }
        );
        let head = *k.lm_head_policy_for(GgmlType::Q6_K).unwrap();
        assert_eq!(head, GgmlRoutingPolicy { dense_row_tiled_q6k: true, ..base });
        let expert = *k.expert_policy_for(GgmlType::Q8_0).unwrap();
        assert_eq!(
            expert,
            GgmlRoutingPolicy {
                expert_grouped_q4k: true,
                expert_grouped_q5_0: true,
                expert_grouped_q8_0: true,
                ..base
            }
        );
        assert!(!body.dense_row_tiled_m1 && !head.dense_row_tiled_m1);
        assert!(matches!(k.attention(), AttentionLever::GqaMixedBucket { .. }));

        let none = Levers {
            dense_row_tiled_q4k: false,
            dense_row_tiled_q6k: false,
            dense_mn_q5_0: false,
            dense_mn_q8_0: false,
            lm_head_row_tiled_q6k: false,
            expert_grouped_q4k: false,
            expert_grouped_q5_0: false,
            expert_grouped_q8_0: false,
            attention: AttentionLever::VecHybrid,
        };
        let k = GemmaDecodeKernels::adopt(base, &AttentionEnv::default(), none).unwrap();
        assert_eq!(*k.body_policy_for(GgmlType::Q6_K).unwrap(), base);
        assert_eq!(*k.expert_policy_for(GgmlType::Q4_K).unwrap(), base);
        assert!(k.lm_head_policy_for(GgmlType::Q6_K).is_none(), "unadopted head stays stock");
        assert_eq!(k.attention(), AttentionLever::VecHybrid, "the stock attention lever is selectable");
    }

    #[test]
    fn decode_kernels_refuse_types_outside_reference() {
        let k = adopted();
        for ty in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q5_0, GgmlType::Q8_0] {
            assert!(k.body_policy_for(ty).is_some(), "{ty:?}");
        }
        for ty in [GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q5_K, GgmlType::Q4_0, GgmlType::Q5_1] {
            assert!(k.body_policy_for(ty).is_none(), "{ty:?}");
            assert!(k.expert_policy_for(ty).is_none(), "{ty:?}");
            assert!(k.lm_head_policy_for(ty).is_none(), "{ty:?}");
        }
        assert!(k.expert_policy_for(GgmlType::Q6_K).is_none(), "Q6_K experts stay stock");
        assert!(k.lm_head_policy_for(GgmlType::Q8_0).is_none(), "non-Q6_K heads stay stock");
    }

    #[test]
    fn decode_kernels_banner_and_report() {
        assert_eq!(banner(None), "gemma decode kernels: HF2Q_GEMMA_DECODE_KERNELS=0 (stock decode kernels)");
        assert_eq!(
            banner(Some(&adopted())),
            "gemma decode kernels: HF2Q_GEMMA_DECODE_KERNELS=1 levers=[dense_row_tiled_q4k,\
             dense_row_tiled_q6k,dense_mn_q5_0,dense_mn_q8_0,lm_head_row_tiled_q6k,\
             expert_grouped_q4k,expert_grouped_q5_0,expert_grouped_q8_0] \
             attention=gqa_mixed_bucket(sliding_g<=2,global_g<=2) \
             base: dense_q6k_mv_nr2=true dense_q8_0_mv_nr2=true expert_q8_0_mv_nr2=false \
             dense_decode_mv_ext=false"
        );
        assert_eq!(report(None), serde_json::json!({"switch": "0"}));
        let r = report(Some(&adopted()));
        assert_eq!(r["switch"], "1");
        assert_eq!(r["levers"].as_array().unwrap().len(), 8);
        assert_eq!(r["attention"], "gqa_mixed_bucket(sliding_g<=2,global_g<=2)");
        assert_eq!(r["base"]["dense_q6k_mv_nr2"], true);
        assert_eq!(r["base"]["expert_q8_0_mv_nr2"], false);
        // A set carrying the stock attention lever reports it.
        let stock_attn = Levers { attention: AttentionLever::VecHybrid, ..ADOPTED_LEVERS };
        let k = GemmaDecodeKernels::adopt(GgmlRoutingPolicy::default(), &AttentionEnv::default(), stock_attn)
            .unwrap();
        assert!(banner(Some(&k)).contains(" attention=vec_hybrid "), "{}", banner(Some(&k)));
        assert_eq!(report(Some(&k))["attention"], "vec_hybrid");
    }

    #[test]
    fn decode_kernels_attention_lever_names() {
        assert_eq!(AttentionLever::VecHybrid.name(), "vec_hybrid");
        assert_eq!(
            AttentionLever::GqaMixedBucket { sliding_group: 2, global_group: 2 }.name(),
            "gqa_mixed_bucket"
        );
        assert!(matches!(ADOPTED_LEVERS.attention, AttentionLever::GqaMixedBucket { .. }));
    }

    #[test]
    fn decode_kernels_attention_group_caps() {
        let lever = AttentionLever::GqaMixedBucket { sliding_group: 2, global_group: 4 };
        assert_eq!(lever.group_cap(256), Some(2));
        assert_eq!(lever.group_cap(512), Some(4));
        assert_eq!(AttentionLever::VecHybrid.group_cap(256), None);
        assert_eq!(
            banner_attention(lever),
            "gqa_mixed_bucket(sliding_g<=2,global_g<=4)"
        );
    }
}
