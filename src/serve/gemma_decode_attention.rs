//! ADR-063 gap program: Gemma 4 decode attention head split.
//!
//! Every d256 (sliding) decode attention dispatch, on all three decode
//! paths, runs mlx-native's NE head-split kernels
//! (`flash_attn_vec_hybrid_{batched_,}ne`, the peer's
//! `kernel_flash_attn_ext_vec` NE argument):
//! - the batched cohort dispatch,
//! - the mixed-bucket per-slot fallback,
//! - the single-slot decode.
//!
//! Both NE kernels share their per-row math, so a slot's decode is bitwise
//! the same on each path (mlx-native `test_flash_attn_vec_hybrid_ne`), as
//! it is with the stock kernels. d512 (global) layers keep the stock
//! kernels. Class B against stock.
//!
//! `HF2Q_GEMMA_DECODE_ATTN_NE` unset selects NE 4, the configuration of the
//! Gemma 4 release gate (ADR-063 Q4-GEMMA-ATTN, campaign
//! `q4-gemma-release-gate-max40-v3`). `0`, `off`, `false` or `stock` is the
//! kill switch (stock kernels); `2` and `4` select that NE.

use crate::inference::models::route_default::Resolved;

pub(crate) const DECODE_ATTN_NE_ENV: &str = "HF2Q_GEMMA_DECODE_ATTN_NE";

/// Decode attention kernel of d256 layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaDecodeAttnNe {
    /// Stock `flash_attn_vec_hybrid{,_batched}` (NE = 1).
    Stock,
    /// `flash_attn_vec_hybrid{,_batched}_ne` at NE 2 or 4.
    Ne(u32),
}

/// Unset default: the release-gate head split.
pub(crate) const DECODE_ATTN_NE_DEFAULT: GemmaDecodeAttnNe = GemmaDecodeAttnNe::Ne(4);

impl GemmaDecodeAttnNe {
    /// Stable wire name (`/hf2q/v1/runtime` active controls, Q1 packets).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Stock => "Stock",
            Self::Ne(2) => "Ne2",
            Self::Ne(4) => "Ne4",
            Self::Ne(_) => unreachable!("only NE 2 and 4 parse"),
        }
    }

    /// The NE to dispatch for a layer of `head_dim`, or `None` for stock.
    pub(crate) fn for_head_dim(self, head_dim: usize) -> Option<u32> {
        match self {
            Self::Ne(ne) if head_dim == 256 => Some(ne),
            Self::Ne(_) | Self::Stock => None,
        }
    }
}

fn parse_switch(raw: Option<&str>) -> std::result::Result<Resolved<GemmaDecodeAttnNe>, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(Resolved::default_value(DECODE_ATTN_NE_DEFAULT));
    };
    match raw.to_ascii_lowercase().as_str() {
        "0" | "1" | "off" | "false" | "stock" => {
            Ok(Resolved::environment(GemmaDecodeAttnNe::Stock))
        }
        "2" => Ok(Resolved::environment(GemmaDecodeAttnNe::Ne(2))),
        "4" => Ok(Resolved::environment(GemmaDecodeAttnNe::Ne(4))),
        _ => Err(format!(
            "{DECODE_ATTN_NE_ENV} must be 2, 4 or 0/off, got {raw:?}"
        )),
    }
}

fn switch_value(raw: Option<&str>) -> Resolved<GemmaDecodeAttnNe> {
    parse_switch(raw).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = DECODE_ATTN_NE_DEFAULT.as_str(),
            "invalid Gemma decode attention NE switch"
        );
        Resolved::default_value(DECODE_ATTN_NE_DEFAULT)
    })
}

/// `HF2Q_GEMMA_DECODE_ATTN_NE`, resolved once at model load.
pub(crate) fn decode_attn_ne_from_env() -> Resolved<GemmaDecodeAttnNe> {
    let raw = std::env::var(DECODE_ATTN_NE_ENV).ok();
    let ne = switch_value(raw.as_deref());
    tracing::info!(
        kernel = ne.value.as_str(),
        source = ne.source.as_str(),
        "Gemma decode attention head split"
    );
    ne
}

/// The value the runtime-controls surface publishes (the environment is
/// fixed for the process, so it equals the value every load resolved).
pub(crate) fn published_decode_attn_ne() -> &'static str {
    static VALUE: std::sync::LazyLock<GemmaDecodeAttnNe> = std::sync::LazyLock::new(|| {
        switch_value(std::env::var(DECODE_ATTN_NE_ENV).ok().as_deref()).value
    });
    VALUE.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::route_default::RouteSource;

    #[test]
    fn gemma_decode_attn_ne_defaults_to_the_release_gate_split() {
        for raw in [None, Some(""), Some(" ")] {
            assert_eq!(
                switch_value(raw),
                Resolved::default_value(GemmaDecodeAttnNe::Ne(4)),
                "{raw:?}"
            );
        }
        assert_eq!(DECODE_ATTN_NE_DEFAULT, GemmaDecodeAttnNe::Ne(4));
    }

    #[test]
    fn gemma_decode_attn_ne_kill_switch_restores_the_stock_kernels() {
        for raw in ["0", "1", "off", "false", "Stock", " 0 "] {
            assert_eq!(
                switch_value(Some(raw)),
                Resolved::environment(GemmaDecodeAttnNe::Stock),
                "{raw:?}"
            );
        }
    }

    #[test]
    fn gemma_decode_attn_ne_explicit_values_and_invalid_input() {
        assert_eq!(
            switch_value(Some("4")),
            Resolved::environment(GemmaDecodeAttnNe::Ne(4))
        );
        assert_eq!(
            switch_value(Some(" 2 ")),
            Resolved::environment(GemmaDecodeAttnNe::Ne(2))
        );
        assert!(parse_switch(Some("8")).is_err());
        // An invalid value warns and keeps the default; it is reported as
        // the default, not as an operator choice.
        let invalid = switch_value(Some("8"));
        assert_eq!(invalid.value, DECODE_ATTN_NE_DEFAULT);
        assert_eq!(invalid.source, RouteSource::Default);
        assert_eq!(GemmaDecodeAttnNe::Ne(4).as_str(), "Ne4");
    }

    #[test]
    fn gemma_decode_attn_ne_applies_to_d256_only() {
        assert_eq!(GemmaDecodeAttnNe::Ne(4).for_head_dim(256), Some(4));
        assert_eq!(GemmaDecodeAttnNe::Ne(4).for_head_dim(512), None);
        assert_eq!(GemmaDecodeAttnNe::Stock.for_head_dim(256), None);
        // A layer geometry without a head-split kernel keeps stock under the
        // default instead of failing.
        for head_dim in [64, 128, 192, 384, 1024] {
            assert_eq!(DECODE_ATTN_NE_DEFAULT.for_head_dim(head_dim), None);
        }
    }
}
