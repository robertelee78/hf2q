//! ADR-063 gap program: Gemma 4 decode attention head split.
//!
//! `HF2Q_GEMMA_DECODE_ATTN_NE=4|2` routes every d256 (sliding) decode
//! attention dispatch, on all three decode paths, through mlx-native's NE
//! head-split kernels (`flash_attn_vec_hybrid_{batched_,}ne`, the peer's
//! `kernel_flash_attn_ext_vec` NE argument):
//! - the batched cohort dispatch,
//! - the mixed-bucket per-slot fallback,
//! - the single-slot decode.
//!
//! Both NE kernels share their per-row math, so a slot's decode is bitwise
//! the same on each path (mlx-native `test_flash_attn_vec_hybrid_ne`), as
//! it is with the stock kernels. d512 (global) layers keep the stock
//! kernels. Class B against stock; unset, `0` or `off` keeps stock.

pub(crate) const DECODE_ATTN_NE_ENV: &str = "HF2Q_GEMMA_DECODE_ATTN_NE";

/// Decode attention kernel of d256 layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmaDecodeAttnNe {
    /// Stock `flash_attn_vec_hybrid{,_batched}` (NE = 1).
    Stock,
    /// `flash_attn_vec_hybrid{,_batched}_ne` at NE 2 or 4.
    Ne(u32),
}

const DECODE_ATTN_NE_DEFAULT: GemmaDecodeAttnNe = GemmaDecodeAttnNe::Stock;

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

fn parse_switch(raw: Option<&str>) -> std::result::Result<GemmaDecodeAttnNe, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(DECODE_ATTN_NE_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "0" | "1" | "off" | "false" | "stock" => Ok(GemmaDecodeAttnNe::Stock),
        "2" => Ok(GemmaDecodeAttnNe::Ne(2)),
        "4" => Ok(GemmaDecodeAttnNe::Ne(4)),
        _ => Err(format!(
            "{DECODE_ATTN_NE_ENV} must be 2, 4 or 0/off, got {raw:?}"
        )),
    }
}

fn switch_value(raw: Option<&str>) -> GemmaDecodeAttnNe {
    parse_switch(raw).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = DECODE_ATTN_NE_DEFAULT.as_str(),
            "invalid Gemma decode attention NE switch"
        );
        DECODE_ATTN_NE_DEFAULT
    })
}

/// `HF2Q_GEMMA_DECODE_ATTN_NE`, resolved once at model load.
pub(crate) fn decode_attn_ne_from_env() -> GemmaDecodeAttnNe {
    let raw = std::env::var(DECODE_ATTN_NE_ENV).ok();
    let ne = switch_value(raw.as_deref());
    tracing::info!(kernel = ne.as_str(), "Gemma decode attention head split");
    ne
}

/// The value the runtime-controls surface publishes (the environment is
/// fixed for the process, so it equals the value every load resolved).
pub(crate) fn published_decode_attn_ne() -> &'static str {
    static VALUE: std::sync::LazyLock<GemmaDecodeAttnNe> = std::sync::LazyLock::new(|| {
        switch_value(std::env::var(DECODE_ATTN_NE_ENV).ok().as_deref())
    });
    VALUE.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_decode_attn_ne_switch() {
        for raw in [
            None,
            Some(""),
            Some(" "),
            Some("0"),
            Some("1"),
            Some("off"),
            Some("Stock"),
        ] {
            assert_eq!(switch_value(raw), GemmaDecodeAttnNe::Stock, "{raw:?}");
        }
        assert_eq!(switch_value(Some("4")), GemmaDecodeAttnNe::Ne(4));
        assert_eq!(switch_value(Some(" 2 ")), GemmaDecodeAttnNe::Ne(2));
        assert!(parse_switch(Some("8")).is_err());
        // an invalid value keeps stock rather than opting in
        assert_eq!(switch_value(Some("8")), GemmaDecodeAttnNe::Stock);
        assert_eq!(GemmaDecodeAttnNe::Ne(4).as_str(), "Ne4");
    }

    #[test]
    fn gemma_decode_attn_ne_applies_to_d256_only() {
        assert_eq!(GemmaDecodeAttnNe::Ne(4).for_head_dim(256), Some(4));
        assert_eq!(GemmaDecodeAttnNe::Ne(4).for_head_dim(512), None);
        assert_eq!(GemmaDecodeAttnNe::Stock.for_head_dim(256), None);
    }
}
