//! Where a promoted (default-on) route value came from.
//!
//! v0.1.30 makes each family's release-gate configuration the default
//! (ADR-063 Q4): a switch that is unset resolves to the gate-qualified
//! route, an explicit value overrides it, and an automatic selection the
//! artifact or device cannot take falls back to the stock route. Every
//! resolver reports which of the three happened so the model-load audit line
//! names the value and its source.

/// Source of one resolved route value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteSource {
    /// The switch is unset (or empty): the family default.
    Default,
    /// The switch is set to a valid value.
    Environment,
    /// The switch is unset and the default route is not available for this
    /// artifact or device, so the stock route serves (reason logged).
    Fallback,
}

impl RouteSource {
    /// Spelling in the audit line.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Environment => "env",
            Self::Fallback => "fallback",
        }
    }
}

/// Off spellings every Gemma 4 route kill switch accepts (after trimming
/// and lower-casing).
pub(crate) fn is_kill_spelling(value: &str) -> bool {
    matches!(value, "0" | "off" | "false" | "stock")
}

/// One default-on route as the load banner, the audit line and
/// `/hf2q/v1/runtime` report it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteAudit {
    /// Control name (`gemma_decode_attn_ne`, `deepseek_row_exact_matvec`, …).
    pub name: &'static str,
    /// Resolved value, in the route's wire spelling.
    pub value: String,
    /// `default`, `env` or `fallback` ([`RouteSource::as_str`]); `caller`
    /// for a value a harness supplied outside the family resolution.
    pub source: &'static str,
}

impl RouteAudit {
    pub(crate) fn new(name: &'static str, value: impl Into<String>, source: &'static str) -> Self {
        Self {
            name,
            value: value.into(),
            source,
        }
    }
}

/// `name=value (source)` for each route, space separated.
pub(crate) fn audit_text(routes: &[RouteAudit]) -> String {
    routes
        .iter()
        .map(|route| format!("{}={} ({})", route.name, route.value, route.source))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Log target of the model-load route audit. `hf2q serve` enables it at
/// `info` by default (`main.rs`), so the line is visible without `-v`.
pub(crate) const AUDIT_TARGET: &str = "hf2q::audit";

/// One resolved route value with its source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Resolved<T> {
    pub(crate) value: T,
    pub(crate) source: RouteSource,
}

impl<T> Resolved<T> {
    pub(crate) fn default_value(value: T) -> Self {
        Self {
            value,
            source: RouteSource::Default,
        }
    }

    pub(crate) fn environment(value: T) -> Self {
        Self {
            value,
            source: RouteSource::Environment,
        }
    }

    pub(crate) fn fallback(value: T) -> Self {
        Self {
            value,
            source: RouteSource::Fallback,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_text_names_value_and_source_of_every_route() {
        let routes = [
            RouteAudit::new("a", "On", RouteSource::Default.as_str()),
            RouteAudit::new("b", "Stock", RouteSource::Environment.as_str()),
            RouteAudit::new("c", "VecHybrid", RouteSource::Fallback.as_str()),
        ];
        assert_eq!(
            audit_text(&routes),
            "a=On (default) b=Stock (env) c=VecHybrid (fallback)"
        );
    }

    #[test]
    fn kill_spellings_are_the_four_off_values() {
        for value in ["0", "off", "false", "stock"] {
            assert!(is_kill_spelling(value), "{value}");
        }
        for value in ["", "1", "no", "none", "disable", "Off"] {
            assert!(!is_kill_spelling(value), "{value}");
        }
    }
}
