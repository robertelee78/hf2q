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
            Self::Environment => "environment",
            Self::Fallback => "fallback",
        }
    }
}

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
