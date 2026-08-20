use std::fmt;
use std::num::NonZeroU64;
use std::path::Path;

use super::fs::{self, RuntimeConfigBinding};
use super::SetupError;

/// A read-only, setup-authenticated session-cache policy decision.
///
/// This type deliberately cannot construct a cache persistor. It prevents the
/// setup wire's zero-disabled value from reaching legacy zero-unlimited APIs.
pub(crate) enum SessionCachePolicyAuthorization {
    Absent,
    Disabled(DisabledSessionCacheAuthorization),
    Enabled(EnabledSessionCacheAuthorization),
}

pub(crate) struct DisabledSessionCacheAuthorization {
    binding: RuntimeConfigBinding,
}

pub(crate) struct EnabledSessionCacheAuthorization {
    limit_bytes: NonZeroU64,
    binding: RuntimeConfigBinding,
}

pub(crate) fn authorize_session_cache_policy(
    state_root: &Path,
) -> Result<SessionCachePolicyAuthorization, SetupError> {
    let Some((config, binding)) = fs::authorize_runtime_config(state_root)? else {
        return Ok(SessionCachePolicyAuthorization::Absent);
    };
    match NonZeroU64::new(config.session_cache.limit_bytes) {
        Some(limit_bytes) => Ok(SessionCachePolicyAuthorization::Enabled(
            EnabledSessionCacheAuthorization {
                limit_bytes,
                binding,
            },
        )),
        None => Ok(SessionCachePolicyAuthorization::Disabled(
            DisabledSessionCacheAuthorization { binding },
        )),
    }
}

impl DisabledSessionCacheAuthorization {
    pub(crate) fn revalidate(&self) -> Result<(), SetupError> {
        self.binding.revalidate()
    }
}

impl EnabledSessionCacheAuthorization {
    #[cfg(test)]
    pub(super) const fn limit_bytes(&self) -> NonZeroU64 {
        self.limit_bytes
    }

    #[cfg(test)]
    pub(super) fn revalidate(&self) -> Result<(), SetupError> {
        self.binding.revalidate()
    }

    #[cfg(test)]
    pub(super) fn retained_regular_files_are_read_only_for_test(&self) -> Result<bool, SetupError> {
        self.binding.retained_regular_files_are_read_only_for_test()
    }
}

impl fmt::Debug for SessionCachePolicyAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => formatter.write_str("SessionCachePolicyAuthorization::Absent"),
            Self::Disabled(_) => {
                formatter.write_str("SessionCachePolicyAuthorization::Disabled(<redacted>)")
            }
            Self::Enabled(_) => {
                formatter.write_str("SessionCachePolicyAuthorization::Enabled(<redacted>)")
            }
        }
    }
}

impl fmt::Debug for DisabledSessionCacheAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DisabledSessionCacheAuthorization(<redacted>)")
    }
}

impl fmt::Debug for EnabledSessionCacheAuthorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("EnabledSessionCacheAuthorization(<redacted>)")
    }
}
