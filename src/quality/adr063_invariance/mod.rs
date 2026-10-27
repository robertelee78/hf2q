//! ADR063 Q1 test-only diagnostic oracle. No production math or service policy.
pub(crate) mod acquisition;
pub(crate) mod capture;
pub(crate) mod compare;
#[cfg(test)]
mod deepseek_actual;
pub(crate) mod deepseek_cache;
#[cfg(test)]
mod deepseek_localizer;
pub(crate) mod deepseek_runner;
pub(crate) mod family;
#[cfg(test)]
mod gemma_localizer;
pub(crate) mod gemma_prefill;
pub(crate) mod gemma_stage;
pub(crate) mod protocol;
pub(crate) mod qwen;
#[cfg(test)]
mod qwen_actual;
pub(crate) mod qwen_cache;
pub(crate) mod qwen_continuation;
#[cfg(test)]
mod qwen_localizer;
pub(crate) mod qwen_runner;
pub(crate) mod receipt;
pub(crate) mod store;
#[cfg(test)]
mod tests;
