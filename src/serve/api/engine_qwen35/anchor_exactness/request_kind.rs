//! Source-declared request purpose: a cancellation/reference is not fabricated
//! successful generation, and a no-advance branch cannot owe fake raw rows.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Window { BeforeCapture, Pending, ResumedDecode, FreshReset }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) enum Purpose { Generation, CancelledRequest(Window), RecoveryReference(Window) }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Recovery { Preserved, Restored { rank: usize }, Reset }
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum Outcome {
    Completed { emitted_tokens: usize },
    Cancelled { window: Window, request: u64, recovery: Recovery, recovered_tokens: usize },
    Reference { window: Window, recovered_tokens: usize },
}
impl Purpose {
    pub(super) fn for_case(case: &str, arm: &str) -> Result<Self> {
        let window = match case {
            "cancel-beforecapture" => Some(Window::BeforeCapture),
            "cancel-pending" => Some(Window::Pending),
            "cancel-resumed-decode" => Some(Window::ResumedDecode),
            "cancel-fresh-reset" => Some(Window::FreshReset),
            _ if case.starts_with("cancel-") => anyhow::bail!("unknown compiled cancellation case"),
            _ => None,
        };
        Ok(match (window, arm) {
            (None, _) => Self::Generation,
            (Some(window), "actual") => Self::CancelledRequest(window),
            (Some(window), "twin") => Self::RecoveryReference(window),
            _ => anyhow::bail!("cancellation has no such arm"),
        })
    }
    pub(super) fn raw_required(self) -> bool {
        matches!(self, Self::Generation | Self::CancelledRequest(Window::Pending | Window::ResumedDecode))
    }
}
impl Outcome {
    pub(super) fn validate(self, purpose: Purpose) -> Result<()> {
        ensure!(match (purpose, self) {
            (Purpose::Generation, Self::Completed { .. }) => true,
            (Purpose::RecoveryReference(expected), Self::Reference { window, .. }) => expected == window,
            (Purpose::CancelledRequest(expected), Self::Cancelled { window, recovery, recovered_tokens, .. }) => {
                expected == window && match window {
                    Window::BeforeCapture => recovery == Recovery::Preserved,
                    Window::Pending | Window::ResumedDecode => matches!(recovery, Recovery::Restored { rank: 0 }),
                    Window::FreshReset => recovery == Recovery::Reset && recovered_tokens == 0,
                }
            }
            _ => false,
        }, "request settlement differs from its independently declared purpose/window");
        Ok(())
    }
    pub(super) fn cancellation(window: Window, request: u64, recovery: &AnchorRecovery,
        recovered_tokens: usize) -> Result<Self> {
        let disposition = match *recovery {
            AnchorRecovery::Preserved { tokens } => {
                ensure!(tokens == recovered_tokens, "preserved recovery token count changed"); Recovery::Preserved
            }
            AnchorRecovery::Restored { rank, tokens } => {
                ensure!(tokens == recovered_tokens, "restored recovery token count changed"); Recovery::Restored { rank }
            }
            AnchorRecovery::Reset => { ensure!(recovered_tokens == 0, "reset recovery retained tokens"); Recovery::Reset }
        };
        Ok(Self::Cancelled { window, request, recovery: disposition, recovered_tokens })
    }
}
