//! Retained-reference-only GPU submissions. No observer or raw Metal escape.
//! Requires the public native 0.15.12 mode authority; absent in 0.15.11.
use super::*;
use mlx_native::CommandEncoder;
use std::cell::RefCell;
// Selected deferral: one Delta CB per linear layer/owner.
// Max48*4=192 per invocation; exactly zero or one per existing row fence.
const PER_FENCE: usize = 1;
const PER_INVOCATION: usize = 192;
struct Pending {
    encoders: Vec<CommandEncoder>,
    submitted: usize,
}
thread_local! {static PENDING:RefCell<Option<Pending>>=const{RefCell::new(None)};}

pub(super) fn active() -> bool {
    PENDING.with(|p| p.try_borrow().map(|p| p.is_some()).unwrap_or(false))
}
/// Exists only around run's actual shared graph, NEVER checked_fixture.
pub(super) struct GraphDeferralScope {
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl GraphDeferralScope {
    pub(super) fn enter() -> Result<Self> {
        ensure!(
            checked_active(),
            "graph deferral requires checked causal scope"
        );
        require_retained_mode(mlx_native::command_buffer_uses_unretained_references())?;
        begin()?;
        Ok(Self {
            _same_thread: std::marker::PhantomData,
        })
    }
}
impl Drop for GraphDeferralScope {
    fn drop(&mut self) {
        end_scope();
    }
}

pub(super) fn require_retained_mode(unretained: bool) -> Result<()> {
    ensure!(
        !unretained,
        "retained-encoder spike refuses actual native unretained mode before mutation"
    );
    Ok(())
}
fn admit(current: usize, total: usize) -> Result<()> {
    ensure!(
        current < PER_FENCE && total < PER_INVOCATION,
        "bounded retained submission capacity exhausted"
    );
    Ok(())
}
fn begin() -> Result<()> {
    PENDING.with(|p| -> Result<()> {
        let mut p = p.try_borrow_mut().context("reentrant retained scope")?;
        ensure!(p.is_none(), "previous retained scope remains");
        let mut encoders = Vec::new();
        encoders
            .try_reserve_exact(PER_FENCE)
            .context("reserve exact retained encoder bound")?;
        *p = Some(Pending {
            encoders,
            submitted: 0,
        });
        Ok(())
    })
}
pub(super) fn submit(enc: CommandEncoder, label: &'static str) -> Result<()> {
    // This constructor mode is fixed by SAME native authority before run begins.
    // Ownership enters bounded storage BEFORE committing (including panic paths).
    PENDING.with(|p| -> Result<()> {
        let mut p = p
            .try_borrow_mut()
            .context("reentrant retained submission")?;
        let q = p.as_mut().context("retained submission outside scope")?;
        admit(q.encoders.len(), q.submitted)?;
        q.encoders.push(enc);
        q.submitted += 1;
        q.encoders
            .last_mut()
            .context("owned encoder disappeared")?
            .commit_labeled(label);
        Ok(())
    })
}
// Actual production check loop; controls exercise the same all-results behavior.
fn check_all<T>(entries: &[T], mut check: impl FnMut(&T) -> Result<()>) -> Result<()> {
    let mut first = None;
    for e in entries {
        if let Err(error) = check(e) {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    match first {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
fn release_after_checks<T>(
    entries: &mut Vec<T>,
    check: impl FnMut(&T) -> Result<()>,
) -> Result<()> {
    check_all(entries, check)?;
    entries.clear(); // Only exact successful checks authorize owner retirement.
    Ok(())
}
/// Call ONLY after existing row/layer anchor succeeded; checks exact earlier CBs.
/// A healthy anchor never substitutes for any earlier Result.
pub(super) fn after_healthy_fence() -> Result<()> {
    PENDING.with(|p| -> Result<()> {
        let mut p = p.try_borrow_mut().context("reentrant retained fence")?;
        let q = p.as_mut().context("retained fence outside scope")?;
        release_after_checks(&mut q.encoders, |enc| {
            enc.wait_until_completed().map_err(Into::into)
        })
        .context("earlier retained helper CB failed despite healthy fence")
    })
}
pub(super) fn require_empty() -> Result<()> {
    PENDING.with(|p| -> Result<()> {
        let p = p.try_borrow().context("retained scope borrowed")?;
        ensure!(
            p.as_ref()
                .context("missing retained scope")?
                .encoders
                .is_empty(),
            "unchecked helper submission at completion"
        );
        Ok(())
    })
}
fn end_scope() {
    PENDING.with(|p| {
        // A reentrant unwind must never release an uncertain command owner.
        let Ok(mut p) = p.try_borrow_mut() else {
            failed();
            return;
        };
        if let Some(q) = p.take() {
            if !q.encoders.is_empty() {
                failed();
                std::mem::forget(q);
            }
        }
    });
}
#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn scalar_checked_fixture_never_starts_graph_deferral() {
        checked_fixture(|| {
            assert!(checked_active());
            assert!(!active());
            Ok(())
        })
        .unwrap();
        assert!(!checked_active());
        assert!(!active());
    }
    #[test]
    fn graph_authority_is_separate_and_scoped() {
        let checked = CheckedScope::enter().unwrap();
        assert!(!active());
        {
            let graph = GraphDeferralScope::enter().unwrap();
            assert!(active());
            assert!(GraphDeferralScope::enter().is_err());
            drop(graph);
        }
        assert!(!active());
        assert!(checked_active());
        drop(checked);
    }
    #[test]
    fn actual_mode_guard_refuses_unretained() {
        assert!(require_retained_mode(false).is_ok());
        assert!(require_retained_mode(true).is_err());
    }
    #[test]
    fn admission_is_before_finite_commit_boundary() {
        assert!(admit(0, 0).is_ok());
        assert!(admit(0, 191).is_ok());
        assert!(admit(1, 0).is_err());
        assert!(admit(0, 192).is_err());
    }
    #[test]
    fn actual_check_loop_checks_every_result_and_keeps_first_error() {
        let mut seen = Vec::new();
        let result = check_all(&[0, 1, 2], |i| {
            seen.push(*i);
            if *i != 1 {
                anyhow::bail!("CB{i}");
            }
            Ok(())
        });
        assert_eq!(seen, vec![0, 1, 2]);
        assert!(result.unwrap_err().to_string().contains("CB0"));
    }
    #[test]
    fn failed_check_retains_every_owner_until_quarantine() {
        let owner = std::rc::Rc::new(());
        let mut entries = vec![owner.clone(), owner.clone()];
        assert!(release_after_checks(&mut entries, |_| anyhow::bail!("earlier Error")).is_err());
        assert_eq!(entries.len(), 2);
        assert_eq!(std::rc::Rc::strong_count(&owner), 3);
        let mut healthy = vec![owner.clone()];
        release_after_checks(&mut healthy, |_| Ok(())).unwrap();
        assert!(healthy.is_empty());
        assert_eq!(entries.len(), 2);
        assert_eq!(std::rc::Rc::strong_count(&owner), 3);
    }
    #[test]
    fn empty_and_all_healthy_check_loops_succeed() {
        assert!(check_all::<u8>(&[], |_| anyhow::bail!("not called")).is_ok());
        assert!(check_all(&[0, 1], |_| Ok(())).is_ok());
    }
}
