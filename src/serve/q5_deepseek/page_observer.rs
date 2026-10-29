//! Dependency seam only. HF must not embed private Mach page-query code or
//! consume an unpublished local native patch to manufacture runtime coverage.
use anyhow::Result;

pub(super) trait PublishedPageObserver {
    fn require_runtime_support(&self) -> Result<()>;
}

pub(super) struct Unavailable;
impl PublishedPageObserver for Unavailable {
    fn require_runtime_support(&self) -> Result<()> {
        anyhow::bail!("native_page_observer=UnavailableUntilPublishedPin")
    }
}
