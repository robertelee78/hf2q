//! Channel-aware `hf2q update` and `hf2q uninstall` command coordination.
//!
//! This module selects one proven installation owner and delegates to that
//! owner. It does not infer package-manager state or mutate source checkouts.

use std::path::Path;

use crate::{cli, setup};

use super::installation::{self, Installation, InstallationError, ManagerCommand};

#[derive(Debug, thiserror::Error)]
pub(crate) enum LifecycleError {
    #[error("{0:#}")]
    Input(anyhow::Error),
    #[error("{0:#}")]
    Operational(anyhow::Error),
}

impl LifecycleError {
    pub(crate) fn is_input(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    fn input(error: impl Into<anyhow::Error>) -> Self {
        Self::Input(error.into())
    }

    fn operational(error: impl Into<anyhow::Error>) -> Self {
        Self::Operational(error.into())
    }
}

pub(crate) fn update(args: cli::UpdateArgs, executable: &Path) -> Result<(), LifecycleError> {
    let installation = installation::detect(executable).map_err(installation_error)?;
    match installation {
        Installation::Standalone { install_dir } => {
            if args.rollback {
                let completion_cleanup = crate::cli::completion_receipt::cleanup_owned();
                super::standalone::rollback(&install_dir).map_err(LifecycleError::operational)?;
                refresh_completion(executable);
                report_completion_cleanup(&completion_cleanup);
                println!(
                    "Restored the previous standalone hf2q in {}. Run `hf2q --version` to inspect it.",
                    install_dir.display()
                );
                return Ok(());
            }
            let outcome = super::standalone::run_update(executable, args.check)
                .map_err(LifecycleError::operational)?;
            match outcome {
                super::standalone::UpdateOutcome::Current { version } => {
                    println!("hf2q {version} is already the current stable standalone release.");
                }
                super::standalone::UpdateOutcome::Available { current, latest } => {
                    println!("Standalone update available: {current} -> {latest}.");
                }
                super::standalone::UpdateOutcome::Updated { previous, current } => {
                    refresh_completion(executable);
                    println!(
                        "Updated standalone hf2q {previous} -> {current}. Roll back with `hf2q update --rollback`."
                    );
                }
            }
        }
        Installation::Cargo {
            root,
            version,
            source,
            options,
        } => {
            if args.rollback {
                return Err(LifecycleError::input(anyhow::anyhow!(
                    "Cargo owns hf2q {} from {} under {}; Cargo has no standalone rollback slot. Re-run the original Cargo install source with an exact version or revision; no files were changed",
                    version,
                    source.description(),
                    root.display()
                )));
            }
            let command = match ManagerCommand::cargo_update(&root, &source, &options) {
                Ok(command) => command,
                Err(error) => {
                    let message = format!(
                        "Cargo owns hf2q {version} from {} under {}, but its source cannot be replayed safely: {error}. Re-run the original Cargo install command with `--root {}`; no files were changed",
                        source.description(),
                        root.display(),
                        root.display()
                    );
                    if args.check {
                        println!("{message}");
                        return Ok(());
                    }
                    return Err(LifecycleError::input(anyhow::anyhow!(message)));
                }
            };
            if args.check {
                println!(
                    "Cargo owns hf2q {version} from {} under {}. Cargo 1.88 has no stable install dry-run; `hf2q update` will delegate exactly: {}",
                    source.description(),
                    root.display(),
                    command.display()
                );
                return Ok(());
            }
            println!("Delegating hf2q update to Cargo: {}", command.display());
            command.run().map_err(installation_error)?;
            let current =
                installation::reconcile_cargo_update(executable, &root, &source, &options)
                    .map_err(installation_error)?;
            refresh_completion(executable);
            if current == version.to_string() {
                println!(
                    "Cargo reports hf2q {current} is already current under {}.",
                    root.display()
                );
            } else {
                println!(
                    "Updated Cargo-managed hf2q {version} -> {current} under {}.",
                    root.display()
                );
            }
        }
        Installation::SourceDevelopment {
            workspace_root,
            profile,
        } => {
            let message = format!(
                "hf2q is a source-development {} build from {}. Update that checkout with its owning VCS workflow, then run `cargo build --release --locked`; hf2q will not edit or delete the checkout.",
                profile.as_str(),
                workspace_root.display()
            );
            if args.check {
                println!("No files changed. {message}");
                return Ok(());
            }
            return Err(LifecycleError::input(anyhow::anyhow!(message)));
        }
        Installation::Unmanaged { executable } => {
            return Err(LifecycleError::input(anyhow::anyhow!(
                "{} has no valid standalone or Cargo ownership receipt and is not the conventional build from its compiled source root. Reinstall through https://hf2q.us/install.sh or the package manager that owns this file; no files were changed",
                executable.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn uninstall(
    args: cli::UninstallArgs,
    state_root: Option<&Path>,
    executable: &Path,
) -> Result<(), LifecycleError> {
    let installation = installation::detect(executable).map_err(installation_error)?;
    let config_purge = args
        .purge_config
        .then(|| setup::prepare_config_purge(state_root))
        .transpose()
        .map_err(setup_error)?;
    let cache_purge = args
        .purge_cache
        .then(super::purge::prepare_cache_purge)
        .transpose()
        .map_err(purge_error)?;
    let purge_preview = uninstall_purge_preview(config_purge.as_ref(), cache_purge.as_ref());
    let completion_preview = completion_uninstall_preview();
    match installation {
        Installation::Standalone { install_dir } => {
            if !args.yes {
                return Err(LifecycleError::input(anyhow::anyhow!(
                    "standalone uninstall would remove only {}, .hf2q-standalone.json, .hf2q-previous, .hf2q-standalone.lock, and exact receipt-owned completion artifacts.{}{} Rerun `hf2q uninstall{}{} --yes` to confirm",
                    executable.display(),
                    completion_preview,
                    purge_preview,
                    if args.purge_config { " --purge-config" } else { "" },
                    if args.purge_cache { " --purge-cache" } else { "" }
                )));
            }
            let completion_cleanup = crate::cli::completion_receipt::cleanup_owned();
            super::standalone::uninstall(&install_dir).map_err(LifecycleError::operational)?;
            execute_uninstall_purges(
                config_purge.as_ref(),
                cache_purge.as_ref(),
                &format!("standalone hf2q from {}", install_dir.display()),
            )?;
            println!(
                "Removed standalone hf2q from {}.{}",
                install_dir.display(),
                uninstall_preservation_summary(&args)
            );
            report_completion_cleanup(&completion_cleanup);
            Ok(())
        }
        Installation::Cargo {
            root, version, ..
        } => {
            let command = ManagerCommand::cargo_uninstall(&root, &version);
            if !args.yes {
                return Err(LifecycleError::input(anyhow::anyhow!(
                    "Cargo owns hf2q {version} under {}. Uninstall would delegate exactly `{}` and remove exact receipt-owned completion artifacts.{}{} Rerun `hf2q uninstall{}{} --yes` to confirm",
                    root.display(),
                    command.display(),
                    completion_preview,
                    purge_preview,
                    if args.purge_config { " --purge-config" } else { "" },
                    if args.purge_cache { " --purge-cache" } else { "" }
                )));
            }
            let completion_cleanup = crate::cli::completion_receipt::cleanup_owned();
            println!("Delegating hf2q uninstall to Cargo: {}", command.display());
            command.run().map_err(installation_error)?;
            installation::reconcile_cargo_uninstall(&root, executable)
                .map_err(installation_error)?;
            execute_uninstall_purges(
                config_purge.as_ref(),
                cache_purge.as_ref(),
                &format!("Cargo-managed hf2q {version} from {}", root.display()),
            )?;
            println!(
                "Removed Cargo-managed hf2q {version} from {}.{}",
                root.display(),
                uninstall_preservation_summary(&args)
            );
            report_completion_cleanup(&completion_cleanup);
            Ok(())
        }
        Installation::SourceDevelopment {
            workspace_root,
            profile,
        } => Err(LifecycleError::input(anyhow::anyhow!(
            "{} is a source-development {} build. hf2q does not own the checkout or run `cargo clean`; remove build artifacts through the checkout at {}",
            executable.display(),
            profile.as_str(),
            workspace_root.display()
        ))),
        Installation::Unmanaged { executable } => Err(LifecycleError::input(anyhow::anyhow!(
            "{} has no valid installation owner, so hf2q will not remove it. Use the package manager or source checkout that owns this file",
            executable.display()
        ))),
    }
}

fn completion_uninstall_preview() -> String {
    match crate::cli::completion_receipt::owned_paths() {
        Ok(paths) if paths.is_empty() => " No completion ownership receipt exists.".to_owned(),
        Ok(paths) => format!(
            " Completion cleanup is limited to: {}.",
            paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Err(error) => format!(
            " Completion cleanup will preserve artifacts because its ownership receipt is invalid ({error})."
        ),
    }
}

fn report_completion_cleanup(cleanup: &crate::cli::completion_receipt::CompletionCleanup) {
    if !cleanup.removed.is_empty() {
        println!(
            "Removed {} receipt-owned completion artifact(s).",
            cleanup.removed.len()
        );
    }
    for preserved in &cleanup.preserved {
        eprintln!("hf2q: preserved {preserved}");
    }
}

fn refresh_completion(executable: &Path) {
    let result = std::process::Command::new(executable)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status();
    match result {
        Ok(status) if status.success() => {}
        Ok(status) => eprintln!(
            "hf2q: updated successfully, but completion refresh exited with {status}; run `hf2q --version` once to retry"
        ),
        Err(error) => eprintln!(
            "hf2q: updated successfully, but completion refresh could not start: {error}; run `hf2q --version` once to retry"
        ),
    }
}

fn uninstall_purge_preview(
    config: Option<&setup::ConfigPurgePlan>,
    cache: Option<&super::purge::CachePurgePlan>,
) -> String {
    let mut clauses = Vec::new();
    if let Some(plan) = config {
        clauses.push(format!(
            " Config purge would remove only: {}.",
            plan.paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    } else {
        clauses.push(" Configuration is preserved.".to_owned());
    }
    if let Some(plan) = cache {
        clauses.push(if plan.contains_data() {
            format!(
                " Cache purge would clear {} and reset {}; locks and external model/Hugging Face/KV data are preserved.",
                plan.models.display(),
                plan.manifest.display()
            )
        } else {
            format!(
                " No manifest-owned cache data exists at {}; cache purge is a no-op.",
                plan.root.display()
            )
        });
    } else {
        clauses.push(" Caches, models, and logs are preserved.".to_owned());
    }
    clauses.concat()
}

fn execute_uninstall_purges(
    config: Option<&setup::ConfigPurgePlan>,
    cache: Option<&super::purge::CachePurgePlan>,
    removed_release: &str,
) -> Result<(), LifecycleError> {
    if let Some(plan) = config {
        let removed = setup::execute_config_purge(plan).map_err(|error| {
            LifecycleError::operational(anyhow::anyhow!(
                "removed {removed_release}, but config purge under {} failed: {error}; cache purge has not run",
                plan.root.display()
            ))
        })?;
        println!(
            "Purged {} setup-owned config file(s) under {}.",
            removed.len(),
            plan.root.display()
        );
    }
    if let Some(plan) = cache {
        let freed = super::purge::execute_cache_purge(plan).map_err(|error| {
            LifecycleError::operational(anyhow::anyhow!(
                "removed {removed_release}, but cache purge at {} failed: {error}",
                plan.root.display()
            ))
        })?;
        println!(
            "Purged manifest-owned hf2q model cache at {} ({} bytes freed).",
            plan.root.display(),
            freed
        );
    }
    Ok(())
}

fn uninstall_preservation_summary(args: &cli::UninstallArgs) -> &'static str {
    match (args.purge_config, args.purge_cache) {
        (false, false) => " Configuration, models, caches, and logs were preserved.",
        (true, false) => " Model data, caches, and logs were preserved.",
        (false, true) => {
            " Configuration, external models, persistent KV data, and logs were preserved."
        }
        (true, true) => " External models, persistent KV data, and logs were preserved.",
    }
}

fn installation_error(error: InstallationError) -> LifecycleError {
    if matches!(
        error,
        InstallationError::Invalid(_) | InstallationError::Ambiguous(_)
    ) {
        LifecycleError::input(error)
    } else {
        LifecycleError::operational(error)
    }
}

fn setup_error(error: setup::SetupError) -> LifecycleError {
    if error.is_input() {
        LifecycleError::input(error)
    } else {
        LifecycleError::operational(error)
    }
}

fn purge_error(error: super::purge::PurgeError) -> LifecycleError {
    if matches!(error, super::purge::PurgeError::Invalid(_)) {
        LifecycleError::input(error)
    } else {
        LifecycleError::operational(error)
    }
}
