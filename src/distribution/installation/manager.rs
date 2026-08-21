//! Exact direct-argv package-manager commands selected from validated receipts.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, Stdio};

use super::{CargoGitSelector, CargoInstallOptions, CargoSource, InstallationError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManagerCommand {
    program: OsString,
    arguments: Vec<OsString>,
}

impl ManagerCommand {
    pub(crate) fn cargo_update(
        root: &Path,
        source: &CargoSource,
        options: &CargoInstallOptions,
    ) -> Result<Self, InstallationError> {
        let mut arguments = vec![
            OsString::from("install"),
            OsString::from("--root"),
            root.as_os_str().to_owned(),
        ];
        let (include_package, supports_version) = match source {
            CargoSource::CratesIo => {
                arguments.extend([OsString::from("--registry"), OsString::from("crates-io")]);
                (true, true)
            }
            CargoSource::Path(path) => {
                arguments.extend([OsString::from("--path"), path.as_os_str().to_owned()]);
                (false, false)
            }
            CargoSource::Git(git) => {
                arguments.extend([OsString::from("--git"), OsString::from(&git.repository)]);
                if let Some(selector) = &git.selector {
                    let (flag, value) = match selector {
                        CargoGitSelector::Branch(value) => ("--branch", value),
                        CargoGitSelector::Tag(value) => ("--tag", value),
                        CargoGitSelector::Rev(value) => ("--rev", value),
                    };
                    arguments.extend([OsString::from(flag), OsString::from(value)]);
                }
                (true, false)
            }
            CargoSource::OtherRegistry(source) => {
                let index = source.strip_prefix("registry+").ok_or_else(|| {
                    InstallationError::Invalid(
                        "Cargo registry receipt has no registry+ source prefix".to_owned(),
                    )
                })?;
                arguments.extend([OsString::from("--index"), OsString::from(index)]);
                (true, true)
            }
            CargoSource::Other(_) => {
                return Err(InstallationError::Invalid(
                    "Cargo receipt source cannot be replayed safely".to_owned(),
                ));
            }
        };
        arguments.extend([
            OsString::from("--locked"),
            OsString::from("--bin"),
            OsString::from("hf2q"),
        ]);
        if let Some(version_req) = &options.version_req {
            if !supports_version {
                return Err(InstallationError::Invalid(
                    "Cargo source receipt unexpectedly contains a version requirement".to_owned(),
                ));
            }
            arguments.push(OsString::from("--version"));
            arguments.push(OsString::from(version_req));
        }
        if !options.features.is_empty() {
            arguments.push(OsString::from("--features"));
            arguments.push(OsString::from(
                options
                    .features
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(","),
            ));
        }
        if options.all_features {
            arguments.push(OsString::from("--all-features"));
        }
        if options.no_default_features {
            arguments.push(OsString::from("--no-default-features"));
        }
        arguments.push(OsString::from("--profile"));
        arguments.push(OsString::from(&options.profile));
        if let Some(target) = &options.target {
            arguments.push(OsString::from("--target"));
            arguments.push(OsString::from(target));
        }
        if include_package {
            arguments.push(OsString::from("hf2q"));
        }
        Ok(Self {
            program: OsString::from("cargo"),
            arguments,
        })
    }

    pub(crate) fn cargo_uninstall(root: &Path, version: &semver::Version) -> Self {
        Self {
            program: OsString::from("cargo"),
            arguments: vec![
                OsString::from("uninstall"),
                OsString::from("--root"),
                root.as_os_str().to_owned(),
                OsString::from("--package"),
                OsString::from(format!("hf2q@{version}")),
                OsString::from("--bin"),
                OsString::from("hf2q"),
            ],
        }
    }

    pub(crate) fn argv(&self) -> Vec<&OsStr> {
        std::iter::once(self.program.as_os_str())
            .chain(self.arguments.iter().map(OsString::as_os_str))
            .collect()
    }

    pub(crate) fn display(&self) -> String {
        self.argv()
            .into_iter()
            .map(shell_word)
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub(crate) fn run(&self) -> Result<(), InstallationError> {
        let status = Command::new(&self.program)
            .args(&self.arguments)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|error| {
                InstallationError::Manager(format!("could not start `{}`: {error}", self.display()))
            })?;
        if !status.success() {
            return Err(InstallationError::Manager(format!(
                "`{}` exited with {status}",
                self.display()
            )));
        }
        Ok(())
    }
}

fn shell_word(value: &OsStr) -> String {
    let text = value.to_string_lossy();
    if !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_+-./:@".contains(&byte))
    {
        text.into_owned()
    } else {
        format!("'{}'", text.replace('\'', "'\"'\"'"))
    }
}
