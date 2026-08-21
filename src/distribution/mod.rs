//! Reachable installation ownership, update, and uninstall behavior.

pub(crate) mod commands;
pub(crate) mod installation;
pub(crate) mod purge;
pub(crate) mod standalone;

#[cfg(test)]
#[path = "installation_tests.rs"]
mod installation_tests;

#[cfg(test)]
#[path = "purge_tests.rs"]
mod purge_tests;
