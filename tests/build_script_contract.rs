#![allow(dead_code)]

// Exercise the std-only build provenance policy as a normal test target. The
// build script itself is not part of Cargo's default test-target discovery.
#[path = "../src/build_script.rs"]
mod build_script;
