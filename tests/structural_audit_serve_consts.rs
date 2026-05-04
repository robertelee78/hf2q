// Structural audit: forbid fn-body-local `const X: &[&str]` declarations
// inside `src/serve/**/*.rs`.
//
// 2026-05-02 — added per the H1 finding in
// `docs/research/decode-test-gap-2026-05-02.md`. The same-day
// `SPECIAL_TOKEN_STOPS` 3-commit thrash (`f8cdebc → d9b23e9 → b254723`) was
// caused by an inline `const SPECIAL_TOKEN_STOPS: &[&str] = &[...]`
// declaration buried inside `cmd_generate_qwen35`'s 488-LOC body in
// `src/serve/mod.rs`. Rust fn-body `const` is fn-local-scope: the name is
// not reachable from `mod tests` (you'd need `use super::cmd_generate_qwen35`
// somehow + then access a private const inside it, which Rust doesn't
// permit). So policy data buried this way is structurally untestable.
//
// The H1 audit identified one un-fixed twin
// (`src/serve/api/handlers.rs:5763` — BOS probe array inside the
// `embeddings` async fn) which was extracted to module scope in commit
// `af99863`. This test pins both extractions by failing if any new
// fn-body-local `const X: &[&str]` declaration lands in `src/serve/`.
//
// Scope: only `&[&str]` (and `&'static [&'static str]`) is checked, since
// (a) those are the typical pattern for marker-policy data, (b) `const N:
// usize = 10` style consts inside fn bodies are idiomatic and harmless.
// Future expansion to `&[u32]` (token-id lists) is in-scope when the
// `SpecialTokenRegistry` work lands per item (d) of the dossier plan.

use std::path::{Path, PathBuf};

use syn::visit::Visit;

/// One offending site found by the AST walker.
///
/// We don't capture line numbers because `proc_macro2::Span::start()` returns
/// stub-only locations outside a proc-macro context unless the `span-locations`
/// feature is enabled (extra dep + compile-time cost). The file + enclosing
/// fn name + const name is sufficient to grep-locate the offender, and that's
/// what the failure message instructs the maintainer to do.
#[derive(Debug)]
struct OffendingConst {
    file: PathBuf,
    fn_name: String,
    const_name: String,
}

/// AST visitor that finds `const X: &[&str]` declarations nested inside
/// function bodies (free fns, methods, or closures).
struct FnBodyConstHunter {
    /// Stack of fn names we're currently descending into. Empty == module
    /// scope (which is fine — those consts are reachable from `#[test]`).
    fn_stack: Vec<String>,
    file: PathBuf,
    hits: Vec<OffendingConst>,
}

impl FnBodyConstHunter {
    fn new(file: PathBuf) -> Self {
        Self {
            fn_stack: Vec::new(),
            file,
            hits: Vec::new(),
        }
    }

    /// Returns true if `ty` is `&[&str]` or `&'static [&'static str]`.
    fn is_slice_of_str_ref(ty: &syn::Type) -> bool {
        let outer_ref = match ty {
            syn::Type::Reference(r) => r,
            _ => return false,
        };
        let slice = match &*outer_ref.elem {
            syn::Type::Slice(s) => s,
            _ => return false,
        };
        let inner_ref = match &*slice.elem {
            syn::Type::Reference(r) => r,
            _ => return false,
        };
        // The inner type should be a path ending in `str`.
        if let syn::Type::Path(tp) = &*inner_ref.elem {
            if let Some(seg) = tp.path.segments.last() {
                return seg.ident == "str";
            }
        }
        false
    }

    fn record(&mut self, fn_name: &str, item_const: &syn::ItemConst) {
        self.hits.push(OffendingConst {
            file: self.file.clone(),
            fn_name: fn_name.to_string(),
            const_name: item_const.ident.to_string(),
        });
    }
}

impl<'ast> Visit<'ast> for FnBodyConstHunter {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.fn_stack.push(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.fn_stack.pop();
    }

    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        // We only care about statement-level consts inside fn bodies.
        if let syn::Stmt::Item(syn::Item::Const(item_const)) = stmt {
            if let Some(fn_name) = self.fn_stack.last() {
                if Self::is_slice_of_str_ref(&item_const.ty) {
                    let fn_name = fn_name.clone();
                    self.record(&fn_name, item_const);
                }
            }
        }
        syn::visit::visit_stmt(self, stmt);
    }
}

/// Walk a directory recursively, return all `.rs` files.
fn rust_files_under(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => panic!("read_dir({}) failed: {e}", dir.display()),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Workspace root, located via CARGO_MANIFEST_DIR.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn no_fn_body_local_str_slice_consts_in_serve_tree() {
    let serve_dir = workspace_root().join("src").join("serve");
    assert!(
        serve_dir.is_dir(),
        "serve dir not found: {} — workspace structure changed?",
        serve_dir.display()
    );

    let files = rust_files_under(&serve_dir);
    assert!(
        !files.is_empty(),
        "no .rs files found under {}",
        serve_dir.display()
    );

    let mut all_hits: Vec<OffendingConst> = Vec::new();
    for file in &files {
        let src = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => panic!("failed to read {}: {e}", file.display()),
        };
        let ast = match syn::parse_file(&src) {
            Ok(a) => a,
            Err(e) => panic!(
                "syn::parse_file failed on {}: {e}\n\
                 (this audit test depends on the file being valid Rust; \
                  if you hit this on a WIP file, fix the parse error first)",
                file.display()
            ),
        };
        let mut hunter = FnBodyConstHunter::new(file.clone());
        hunter.visit_file(&ast);
        all_hits.extend(hunter.hits);
    }

    if all_hits.is_empty() {
        return;
    }

    // Render a clear failure message — including a fix recipe — so a
    // future maintainer who hits this test knows exactly what to do.
    let workspace_root = workspace_root();
    let mut msg = String::from(
        "\n\
         ===================================================================\n\
         H1 STRUCTURAL AUDIT — fn-body-local `const X: &[&str]` declarations\n\
         ===================================================================\n\n\
         The following sites declare `const X: &[&str] = &[...]` (or\n\
         `&'static [&'static str]`) INSIDE a function body:\n\n",
    );
    for hit in &all_hits {
        let rel = hit
            .file
            .strip_prefix(&workspace_root)
            .unwrap_or(&hit.file)
            .display();
        msg.push_str(&format!(
            "  • {rel} — fn `{}` declares `const {}`\n",
            hit.fn_name, hit.const_name
        ));
    }
    msg.push_str(
        "\n\
         WHY THIS FAILS:\n\
         -----------------\n\
         Rust fn-body `const` is fn-local-scope. The name cannot be reached\n\
         from `mod tests` (`use super::*` only sees module-scope items, not\n\
         fn-local items). So this policy data is STRUCTURALLY UNTESTABLE\n\
         from a unit test. The same-day 2026-05-02 `SPECIAL_TOKEN_STOPS`\n\
         3-commit thrash (commits f8cdebc → d9b23e9 → b254723) is the\n\
         direct evidence of the regression class this audit prevents — see\n\
         docs/research/decode-test-gap-2026-05-02.md for the full audit.\n\n\
         FIX RECIPE:\n\
         -----------\n\
         1. Move the `const` to module scope (top of the file is fine).\n\
         2. If it's policy data, also extract a helper fn that consumes it\n\
            and add a #[test] module pinning the helper's behavior.\n\
         3. Update the original call site to reference the moved const.\n\
         4. Re-run `cargo test --release --test structural_audit_serve_consts`\n\
            to confirm the audit passes.\n\n\
         See `BOS_PROBE_FRAGMENTS` + `probe_bos_token_id` + the\n\
         `bos_probe_tests` module in src/serve/api/handlers.rs (commit\n\
         af99863) for a worked example of the fix shape.\n\
         ===================================================================\n",
    );
    panic!("{msg}");
}

#[test]
fn structural_audit_actually_walks_serve_files() {
    // Sanity: ensure the audit's file walker is actually finding files —
    // a future refactor that moves serve/ elsewhere would silently make
    // the audit vacuous (passes on zero files = always passes). Pin a
    // floor count so accidental empty-set passes fail loudly.
    let serve_dir = workspace_root().join("src").join("serve");
    let files = rust_files_under(&serve_dir);
    assert!(
        files.len() >= 10,
        "expected at least 10 .rs files under src/serve/ (current count is \
         {}); if the serve tree was intentionally split or shrunk, lower \
         the floor here, but DO NOT silence the audit by skipping the walk.",
        files.len()
    );
}

#[test]
fn fn_body_const_hunter_detects_planted_offender() {
    // Defense-in-depth: feed the hunter a hand-crafted Rust source string
    // containing the exact pattern we want to catch, and assert it gets
    // detected. This guards against a future refactor of `is_slice_of_str_ref`
    // that silently makes the audit blind.
    let src = r#"
        fn legitimate_module_consumer() {
            const PLANTED_OFFENDER: &[&str] = &["<bos>", "<|im_start|>"];
            let _ = PLANTED_OFFENDER;
        }
    "#;
    let ast = syn::parse_file(src).expect("planted source must parse");
    let mut hunter = FnBodyConstHunter::new(PathBuf::from("planted.rs"));
    hunter.visit_file(&ast);
    assert_eq!(hunter.hits.len(), 1, "must catch the planted offender");
    assert_eq!(hunter.hits[0].const_name, "PLANTED_OFFENDER");
    assert_eq!(hunter.hits[0].fn_name, "legitimate_module_consumer");
}

#[test]
fn fn_body_const_hunter_ignores_module_scope_const() {
    // Negative case: a module-scope const with the same type must NOT
    // be flagged — those ARE reachable from #[test].
    let src = r#"
        const MODULE_SCOPE_OK: &[&str] = &["<bos>"];
        fn consumer() {
            let _ = MODULE_SCOPE_OK;
        }
    "#;
    let ast = syn::parse_file(src).expect("must parse");
    let mut hunter = FnBodyConstHunter::new(PathBuf::from("planted.rs"));
    hunter.visit_file(&ast);
    assert!(
        hunter.hits.is_empty(),
        "module-scope consts must not be flagged; got {} hits",
        hunter.hits.len()
    );
}

#[test]
fn fn_body_const_hunter_ignores_unrelated_types() {
    // Negative case: a fn-body-local `const N: usize = 10` is idiomatic
    // and harmless; only `&[&str]` should fail this audit.
    let src = r#"
        fn unrelated() {
            const N: usize = 10;
            const SLICE_OF_U32: &[u32] = &[1, 2, 3];
            let _ = (N, SLICE_OF_U32);
        }
    "#;
    let ast = syn::parse_file(src).expect("must parse");
    let mut hunter = FnBodyConstHunter::new(PathBuf::from("planted.rs"));
    hunter.visit_file(&ast);
    assert!(
        hunter.hits.is_empty(),
        "non-&[&str] consts must not be flagged; got {} hits",
        hunter.hits.len()
    );
}
