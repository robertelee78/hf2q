//! ADR-012 R1 + R10 upstream-drift canary.
//!
//! Catalog entries in `src/arch/entries/{qwen35,qwen35moe}.rs` carry
//! hand-transcribed citations of the form "llama-arch.cpp:NNN LLM_TENSOR_X".
//! Per `feedback_hf2q_sovereignty.md` we read llama.cpp as a spec source
//! only; there is no build-time or runtime link. But spec sources DO drift:
//! if upstream llama.cpp renumbers a tensor constant, our hand-transcribed
//! citation becomes misleading even though the mapping itself might still
//! be correct by string value.
//!
//! This canary does three things:
//!   1. If `/opt/llama.cpp` is not present on the build machine, the tests
//!      silent-skip (pass) so CI without llama.cpp's source remains green.
//!   2. If present, it opens `/opt/llama.cpp/src/llama-arch.cpp` and walks
//!      every catalog entry's citation, confirming the cited line genuinely
//!      contains the referenced LLM_TENSOR_X constant.
//!   3. On mismatch, it reports the affected citation and the offending
//!      constant so the amendment (per R1 mitigation) is actionable on
//!      the first read.
//!
//! The test explicitly DOES NOT verify that llama.cpp's GGUF string
//! matches our mapped name — that'd be brittle to whitespace / format
//! changes. The "constant name appears on cited line" check is enough
//! to catch a meaningful drift.

use std::path::Path;

fn llama_arch_cpp_path() -> Option<&'static Path> {
    const P: &str = "/opt/llama.cpp/src/llama-arch.cpp";
    if Path::new(P).is_file() {
        Some(Path::new(P))
    } else {
        None
    }
}

fn llama_arch_cpp_source() -> Option<String> {
    llama_arch_cpp_path().and_then(|p| std::fs::read_to_string(p).ok())
}

/// Parse a citation string like "llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM"
/// into `(line_number, constant_name)`. Returns `None` if the string does
/// not reference llama-arch.cpp.
fn parse_llama_arch_citation(s: &str) -> Option<(usize, String)> {
    // Loose parsing — citations can be "llama-arch.cpp:449 LLM_TENSOR_X"
    // or "llama-arch.cpp:449 LLM_TENSOR_X; ...rest" or include file paths
    // with ":". Find the first "llama-arch.cpp:" occurrence and parse from
    // there.
    let idx = s.find("llama-arch.cpp:")?;
    let after = &s[idx + "llama-arch.cpp:".len()..];
    let (num_str, rest) = after.split_at(after.find(|c: char| !c.is_ascii_digit())?);
    let line: usize = num_str.parse().ok()?;
    // Find LLM_TENSOR_* constant after the line number.
    let rest = rest.trim_start_matches(|c: char| c == ' ' || c == ';');
    // Grab first LLM_ or LLM_TENSOR_ identifier.
    let start = rest.find("LLM_")?;
    let ident_body = &rest[start..];
    let end = ident_body
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(ident_body.len());
    let ident = &ident_body[..end];
    if ident.starts_with("LLM_TENSOR_") || ident.starts_with("LLM_KV_") {
        Some((line, ident.to_string()))
    } else {
        None
    }
}

/// Gather all citation strings from a catalog file's source.
fn citations_in_source(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| {
            let ll = l.trim();
            if !ll.starts_with("citation:") {
                return None;
            }
            // `citation: "foo",`  → extract the quoted body.
            let q1 = ll.find('"')?;
            let q2 = ll[q1 + 1..].find('"')?;
            Some(ll[q1 + 1..q1 + 1 + q2].to_string())
        })
        .collect()
}

fn audit_catalog_file(
    file_path: &str,
    arch_cpp_src: &str,
) -> Result<usize, String> {
    let src = std::fs::read_to_string(file_path)
        .map_err(|e| format!("read {}: {}", file_path, e))?;
    let arch_lines: Vec<&str> = arch_cpp_src.lines().collect();

    let mut checked = 0usize;
    for cite in citations_in_source(&src) {
        let Some((line_no, ident)) = parse_llama_arch_citation(&cite) else {
            continue;
        };
        if line_no == 0 || line_no > arch_lines.len() {
            return Err(format!(
                "{}: citation {:?} points at line {} but llama-arch.cpp has {} lines",
                file_path, cite, line_no, arch_lines.len()
            ));
        }
        let found = arch_lines[line_no - 1];
        if !found.contains(&ident) {
            // ± 2-line tolerance — minor upstream shuffles shouldn't block.
            let lo = line_no.saturating_sub(3);
            let hi = (line_no + 2).min(arch_lines.len());
            let window_hit = arch_lines[lo..hi].iter().any(|l| l.contains(&ident));
            if !window_hit {
                return Err(format!(
                    "{}: citation {:?} claims {} at line {}, but llama-arch.cpp does NOT contain {} \
                     anywhere in the ±2-line window. Upstream drift detected — read R1 mitigation \
                     in docs/ADR-012-qwen35moe-conversion.md and re-transcribe the catalog.",
                    file_path, cite, ident, line_no, ident
                ));
            }
        }
        checked += 1;
    }
    Ok(checked)
}

#[test]
fn qwen35_catalog_citations_resolve_in_upstream_llama_arch_cpp() {
    let Some(arch_src) = llama_arch_cpp_source() else {
        eprintln!("/opt/llama.cpp/src/llama-arch.cpp absent — silent skip");
        return;
    };
    let checked =
        audit_catalog_file("src/arch/entries/qwen35.rs", &arch_src).expect("audit");
    // Most qwen35.rs linear-attn catalog entries cite llama-arch.cpp with a
    // named LLM_TENSOR_* constant. Loosest invariant: at least one named
    // constant resolves — any zero-check would pass trivially even on an
    // empty catalog.
    assert!(
        checked >= 1,
        "expected ≥ 1 resolved llama-arch.cpp citation in qwen35.rs, got {}",
        checked
    );
}

#[test]
fn qwen35moe_catalog_citations_resolve_in_upstream_llama_arch_cpp() {
    let Some(arch_src) = llama_arch_cpp_source() else {
        eprintln!("/opt/llama.cpp/src/llama-arch.cpp absent — silent skip");
        return;
    };
    let checked =
        audit_catalog_file("src/arch/entries/qwen35moe.rs", &arch_src).expect("audit");
    assert!(
        checked >= 1,
        "expected ≥ 1 resolved llama-arch.cpp citation in qwen35moe.rs, got {}",
        checked
    );
}

#[test]
fn parse_llama_arch_citation_handles_canonical_form() {
    let (line, ident) = parse_llama_arch_citation(
        "llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM",
    )
    .expect("parse");
    assert_eq!(line, 449);
    assert_eq!(ident, "LLM_TENSOR_NEXTN_ENORM");
}

#[test]
fn parse_llama_arch_citation_handles_variant_forms() {
    // Multi-citation line
    let (line, ident) = parse_llama_arch_citation(
        "llama-arch.cpp:382 LLM_TENSOR_ATTN_QKV; src/models/qwen35/dense.rs:136",
    )
    .expect("parse");
    assert_eq!(line, 382);
    assert_eq!(ident, "LLM_TENSOR_ATTN_QKV");

    // LLM_KV_* (metadata keys) also supported.
    let (line, ident) =
        parse_llama_arch_citation("llama-arch.cpp:194 LLM_KV_NEXTN_PREDICT_LAYERS")
            .expect("parse");
    assert_eq!(line, 194);
    assert_eq!(ident, "LLM_KV_NEXTN_PREDICT_LAYERS");
}

#[test]
fn parse_llama_arch_citation_rejects_non_llama_arch() {
    assert!(parse_llama_arch_citation("src/models/qwen35/dense.rs:50").is_none());
    assert!(parse_llama_arch_citation("random text").is_none());
}

#[test]
fn citations_in_source_extracts_all_citation_strings() {
    let src = r#"
        TensorCatalogEntry {
            name_template: "foo",
            scope: LayerScope::Global,
            dtype: TensorDtype::F16,
            citation: "llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM",
        },
        TensorCatalogEntry {
            citation: "src/models/qwen35/dense.rs:50 (model.embed_tokens.weight → token_embd.weight)",
        }
    "#;
    let cites = citations_in_source(src);
    assert_eq!(cites.len(), 2);
    assert!(cites[0].contains("LLM_TENSOR_NEXTN_ENORM"));
    assert!(cites[1].contains("token_embd.weight"));
}
