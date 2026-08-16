//! Catalog-citation ownership contract.
//!
//! Architecture catalog entries are release evidence. Their citations must
//! resolve to source files owned by this repository so the checked-in mapper
//! and native consumer remain the authority in every build environment.

use std::path::Path;

fn citations_in_source(source: &str) -> Vec<String> {
    let mut rest = source;
    let mut citations = Vec::new();

    while let Some(offset) = rest.find("citation:") {
        rest = &rest[offset + "citation:".len()..];
        let Some(open) = rest.find('"') else {
            break;
        };
        rest = &rest[open + 1..];
        let Some(close) = rest.find('"') else {
            break;
        };
        citations.push(rest[..close].to_owned());
        rest = &rest[close + 1..];
    }

    citations
}

fn cited_source_path(citation: &str) -> &str {
    citation
        .trim()
        .split(|character: char| character.is_whitespace() || character == '(')
        .next()
        .unwrap_or_default()
}

fn assert_repo_owned_catalog_citations(catalog_path: &str) {
    let source = std::fs::read_to_string(catalog_path)
        .unwrap_or_else(|error| panic!("read {catalog_path}: {error}"));
    let entry_count = source.matches("TensorCatalogEntry {").count();
    let citations = citations_in_source(&source);

    assert!(entry_count > 0, "{catalog_path}: catalog must not be empty");
    assert_eq!(
        citations.len(),
        entry_count,
        "{catalog_path}: every tensor catalog entry must carry one citation"
    );

    for citation in citations {
        for source_reference in citation.split(';') {
            let path = cited_source_path(source_reference);
            assert!(
                path.starts_with("src/"),
                "{catalog_path}: citation {citation:?} must name a repository-owned src/ path"
            );
            assert!(
                !Path::new(path).is_absolute() && !path.contains(".."),
                "{catalog_path}: citation {citation:?} escapes the repository"
            );
            assert!(
                Path::new(path).is_file(),
                "{catalog_path}: cited source {path:?} does not exist"
            );
        }
    }
}

#[test]
fn qwen35_catalog_citations_are_repo_owned() {
    assert_repo_owned_catalog_citations("src/arch/entries/qwen35.rs");
}

#[test]
fn qwen35moe_catalog_citations_are_repo_owned() {
    assert_repo_owned_catalog_citations("src/arch/entries/qwen35moe.rs");
}

#[test]
fn citation_extractor_handles_multiline_and_multiple_sources() {
    let source = r#"
        TensorCatalogEntry {
            citation:
                "src/convert/arch/qwen35_dense.rs (tensor mapping)",
        },
        TensorCatalogEntry {
            citation: "src/convert/arch/qwen35moe_full.rs; src/inference/models/qwen35/mtp_weights_load.rs",
        },
    "#;
    let citations = citations_in_source(source);
    assert_eq!(citations.len(), 2);
    assert_eq!(
        cited_source_path(&citations[0]),
        "src/convert/arch/qwen35_dense.rs"
    );
    assert_eq!(citations[1].split(';').count(), 2);
}
