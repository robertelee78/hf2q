use super::*;

struct MockHub {
    revision: String,
    filenames: Vec<String>,
    downloads: Vec<(String, String, String)>,
    inventories: usize,
}

impl MockHub {
    fn new(filenames: &[&str]) -> Self {
        Self {
            revision: "a".repeat(40),
            filenames: filenames.iter().map(|name| name.to_string()).collect(),
            downloads: Vec::new(),
            inventories: 0,
        }
    }
}

impl GlpHub for MockHub {
    fn search(&mut self, _: &str, _: &str, _: usize) -> Result<Vec<String>> {
        bail!("unexpected Hub search")
    }

    fn inventory(&mut self, _: &HfModelReference) -> Result<RepositoryInventory> {
        self.inventories += 1;
        Ok(RepositoryInventory {
            revision: self.revision.clone(),
            filenames: self.filenames.clone(),
        })
    }

    fn download(
        &mut self,
        reference: &ResolvedHfModelReference,
        filename: &str,
    ) -> Result<PathBuf> {
        self.downloads.push((
            reference.repo_id().into(),
            reference.revision().into(),
            filename.into(),
        ));
        Ok(PathBuf::from("mock-cache/vector.gguf"))
    }
}

#[test]
fn explicit_repository_resolves_and_downloads_exact_inventory_commit() {
    let mut hub = MockHub::new(&["README.md", "vector.gguf"]);
    let resolved = resolve_with(
        Path::new("example/vector"),
        Path::new("unused-model"),
        &mut hub,
    )
    .unwrap();
    assert_eq!(
        hub.downloads,
        vec![(
            "example/vector".into(),
            "a".repeat(40),
            "vector.gguf".into()
        )]
    );
    assert_eq!(resolved.source_revision, Some("a".repeat(40)));
}

#[test]
fn explicit_file_url_selects_among_multiple_files_and_pins_branch() {
    let mut hub = MockHub::new(&["vectors/first.gguf", "vectors/second.gguf"]);
    let url = "https://huggingface.co/example/vector/resolve/release/vectors/second.gguf";
    resolve_with(Path::new(url), Path::new("unused-model"), &mut hub).unwrap();
    assert_eq!(
        hub.downloads,
        vec![(
            "example/vector".into(),
            "a".repeat(40),
            "vectors/second.gguf".into()
        )]
    );
}

#[test]
fn ambiguous_files_fail_before_any_download_in_any_inventory_order() {
    for names in [["first.gguf", "second.gguf"], ["second.gguf", "first.gguf"]] {
        let mut hub = MockHub::new(&names);
        let error = resolve_hub(
            HfModelReference::parse("example/vector", None).unwrap(),
            &mut hub,
        )
        .unwrap_err();
        assert!(error.to_string().contains("ambiguous GLP files"));
        assert!(hub.downloads.is_empty());
    }
}

#[test]
fn wrong_explicit_commit_is_rejected_before_download() {
    let mut hub = MockHub::new(&["vector.gguf"]);
    let url = format!(
        "https://huggingface.co/example/vector/resolve/{}/vector.gguf",
        "b".repeat(40)
    );
    assert!(resolve_hub(HfModelReference::parse(&url, None).unwrap(), &mut hub).is_err());
    assert!(hub.downloads.is_empty());
}

#[test]
fn missing_or_mutable_inventory_commit_is_rejected() {
    for revision in ["", "main", "xyz"] {
        let mut hub = MockHub::new(&["vector.gguf"]);
        hub.revision = revision.into();
        assert!(resolve_hub(
            HfModelReference::parse("example/vector", None).unwrap(),
            &mut hub
        )
        .is_err());
        assert!(hub.downloads.is_empty());
    }
}

#[test]
fn unsafe_duplicate_missing_or_non_gguf_inventory_never_downloads() {
    for files in [
        vec!["../vector.gguf"],
        vec!["vector.gguf", "vector.gguf"],
        vec!["README.md"],
        vec![],
    ] {
        let mut hub = MockHub::new(&files);
        assert!(resolve_hub(
            HfModelReference::parse("example/vector", None).unwrap(),
            &mut hub
        )
        .is_err());
        assert!(hub.downloads.is_empty());
    }
    let mut hub = MockHub::new(&["vector.gguf"]);
    let url = "https://huggingface.co/example/vector/blob/main/missing.gguf";
    assert!(resolve_hub(HfModelReference::parse(url, None).unwrap(), &mut hub).is_err());
    assert!(hub.downloads.is_empty());
}

#[test]
fn existing_local_file_uses_no_hub_operations() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut hub = MockHub::new(&[]);
    let resolved = resolve_with(file.path(), Path::new("unused-model"), &mut hub).unwrap();
    assert_eq!(resolved.path, file.path());
    assert!(resolved.source_repo.is_none());
    assert_eq!(hub.inventories, 0);
    assert!(hub.downloads.is_empty());
}

#[test]
fn missing_local_paths_and_untrusted_urls_do_not_fall_back_to_hub() {
    let mut hub = MockHub::new(&[]);
    for value in [
        "./missing.gguf",
        "/missing/vector.gguf",
        "dir/missing.gguf",
        "https://example.com/vector.gguf",
    ] {
        assert!(resolve_with(Path::new(value), Path::new("unused-model"), &mut hub).is_err());
    }
    assert_eq!(hub.inventories, 0);
    assert!(hub.downloads.is_empty());
}

#[test]
fn multiple_repositories_are_ambiguous_regardless_of_coverage_or_order() {
    let search = "Example-Model-abliterated-cyber-GLP-";
    let first = format!("msuiche/{search}1");
    let second = format!("msuiche/{search}2");
    for repos in [vec![first.clone(), second.clone()], vec![second, first]] {
        assert!(unique_discovery_repo(search, repos)
            .unwrap_err()
            .to_string()
            .contains("ambiguous GLP repositories"));
    }
}

#[test]
fn published_deepseek_repository_is_a_discovery_candidate() {
    let search = "DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-";
    let shipped = "msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4";
    let wrong_base = "msuiche/DeepSeek-V4-Flash-0731X-abliterated-cyber-GLP-29-L10-38-a4";
    assert_eq!(
        unique_discovery_repo(search, vec![wrong_base.into(), shipped.into()]).unwrap(),
        shipped
    );
    assert!(unique_discovery_repo(search, vec![wrong_base.into()]).is_err());
}

#[test]
fn published_repository_and_alternate_same_base_are_ambiguous() {
    let search = "DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-";
    let shipped = "msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4";
    let alternate = "msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-28-L11-38-a4";
    for repos in [
        vec![shipped.into(), alternate.into()],
        vec![alternate.into(), shipped.into()],
    ] {
        assert!(unique_discovery_repo(search, repos)
            .unwrap_err()
            .to_string()
            .contains("ambiguous GLP repositories"));
    }
}

#[test]
fn only_recognized_layer_and_dose_suffixes_are_candidates() {
    for tail in ["29", "29-L10-38-a4", "29-L10-38-a0.25", "29-L10-38-a0"] {
        assert!(weightless_artifact_suffix(tail), "{tail}");
    }
    for tail in [
        "0",
        "29-extra",
        "29-L10-38",
        "29-L38-10-a4",
        "29-L0-28-a4",
        "29-L10-38-a-1",
        "29-L10-38-a1.2.3",
        "29-L10-38-a4-extra",
    ] {
        assert!(!weightless_artifact_suffix(tail), "{tail}");
    }
}

#[test]
fn discovery_filters_author_and_full_convention_and_rejects_incomplete_search() {
    let search = "Example-Model-abliterated-cyber-GLP-";
    let target = format!("msuiche/{search}12");
    let repos = vec![
        target.clone(),
        format!("other/{search}1"),
        format!("msuiche/{search}2-extra"),
    ];
    assert_eq!(unique_discovery_repo(search, repos).unwrap(), target);
    assert!(unique_discovery_repo(search, vec![]).is_err());
    assert!(unique_discovery_repo(search, vec![target; MAX_DISCOVERY_REPOS + 1]).is_err());
}

#[test]
fn transport_failure_is_fatal_without_selecting_cached_or_other_files() {
    struct Unavailable;
    impl GlpHub for Unavailable {
        fn search(&mut self, _: &str, _: &str, _: usize) -> Result<Vec<String>> {
            bail!("offline")
        }
        fn inventory(&mut self, _: &HfModelReference) -> Result<RepositoryInventory> {
            bail!("offline")
        }
        fn download(&mut self, _: &ResolvedHfModelReference, _: &str) -> Result<PathBuf> {
            panic!("must not download")
        }
    }
    let error = resolve_with(
        Path::new("example/vector"),
        Path::new("unused-model"),
        &mut Unavailable,
    )
    .unwrap_err();
    assert!(error.to_string().contains("offline"));
}
