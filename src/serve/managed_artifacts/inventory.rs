use super::*;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};

pub(super) struct ExactLooseFile {
    pub(super) path: PathBuf,
    pub(super) retained: crate::core::bounded_file::StableRegularFile,
}

pub(super) fn find_matching_loose(
    artifact: &HubGgufArtifact,
    model_dirs: &[PathBuf],
) -> Result<Option<ExactLooseFile>> {
    for root in scan_roots(model_dirs)? {
        let mut found = None;
        visit_files(&root, |path, metadata, mut file| {
            if found.is_some()
                || metadata.len() != artifact.bytes
                || path
                    .extension()
                    .and_then(|v| v.to_str())
                    .is_none_or(|ext| !ext.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            if file
                .sha256()?
                .is_some_and(|digest| digest.eq_ignore_ascii_case(&artifact.sha256))
            {
                found = Some(ExactLooseFile {
                    path: path.to_path_buf(),
                    retained: file,
                });
            }
            Ok(())
        })?;
        if found.is_some() {
            return Ok(found);
        }
    }
    Ok(None)
}

pub(crate) fn print_inventory(model_dirs: &[PathBuf]) -> Result<()> {
    let cache_root = crate::serve::cache::default_root()?;
    let manifest_path = cache_root.join("manifest.json");
    let cache_manifest = manifest_path
        .is_file()
        .then(|| crate::serve::cache::read_manifest(&manifest_path))
        .transpose()?;
    let inventory = LocalArtifactInventory::for_serve(model_dirs)?;
    let local = inventory.discover(
        None,
        cache_manifest
            .as_ref()
            .map(|manifest| (cache_root.as_path(), manifest)),
    );
    let managed = scan_bindings(model_dirs, None)?;
    println!("REPOSITORY\tREVISION\tQUANT\tORIGIN\tLAST_USED\tMMPROJ\tPATH");
    let mut paths = BTreeSet::new();
    for candidate in managed {
        paths.insert(candidate.path.clone());
        println!(
            "{}",
            inventory_row([
                candidate.repository,
                short_revision(&candidate.revision).to_owned(),
                candidate.quant.to_string(),
                candidate.origin,
                display_timestamp(candidate.last_used_at_secs),
                if candidate.projector.is_some() {
                    "yes".to_owned()
                } else {
                    "no".to_owned()
                },
                candidate.path.to_string_lossy().into_owned(),
            ])
        );
    }
    for artifact in local.artifacts {
        if paths.insert(artifact.path.clone()) {
            let last_used = cache_manifest
                .as_ref()
                .and_then(|manifest| manifest.models.get(&artifact.repository))
                .and_then(|model| model.quantizations.get(&artifact.quant_hint))
                .map_or(0, |entry| entry.last_used_at_secs);
            let cached_projector = cache_manifest
                .as_ref()
                .and_then(|manifest| manifest.models.get(&artifact.repository))
                .and_then(|model| model.quantizations.get(&artifact.quant_hint))
                .and_then(|entry| entry.mmproj_path.as_ref())
                .and_then(|path| {
                    projector_authority_from_receipt(path, &artifact.repository, &artifact.revision)
                        .ok()
                        .flatten()
                });
            let receipt_projector = paired_projector_path(&artifact.path).and_then(|path| {
                projector_authority_from_receipt(&path, &artifact.repository, &artifact.revision)
                    .ok()
                    .flatten()
            });
            println!(
                "{}",
                inventory_row([
                    artifact.repository,
                    short_revision(&artifact.revision).to_owned(),
                    artifact.quant_hint,
                    artifact.provenance.as_str().to_owned(),
                    display_timestamp(last_used),
                    if cached_projector.or(receipt_projector).is_some() {
                        "yes".to_owned()
                    } else {
                        "no".to_owned()
                    },
                    artifact.path.to_string_lossy().into_owned(),
                ])
            );
        }
    }
    print_hub_cache_inventory(&mut paths)?;
    for root in scan_roots(model_dirs)? {
        visit_files(&root, |path, _, file| {
            if paths.contains(path)
                || path
                    .extension()
                    .and_then(|v| v.to_str())
                    .is_none_or(|ext| !ext.eq_ignore_ascii_case("gguf"))
            {
                return Ok(());
            }
            let quant = quant_type_from_gguf_file(file.try_clone()?, path)
                .map(|quant| quant.to_string())
                .unwrap_or_else(|_| "?".into());
            println!(
                "{}",
                inventory_row([
                    "?".to_owned(),
                    "?".to_owned(),
                    quant,
                    "unbound".to_owned(),
                    "-".to_owned(),
                    "?".to_owned(),
                    path.to_string_lossy().into_owned(),
                ])
            );
            Ok(())
        })?;
    }
    for warning in local.warnings {
        eprintln!("Warning: {warning}");
    }
    Ok(())
}

fn print_hub_cache_inventory(paths: &mut BTreeSet<PathBuf>) -> Result<()> {
    use rustix::fs::{Mode, OFlags};

    let hub_root = crate::input::hf_download::hf_hub_cache_dir();
    let hub_fd = match rustix::fs::open(
        &hub_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(()),
    };
    let mut model_dirs = directory_names(&hub_fd)?;
    model_dirs.sort();
    let mut visited = 0_usize;
    for name in model_dirs {
        if visited >= MAX_SCAN_ENTRIES {
            break;
        }
        let Some(encoded) = name.to_str().and_then(|name| name.strip_prefix("models--")) else {
            continue;
        };
        let Some((owner, model)) = encoded.split_once("--") else {
            continue;
        };
        let repository = format!("{owner}/{model}");
        let Some(model_fd) = open_directory_at(&hub_fd, &name)? else {
            continue;
        };
        let model_root = hub_root.join(&name);
        let Some(snapshots_fd) = open_directory_at(&model_fd, "snapshots")? else {
            continue;
        };
        let snapshots = model_root.join("snapshots");
        let blob_root = open_directory_at(&model_fd, "blobs")?.map(|file| HubBlobAuthority {
            path: model_root.join("blobs"),
            file,
        });
        let mut revisions = directory_names(&snapshots_fd)?;
        revisions.sort();
        for revision_name in revisions {
            let Some(revision_name) = revision_name.to_str() else {
                continue;
            };
            if revision_name.len() != 40
                || !revision_name.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                continue;
            }
            let Some(revision_fd) = open_directory_at(&snapshots_fd, revision_name)? else {
                continue;
            };
            print_hub_snapshot_inventory_from_authority(
                &snapshots.join(revision_name),
                revision_fd,
                blob_root.as_ref(),
                &repository,
                revision_name,
                paths,
                &mut visited,
                |_| {},
                |_| {},
            )?;
        }
    }
    Ok(())
}

struct HubBlobAuthority {
    path: PathBuf,
    file: fs::File,
}

fn directory_names(directory: &fs::File) -> Result<Vec<OsString>> {
    Ok(rustix::fs::Dir::read_from(directory)?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let bytes = entry.file_name().to_bytes();
            (!matches!(bytes, b"." | b"..")).then(|| OsString::from_vec(bytes.to_vec()))
        })
        .collect())
}

fn open_directory_at(
    directory: &fs::File,
    name: impl AsRef<std::ffi::OsStr>,
) -> Result<Option<fs::File>> {
    use rustix::fs::{Mode, OFlags};

    match rustix::fs::openat(
        directory,
        name.as_ref(),
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => Ok(Some(fs::File::from(fd))),
        Err(error)
            if matches!(
                error,
                rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR | rustix::io::Errno::LOOP
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into()),
    }
}

#[cfg(test)]
fn print_hub_snapshot_inventory_with_hooks(
    _repository_root: &Path,
    revision_root: &Path,
    blob_root: Option<&Path>,
    repository: &str,
    revision: &str,
    paths: &mut BTreeSet<PathBuf>,
    visited: &mut usize,
    before_directory: impl FnMut(&Path),
    after_symlink_parse: impl FnMut(&Path),
) -> Result<()> {
    use rustix::fs::{Mode, OFlags};

    let root_fd = match rustix::fs::open(
        revision_root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(()),
    };
    let blob_root = blob_root.and_then(|path| {
        rustix::fs::open(
            path,
            OFlags::RDONLY
                | OFlags::DIRECTORY
                | OFlags::NOFOLLOW
                | OFlags::NONBLOCK
                | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .ok()
        .map(|fd| HubBlobAuthority {
            path: path.to_path_buf(),
            file: fs::File::from(fd),
        })
    });
    print_hub_snapshot_inventory_from_authority(
        revision_root,
        root_fd,
        blob_root.as_ref(),
        repository,
        revision,
        paths,
        visited,
        before_directory,
        after_symlink_parse,
    )
}

fn print_hub_snapshot_inventory_from_authority(
    revision_root: &Path,
    root_fd: fs::File,
    blob_root: Option<&HubBlobAuthority>,
    repository: &str,
    revision: &str,
    paths: &mut BTreeSet<PathBuf>,
    visited: &mut usize,
    mut before_directory: impl FnMut(&Path),
    mut after_symlink_parse: impl FnMut(&Path),
) -> Result<()> {
    use rustix::fs::{Mode, OFlags};

    let root_authority = root_fd.try_clone()?;
    let mut queue = std::collections::VecDeque::from([(
        revision_root.to_path_buf(),
        PathBuf::new(),
        root_fd,
        0_usize,
    )]);
    while let Some((directory, relative_directory, directory_fd, depth)) = queue.pop_front() {
        before_directory(&directory);
        let mut entries = match rustix::fs::Dir::read_from(&directory_fd) {
            Ok(entries) => entries
                .filter_map(std::result::Result::ok)
                .filter_map(|entry| {
                    let bytes = entry.file_name().to_bytes();
                    (!matches!(bytes, b"." | b"..")).then(|| OsString::from_vec(bytes.to_vec()))
                })
                .collect::<Vec<_>>(),
            Err(_) => continue,
        };
        entries.sort();
        for name in entries {
            *visited += 1;
            if *visited > MAX_SCAN_ENTRIES {
                return Ok(());
            }
            let display_path = directory.join(&name);
            let relative_path = relative_directory.join(&name);
            let gguf_name = display_path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"));
            match rustix::fs::openat(
                &directory_fd,
                &name,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => {
                    let file = fs::File::from(fd);
                    let metadata = file.metadata()?;
                    if metadata.is_dir() {
                        if depth < MAX_SCAN_DEPTH {
                            queue.push_back((display_path, relative_path, file, depth + 1));
                        }
                        continue;
                    }
                    if !gguf_name || !metadata.is_file() {
                        continue;
                    }
                    let Some(retained) =
                        crate::core::bounded_file::StableRegularFile::from_walked_file(
                            root_authority.try_clone()?,
                            revision_root.to_path_buf(),
                            relative_path,
                            &display_path,
                            file,
                            metadata.len(),
                        )?
                    else {
                        continue;
                    };
                    print_retained_hub_row(retained, display_path, repository, revision, paths)?;
                }
                Err(error) if error == rustix::io::Errno::LOOP && gguf_name => {
                    let Some(blob_root) = blob_root else {
                        continue;
                    };
                    let target = match rustix::fs::readlinkat(&directory_fd, &name, Vec::new()) {
                        Ok(target) => OsString::from_vec(target.into_bytes()),
                        Err(_) => continue,
                    };
                    let Some(blob_relative) =
                        resolve_hub_blob_target(revision, &relative_directory, Path::new(&target))
                    else {
                        continue;
                    };
                    let Some(blob_file) = open_relative_regular(&blob_root.file, &blob_relative)?
                    else {
                        continue;
                    };
                    let followed = match rustix::fs::openat(
                        &directory_fd,
                        &name,
                        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
                        Mode::empty(),
                    ) {
                        Ok(fd) => fs::File::from(fd),
                        Err(_) => continue,
                    };
                    let metadata = followed.metadata()?;
                    let blob_metadata = blob_file.metadata()?;
                    use std::os::unix::fs::MetadataExt as _;
                    if !metadata.is_file()
                        || metadata.dev() != blob_metadata.dev()
                        || metadata.ino() != blob_metadata.ino()
                    {
                        continue;
                    }
                    let canonical = blob_root.path.join(&blob_relative);
                    let Some(retained) =
                        crate::core::bounded_file::StableRegularFile::from_walked_file(
                            blob_root.file.try_clone()?,
                            blob_root.path.clone(),
                            blob_relative.clone(),
                            &canonical,
                            followed,
                            metadata.len(),
                        )?
                    else {
                        continue;
                    };
                    let original_identity = retained.identity();
                    let Ok(quant) = quant_type_from_gguf_file(retained.try_clone()?, &display_path)
                    else {
                        continue;
                    };
                    after_symlink_parse(&display_path);
                    let link_unchanged = rustix::fs::readlinkat(&directory_fd, &name, Vec::new())
                        .is_ok_and(|current| current.into_bytes() == target.as_bytes());
                    let reopened_same = rustix::fs::openat(
                        &directory_fd,
                        &name,
                        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
                        Mode::empty(),
                    )
                    .ok()
                    .and_then(|fd| {
                        crate::core::bounded_file::StableRegularFile::from_open_file(
                            fs::File::from(fd),
                            &canonical,
                            metadata.len(),
                        )
                        .ok()
                        .flatten()
                    })
                    .is_some_and(|file| file.identity().same_inode(original_identity));
                    if !link_unchanged
                        || !reopened_same
                        || !retained.is_stable()?
                        || !crate::core::bounded_file::walked_directory_is_current(
                            &root_authority,
                            revision_root,
                            &relative_directory,
                        )?
                        || !paths.insert(display_path.clone())
                    {
                        continue;
                    }
                    print_hub_row(repository, revision, quant, &display_path);
                }
                Err(_) => {}
            }
        }
    }
    Ok(())
}

fn print_retained_hub_row(
    retained: crate::core::bounded_file::StableRegularFile,
    path: PathBuf,
    repository: &str,
    revision: &str,
    paths: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let Ok(quant) = quant_type_from_gguf_file(retained.try_clone()?, &path) else {
        return Ok(());
    };
    if retained.is_stable()? && paths.insert(path.clone()) {
        print_hub_row(repository, revision, quant, &path);
    }
    Ok(())
}

fn print_hub_row(repository: &str, revision: &str, quant: QuantType, path: &Path) {
    println!(
        "{}",
        inventory_row([
            repository.to_owned(),
            short_revision(revision).to_owned(),
            quant.to_string(),
            "hf_hub_cache_unverified".to_owned(),
            "-".to_owned(),
            "?".to_owned(),
            path.to_string_lossy().into_owned(),
        ])
    );
}

fn resolve_hub_blob_target(
    revision: &str,
    relative_directory: &Path,
    target: &Path,
) -> Option<PathBuf> {
    use std::path::Component;

    if target.is_absolute() {
        return None;
    }
    let mut components = vec![OsString::from("snapshots"), OsString::from(revision)];
    for component in relative_directory.components() {
        let Component::Normal(name) = component else {
            return None;
        };
        components.push(name.to_os_string());
    }
    for component in target.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                components.pop()?;
            }
            Component::Normal(name) => components.push(name.to_os_string()),
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    if components.first().is_none_or(|value| value != "blobs") || components.len() < 2 {
        return None;
    }
    Some(components.into_iter().skip(1).collect())
}

fn open_relative_regular(root: &fs::File, relative: &Path) -> Result<Option<fs::File>> {
    use rustix::fs::{Mode, OFlags};

    let components = relative.components().collect::<Vec<_>>();
    if components.is_empty() {
        return Ok(None);
    }
    let mut directory = root.try_clone()?;
    for (index, component) in components.iter().enumerate() {
        let std::path::Component::Normal(name) = component else {
            return Ok(None);
        };
        let last = index + 1 == components.len();
        let mut flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
        if !last {
            flags |= OFlags::DIRECTORY;
        }
        let opened = match rustix::fs::openat(&directory, *name, flags, Mode::empty()) {
            Ok(fd) => fs::File::from(fd),
            Err(error)
                if matches!(
                    error,
                    rustix::io::Errno::NOENT | rustix::io::Errno::NOTDIR | rustix::io::Errno::LOOP
                ) =>
            {
                return Ok(None);
            }
            Err(error) => {
                return Err(std::io::Error::from_raw_os_error(error.raw_os_error()).into());
            }
        };
        if last {
            return Ok(opened.metadata()?.is_file().then_some(opened));
        }
        directory = opened;
    }
    Ok(None)
}

fn inventory_row<const N: usize>(fields: [String; N]) -> String {
    fields
        .iter()
        .map(|field| escape_inventory_field(field))
        .collect::<Vec<_>>()
        .join("\t")
}

fn escape_inventory_field(field: &str) -> String {
    let mut escaped = String::with_capacity(field.len());
    for character in field.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\r' => escaped.push_str("\\r"),
            '\n' => escaped.push_str("\\n"),
            character if character.is_control() && (character as u32) <= 0xff => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\x{:02X}", character as u32);
            }
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{{{:X}}}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod inventory_output_tests {
    use super::{
        directory_names, escape_inventory_field, inventory_row, open_directory_at,
        print_hub_snapshot_inventory_from_authority, print_hub_snapshot_inventory_with_hooks,
        visit_files_with_directory_hook,
    };
    use std::collections::BTreeSet;
    use std::ffi::OsString;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::Path;

    fn write_quant_gguf(path: &Path, file_type: u32) {
        let mut gguf = Vec::new();
        gguf.extend_from_slice(b"GGUF");
        gguf.extend_from_slice(&3_u32.to_le_bytes());
        gguf.extend_from_slice(&0_u64.to_le_bytes());
        gguf.extend_from_slice(&1_u64.to_le_bytes());
        gguf.extend_from_slice(&17_u64.to_le_bytes());
        gguf.extend_from_slice(b"general.file_type");
        gguf.extend_from_slice(&4_u32.to_le_bytes());
        gguf.extend_from_slice(&file_type.to_le_bytes());
        gguf.resize(256, 0);
        fs::write(path, gguf).unwrap();
    }

    #[test]
    fn inventory_fields_escape_row_and_terminal_control_characters() {
        let row = inventory_row([
            "owner/model".to_owned(),
            "artifact\tname\n\u{1b}\u{009b}[31m.gguf".to_owned(),
        ]);
        assert_eq!(row.lines().count(), 1);
        assert_eq!(row.matches('\t').count(), 1);
        assert!(!row.contains('\u{1b}'));
        assert!(!row.contains('\u{009b}'));
        assert_eq!(
            escape_inventory_field("artifact\\name\r\n\t\u{1b}"),
            "artifact\\\\name\\r\\n\\t\\x1B"
        );
    }

    #[test]
    fn queued_directory_replacement_cannot_escape_the_scan_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let queued = root.path().join("queued");
        fs::create_dir(&queued).unwrap();
        fs::write(queued.join("owned.gguf"), b"owned").unwrap();
        fs::write(outside.path().join("escape.gguf"), b"escape").unwrap();

        let mut replaced = false;
        let mut visited = Vec::new();
        visit_files_with_directory_hook(
            root.path(),
            |path, _, file| {
                visited.push((
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    file.is_stable().unwrap(),
                ));
                Ok(())
            },
            |directory| {
                if !replaced && directory == queued {
                    fs::rename(&queued, root.path().join("parked")).unwrap();
                    symlink(outside.path(), &queued).unwrap();
                    replaced = true;
                }
            },
        )
        .unwrap();

        assert_eq!(visited, vec![("owned.gguf".to_owned(), false)]);
        assert!(!visited.iter().any(|(name, _)| name == "escape.gguf"));
    }

    #[test]
    fn hub_cache_queued_directory_replacement_cannot_escape_snapshot() {
        let repository = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let revision = "a".repeat(40);
        let revision_root = repository.path().join("snapshots").join(&revision);
        let queued = revision_root.join("queued");
        let blobs = repository.path().join("blobs");
        fs::create_dir_all(&queued).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        write_quant_gguf(&queued.join("owned.gguf"), 15);
        write_quant_gguf(&outside.path().join("escape.gguf"), 15);
        let mut replaced = false;
        let mut paths = BTreeSet::new();
        let mut visited = 0;
        print_hub_snapshot_inventory_with_hooks(
            repository.path(),
            &revision_root,
            Some(&blobs.canonicalize().unwrap()),
            "owner/model",
            &revision,
            &mut paths,
            &mut visited,
            |directory| {
                if !replaced && directory == queued {
                    fs::rename(&queued, revision_root.join("parked")).unwrap();
                    symlink(outside.path(), &queued).unwrap();
                    replaced = true;
                }
            },
            |_| {},
        )
        .unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn hub_cache_model_ancestor_replacement_cannot_redirect_snapshot_authority() {
        use rustix::fs::{Mode, OFlags};

        let cache = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let model_name = "models--owner--model";
        let revision = "a".repeat(40);
        let model_root = cache.path().join(model_name);
        let revision_root = model_root.join("snapshots").join(&revision);
        fs::create_dir_all(&revision_root).unwrap();
        write_quant_gguf(&revision_root.join("owned.gguf"), 15);
        let outside_revision = outside.path().join("snapshots").join(&revision);
        fs::create_dir_all(&outside_revision).unwrap();
        write_quant_gguf(&outside_revision.join("escape.gguf"), 15);

        let cache_fd = fs::File::from(
            rustix::fs::open(
                cache.path(),
                OFlags::RDONLY
                    | OFlags::DIRECTORY
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK
                    | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .unwrap(),
        );
        let model_fd = open_directory_at(&cache_fd, model_name).unwrap().unwrap();
        fs::rename(&model_root, cache.path().join("parked")).unwrap();
        symlink(outside.path(), &model_root).unwrap();

        let snapshots_fd = open_directory_at(&model_fd, "snapshots").unwrap().unwrap();
        assert_eq!(
            directory_names(&snapshots_fd).unwrap(),
            vec![OsString::from(revision.clone())]
        );
        let revision_fd = open_directory_at(&snapshots_fd, &revision)
            .unwrap()
            .unwrap();
        let mut paths = BTreeSet::new();
        let mut visited = 0;
        print_hub_snapshot_inventory_from_authority(
            &revision_root,
            revision_fd,
            None,
            "owner/model",
            &revision,
            &mut paths,
            &mut visited,
            |_| {},
            |_| {},
        )
        .unwrap();

        assert!(paths.is_empty());
    }

    #[test]
    fn hub_cache_snapshot_retarget_after_parse_is_not_listed() {
        let repository = tempfile::tempdir().unwrap();
        let revision = "a".repeat(40);
        let revision_root = repository.path().join("snapshots").join(&revision);
        let blobs = repository.path().join("blobs");
        fs::create_dir_all(&revision_root).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        write_quant_gguf(&blobs.join("one"), 15);
        write_quant_gguf(&blobs.join("two"), 8);
        let snapshot = revision_root.join("model-q4_k_m.gguf");
        symlink("../../blobs/one", &snapshot).unwrap();
        let mut retargeted = false;
        let mut paths = BTreeSet::new();
        let mut visited = 0;
        print_hub_snapshot_inventory_with_hooks(
            repository.path(),
            &revision_root,
            Some(&blobs.canonicalize().unwrap()),
            "owner/model",
            &revision,
            &mut paths,
            &mut visited,
            |_| {},
            |path| {
                if !retargeted && path == snapshot {
                    fs::remove_file(&snapshot).unwrap();
                    symlink("../../blobs/two", &snapshot).unwrap();
                    retargeted = true;
                }
            },
        )
        .unwrap();
        assert!(paths.is_empty());
    }
}

pub(super) fn scan_roots(model_dirs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut roots = vec![
        managed_model_root()?,
        std::env::current_dir()?.join("models"),
    ];
    roots.extend(model_dirs.iter().cloned());
    roots.sort();
    roots.dedup();
    Ok(roots)
}

pub(super) fn visit_files(
    root: &Path,
    visit: impl FnMut(&Path, &fs::Metadata, crate::core::bounded_file::StableRegularFile) -> Result<()>,
) -> Result<()> {
    visit_files_with_directory_hook(root, visit, |_| {})
}

fn visit_files_with_directory_hook(
    root: &Path,
    mut visit: impl FnMut(
        &Path,
        &fs::Metadata,
        crate::core::bounded_file::StableRegularFile,
    ) -> Result<()>,
    mut before_directory: impl FnMut(&Path),
) -> Result<()> {
    use rustix::fs::{Mode, OFlags};

    let root_fd = match rustix::fs::open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(fd) => fs::File::from(fd),
        Err(_) => return Ok(()),
    };
    let root_authority = root_fd.try_clone()?;
    let root_path = root.to_path_buf();
    let mut queue =
        std::collections::VecDeque::from([(root_path.clone(), PathBuf::new(), root_fd, 0usize)]);
    let mut visited = 0usize;
    while let Some((directory, relative_directory, directory_fd, depth)) = queue.pop_front() {
        before_directory(&directory);
        let mut entries = match rustix::fs::Dir::read_from(&directory_fd) {
            Ok(entries) => entries
                .filter_map(std::result::Result::ok)
                .filter_map(|entry| {
                    let bytes = entry.file_name().to_bytes();
                    (!matches!(bytes, b"." | b"..")).then(|| OsString::from_vec(bytes.to_vec()))
                })
                .collect::<Vec<_>>(),
            Err(_) => continue,
        };
        entries.sort();
        for name in entries {
            visited += 1;
            if visited > MAX_SCAN_ENTRIES {
                return Ok(());
            }
            let entry_fd = match rustix::fs::openat(
                &directory_fd,
                &name,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            ) {
                Ok(fd) => fs::File::from(fd),
                _ => continue,
            };
            let metadata = match entry_fd.metadata() {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };
            let path = directory.join(&name);
            let relative = relative_directory.join(&name);
            if metadata.is_dir() && depth < MAX_SCAN_DEPTH {
                queue.push_back((path, relative, entry_fd, depth + 1));
            } else if metadata.is_file() {
                let Some(stable) = crate::core::bounded_file::StableRegularFile::from_walked_file(
                    root_authority.try_clone()?,
                    root_path.clone(),
                    relative,
                    &path,
                    entry_fd,
                    metadata.len(),
                )?
                else {
                    continue;
                };
                visit(&path, &metadata, stable)?;
            }
        }
    }
    Ok(())
}
