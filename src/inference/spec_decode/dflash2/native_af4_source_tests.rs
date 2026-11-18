// Read-only Git object inspection is part of the parent-run fixture validation,
// never a build or model execution. Bind source bytes to the pinned tree, then
// derive host TUs from THAT Makefile rather than trusting capture-supplied names.
fn af4_verify_pinned_build(base: &Path, build: &serde_json::Value) -> Result<()> {
    let peer = Path::new(build["shader_provenance_contents"]["peer_root"].as_str().context("pinned peer root missing")?);
    let binary_path = Path::new(build["binaries"]["plain"]["path"].as_str().context("plain binary path missing")?);
    let directory = binary_path.parent().context("native build directory missing")?;
    let tree = std::process::Command::new("git").args(["--no-replace-objects", "-C"])
        .arg(peer).args(["ls-tree", "-rz", "--full-tree", AF4_PEER, "--", "runtime", "Makefile"])
        .env_remove("GIT_DIR").env_remove("GIT_WORK_TREE").output().context("inspect pinned peer Git source tree")?;
    ensure!(tree.status.success(), "cannot inspect pinned peer source tree: {}", String::from_utf8_lossy(&tree.stderr));
    let originals = build["source_inputs"].as_array().context("source inventory missing")?;
    let frozen = build["frozen_source_inputs"].as_array().context("frozen source inventory missing")?;
    let mut expected_sources = std::collections::BTreeSet::new();
    for entry in tree.stdout.split(|&byte| byte == 0).filter(|entry| !entry.is_empty()) {
        let entry = std::str::from_utf8(entry).context("non-UTF8 pinned source entry")?;
        let (metadata, name) = entry.split_once('\t').context("invalid pinned tree record")?;
        let fields = metadata.split_whitespace().collect::<Vec<_>>();
        ensure!(fields.len() == 3 && matches!(fields[0], "100644" | "100755") && fields[1] == "blob",
            "pinned production source is not a regular blob");
        ensure!(expected_sources.insert(name.to_string()), "duplicate pinned source entry");
        let original = af4_inventory_path(originals, &peer.join(name))?;
        let copy = af4_inventory_path(frozen, &directory.join("source").join(name))?;
        ensure!(original["bytes"] == copy["bytes"] && original["sha256"] == copy["sha256"], "frozen source differs: {name}");
        // SHA-1 here is Git's blob identity, NOT the artifact SHA-256 binding.
        let path = af4_descriptor(base, copy, false)?;
        let mut file = std::fs::File::open(&path)?;
        let mut digest = sha1::Sha1::new();
        digest.update(format!("blob {}\0", file.metadata()?.len()).as_bytes());
        let mut buffer = [0_u8; 65536];
        loop {
            let count = std::io::Read::read(&mut file, &mut buffer)?;
            if count == 0 { break; }
            digest.update(&buffer[..count]);
        }
        ensure!(hex::encode(digest.finalize()) == fields[2], "source is not the pinned Git blob: {name}");
    }
    ensure!(!expected_sources.is_empty() && originals.len() == expected_sources.len() && frozen.len() == expected_sources.len(),
        "complete pinned/frozen source inventory differs");
    let make = std::fs::read_to_string(directory.join("source/Makefile"))?;
    let mut units = af4_make_sources(&make, "ENGINE_CPP_SOURCES")?;
    units.extend(af4_make_sources(&make, "ENGINE_MM_SOURCES")?);
    let backend_rule = make.lines().find(|line| line.starts_with("$(ENGINE_METAL_RUNTIME_OBJECT):"))
        .context("pinned production MetalBackend rule missing")?;
    let backend = backend_rule.split_once(':').context("invalid MetalBackend rule")?.1.split_whitespace().next()
        .context("MetalBackend production source missing")?;
    ensure!(backend.starts_with("runtime/") && backend.ends_with(".mm"), "invalid MetalBackend source rule");
    units.push(backend.to_string());
    ensure!(units.len() == 90 && units.iter().collect::<std::collections::BTreeSet<_>>().len() == 90
        && build["translation_units"] == serde_json::to_value(&units)?, "translation units differ from the complete pinned Makefile");
    let harness = build["harness_inputs"].as_array().context("harness inventory missing")?;
    let frozen_harness = build["frozen_harness_inputs"].as_array().context("frozen harness inventory missing")?;
    let builder = af4_inventory_path(frozen_harness, &directory.join("harness/build.py"))?;
    let builder = std::fs::read_to_string(af4_descriptor(base, builder, false)?)?;
    let declaration = builder.lines().find(|line| line.trim_start().starts_with("harness = [artifact(ROOT / n) for n in ["))
        .context("source-bound builder harness declaration missing")?;
    let declared = declaration.split_once("for n in [").context("invalid harness declaration")?.1
        .split_once(']').context("invalid harness declaration extent")?.0;
    let mut harness_names = std::collections::BTreeSet::new();
    for name in declared.split(',') {
        let name = name.trim().trim_matches('\'');
        ensure!(!name.is_empty() && Path::new(name).components().count() == 1 && harness_names.insert(name), "invalid/duplicate harness source name");
        let copy = af4_inventory_path(frozen_harness, &directory.join("harness").join(name))?;
        let original = harness.iter().find(|record| record["path"].as_str().is_some_and(|path| Path::new(path).file_name() == Some(std::ffi::OsStr::new(name))))
            .context("declared original harness source missing")?;
        ensure!(original["sha256"] == copy["sha256"] && original["bytes"] == copy["bytes"], "frozen harness source differs");
    }
    ensure!(harness_names.len() == 6 && harness.len() == 6 && frozen_harness.len() == 6
        && harness_names.contains("build.py") && harness_names.contains("main.mm"), "complete source-declared frozen harness closure differs");
    let objects = build["objects"].as_array().context("production objects missing")?;
    let commands = build["commands"].as_array().context("build commands missing")?;
    let mut expected_objects = std::collections::BTreeMap::new();
    for unit in units {
        ensure!(expected_sources.contains(&unit), "Makefile TU absent from pinned source closure");
        let basename = unit.replace('/', "_");
        let basename = basename.rsplit_once('.').context("production source extension missing")?.0.to_string() + ".o";
        expected_objects.insert(directory.join(basename), directory.join("source").join(unit));
    }
    expected_objects.insert(directory.join("main.o"), directory.join("harness/main.mm"));
    expected_objects.insert(directory.join("DFlashDraft.observed.o"), directory.join("DFlashDraft.observed.cpp"));
    ensure!(objects.len() == 92 && expected_objects.len() == 92, "complete production object count differs");
    for (object, source) in &expected_objects {
        af4_inventory_path(objects, object)?;
        let matching = commands.iter().filter(|command| {
            let Some(args) = command.as_array() else { return false; };
            args.windows(2).any(|pair| pair[0] == "-c" && pair[1].as_str().is_some_and(|s| Path::new(s) == source))
                && args.windows(2).any(|pair| pair[0] == "-o" && pair[1].as_str().is_some_and(|s| Path::new(s) == object))
        }).count();
        ensure!(matching == 1, "source/object compilation binding missing or duplicated: {}", object.display());
    }
    ensure!(commands.iter().filter(|command| command.as_array().is_some_and(|args| args.iter().any(|arg| arg == "-c"))).count() == 92,
        "extra/missing production compilation command");
    let maps = build["link_maps"].as_array().context("link map inventory missing")?;
    let binaries = build["binaries"].as_object().context("binary inventory missing")?;
    ensure!(maps.len() == 2 && binaries.len() == 2 && binaries.contains_key("plain") && binaries.contains_key("observed"),
        "complete link map/binary inventory differs");
    for mode in ["plain", "observed"] {
        let map = directory.join(format!("link-{mode}.map"));
        af4_inventory_path(maps, &map)?;
        ensure!(binaries[mode]["path"].as_str().is_some_and(|path| Path::new(path) == directory.join(format!("native-draft-{mode}"))),
            "binary build directory differs");
        let flag = format!("-Wl,-map,{}", map.display());
        ensure!(commands.iter().filter(|command| command.as_array().is_some_and(|args|
            args.iter().any(|arg| arg == &flag) && args.windows(2).any(|pair| pair[0] == "-o" && pair[1] == binaries[mode]["path"]))).count() == 1,
            "binary/link map command binding missing or duplicated");
    }
    let shader_sources = build["shader_provenance_contents"]["source_inputs"].as_array().context("shader source closure missing")?;
    ensure!(shader_sources.len() == originals.len(), "shader/host pinned source closure count differs");
    for original in originals {
        let path = Path::new(original["path"].as_str().context("source path missing")?);
        let shader = af4_inventory_path(shader_sources, path)?;
        ensure!(original["sha256"] == shader["sha256"] && original["bytes"] == shader["bytes"], "shader/host pinned source closure differs");
    }
    Ok(())
}

fn af4_inventory_path<'a>(records: &'a [serde_json::Value], path: &Path) -> Result<&'a serde_json::Value> {
    let mut matching = records.iter().filter(|record| record["path"].as_str().is_some_and(|name| Path::new(name) == path));
    let record = matching.next().with_context(|| format!("complete inventory missing {}", path.display()))?;
    ensure!(matching.next().is_none(), "duplicate inventory path: {}", path.display());
    Ok(record)
}

fn af4_make_sources(make: &str, variable: &str) -> Result<Vec<String>> {
    let declaration = format!("{variable} := \\");
    let mut lines = make.lines().skip_while(|line| *line != declaration);
    ensure!(lines.next().is_some(), "pinned Makefile declaration missing: {variable}");
    let mut sources = Vec::new();
    loop {
        let line = lines.next().context("unterminated pinned Makefile source declaration")?.trim();
        let continued = line.ends_with('\\');
        let source = line.trim_end_matches('\\').trim();
        ensure!(source.starts_with("runtime/") && (source.ends_with(".cpp") || source.ends_with(".mm"))
            && source.split_whitespace().count() == 1, "invalid pinned Makefile source declaration");
        sources.push(source.to_string());
        if !continued { break; }
    }
    Ok(sources)
}
