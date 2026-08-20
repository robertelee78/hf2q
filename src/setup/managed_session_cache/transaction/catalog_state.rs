fn load_highest_catalog(
    catalogs: &Directory,
) -> Result<(CatalogV1, Vec<RetainedCatalog>), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(catalogs, MAX_RETAINED_CATALOGS + 2)?;
    if names.is_empty() {
        return Ok((CatalogV1::empty(), Vec::new()));
    }
    let mut generations = BTreeMap::new();
    let mut parsed = Vec::new();
    for name in names {
        let (generation, digest) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if generations.insert(generation, digest.to_owned()).is_some() {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalogs contain an ambiguous generation",
            ));
        }
        let identity = unix::inspect_owned_regular(catalogs, &name, 1)?;
        if identity.size() == 0 || identity.size() > MAX_CATALOG_BYTES as u64 {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog final exceeds its byte cap",
            ));
        }
        let bytes = read_exact_owned(catalogs, &name, identity.size() as usize)?;
        if sha256_hex(&bytes) != digest {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog filename digest mismatched",
            ));
        }
        let catalog = CatalogV1::parse_exact(&bytes)?;
        if catalog.generation() != generation {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog filename generation mismatched",
            ));
        }
        parsed.push((generation, name, identity, catalog));
    }
    parsed.sort_by_key(|entry| entry.0);
    let selected = parsed
        .last()
        .ok_or(ManagedSessionCacheError::InvalidLayout(
            "managed catalog set is empty",
        ))?
        .3
        .clone();
    Ok((
        selected,
        parsed
            .into_iter()
            .map(|entry| RetainedCatalog {
                name: entry.1,
                identity: entry.2,
            })
            .collect(),
    ))
}

fn inventory_objects(
    objects: &Directory,
) -> Result<BTreeMap<String, RetainedObject>, ManagedSessionCacheError> {
    let shards = unix::list_names_bounded(objects, MAX_OBJECT_SHARDS)?;
    let mut found = BTreeMap::new();
    for shard_name in shards {
        if shard_name.len() != 2 || !is_lower_hex(&shard_name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed object shard name is invalid",
            ));
        }
        let shard = unix::open_recoverable_directory(objects, &shard_name)?;
        let names = unix::list_names_bounded(&shard, MAX_OBJECTS + 1)?;
        for name in names {
            let digest =
                name.strip_suffix(".checkpoint")
                    .ok_or(ManagedSessionCacheError::InvalidLayout(
                        "managed object name is invalid",
                    ))?;
            if digest.len() != 64 || !is_lower_hex(digest) || &digest[..2] != shard_name {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed object name does not match its shard",
                ));
            }
            if found.len() >= MAX_OBJECTS {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "managed object inventory exceeds its cap",
                ));
            }
            let identity = unix::inspect_owned_regular(&shard, &name, 1)?;
            found.insert(
                digest.to_owned(),
                RetainedObject {
                    shard: shard_name.clone(),
                    shard_identity: shard.identity(),
                    name,
                    identity,
                },
            );
        }
        unix::verify_directory(objects, &shard_name, &shard)?;
    }
    Ok(found)
}

fn verify_named_catalog_objects(
    store: &ManagedSessionCache,
    catalog: &CatalogV1,
    retained_objects: &BTreeMap<String, RetainedObject>,
    candidate: Option<(&str, &RetainedObject)>,
) -> Result<(), ManagedSessionCacheError> {
    for entry in catalog.entries() {
        let retained = candidate
            .filter(|(digest, _)| *digest == entry.object_sha256)
            .map(|(_, object)| object)
            .or_else(|| retained_objects.get(&entry.object_sha256))
            .ok_or(ManagedSessionCacheError::InvalidLayout(
                "managed catalog references an unretained object",
            ))?;
        if retained.identity.size() != entry.object_bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object length mismatched",
            ));
        }
        verify_retained_object(&store.directories.objects, &entry.object_sha256, retained)?;
    }
    Ok(())
}

fn validate_catalog_object_shapes(
    catalog: &CatalogV1,
    objects: &BTreeMap<String, RetainedObject>,
    limit_bytes: NonZeroU64,
) -> Result<(), ManagedSessionCacheError> {
    let mut referenced_bytes = 0u64;
    for entry in catalog.entries() {
        if entry.object_bytes > limit_bytes.get() {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        referenced_bytes = referenced_bytes
            .checked_add(entry.object_bytes)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if referenced_bytes > limit_bytes.get() {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        let object =
            objects
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog references a missing object",
                ))?;
        if object.identity.size() != entry.object_bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object length mismatched",
            ));
        }
    }
    Ok(())
}

fn require_retained_objects_match(
    catalog: &CatalogV1,
    retained: &BTreeMap<String, RetainedObject>,
    observed: &BTreeMap<String, RetainedObject>,
) -> Result<(), ManagedSessionCacheError> {
    for entry in catalog.entries() {
        let expected =
            retained
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog object lost its retained identity",
                ))?;
        let actual =
            observed
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog references a missing object",
                ))?;
        if expected.shard != actual.shard
            || !expected.shard_identity.same_node(actual.shard_identity)
            || expected.name != actual.name
            || expected.identity != actual.identity
        {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object changed after authorization",
            ));
        }
    }
    Ok(())
}

fn verify_retained_object(
    objects: &Directory,
    digest: &str,
    expected: &RetainedObject,
) -> Result<(), ManagedSessionCacheError> {
    if expected.shard != digest[..2] || expected.name != format!("{digest}.checkpoint") {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint retained name binding mismatched",
        ));
    }
    let shard = unix::open_directory(objects, &expected.shard)?;
    if !shard.identity().same_node(expected.shard_identity) {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint shard changed after authorization",
        ));
    }
    unix::verify_named(&shard, &expected.name, expected.identity)?;
    unix::verify_directory(objects, &expected.shard, &shard)
}

fn verify_catalog_objects(
    objects_dir: &Directory,
    catalog: &CatalogV1,
    objects: &BTreeMap<String, RetainedObject>,
    maximum_bytes: u64,
) -> Result<(), ManagedSessionCacheError> {
    for entry in catalog.entries() {
        let object =
            objects
                .get(&entry.object_sha256)
                .ok_or(ManagedSessionCacheError::InvalidLayout(
                    "managed catalog references a missing object",
                ))?;
        if object.identity.size() != entry.object_bytes {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalog object length mismatched",
            ));
        }
        let key = ManagedCheckpointKey::from_hex(&entry.key_sha256)?;
        verify_object(
            objects_dir,
            key,
            &entry.object_sha256,
            entry.object_bytes,
            maximum_bytes,
            Some(object),
        )?;
    }
    Ok(())
}

fn committed_unknown(error: ManagedSessionCacheError) -> ManagedSessionCacheError {
    match error {
        ManagedSessionCacheError::CommittedDurabilityUnknown(_) => error,
        other => ManagedSessionCacheError::CommittedDurabilityUnknown(other.to_string()),
    }
}
