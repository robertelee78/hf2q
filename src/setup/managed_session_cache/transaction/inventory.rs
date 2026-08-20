fn clean_orphans_with_parent(
    objects_dir: &Directory,
    catalog: &CatalogV1,
    objects: &BTreeMap<String, RetainedObject>,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let referenced: BTreeSet<_> = catalog
        .entries()
        .iter()
        .map(|entry| entry.object_sha256.as_str())
        .collect();
    for (digest, object) in objects {
        if !referenced.contains(digest.as_str()) {
            revalidate()?;
            let shard = unix::open_directory(objects_dir, &object.shard)?;
            unix::remove_file(&shard, &object.name, object.identity)?;
            unix::sync_directory(&shard)?;
            revalidate()?;
        }
    }
    Ok(())
}

fn clean_pending(
    pending: &Directory,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(pending, MAX_PENDING_NAMES + 1)?;
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        let identity = unix::inspect_reserved_partial(pending, &name)?;
        entries.push((name, identity));
    }
    for (name, identity) in entries {
        revalidate()?;
        unix::remove_file(pending, &name, identity)?;
        revalidate()?;
    }
    unix::sync_directory(pending)
}

fn verify_pending_inventory(pending: &Directory) -> Result<(), ManagedSessionCacheError> {
    for name in unix::list_names_bounded(pending, MAX_PENDING_NAMES + 1)? {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        unix::inspect_reserved_partial(pending, &name)?;
    }
    Ok(())
}

fn preflight_catalog_inventory(catalogs: &Directory) -> Result<(), ManagedSessionCacheError> {
    let mut generations = BTreeSet::new();
    for name in unix::list_names_bounded(catalogs, MAX_RETAINED_CATALOGS + 2)? {
        let (generation, _) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if !generations.insert(generation) {
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
    }
    Ok(())
}

fn verify_quarantine(quarantine: &Directory) -> Result<(), ManagedSessionCacheError> {
    let names = unix::list_names_bounded(quarantine, MAX_QUARANTINE_ENTRIES)?;
    for name in names {
        if !valid_quarantine_name(&name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed quarantine contains an unknown name",
            ));
        }
        unix::inspect_owned_regular(quarantine, &name, 1)?;
    }
    Ok(())
}

fn prune_old_catalogs(
    catalogs: &Directory,
    mut retained: Vec<RetainedCatalog>,
    revalidate: &impl Fn() -> Result<(), ManagedSessionCacheError>,
) -> Result<Vec<RetainedCatalog>, ManagedSessionCacheError> {
    retained.sort_by_key(|catalog| parse_catalog_name(&catalog.name).map(|entry| entry.0));
    while retained.len() > MAX_RETAINED_CATALOGS {
        let catalog = retained.remove(0);
        revalidate()?;
        unix::remove_file(catalogs, &catalog.name, catalog.identity)?;
        unix::sync_directory(catalogs)?;
        revalidate()?;
    }
    Ok(retained)
}

fn remove_object_expected(
    objects: &Directory,
    digest: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    unix::remove_file(&shard, &name, expected)?;
    #[cfg(test)]
    test_io(TestIoFault::DeletionDirectorySync)?;
    unix::sync_directory(&shard)
}

fn remove_if_present(parent: &Directory, name: &str) -> Result<(), ManagedSessionCacheError> {
    match unix::entry_identity(parent, name)? {
        Some(_) => {
            let identity = unix::inspect_reserved_partial(parent, name)?;
            unix::remove_file(parent, name, identity)?;
            unix::sync_directory(parent)
        }
        None => Ok(()),
    }
}

fn remove_exact_if_present(
    parent: &Directory,
    name: &str,
    expected: EntryIdentity,
) -> Result<(), ManagedSessionCacheError> {
    match unix::entry_identity(parent, name)? {
        Some(actual) if actual == expected => {
            unix::remove_file(parent, name, expected)?;
            unix::sync_directory(parent)
        }
        Some(_) => Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache partial changed before cleanup",
        )),
        None => Ok(()),
    }
}

fn revalidate_recovery(
    binding: &RuntimeConfigBinding,
    sessions: &Directory,
    directories: &StoreDirectories,
    lock: &StoreLock,
) -> Result<(), ManagedSessionCacheError> {
    binding
        .revalidate()
        .map_err(|error| ManagedSessionCacheError::StaleAuthorization(error.to_string()))?;
    verify_managed_root_inventory(sessions)?;
    unix::verify_directory(sessions, PENDING_DIR, &directories.pending)?;
    unix::verify_directory(sessions, OBJECTS_DIR, &directories.objects)?;
    unix::verify_directory(sessions, CATALOGS_DIR, &directories.catalogs)?;
    unix::verify_directory(sessions, QUARANTINE_DIR, &directories.quarantine)?;
    unix::verify_lock(sessions, LOCK_NAME, lock)
}

pub(super) fn verify_managed_root_inventory(
    sessions: &Directory,
) -> Result<(), ManagedSessionCacheError> {
    let expected = BTreeSet::from([
        LOCK_NAME.to_owned(),
        PENDING_DIR.to_owned(),
        OBJECTS_DIR.to_owned(),
        CATALOGS_DIR.to_owned(),
        QUARANTINE_DIR.to_owned(),
    ]);
    if unix::list_names_bounded(sessions, expected.len() + 1)? != expected {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "sessions directory contains an unknown managed entry",
        ));
    }
    Ok(())
}

pub(super) fn inventory_charge(
    sessions: &Directory,
    directories: &StoreDirectories,
) -> Result<u64, ManagedSessionCacheError> {
    verify_managed_root_inventory(sessions)?;
    let mut total = 0u64;
    for directory in [
        &directories.pending,
        &directories.objects,
        &directories.catalogs,
        &directories.quarantine,
    ] {
        total = total
            .checked_add(unix::directory_charge(directory)?)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let lock = unix::inspect_owned_regular(sessions, LOCK_NAME, 1)?;
    total = total
        .checked_add(lock.charge())
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    for name in unix::list_names_bounded(&directories.pending, MAX_PENDING_NAMES + 1)? {
        if name != OBJECT_PARTIAL && name != CATALOG_PARTIAL {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed pending directory contains an unknown name",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.pending, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let mut generations = BTreeSet::new();
    for name in unix::list_names_bounded(&directories.catalogs, MAX_RETAINED_CATALOGS + 2)? {
        let (generation, _) = parse_catalog_name(&name).ok_or(
            ManagedSessionCacheError::InvalidLayout("managed catalog name is invalid"),
        )?;
        if !generations.insert(generation) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed catalogs contain an ambiguous generation",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.catalogs, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    for name in unix::list_names_bounded(&directories.quarantine, MAX_QUARANTINE_ENTRIES)? {
        if !valid_quarantine_name(&name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed quarantine contains an unknown name",
            ));
        }
        total = total
            .checked_add(unix::inspect_owned_regular(&directories.quarantine, &name, 1)?.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    let shard_names = unix::list_names_bounded(&directories.objects, MAX_OBJECT_SHARDS)?;
    for shard_name in &shard_names {
        if shard_name.len() != 2 || !is_lower_hex(shard_name) {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed object shard name is invalid",
            ));
        }
    }
    let objects = inventory_objects(&directories.objects)?;
    for shard_name in shard_names {
        let shard = unix::open_directory(&directories.objects, &shard_name)?;
        total = total
            .checked_add(unix::directory_charge(&shard)?)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    for object in objects.values() {
        total = total
            .checked_add(object.identity.charge())
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    }
    Ok(total)
}
