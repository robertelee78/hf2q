pub(super) fn restore(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
) -> Result<Option<Vec<u8>>, ManagedSessionCacheError> {
    let state = store.state.lock().map_err(|_| {
        ManagedSessionCacheError::Filesystem("managed cache mutex poisoned".to_owned())
    })?;
    if state.needs_recovery {
        return Err(ManagedSessionCacheError::RecoveryRequired);
    }
    store.revalidate_endpoint()?;
    let Some(entry) = state.catalog.find(&key.hex()) else {
        return Ok(None);
    };
    let payload = read_object(
        &store.directories.objects,
        key,
        &entry.object_sha256,
        entry.object_bytes,
        store.limit_bytes.get(),
        state.retained_objects.get(&entry.object_sha256),
    )?;
    store.revalidate_endpoint()?;
    Ok(Some(payload))
}

fn reconcile_uncommitted(
    store: &ManagedSessionCache,
    state: &super::StoreState,
) -> Result<(), ManagedSessionCacheError> {
    let revalidate = || store.revalidate_endpoint();
    revalidate()?;
    clean_pending(&store.directories.pending, &revalidate)?;
    let objects = inventory_objects(&store.directories.objects)?;
    validate_catalog_object_shapes(&state.catalog, &objects, store.limit_bytes)?;
    require_retained_objects_match(&state.catalog, &state.retained_objects, &objects)?;
    clean_orphans_with_parent(
        &store.directories.objects,
        &state.catalog,
        &objects,
        &revalidate,
    )?;
    revalidate()?;
    if inventory_charge(&store.sessions, &store.directories)? > store.limit_bytes.get() {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    Ok(())
}

fn make_room(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    key_hex: String,
    object_digest: String,
    object_bytes: u64,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(CatalogV1, Vec<u8>), ManagedSessionCacheError> {
    loop {
        if state.catalog.find(&key_hex).is_none()
            && state.catalog.entries().len() >= MAX_LIVE_ENTRIES
        {
            evict_one(store, state, hook)?;
            continue;
        }
        let proposed = state.catalog.with_entry(CatalogEntryV1::new(
            key_hex.clone(),
            object_digest.clone(),
            object_bytes,
        ))?;
        let proposed_bytes = proposed.to_canonical_bytes()?;
        let charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let required = round_up(object_bytes, block)?
            .checked_add(round_up(proposed_bytes.len() as u64, block)?)
            .and_then(|value| value.checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN)))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if charge
            .checked_add(required)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            <= store.limit_bytes.get()
        {
            return Ok((proposed, proposed_bytes));
        }
        evict_one(store, state, hook)?;
    }
}

fn evict_one(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    let victim = state
        .catalog
        .entries()
        .iter()
        .min_by_key(|entry| (entry.last_committed_generation, entry.key_sha256.as_str()))
        .cloned()
        .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
    let victim_identity = state
        .retained_objects
        .get(&victim.object_sha256)
        .ok_or(ManagedSessionCacheError::InvalidLayout(
            "managed catalog object lost its retained identity",
        ))?
        .identity;
    let removed = BTreeSet::from([victim.key_sha256.clone()]);
    let pruned = state.catalog.without_keys(&removed)?;
    let bytes = pruned.to_canonical_bytes()?;
    enforce_free_floor(&store.directories.pending, 0, bytes.len())?;
    reserve_catalog_transaction(store, state, bytes.len(), hook)?;
    let name = publish_catalog(store, &pruned, &bytes, &state.retained_objects, None, hook)?;
    state.catalog = pruned;
    state.retained_catalogs.push(name);
    if !state
        .catalog
        .entries()
        .iter()
        .any(|entry| entry.object_sha256 == victim.object_sha256)
    {
        hook(ManagedCacheBarrier::BeforeObjectDelete).map_err(committed_unknown)?;
        store.revalidate_endpoint().map_err(committed_unknown)?;
        remove_object_expected(
            &store.directories.objects,
            &victim.object_sha256,
            victim_identity,
        )
        .map_err(committed_unknown)?;
        note_test_space_reclaimed(victim_identity.charge());
        state.retained_objects.remove(&victim.object_sha256);
        hook(ManagedCacheBarrier::ObjectDeleted).map_err(committed_unknown)?;
    }
    Ok(())
}

fn reserve_catalog_transaction(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    bytes: usize,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    while state.retained_catalogs.len() >= MAX_RETAINED_CATALOGS {
        prune_one_catalog_history(store, state, hook)?;
    }
    loop {
        let charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let planned = round_up(bytes as u64, block)?
            .checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if charge
            .checked_add(planned)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            <= store.limit_bytes.get()
        {
            return Ok(());
        }
        if state.retained_catalogs.len() <= 1 {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }
        prune_one_catalog_history(store, state, hook)?;
    }
}

fn prune_one_catalog_history(
    store: &ManagedSessionCache,
    state: &mut super::StoreState,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<(), ManagedSessionCacheError> {
    if state.retained_catalogs.len() <= 1 {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    state.retained_catalogs.sort_by_key(|catalog| {
        parse_catalog_name(&catalog.name)
            .map(|entry| entry.0)
            .unwrap_or(u64::MAX)
    });
    let stale_name = state.retained_catalogs[0].name.clone();
    let stale_identity = state.retained_catalogs[0].identity;
    hook(ManagedCacheBarrier::BeforeCatalogHistoryPrune)?;
    store.revalidate_endpoint()?;
    unix::remove_file(&store.directories.catalogs, &stale_name, stale_identity)
        .map_err(committed_unknown)?;
    #[cfg(test)]
    test_io(TestIoFault::CatalogHistoryDirectorySync).map_err(committed_unknown)?;
    unix::sync_directory(&store.directories.catalogs).map_err(committed_unknown)?;
    store.revalidate_endpoint().map_err(committed_unknown)?;
    state.retained_catalogs.remove(0);
    hook(ManagedCacheBarrier::CatalogHistoryPruned).map_err(committed_unknown)?;
    Ok(())
}

fn publish_catalog(
    store: &ManagedSessionCache,
    catalog: &CatalogV1,
    bytes: &[u8],
    retained_objects: &BTreeMap<String, RetainedObject>,
    candidate: Option<(&str, &RetainedObject)>,
    hook: &mut impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<RetainedCatalog, ManagedSessionCacheError> {
    let directories = &store.directories;
    let digest = sha256_hex(bytes);
    let final_name = catalog_name(catalog.generation(), &digest);
    let partial_identity = write_partial(&directories.pending, CATALOG_PARTIAL, bytes)?;
    hook(ManagedCacheBarrier::CatalogPartialSynced)?;
    store.revalidate_endpoint()?;
    unix::verify_named(&directories.pending, CATALOG_PARTIAL, partial_identity)?;
    hook(ManagedCacheBarrier::BeforeCatalogPublish)?;
    store.revalidate_endpoint()?;
    unix::verify_named(&directories.pending, CATALOG_PARTIAL, partial_identity)?;
    verify_named_catalog_objects(store, catalog, retained_objects, candidate)?;
    let created = unix::rename_noreplace(
        &directories.pending,
        CATALOG_PARTIAL,
        &directories.catalogs,
        &final_name,
    )?;
    let final_identity = if created {
        unix::verify_named(&directories.catalogs, &final_name, partial_identity)
            .map_err(committed_unknown)?;
        partial_identity
    } else {
        let identity = unix::inspect_owned_regular(&directories.catalogs, &final_name, 1)
            .map_err(committed_unknown)?;
        if identity.size() != bytes.len() as u64 {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "immutable managed catalog has the wrong length".to_owned(),
            ));
        }
        identity
    };
    hook(ManagedCacheBarrier::CatalogPublished)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    if !created {
        unix::full_sync_named(&directories.catalogs, &final_name, final_identity)
            .map_err(committed_unknown)?;
        let existing = read_exact_owned(&directories.catalogs, &final_name, bytes.len())
            .map_err(committed_unknown)?;
        if existing != bytes {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "immutable managed catalog name was reused for different bytes".to_owned(),
            ));
        }
        remove_exact_if_present(&directories.pending, CATALOG_PARTIAL, partial_identity)
            .map_err(committed_unknown)?;
    }
    #[cfg(test)]
    test_io(TestIoFault::CatalogDirectorySync).map_err(committed_unknown)?;
    unix::sync_directory(&directories.catalogs)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    hook(ManagedCacheBarrier::CatalogDirectorySynced)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    #[cfg(test)]
    test_io(TestIoFault::PendingDirectorySync).map_err(committed_unknown)?;
    unix::sync_directory(&directories.pending)
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    #[cfg(test)]
    test_io(TestIoFault::CatalogFinalFullSync).map_err(committed_unknown)?;
    unix::full_sync_named(&directories.catalogs, &final_name, final_identity)
        .map_err(committed_unknown)?;
    let existing = read_exact_owned(&directories.catalogs, &final_name, bytes.len())
        .map_err(|error| ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string()))?;
    if existing != bytes {
        return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
            "published catalog bytes changed".to_owned(),
        ));
    }
    unix::verify_named(&directories.catalogs, &final_name, final_identity)
        .map_err(committed_unknown)?;
    Ok(RetainedCatalog {
        name: final_name,
        identity: final_identity,
    })
}
