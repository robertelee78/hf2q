pub(super) fn commit(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    commit_with_hook(store, key, payload, |_barrier| {
        #[cfg(test)]
        super::tests::abort_at_managed_cache_barrier(_barrier);
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn commit_with_test_hook(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
    hook: impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    commit_with_hook(store, key, payload, hook)
}

fn commit_with_hook(
    store: &ManagedSessionCache,
    key: ManagedCheckpointKey,
    payload: &[u8],
    mut hook: impl FnMut(ManagedCacheBarrier) -> Result<(), ManagedSessionCacheError>,
) -> Result<ManagedCommitOutcome, ManagedSessionCacheError> {
    let payload_bytes =
        u64::try_from(payload.len()).map_err(|_| ManagedSessionCacheError::QuotaExceeded)?;
    if payload_bytes == 0
        || payload_bytes
            .checked_add(OBJECT_HEADER_BYTES as u64)
            .is_none_or(|bytes| bytes > store.limit_bytes.get())
    {
        return Err(ManagedSessionCacheError::QuotaExceeded);
    }
    let object = encode_object(key, payload)?;
    let object_bytes = object.total_bytes;
    let object_digest = object.digest.clone();
    let key_hex = key.hex();
    let mut state = store.state.lock().map_err(|_| {
        ManagedSessionCacheError::Filesystem("managed cache mutex poisoned".to_owned())
    })?;
    if state.needs_recovery {
        return Err(ManagedSessionCacheError::RecoveryRequired);
    }
    let result = (|| {
        reconcile_uncommitted(store, &state)?;
        store.revalidate_endpoint()?;
        hook(ManagedCacheBarrier::AuthorizationValidated)?;
        let replaced_digest = state
            .catalog
            .find(&key_hex)
            .map(|entry| entry.object_sha256.clone());
        if let Some(existing) = state.catalog.find(&key_hex) {
            if existing.object_sha256 == object_digest && existing.object_bytes == object_bytes {
                let restored = read_object(
                    &store.directories.objects,
                    key,
                    &existing.object_sha256,
                    existing.object_bytes,
                    store.limit_bytes.get(),
                    state.retained_objects.get(&existing.object_sha256),
                )?;
                if restored == payload {
                    store.revalidate_endpoint()?;
                    return Ok(ManagedCommitOutcome::AlreadyPresent);
                }
            }
        }

        let (proposed, proposed_bytes) = loop {
            let candidate = make_room(
                store,
                &mut state,
                key_hex.clone(),
                object_digest.clone(),
                object_bytes,
                &mut hook,
            )?;
            match enforce_free_floor(&store.directories.pending, object_bytes, candidate.1.len()) {
                Ok(()) => break candidate,
                Err(ManagedSessionCacheError::FreeSpaceFloor)
                    if !state.catalog.entries().is_empty() =>
                {
                    evict_one(store, &mut state, &mut hook)?;
                }
                Err(error) => return Err(error),
            }
        };

        let shard_name = &object_digest[..2];
        let shard = unix::ensure_directory(&store.directories.objects, shard_name)?;
        unix::verify_directory(&store.directories.objects, shard_name, &shard)?;
        let current_charge = inventory_charge(&store.sessions, &store.directories)?;
        let stat = unix::volume_space(&store.directories.pending)?;
        let block = stat.f_frsize.max(1);
        let planned = round_up(object_bytes, block)?
            .checked_add(round_up(proposed_bytes.len() as u64, block)?)
            .and_then(|value| value.checked_add(block.saturating_mul(METADATA_BLOCK_MARGIN)))
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?;
        if current_charge
            .checked_add(planned)
            .ok_or(ManagedSessionCacheError::QuotaExceeded)?
            > store.limit_bytes.get()
        {
            return Err(ManagedSessionCacheError::QuotaExceeded);
        }

        let partial_identity =
            write_object_partial(&store.directories.pending, OBJECT_PARTIAL, &object)?;
        hook(ManagedCacheBarrier::ObjectPartialSynced)?;
        store.revalidate_endpoint()?;
        unix::verify_named(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        hook(ManagedCacheBarrier::BeforeObjectPublish)?;
        store.revalidate_endpoint()?;
        unix::verify_named(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        let final_name = format!("{object_digest}.checkpoint");
        let created = unix::rename_noreplace(
            &store.directories.pending,
            OBJECT_PARTIAL,
            &shard,
            &final_name,
        )?;
        if created {
            unix::verify_named(&shard, &final_name, partial_identity)?;
        }
        hook(ManagedCacheBarrier::ObjectPublished)?;
        if created {
            #[cfg(test)]
            test_io(TestIoFault::ObjectDirectorySync)?;
            unix::sync_directory(&shard)?;
            #[cfg(test)]
            test_io(TestIoFault::ObjectPendingDirectorySync)?;
            unix::sync_directory(&store.directories.pending)?;
        } else {
            remove_exact_if_present(&store.directories.pending, OBJECT_PARTIAL, partial_identity)?;
        }
        unix::verify_directory(&store.directories.objects, shard_name, &shard)?;
        if created {
            unix::verify_named(&shard, &final_name, partial_identity)?;
        }
        let final_identity = if created {
            partial_identity
        } else {
            let identity = unix::inspect_owned_regular(&shard, &final_name, 1)?;
            if identity.size() != object_bytes {
                return Err(ManagedSessionCacheError::InvalidLayout(
                    "content-addressed managed object has the wrong length",
                ));
            }
            identity
        };
        let retained_object = RetainedObject {
            shard: shard_name.to_owned(),
            shard_identity: shard.identity(),
            name: final_name.clone(),
            identity: final_identity,
        };
        #[cfg(test)]
        test_io(TestIoFault::ObjectFinalFullSync)?;
        unix::full_sync_named(&shard, &final_name, final_identity)?;
        hook(ManagedCacheBarrier::ObjectDirectorySynced)?;
        let adopted = read_object(
            &store.directories.objects,
            key,
            &object_digest,
            object_bytes,
            store.limit_bytes.get(),
            Some(&retained_object),
        )?;
        if adopted != payload {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "content-addressed managed object did not match its digest",
            ));
        }
        let publication = (|| {
            store.revalidate_endpoint()?;
            enforce_free_floor(&store.directories.pending, 0, proposed_bytes.len())?;
            if inventory_charge(&store.sessions, &store.directories)? > store.limit_bytes.get() {
                return Err(ManagedSessionCacheError::QuotaExceeded);
            }

            reserve_catalog_transaction(store, &mut state, proposed_bytes.len(), &mut hook)?;
            let replaced_identity = replaced_digest
                .as_deref()
                .filter(|digest| *digest != object_digest)
                .filter(|digest| {
                    state
                        .catalog
                        .entries()
                        .iter()
                        .any(|entry| entry.object_sha256 == **digest)
                })
                .filter(|digest| {
                    !proposed
                        .entries()
                        .iter()
                        .any(|entry| entry.object_sha256 == **digest)
                })
                .map(|digest| {
                    state
                        .retained_objects
                        .get(digest)
                        .map(|object| (digest.to_owned(), object.identity))
                        .ok_or(ManagedSessionCacheError::InvalidLayout(
                            "replaced managed object lost its retained identity",
                        ))
                })
                .transpose()?;
            let published_name = publish_catalog(
                store,
                &proposed,
                &proposed_bytes,
                &state.retained_objects,
                Some((&object_digest, &retained_object)),
                &mut hook,
            )?;
            Ok((published_name, replaced_identity))
        })();
        let (published_name, replaced_identity) = match publication {
            Ok(published) => published,
            Err(error @ ManagedSessionCacheError::CommittedDurabilityUnknown(_)) => {
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        state.catalog = proposed;
        state.retained_catalogs.push(published_name);
        state
            .retained_objects
            .insert(object_digest.clone(), retained_object);
        if let Some((digest, identity)) = replaced_identity {
            hook(ManagedCacheBarrier::BeforeObjectDelete).map_err(committed_unknown)?;
            store.revalidate_endpoint().map_err(committed_unknown)?;
            remove_object_expected(&store.directories.objects, &digest, identity)
                .map_err(committed_unknown)?;
            state.retained_objects.remove(&digest);
            hook(ManagedCacheBarrier::ObjectDeleted).map_err(committed_unknown)?;
        }
        store.revalidate_endpoint().map_err(committed_unknown)?;
        #[cfg(test)]
        test_io(TestIoFault::EndpointLockFullSync).map_err(committed_unknown)?;
        unix::full_sync_lock(&store.lock).map_err(committed_unknown)?;
        hook(ManagedCacheBarrier::EndpointSynced).map_err(|error| {
            ManagedSessionCacheError::CommittedDurabilityUnknown(error.to_string())
        })?;
        if inventory_charge(&store.sessions, &store.directories).map_err(committed_unknown)?
            > store.limit_bytes.get()
        {
            return Err(ManagedSessionCacheError::CommittedDurabilityUnknown(
                "committed managed cache exceeds its aggregate cap".to_owned(),
            ));
        }
        Ok(ManagedCommitOutcome::Published)
    })();
    let mut cleanup_failed = false;
    if let Err(error) = &result {
        if !matches!(
            error,
            ManagedSessionCacheError::CommittedDurabilityUnknown(_)
        ) {
            cleanup_failed = reconcile_uncommitted(store, &state).is_err();
        }
    }
    if cleanup_failed
        || matches!(
            result,
            Err(ManagedSessionCacheError::CommittedDurabilityUnknown(_))
        )
    {
        state.needs_recovery = true;
    }
    result
}
