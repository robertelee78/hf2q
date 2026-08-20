fn write_partial(
    pending: &Directory,
    name: &str,
    bytes: &[u8],
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    remove_if_present(pending, name)?;
    let (mut file, identity) = unix::create_private(pending, name)?;
    #[cfg(test)]
    if test_io_fault_active(TestIoFault::CatalogWrite) {
        let prefix = bytes.len().div_ceil(2).max(1);
        file.write_all(&bytes[..prefix]).map_err(|error| {
            ManagedSessionCacheError::Filesystem(format!(
                "write injected managed catalog prefix: {error}"
            ))
        })?;
        return Err(ManagedSessionCacheError::StorageFull);
    }
    file.write_all(bytes)
        .map_err(|error| std_io("write managed cache partial", error))?;
    file.flush()
        .map_err(|error| std_io("flush managed cache partial", error))?;
    #[cfg(test)]
    test_io(TestIoFault::CatalogFullSync)?;
    unix::full_sync(&file)?;
    let actual = unix::identity_after_io(&file, pending, identity)?;
    if actual.size() != bytes.len() as u64 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache partial has the wrong exact length",
        ));
    }
    unix::verify_named(pending, name, actual)?;
    Ok(actual)
}

fn write_object_partial(
    pending: &Directory,
    name: &str,
    object: &EncodedObject<'_>,
) -> Result<EntryIdentity, ManagedSessionCacheError> {
    remove_if_present(pending, name)?;
    let (mut file, identity) = unix::create_private(pending, name)?;
    #[cfg(test)]
    if test_io_fault_active(TestIoFault::ObjectWrite) {
        file.write_all(&object.header).map_err(|error| {
            ManagedSessionCacheError::Filesystem(format!(
                "write injected managed object header: {error}"
            ))
        })?;
        let prefix = object.payload.len().div_ceil(2).max(1);
        file.write_all(&object.payload[..prefix]).map_err(|error| {
            ManagedSessionCacheError::Filesystem(format!(
                "write injected managed object prefix: {error}"
            ))
        })?;
        return Err(ManagedSessionCacheError::StorageFull);
    }
    for bytes in std::iter::once(object.header.as_slice())
        .chain(object.payload.chunks(OBJECT_WRITE_CHUNK_BYTES))
    {
        if let Err(error) = file.write_all(bytes) {
            #[cfg(test)]
            TEST_OBJECT_WRITE_FAILURE_LEN.with(|slot| {
                slot.set(file.metadata().ok().map(|metadata| metadata.len()));
            });
            return Err(std_io("write managed checkpoint partial", error));
        }
    }
    file.flush()
        .map_err(|error| std_io("flush managed checkpoint partial", error))?;
    #[cfg(test)]
    test_io(TestIoFault::ObjectFullSync)?;
    unix::full_sync(&file)?;
    let actual = unix::identity_after_io(&file, pending, identity)?;
    if actual.size() != object.total_bytes {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint partial has the wrong exact length",
        ));
    }
    unix::verify_named(pending, name, actual)?;
    Ok(actual)
}

fn read_object(
    objects: &Directory,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
    expected: Option<&RetainedObject>,
) -> Result<Vec<u8>, ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    if let Some(expected) = expected {
        verify_retained_object(objects, digest, expected)?;
        if !shard.identity().same_node(expected.shard_identity) || name != expected.name {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed checkpoint retained binding mismatched",
            ));
        }
    }
    let payload = read_and_verify_object(
        &shard,
        &name,
        key,
        digest,
        expected_bytes,
        maximum_bytes,
        true,
    )?
    .ok_or(ManagedSessionCacheError::InvalidLayout(
        "managed checkpoint payload was not retained",
    ))?;
    unix::verify_directory(objects, &digest[..2], &shard)?;
    if let Some(expected) = expected {
        verify_retained_object(objects, digest, expected)?;
    }
    Ok(payload)
}

fn verify_object(
    objects: &Directory,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
    expected: Option<&RetainedObject>,
) -> Result<(), ManagedSessionCacheError> {
    let shard = unix::open_directory(objects, &digest[..2])?;
    let name = format!("{digest}.checkpoint");
    if let Some(expected) = expected {
        verify_retained_object(objects, digest, expected)?;
        if !shard.identity().same_node(expected.shard_identity) || name != expected.name {
            return Err(ManagedSessionCacheError::InvalidLayout(
                "managed checkpoint retained binding mismatched",
            ));
        }
    }
    read_and_verify_object(
        &shard,
        &name,
        key,
        digest,
        expected_bytes,
        maximum_bytes,
        false,
    )?;
    unix::verify_directory(objects, &digest[..2], &shard)?;
    if let Some(expected) = expected {
        verify_retained_object(objects, digest, expected)?;
    }
    Ok(())
}

fn read_and_verify_object(
    shard: &Directory,
    name: &str,
    key: ManagedCheckpointKey,
    digest: &str,
    expected_bytes: u64,
    maximum_bytes: u64,
    retain_payload: bool,
) -> Result<Option<Vec<u8>>, ManagedSessionCacheError> {
    if expected_bytes < OBJECT_HEADER_BYTES as u64
        || expected_bytes > MAX_OBJECT_BYTES
        || expected_bytes > maximum_bytes
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed object exceeds its authorized exact bound",
        ));
    }
    let (mut file, identity) = unix::open_private(shard, name, false)?;
    if identity.size() != expected_bytes {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed object length mismatched",
        ));
    }
    let mut header = [0u8; OBJECT_HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    if &header[..8] != OBJECT_MAGIC
        || u32::from_le_bytes(header[8..12].try_into().unwrap()) != OBJECT_VERSION
        || header[12..44] != key.0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint compatibility binding mismatched",
        ));
    }
    let payload_len = u64::from_le_bytes(header[44..52].try_into().unwrap());
    if payload_len == 0
        || payload_len > MAX_CHECKPOINT_BYTES
        || payload_len.checked_add(OBJECT_HEADER_BYTES as u64) != Some(expected_bytes)
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint declares an invalid length",
        ));
    }
    let mut object_hasher = Sha256::new();
    object_hasher.update(header);
    let mut payload_hasher = Sha256::new();
    let mut payload = if retain_payload {
        let capacity = usize::try_from(payload_len).map_err(|_| {
            ManagedSessionCacheError::InvalidLayout("managed checkpoint is not addressable")
        })?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity).map_err(|_| {
            ManagedSessionCacheError::Filesystem(
                "managed checkpoint payload allocation failed".to_owned(),
            )
        })?;
        Some(bytes)
    } else {
        None
    };
    let mut remaining = payload_len;
    let mut buffer = [0u8; 64 * 1024];
    while remaining != 0 {
        let chunk = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        file.read_exact(&mut buffer[..chunk])
            .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
        object_hasher.update(&buffer[..chunk]);
        payload_hasher.update(&buffer[..chunk]);
        if let Some(payload) = &mut payload {
            payload.extend_from_slice(&buffer[..chunk]);
        }
        remaining -= chunk as u64;
    }
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?
        != 0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint has trailing bytes",
        ));
    }
    if hex::encode(object_hasher.finalize()) != digest
        || payload_hasher.finalize().as_slice() != &header[52..84]
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint digest mismatched",
        ));
    }
    let after = unix::identity_after_io(&file, shard, identity)?;
    if after != identity {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed checkpoint changed while reading",
        ));
    }
    unix::verify_named(shard, name, identity)?;
    Ok(payload)
}

fn read_exact_owned(
    parent: &Directory,
    name: &str,
    expected: usize,
) -> Result<Vec<u8>, ManagedSessionCacheError> {
    let (mut file, identity) = unix::open_private(parent, name, false)?;
    if identity.size() != expected as u64 {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file length mismatched",
        ));
    }
    let mut bytes = vec![0; expected];
    file.read_exact(&mut bytes)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?;
    let mut extra = [0u8; 1];
    if file
        .read(&mut extra)
        .map_err(|error| ManagedSessionCacheError::Filesystem(error.to_string()))?
        != 0
    {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file has trailing bytes",
        ));
    }
    let after = unix::identity_after_io(&file, parent, identity)?;
    if after != identity {
        return Err(ManagedSessionCacheError::InvalidLayout(
            "managed cache file changed while reading",
        ));
    }
    unix::verify_named(parent, name, identity)?;
    Ok(bytes)
}
