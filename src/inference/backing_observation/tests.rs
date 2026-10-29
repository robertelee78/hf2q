use super::*;
#[test]
fn actual_file_offset_join_keeps_parent_padding_and_checks_overflow() {
    // Actual-used join arithmetic: mmap begins4096, Metal begins8192,
    // logical view begins8200. Root tail remains explicit, not content bytes.
    assert_eq!(join_offsets(4096,8,16,4120,4096,8200,8216).unwrap(),(4096,8192,4072));
    assert!(join_offsets(4096,8,16,4121,4096,8200,8216).is_err());
    assert!(join_offsets(0,9,16,32,32,8,32).is_err());
    assert!(join_offsets(4096,0,1,1,1,0,10).is_err());
    assert!(join_offsets(0,0,2,2,2,u64::MAX, u64::MAX).is_err());
}
#[test]
fn actual_retained_fd_identity_is_not_content_or_path_identity() {
    use std::io::Write;
    let mut a=tempfile::tempfile().unwrap();
    let mut b=tempfile::tempfile().unwrap();
    a.write_all(b"same").unwrap();b.write_all(b"same").unwrap();
    let sa=FileStat::read(&a).unwrap();
    assert_eq!(sa,FileStat::read(&a.try_clone().unwrap()).unwrap());
    assert_ne!((sa.device,sa.inode),(FileStat::read(&b).unwrap().device,FileStat::read(&b).unwrap().inode));
    a.set_len(5).unwrap(); assert_ne!(sa,FileStat::read(&a).unwrap());
}
#[test]
fn actual_header_owner_binds_same_retained_fd_and_refuses_changed_stat() {
    use std::io::Write;
    let mut tmp=tempfile::NamedTempFile::new().unwrap();
    let file=tmp.as_file_mut();
    file.write_all(b"GGUF").unwrap();file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();file.write_all(&0u64.to_le_bytes()).unwrap();
    let retained=file.try_clone().unwrap();
    assert!(ObservedGgufOwner::from_file(retained.try_clone().unwrap()).is_err());
    let owner=ObservedGgufOwner::from_file(File::open(tmp.path()).unwrap()).unwrap();
    assert_eq!(owner.file_stat(),FileStat::read(&retained).unwrap());
    owner.require_unchanged().unwrap();retained.set_len(33).unwrap();
    assert!(owner.require_unchanged().is_err());
}

#[test]
#[ignore = "fresh-process <=64KiB same-FD mapped weight ownership join; root supervised only"]
fn actual_mapped_weight_owner_aliases_join_same_inode_and_file_offsets() {
    use std::io::Write;
    use mlx_native::{NativeRootAuthority,RootCreationLimits};
    let authority=NativeRootAuthority::install_early(RootCreationLimits{
        max_attempted_requested_bytes:65536,max_root_attempts:4,max_raw_constructor_attempts:4,
        max_registered_devices:1,max_rope_cache_entries:1}).unwrap();
    let mut tmp=tempfile::NamedTempFile::new().unwrap();
    let mut bytes=Vec::new();bytes.extend_from_slice(b"GGUF");bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());bytes.push(b'w');
    bytes.extend_from_slice(&1u32.to_le_bytes());bytes.extend_from_slice(&4u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(64,0);for value in [1f32,2.,3.,4.] {bytes.extend_from_slice(&value.to_le_bytes());}
    tmp.write_all(&bytes).unwrap();tmp.flush().unwrap();
    let owner=ObservedGgufOwner::from_file(File::open(tmp.path()).unwrap()).unwrap();
    let device=MlxDevice::new().expect("actual Metal required; no skip");authority.register_device(&device).unwrap();
    let mapped=owner.map_weights(&device).unwrap();
    let first=mapped.load("w").unwrap();let alias=mapped.load("w").unwrap();
    let a=first.snapshot().unwrap();let b=alias.snapshot().unwrap();
    assert!(a.role.native_origin_known);assert_eq!(a.role.backing.origin,b.role.backing.origin);
    assert_eq!(a.file,owner.file_stat());assert_eq!(a.logical_file_offset,64);
    assert_eq!(a.mapping_file_offset,0);assert_eq!(a.registered_file_offset,0);
    assert_eq!(first.buffer().as_slice::<f32>().unwrap(),&[1.,2.,3.,4.]);
    drop(mapped);drop(owner); // Each observed weight still retains its exact FD + mmap.
    assert_eq!(first.snapshot().unwrap().file,b.file);
    println!("HF_BACKING_ROLE_JOIN actual_same_fd=true alias_root_same=true logical_file_offset=64 physical_credit_bytes=0 gpu_work=false");
}
