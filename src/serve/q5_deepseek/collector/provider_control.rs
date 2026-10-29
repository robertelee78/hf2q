//! Root-supervised constructor/page-metadata control; never model admission.
use super::*;
use crate::serve::q5_deepseek::{packet::FileIdentity, page_observer};
use std::{fs::File, io::Write, os::unix::fs::MetadataExt};
const DATA: usize = 2 * 1024 * 1024;
const FILE_BYTES: usize = DATA + 65536;
const NATIVE_CEILING: usize = 8 * 1024 * 1024;
fn fixture_header() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.push(b'w');
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&((DATA / 4) as u64).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.resize(64, 0);
    b
}
#[test]
fn actual_provider_fixture_plan_is_finite_before_device() {
    assert_eq!(fixture_header().len(), 64);
    // Worst64KiB host page: native mapped parent<=DATA+64KiB, owned DATA.
    assert!(FILE_BYTES + DATA <= NATIVE_CEILING);
    // File, mapped/owned requested parents, transient owned read, one alias and
    // 1MiB metadata reserve. Not process/driver/PTE or physical-memory authority.
    assert!(
        FILE_BYTES * 2 + DATA * 2 + page_observer::PAGES_PER_ROOT as usize * 65536 + 1024 * 1024
            < 16 * 1024 * 1024
    );
}
#[test]
#[ignore = "root-supervised fresh-process <=8MiB native roots; actual SameFD paired metadata, no model/commands"]
fn actual_coalesced_nocopy_weight_joins_paired_samefd_page_provider() {
    paired_provider(false);
}
#[test]
#[ignore = "root-supervised NEW EOF-tail fixture/filter;16KiB host, no model/commands; old padded case unchanged"]
fn actual_coalesced_nocopy_eof_tail_keeps_full_requested_charge() {
    paired_provider(true);
}
fn paired_provider(eof_tail: bool) {
    use mlx_native::{
        LiveRootLimits, MlxDevice, NativeRootAuthority, RootCreationLimits, RootOriginSnapshot,
    };
    let file_bytes = if eof_tail { DATA + 64 } else { FILE_BYTES };
    if eof_tail {
        // SAFETY: fixed OS page metadata only, before Device/native allocation.
        assert_eq!(
            unsafe { libc::sysconf(libc::_SC_PAGESIZE) },
            16384,
            "selected actual host required; no skip"
        );
        assert_eq!(DATA + 16384 + DATA, 4_210_688);
    }
    let header = fixture_header();
    assert_eq!(header.len(), 64);
    assert!(file_bytes + DATA <= NATIVE_CEILING); // BEFORE Device or native creation.
    let authority = NativeRootAuthority::install_early_with_live(
        RootCreationLimits {
            max_attempted_requested_bytes: NATIVE_CEILING,
            max_root_attempts: 4,
            max_raw_constructor_attempts: 4,
            max_registered_devices: 1,
            max_rope_cache_entries: 1,
        },
        LiveRootLimits {
            max_live_and_pending_requested_bytes: NATIVE_CEILING,
            max_live_and_pending_roots: 4,
        },
    )
    .unwrap();
    let before = (
        mlx_native::dispatch_count(),
        mlx_native::commit_count(),
        mlx_native::sync_count(),
    );
    let mut fixture = tempfile::NamedTempFile::new().unwrap();
    fixture.write_all(&header).unwrap();
    fixture.write_all(&1f32.to_le_bytes()).unwrap();
    fixture.as_file().set_len(file_bytes as u64).unwrap();
    fixture.flush().unwrap();
    let fd = File::open(fixture.path()).unwrap();
    let stat = fd.metadata().unwrap();
    let identity = FileIdentity {
        device: stat.dev(),
        inode: stat.ino(),
        bytes: stat.len(),
        mtime_seconds: stat.mtime(),
        mtime_nanoseconds: stat.mtime_nsec(),
        ctime_seconds: stat.ctime(),
        ctime_nanoseconds: stat.ctime_nsec(),
    };
    let gguf = mlx_native::gguf::GgufFile::from_file(fd.try_clone().unwrap()).unwrap();
    assert_eq!(gguf.tensor_data_offset(), 64);
    let scope = HeaderScope::enter().unwrap();
    bind_actual_for_tiny(authority).unwrap();
    let mut wrong = identity;
    wrong.inode = wrong.inode.checked_add(1).unwrap();
    assert!(attach_source(&fd, wrong, 64).is_err());
    attach_source(&fd, identity, 64).unwrap();
    let device = MlxDevice::new().expect("actual Metal required; no skip");
    authority.register_device(&device).unwrap();
    let mapped = gguf.map_tensor_data(&device).unwrap();
    let weight = mapped.load_tensor("w").unwrap();
    let alias = mapped.load_tensor("w").unwrap();
    assert_eq!(weight.root_origin_snapshot(), alias.root_origin_snapshot());
    let (id, requested) = match weight.root_origin_snapshot() {
        RootOriginSnapshot::KnownNative {
            id,
            requested_bytes,
        } => (id, requested_bytes),
        other => panic!("actual known NoCopy root required: {other:?}"),
    };
    if eof_tail {
        assert_eq!(requested, DATA + 16384);
        assert!(requested > file_bytes);
    } else {
        assert!(requested <= file_bytes && requested >= DATA);
    }
    weight_observation::weight(&gguf, "w", &weight).unwrap();
    let mut raw = None;
    with_current(|c| {
        assert_eq!(c.weights.len(), 1);
        assert_eq!(c.weights[&id].bytes, requested);
        let span = c.weights[&id].file_span.unwrap();
        assert_eq!((span.0, span.1), (identity.device, identity.inode));
        assert_eq!(span.2, 0);
        assert_eq!(span.3, requested as u64);
        let actual = c.page_observations.get(&id).unwrap();
        assert!(matches!(
            actual,
            page_observer::RootObservation::PartialNoCredit { .. }
        ));
        raw = Some(serde_json::to_value(actual)?);
        Ok(())
    })
    .unwrap();
    let raw = raw.unwrap();
    assert_eq!(raw["root_id"].as_u64(), Some(id as u64));
    assert_eq!(raw["requested_bytes"].as_u64(), Some(requested as u64));
    for role in ["root", "ephemeral_alias"] {
        assert_eq!(raw[role]["returned_count"].as_u64(), Some(64));
        assert_eq!(raw[role]["dispositions"].as_array().unwrap().len(), 64);
    }
    weight_observation::weight(&gguf, "w", &alias).unwrap();
    with_current(|c| {
        assert_eq!(c.weights.len(), 1);
        assert_eq!(c.page_observations.len(), 1);
        assert_eq!(c.weights[&id].bytes, requested);
        Ok(())
    })
    .unwrap();
    // Data oracle AFTER the first observations; no prequery CPU faulting.
    assert_eq!(weight.as_slice::<f32>().unwrap()[0], 1.0);
    assert!(matches!(
        page_observer::actual(&weight, &fd, wrong, 0, id, requested),
        page_observer::RootObservation::UnknownFullCharge { .. }
    ));
    assert!(matches!(
        page_observer::actual(&weight, &fd, identity, 1, id, requested),
        page_observer::RootObservation::UnknownFullCharge { .. }
    ));
    let owned = gguf.load_tensor("w", &device).unwrap();
    weight_observation::weight(&gguf, "w", &owned).unwrap();
    with_current(|c| {
        assert_eq!(c.weights.len(), 2);
        let unknown = c
            .page_observations
            .values()
            .filter(|o| matches!(o, page_observer::RootObservation::UnknownFullCharge { .. }))
            .count();
        assert_eq!(unknown, 1);
        assert_eq!(
            c.weights.values().map(|r| r.bytes).sum::<usize>(),
            requested + DATA
        );
        Ok(())
    })
    .unwrap();
    if eof_tail {
        with_current(|c| {
            let v = serde_json::to_value(coverage::actual(c)?)?;
            assert_eq!(v["full_requested_bytes"], 4_210_688);
            assert_eq!(
                v["mapped_logical_file_byte_union_bytes"],
                (DATA + 64) as u64
            );
            assert_eq!(v["registered_file_page_union_bytes"], requested as u64);
            assert_eq!(v["physical_credit_bytes"], 0);
            println!("Q5_HF_PROVIDER_EOF_COVERAGE {}", v);
            Ok(())
        })
        .unwrap();
    }
    capture_native_for_tiny().unwrap();
    identity.matches(&fd.metadata().unwrap()).unwrap();
    let after = (
        mlx_native::dispatch_count(),
        mlx_native::commit_count(),
        mlx_native::sync_count(),
    );
    assert_eq!(before, after);
    let native = authority.live_snapshot().unwrap();
    assert!(native.covered);
    assert_eq!(native.pending_roots, 0);
    assert_eq!(native.live_roots, 2);
    assert_eq!(native.live_requested_bytes, requested + DATA);
    println!("Q5_HF_PROVIDER_COUNTERS before_dispatch={} before_commit={} before_sync={} after_dispatch={} after_commit={} after_sync={} live_roots={} live_requested_bytes={} pending_roots={}",before.0,before.1,before.2,after.0,after.1,after.2,native.live_roots,native.live_requested_bytes,native.pending_roots);
    println!("Q5_HF_PROVIDER_PAIRED {}", raw);
    println!("Q5_HF_PROVIDER_COMPLETE roots=2 queried_roots=1 paired_pages=64 requested_bytes={} dispatch={} commit={} sync={} physical_credit_bytes=0 model=false commands=false",requested+DATA,after.0-before.0,after.1-before.1,after.2-before.2);
    // Explicit owners kept through every query/assertion; dropping is NOT retirement proof.
    drop(owned);
    drop(alias);
    drop(weight);
    drop(mapped);
    drop(gguf);
    drop(scope);
    drop(fd);
    drop(fixture);
}
