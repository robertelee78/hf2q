//! Full ORIGINAL manifest/payload audit before canonical projection.
use super::*;
use sha2::{Digest,Sha256};
use std::{collections::BTreeSet,fs::File,io::Read};
pub(super) fn original(root:&Path,trace:&Trace,expected:&[ExpectedTensor])->Result<Value> {
    ensure!(trace.schema=="hf2q.adr063.invariance.v2" && trace.coverage.expected==expected
        && trace.coverage.captured.len()==expected.len() && trace.coverage.errors.is_empty(),
        "complete original source-derived manifest required");
    let required=expected.iter().map(|e|e.at.clone()).collect::<BTreeSet<_>>();
    ensure!(required.len()==expected.len(),"duplicate original expected coordinate");
    let mut seen=BTreeSet::new();let mut files=BTreeSet::new();let mut total=0u64;let mut pages=0usize;
    let mut buffer=vec![0u8;super::super::store::PAGE_BYTES];
    for t in &trace.coverage.captured {
        let n=t.shape.iter().try_fold(t.scalar.bytes() as u64,|n,&d|n.checked_mul(d as u64))
            .context("original shape overflow")?;
        ensure!(expected.iter().any(|e|e.at==t.at && e.scalar==t.scalar && e.shape==t.shape)
            && seen.insert(t.at.clone()) && files.insert(t.file.clone())
            && !t.shape.is_empty() && !t.shape.contains(&0) && n==t.logical_bytes && n>0
            && t.parent_identity>0 && t.parent_offset.checked_add(n).is_some_and(|end|end<=t.parent_bytes)
            && !t.checked_drain.is_empty(),"original full extent/identity/duplicate");
        ensure!(Path::new(&t.file).components().count()==1 && t.file.starts_with("tensor-")
            && t.file.ends_with(".bin"),"original payload path");
        let path=root.join(&t.file);let meta=std::fs::symlink_metadata(&path)?;
        ensure!(meta.is_file() && !meta.file_type().is_symlink() && meta.len()==n
            && path.canonicalize()?.parent()==Some(root),"original payload size/root");
        let mut input=File::open(&path)?;let mut full=Sha256::new();let mut offset=0u64;let mut index=0;
        while offset<n {
            let count=usize::try_from((n-offset).min(buffer.len() as u64))?;
            input.read_exact(&mut buffer[..count])?;let bytes=&buffer[..count];
            let page=t.pages.get(index).context("original missing page")?;
            ensure!(page.offset==offset && page.bytes==count && page.sha256==hash(bytes),"original page hash/extent");
            match t.scalar {
                Scalar::F32=>ensure!(bytes.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()),"original nonfinite F32"),
                Scalar::F16=>ensure!(bytes.chunks_exact(2).all(|b|half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),"original nonfinite F16"),
                Scalar::Bf16=>ensure!(bytes.chunks_exact(2).all(|b|f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32)<<16).is_finite()),"original nonfinite BF16"),
                Scalar::U8=>ensure!(t.at.stage==Stage::CacheValuePacked,"selected Gemma U8 is packed floating V only"),
                _=>anyhow::bail!("selected Gemma unexpected scalar"),
            }
            full.update(bytes);offset+=count as u64;index+=1;
        }
        ensure!(index==t.pages.len() && hex::encode(full.finalize())==t.sha256,"original full payload hash/pagecount");
        pages+=index;total=total.checked_add(n).context("original total overflow")?;
    }
    ensure!(seen==required,"original missing/excess coordinate");
    Ok(json!({"records":seen.len(),"payload_bytes":total,"pages":pages,
        "all_original_manifests_payloads_pages_finite_checked":true,"projection_performed":false}))
}
pub(super) fn restore64(root:&Path,trace:&Trace)->Result<Value> {
    let hashed=gemma_cache::require_restore64_exact(&trace.coverage.captured).map_err(anyhow::Error::msg)?;
    let before=trace.coverage.captured.iter().filter(|t|t.at.logical_step==64 && t.at.segment<1000).collect::<Vec<_>>();
    ensure!(before.len()==2520,"full four-lane selected2520 checkpoint role pairs");
    let mut a=vec![0u8;super::super::store::PAGE_BYTES];let mut b=a.clone();let mut compared=0u64;
    for source in before {
        let mut at=source.at.clone();at.segment+=1000;
        let target=trace.coverage.captured.iter().find(|t|t.at==at).context("actual restored role")?;
        ensure!(source.file!=target.file,"separate same-store captured64 files required");
        let mut left=File::open(root.join(&source.file))?;let mut right=File::open(root.join(&target.file))?;
        let mut remaining=source.logical_bytes;
        while remaining>0 {let n=usize::try_from(remaining.min(a.len() as u64))?;
            left.read_exact(&mut a[..n])?;right.read_exact(&mut b[..n])?;
            ensure!(a[..n]==b[..n],"actual restored64 bytes differ");remaining-=n as u64;compared+=n as u64;}
    }
    ensure!(compared==hashed,"complete restored64 bytes");
    Ok(json!({"pairs":2520,"actual_bytes_equal":compared,"within_one_original_acquisition":true}))
}
