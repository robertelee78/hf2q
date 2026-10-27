//! Complete extent comparison with bounded paired pages; no sampled-stage PASS.
use super::{
    receipt::*,
    store::{hash, PAGE_BYTES},
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};

fn blank(disposition: Disposition, reason: Option<String>) -> Comparison {
    Comparison {
        disposition,
        first_difference: None,
        compared_bytes: 0,
        compared_elements: 0,
        max_abs: 0.0,
        rms: 0.0,
        relative_rms: None,
        cosine: None,
        nonfinite_elements: 0,
        reason,
        oracle_qualified: false,
        captured_extent_qualified: false,
    }
}
fn validate(trace: &Trace) -> Result<BTreeMap<Coordinate, &Tensor>, String> {
    trace.identity.validate()?;
    if trace.schema != "hf2q.adr063.invariance.v1"
        || !trace.coverage.errors.is_empty()
        || trace.coverage.expected.is_empty()
    {
        return Err("invalid/error/incomplete trace".into());
    }
    let expected: BTreeSet<_> = trace
        .coverage
        .expected
        .iter()
        .map(|e| e.at.clone())
        .collect();
    let mut result = BTreeMap::new();
    for t in &trace.coverage.captured {
        let size = t
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .and_then(|n| n.checked_mul(t.scalar.bytes()))
            .ok_or("tensor shape overflow")?;
        if !trace
            .coverage
            .expected
            .iter()
            .any(|e| e.at == t.at && e.scalar == t.scalar && e.shape == t.shape)
            || size as u64 != t.logical_bytes
            || t.shape.contains(&0)
            || t.parent_identity == 0
            || t.parent_offset
                .checked_add(t.logical_bytes)
                .is_none_or(|end| end > t.parent_bytes)
            || t.checked_drain.is_empty()
            || t.pages.is_empty()
            || result.insert(t.at.clone(), t).is_some()
        {
            return Err("invalid tensor coverage/extent".into());
        }
    }
    if expected.len() != trace.coverage.expected.len()
        || result.keys().cloned().collect::<BTreeSet<_>>() != expected
    {
        return Err("required stage/row missing or duplicated".into());
    }
    Ok(result)
}
fn open(root: &Path, t: &Tensor) -> Result<File, String> {
    if Path::new(&t.file).components().count() != 1
        || !t.file.starts_with("tensor-")
        || !t.file.ends_with(".bin")
    {
        return Err("invalid trace payload name".into());
    }
    let path = root.join(&t.file);
    let m = std::fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !m.is_file()
        || m.file_type().is_symlink()
        || m.len() != t.logical_bytes
        || path.canonicalize().map_err(|e| e.to_string())?.parent() != Some(root)
    {
        return Err("payload path/size identity".into());
    }
    File::open(path).map_err(|e| e.to_string())
}
fn number(scalar: Scalar, b: &[u8]) -> Option<f64> {
    Some(match scalar {
        Scalar::F32 => f32::from_le_bytes(b.try_into().ok()?) as f64,
        Scalar::F16 => half::f16::from_bits(u16::from_le_bytes(b.try_into().ok()?)).to_f64(),
        Scalar::Bf16 => {
            f32::from_bits((u16::from_le_bytes(b.try_into().ok()?) as u32) << 16) as f64
        }
        _ => return None,
    })
}
fn is_float(scalar: Scalar) -> bool {
    matches!(scalar, Scalar::F32 | Scalar::F16 | Scalar::Bf16)
}

pub fn compare(
    left_root: &Path,
    left: &Trace,
    right_root: &Path,
    right: &Trace,
    policies: &BTreeMap<Stage, NumericPolicy>,
) -> Comparison {
    if left.identity != right.identity {
        return blank(
            Disposition::IdentityMismatch,
            Some("source/model/config/token/device identity differs".into()),
        );
    }
    let mut out = blank(Disposition::ExactEqual, None);
    match compare_inner(left_root, left, right_root, right, policies, &mut out) {
        Ok(()) => out,
        Err(error) => {
            out.disposition = Disposition::InvalidEvidence;
            out.reason = Some(error);
            out.captured_extent_qualified = false;
            out.oracle_qualified = false;
            out
        }
    }
}
fn compare_inner(
    lroot: &Path,
    left: &Trace,
    rroot: &Path,
    right: &Trace,
    policies: &BTreeMap<Stage, NumericPolicy>,
    out: &mut Comparison,
) -> Result<(), String> {
    let l = validate(left)?;
    let r = validate(right)?;
    if l.keys().ne(r.keys()) {
        out.disposition = Disposition::LocalizationIncomplete;
        out.reason = Some("logical coordinate populations differ".into());
        return Ok(());
    }
    if left.coverage.expected != right.coverage.expected {
        return Err("frozen required coordinate order/shape differs".into());
    }
    let mut a = vec![0u8; PAGE_BYTES];
    let mut b = vec![0u8; PAGE_BYTES];
    let (mut diff_sq, mut reference_sq, mut dot, mut candidate_sq, mut float_count) =
        (0.0, 0.0, 0.0, 0.0, 0u64);
    let (mut unknown, mut exact_failed, mut bound_failed, mut bounded) =
        (false, false, false, false);
    // The independently reviewed packet freezes token -> layer -> source
    // stage -> lane/row order. Hash-map/file creation order cannot silently
    // redefine the first divergence, and families need not share an ordinal.
    for expected in &left.coverage.expected {
        let at = expected.at.clone();
        let lt = *l.get(&at).ok_or("missing left coordinate")?;
        let rt = r.get(&at).ok_or("missing right coordinate")?;
        if lt.scalar != rt.scalar || lt.shape != rt.shape || lt.logical_bytes != rt.logical_bytes {
            out.disposition = Disposition::LocalizationIncomplete;
            out.reason = Some("logical dtype/shape/extent differs".into());
            return Ok(());
        }
        let policy = policies
            .get(&at.stage)
            .unwrap_or(&NumericPolicy::Unspecified);
        policy.validate()?;
        let (mut lf, mut rf) = (open(lroot, lt)?, open(rroot, rt)?);
        let (mut lh, mut rh) = (Sha256::new(), Sha256::new());
        let (mut offset, mut page) = (0usize, 0usize);
        let (mut tensor_diff, mut tensor_ref, mut tensor_n, mut tensor_max) =
            (0.0, 0.0, 0u64, 0.0f64);
        let mut changed = false;
        while offset < lt.logical_bytes as usize {
            let n = PAGE_BYTES.min(lt.logical_bytes as usize - offset);
            lf.read_exact(&mut a[..n]).map_err(|e| e.to_string())?;
            rf.read_exact(&mut b[..n]).map_err(|e| e.to_string())?;
            for (tensor, bytes) in [(lt, &a[..n]), (*rt, &b[..n])] {
                let p = tensor.pages.get(page).ok_or("missing page receipt")?;
                if p.offset != offset as u64 || p.bytes != n || p.sha256 != hash(bytes) {
                    return Err("corrupt page receipt/payload".into());
                }
            }
            lh.update(&a[..n]);
            rh.update(&b[..n]);
            let width = lt.scalar.bytes();
            for (j, (av, bv)) in a[..n]
                .chunks_exact(width)
                .zip(b[..n].chunks_exact(width))
                .enumerate()
            {
                if av != bv {
                    changed = true;
                    if out.first_difference.is_none() {
                        out.first_difference = Some(Difference {
                            at: at.clone(),
                            logical_element: ((offset / width) + j) as u64,
                            left_bits: hex::encode(av),
                            right_bits: hex::encode(bv),
                        });
                    }
                }
                out.compared_elements = out
                    .compared_elements
                    .checked_add(1)
                    .ok_or("element count overflow")?;
                if let (Some(x), Some(y)) = (number(lt.scalar, av), number(lt.scalar, bv)) {
                    if !x.is_finite() || !y.is_finite() {
                        // Compressor score states deliberately initialize the
                        // valid recurrence sentinel to negative infinity.
                        // Only bit-identical, source-named sentinels can pass
                        // an EXACT policy; NaNs/other stages remain invalid.
                        let sentinel =
                            matches!(at.stage, Stage::CacheMainScore | Stage::CacheIndexerScore)
                                && matches!(policy, NumericPolicy::Exact)
                                && av == bv
                                && x == f64::NEG_INFINITY
                                && y == f64::NEG_INFINITY;
                        if !sentinel {
                            out.nonfinite_elements += 1;
                        }
                        continue;
                    }
                    let d = (x - y).abs();
                    out.max_abs = out.max_abs.max(d);
                    tensor_max = tensor_max.max(d);
                    diff_sq += (x - y) * (x - y);
                    reference_sq += x * x;
                    dot += x * y;
                    candidate_sq += y * y;
                    float_count += 1;
                    tensor_diff += (x - y) * (x - y);
                    tensor_ref += x * x;
                    tensor_n += 1;
                }
            }
            out.compared_bytes = out
                .compared_bytes
                .checked_add(n as u64)
                .ok_or("byte count overflow")?;
            offset += n;
            page += 1;
        }
        if page != lt.pages.len()
            || page != rt.pages.len()
            || hex::encode(lh.finalize()) != lt.sha256
            || hex::encode(rh.finalize()) != rt.sha256
        {
            return Err("full payload hash/page count mismatch".into());
        }
        if !is_float(lt.scalar) {
            exact_failed |= changed;
            continue;
        }
        match policy {
            NumericPolicy::Exact => exact_failed |= changed,
            NumericPolicy::Unspecified => unknown = true,
            NumericPolicy::SourceQualified {
                max_abs,
                relative_rms,
                ..
            } => {
                bounded = true;
                let relative = if tensor_ref > 0.0 {
                    (tensor_diff / tensor_ref).sqrt()
                } else if tensor_diff == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                };
                bound_failed |= tensor_n == 0 || tensor_max > *max_abs || relative > *relative_rms;
            }
        }
    }
    if float_count > 0 {
        out.rms = (diff_sq / float_count as f64).sqrt();
    }
    if reference_sq > 0.0 {
        out.relative_rms = Some((diff_sq / reference_sq).sqrt());
    }
    if reference_sq > 0.0 && candidate_sq > 0.0 {
        out.cosine = Some(dot / (reference_sq * candidate_sq).sqrt());
    }
    if ![out.max_abs, out.rms].iter().all(|n| n.is_finite()) {
        return Err("numeric accumulator overflow".into());
    }
    if out.nonfinite_elements > 0 {
        out.disposition = Disposition::InvalidEvidence;
        out.reason = Some("nonfinite active elements".into());
    } else if exact_failed {
        out.disposition = Disposition::DivergedExactInvariant;
    } else if bound_failed {
        out.disposition = Disposition::DivergedSourceQualifiedPolicy;
    } else if unknown
        || !left.coverage.missing_native_capabilities.is_empty()
        || !right.coverage.missing_native_capabilities.is_empty()
    {
        out.disposition = Disposition::ExplicitInconclusiveNumericPolicy;
        out.reason =
            Some("no approved policy or required native routing capture unavailable".into());
    } else if bounded {
        out.disposition = Disposition::WithinSourceQualifiedPolicy;
    }
    out.oracle_qualified = out.disposition != Disposition::InvalidEvidence;
    out.captured_extent_qualified = matches!(
        out.disposition,
        Disposition::ExactEqual | Disposition::WithinSourceQualifiedPolicy
    );
    Ok(())
}
