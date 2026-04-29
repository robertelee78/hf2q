//! ADR-014 P11 iter-98 — dump GGUF tensor types + sizes for debugging.

use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: dump_gguf_types <gguf_path>"))?
        .into();
    let gguf = GgufFile::open(&path)?;

    let mut hist: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut total: u64 = 0;
    for name in gguf.tensor_names() {
        let info = gguf.tensor_info(name).unwrap();
        let entry = hist.entry(format!("{:?}", info.ggml_type)).or_default();
        entry.0 += 1;
        entry.1 += info.byte_len as u64;
        total += info.byte_len as u64;
    }

    eprintln!("[dump] tensors: {}", gguf.tensor_count());
    eprintln!("[dump] total tensor bytes: {} ({:.2} GB)", total, total as f64 / (1024.0 * 1024.0 * 1024.0));
    eprintln!("[dump] type histogram:");
    for (t, (n, b)) in &hist {
        eprintln!("  {t:>6}: {n:5} tensors  {b:>14} bytes  ({:.2} GB)", *b as f64 / (1024.0 * 1024.0 * 1024.0));
    }

    // Sample a few large tensors.
    let mut tensors: Vec<_> = gguf.tensor_names().iter().map(|n| (n.to_string(), gguf.tensor_info(n).unwrap().clone())).collect();
    tensors.sort_by_key(|(_, i)| std::cmp::Reverse(i.byte_len));
    eprintln!("[dump] top 10 by size:");
    for (name, info) in tensors.iter().take(10) {
        eprintln!("  {} offset={} bytes={} type={:?} shape={:?}", name, info.offset, info.byte_len, info.ggml_type, info.shape);
    }

    // Check for offset gaps
    let mut sorted_by_offset: Vec<_> = gguf.tensor_names().iter().map(|n| (n.to_string(), gguf.tensor_info(n).unwrap().clone())).collect();
    sorted_by_offset.sort_by_key(|(_, i)| i.offset);
    eprintln!("[dump] offset coverage check (gap > 0 only):");
    let mut prev_end: u64 = 0;
    let mut total_gap: u64 = 0;
    let mut gap_count = 0;
    for (name, info) in sorted_by_offset.iter() {
        let gap = info.offset.saturating_sub(prev_end);
        if gap > 0 {
            eprintln!("  offset={:>14} (gap={:>10}) end={:>14} bytes={:>10} type={:?} {}", info.offset, gap, info.offset + info.byte_len as u64, info.byte_len, info.ggml_type, name);
            total_gap += gap;
            gap_count += 1;
        }
        prev_end = info.offset + info.byte_len as u64;
    }
    eprintln!("[dump] total gaps: {} bytes ({:.2} GB) across {} gap entries", total_gap, total_gap as f64 / 1024.0 / 1024.0 / 1024.0, gap_count);
    eprintln!("[dump] last few in offset order:");
    for (name, info) in sorted_by_offset.iter().rev().take(5) {
        eprintln!("  offset={:>14} end={:>14} bytes={:>10} {}", info.offset, info.offset + info.byte_len as u64, info.byte_len, name);
    }
    let last = sorted_by_offset.last().unwrap();
    eprintln!("[dump] last tensor end offset: {} = {:.2} GB", last.1.offset + last.1.byte_len as u64, (last.1.offset + last.1.byte_len as u64) as f64 / 1024.0 / 1024.0 / 1024.0);

    // Summary of any F16/F32 tensors (likely fallbacks)
    let f16_tensors: Vec<_> = gguf.tensor_names().iter()
        .map(|n| gguf.tensor_info(n).unwrap())
        .filter(|i| matches!(i.ggml_type, GgmlType::F16 | GgmlType::F32))
        .collect();
    eprintln!("[dump] F16/F32 tensors: {} (total bytes: {})",
        f16_tensors.len(),
        f16_tensors.iter().map(|i| i.byte_len as u64).sum::<u64>()
    );

    Ok(())
}
