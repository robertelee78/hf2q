use mlx_native::gguf::GgufFile;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: dump_gguf_blk0 <gguf>"))?
        .into();
    let gguf = GgufFile::open(&path)?;
    let mut blk: Vec<_> = gguf
        .tensor_names()
        .iter()
        .filter(|n| n.starts_with("blk.0.") || n.starts_with("blk.1.") || n.starts_with("blk.2."))
        .map(|n| (n.to_string(), gguf.tensor_info(n).unwrap().clone()))
        .collect();
    blk.sort_by_key(|(n, _)| n.clone());
    for (name, info) in &blk {
        println!("{name}  shape={:?}  type={:?}", info.shape, info.ggml_type);
    }
    Ok(())
}
