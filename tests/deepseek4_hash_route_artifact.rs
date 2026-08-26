use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const DEFAULT_MODEL: &str = "/opt/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-agentic-q2.gguf";

#[test]
fn real_deepseek4_hash_rows_have_distinct_in_range_experts() {
    let path = std::env::var_os("DEEPSEEK4_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    if !path.exists() {
        eprintln!(
            "[skip] DeepSeek-V4 artifact not present at {}",
            path.display()
        );
        return;
    }

    let gguf = mlx_native::gguf::GgufFile::open(Path::new(&path)).expect("open DeepSeek GGUF");
    let hash_layers = gguf
        .metadata_u32("deepseek4.hash_layer_count")
        .expect("deepseek4.hash_layer_count") as usize;
    let expert_count = gguf
        .metadata_u32("deepseek4.expert_count")
        .expect("deepseek4.expert_count") as i32;
    let top_k = gguf
        .metadata_u32("deepseek4.expert_used_count")
        .expect("deepseek4.expert_used_count") as usize;
    let mut file = File::open(&path).expect("reopen DeepSeek GGUF");

    for layer in 0..hash_layers {
        let name = format!("blk.{layer}.ffn_gate_tid2eid.weight");
        let info = gguf
            .tensor_info(&name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.ggml_type, mlx_native::GgmlType::I32);
        assert_eq!(info.shape.last().copied(), Some(top_k));
        let rows = info.shape[..info.shape.len() - 1].iter().product::<usize>();
        assert_eq!(info.byte_len, rows * top_k * 4);

        file.seek(SeekFrom::Start(gguf.tensor_data_offset() + info.offset))
            .expect("seek lookup tensor");
        let mut payload = vec![0_u8; info.byte_len];
        file.read_exact(&mut payload).expect("read lookup tensor");

        let mut duplicate_rows = 0usize;
        let mut out_of_range = 0usize;
        for row in payload.chunks_exact(top_k * 4) {
            let mut seen = HashSet::with_capacity(top_k);
            for bytes in row.chunks_exact(4) {
                let expert = i32::from_le_bytes(bytes.try_into().unwrap());
                if !(0..expert_count).contains(&expert) {
                    out_of_range += 1;
                }
                if !seen.insert(expert) {
                    duplicate_rows += 1;
                    break;
                }
            }
        }
        eprintln!(
            "{name}: rows={rows} duplicate_rows={duplicate_rows} out_of_range={out_of_range}"
        );
        assert_eq!(out_of_range, 0, "{name} contains out-of-range experts");
        assert_eq!(duplicate_rows, 0, "{name} contains duplicate expert IDs");
    }
}
