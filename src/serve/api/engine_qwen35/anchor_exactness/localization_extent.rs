//! Header-only independent C23 raw extents, using the allocator's own plan.
//! No weights, device construction, Metal allocation, or inference is used.
use super::*;
use crate::inference::models::qwen35::{Qwen35Config, c23_layer_capture, kv_cache::{LayerSlot,q1_capture_layout}};

pub(super) fn anchor_coordinate(boundary:u64,segment:u32)->Coordinate {
    Coordinate{logical_step:boundary,layer:0,stage:Stage::CacheState,agent:"c23-lane-2-anchor914".into(),logical_row_start:914,segment}
}
fn tensor(boundary:u64,segment:u32,scalar:Scalar,len:usize)->ExpectedTensor {
    ExpectedTensor{at:anchor_coordinate(boundary,segment),scalar,shape:vec![len]}
}
fn product(shape:&[usize])->Result<usize> {
    shape.iter().try_fold(1usize,|n,d|n.checked_mul(*d)).context("localization allocator extent overflow")
}
pub(super) fn anchor_expected(cfg:&Qwen35Config,keep:usize)->Result<Vec<ExpectedTensor>> {
    ensure!(matches!(keep,1|4),"C23 extent only admits K1/K4");
    let layouts=q1_capture_layout::layouts(cfg,262144)?;
    let full=layouts.iter().filter(|l|matches!(l.slot,LayerSlot::Full(_))&&!l.mtp).count();
    let mtp=layouts.iter().any(|l|l.mtp);let linear=layouts.iter().filter(|l|matches!(l.slot,LayerSlot::Linear(_))).count();
    ensure!(full>0&&linear>0,"C23 dense hybrid geometry required");
    let mut kv=Vec::new();let mut segment=2u32;
    for is_mtp in [false,true] {for layout in layouts.iter().filter(|l|matches!(l.slot,LayerSlot::Full(_))&&l.mtp==is_mtp) {
        // OFF target-only forwards never move the initialized MTP cursor.
        let valid=if is_mtp {0}else{914};
        for (dtype,shape) in &layout.roots {ensure!(shape.len()==4&&shape[0]==1&&shape[2]==262144,"allocator full-KV descriptor");
            let bytes=shape[1].checked_mul(valid).and_then(|n|n.checked_mul(shape[3])).and_then(|n|n.checked_mul(dtype.size_of())).context("C23 full-KV extent overflow")?;
            if bytes>0 {kv.push((segment,bytes));}segment=segment.checked_add(1).context("C23 segment overflow")?;
        }
    }}
    let mut image_words=0usize;let mut live=Vec::new();
    for layout in layouts.iter().filter(|l|matches!(l.slot,LayerSlot::Linear(_))) {
        ensure!(layout.roots.len()==4,"allocator linear ping-pong roots");
        let (conv_dtype,conv)=&layout.roots[0];let (rec_dtype,rec)=&layout.roots[2];
        ensure!(conv.len()==3&&conv[2]==1&&rec.len()==4&&rec[3]==1&&conv_dtype.size_of()==4&&rec_dtype.size_of()==4,"allocator linear descriptors");
        let conv_words=product(conv)?;let rec_words=product(rec)?;
        image_words=image_words.checked_add(conv_words).and_then(|n|n.checked_add(rec_words)).context("C23 raw image extent overflow")?;
        live.push((segment,conv_words.checked_mul(4).context("conv byte extent overflow")?));segment+=1;
        live.push((segment,rec_words.checked_mul(4).context("recurrent byte extent overflow")?));segment+=1;
    }
    let mut expected=vec![tensor(0,0,Scalar::U32,full+usize::from(mtp)),tensor(0,u32::MAX-1,Scalar::U32,image_words),tensor(0,u32::MAX,Scalar::U32,914)];
    expected.extend(kv.iter().map(|(segment,bytes)|tensor(0,*segment,Scalar::U8,*bytes)));
    if keep==4 {
        expected.push(tensor(1,0,Scalar::U32,full+usize::from(mtp)));expected.push(tensor(1,1,Scalar::U8,linear));
        expected.extend(kv.iter().chain(live.iter()).map(|(segment,bytes)|tensor(1,*segment,Scalar::U8,*bytes)));
        expected.push(tensor(1,u32::MAX,Scalar::U32,914));
    }Ok(expected)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {schema:String,model:PathBuf,file_identity:acquisition::FileIdentity,output:PathBuf}
#[test]
#[ignore="C23 GGUF header-only independent raw extent export; no model/device/inference"]
fn qwen_c23_localization_extent_export() {run().unwrap();}
fn run()->Result<()> {
    let path=PathBuf::from(std::env::var_os("HF2Q_C23_LOCALIZATION_EXTENT_PACKET").context("C23 extent packet missing")?);
    let mut raw=Vec::new();fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;ensure!(raw.len()<=1024*1024,"bounded C23 extent packet required");
    let packet:Packet=serde_json::from_slice(&raw)?;ensure!(packet.schema=="hf2q.c23.localization-extent-packet.v1","C23 extent schema");
    output_preflight(&packet.output)?;let file=fs::File::open(&packet.model)?;let before=acquisition::FileIdentity::observed(&file)?;
    ensure!(before==packet.file_identity,"C23 header artifact identity changed");
    let header=mlx_native::gguf::GgufFile::from_file(file.try_clone()?)?;ensure!(header.tensor_data_offset()<=32*1024*1024,"C23 metadata header bound");
    let cfg=Qwen35Config::from_gguf(&header)?;let config_sha256=hash(&serde_json::to_vec(&cfg)?);
    let stores=[("layers",c23_layer_capture::expected(&cfg,2,3,914,1923)?),("anchor-k1",anchor_expected(&cfg,1)?),("anchor-k4",anchor_expected(&cfg,4)?)];
    let after=acquisition::FileIdentity::observed(&file)?;ensure!(after==before,"C23 header changed during export");
    fs::create_dir(&packet.output)?;fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    let mut bindings=std::collections::BTreeMap::new();
    for (name,expected) in stores {let bytes=serde_json::to_vec_pretty(&expected)?;let path=packet.output.join(format!("{name}.expected.json"));
        let mut f=fs::OpenOptions::new().write(true).create_new(true).open(&path)?;use std::io::Write;f.write_all(&bytes)?;
        let payload_bytes=expected.iter().try_fold(0u64,|total,e|e.shape.iter().try_fold(e.scalar.bytes() as u64,|n,d|n.checked_mul(*d as u64)).and_then(|n|total.checked_add(n)).context("C23 planned payload overflow"))?;
        bindings.insert(name,serde_json::json!({"path":path,"sha256":hash(&bytes),"records":expected.len(),"payload_bytes":payload_bytes}));
    }
    let receipt=serde_json::json!({"schema":"hf2q.c23.localization-extent-export.v1","packet_sha256":hash(&raw),"file_identity_before":before,"file_identity_after":after,
        "model_config_sha256":config_sha256,"stores":bindings,"source_geometry":"HybridCacheAllocationPlan(cfg,262144,1,true,true) via q1_capture_layout::layouts",
        "model_loaded":false,"metal_allocated":false,"inference_executed":false,"target_valid_rows":914,"off_mtp_valid_rows":0,
        "creation_live_parity":null,"linear_slot_addressing":"contiguous slot_index*(element_count/nseq), matching la_restore.rs; descriptor nseq axis last"});
    let mut f=fs::OpenOptions::new().write(true).create_new(true).open(packet.output.join("extent-export.json"))?;use std::io::Write;f.write_all(&serde_json::to_vec_pretty(&receipt)?)?;Ok(())
}
#[test]
fn c23_localization_header_extent_rejects_product_overflow() {assert!(product(&[usize::MAX,2]).is_err());assert_eq!(product(&[3,2,1]).unwrap(),6);}
