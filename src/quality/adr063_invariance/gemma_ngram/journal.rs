//! Bounded dynamic transaction journal, separate from comparable Q1 Store.
use anyhow::{ensure,Context,Result};
use mlx_native::{DType,MlxBuffer};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs::{File,OpenOptions},io::Write,os::unix::fs::OpenOptionsExt,path::{Path,PathBuf}};
pub const PAYLOAD:u64=4*1024*1024*1024;
pub const METADATA:u64=128*1024*1024;
// alloc_hybrid_kv_for_layer stores singleton norms as [heads, capacity].
// Other roles and multi-norm rows retain their actual three-dimensional tag.
pub(super) fn head_role_shape(shape:&[usize],role:&str,heads:usize,cap:usize,width:usize)->bool {
    heads>0 && cap>0 && width>0 && if role=="CacheValueNorm" && width==1 {
        shape==[heads,cap]
    } else {shape==[heads,cap,width]}
}
/// Where a cache snapshot's head tensors go: the v3 journal, a T5 Q1 store
/// (with the manifest's per-lane-boundary record count), or nowhere (T5
/// events-only journal; the cursor/profile checks and ownership event remain).
pub enum Sink<'a> {Journal,Events,Store(&'a mut crate::adr063_invariance::store::Store,usize)}
/// Q1 store coordinate of one journal role/head: Hybrid roles keep live head
/// segments; dense companions use `2000 + head`.
pub(super) fn store_role(role:&str,head:usize)->Result<(crate::adr063_invariance::receipt::Stage,u32)> {
    use crate::adr063_invariance::receipt::Stage;
    let dense=super::t5::DENSE_SEGMENT+head as u32;
    Ok(match role {
        "CacheKey"=>(Stage::CacheKey,head as u32),"CacheValuePacked"=>(Stage::CacheValuePacked,head as u32),
        "CacheValueNorm"=>(Stage::CacheValueNorm,head as u32),"PrefillDenseKey"=>(Stage::CacheKey,dense),
        "PrefillDenseValue"=>(Stage::CacheValue,dense),_=>anyhow::bail!("unmodeled Gemma cache role {role}"),
    })
}
pub struct Journal {root:PathBuf,meta:File,metadata:u64,payload:u64,records:usize}
impl Journal {
    pub fn new(root:&Path)->Result<Self> {
        std::fs::create_dir(root)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(root,std::fs::Permissions::from_mode(0o700))?;
        let meta=OpenOptions::new().write(true).create_new(true).mode(0o600).open(root.join("records.jsonl"))?;
        Ok(Self {root:root.into(),meta,metadata:0,payload:0,records:0})
    }
    pub fn event(&mut self,kind:&str,payload:Value)->Result<()> {
        let data=serde_json::to_vec(&json!({"kind":kind,"payload":payload}))?;
        ensure!(data.len()<=65536 && self.metadata.checked_add(data.len() as u64+1).is_some_and(|n|n<=METADATA),"journal metadata bound");
        self.meta.write_all(&data)?;self.meta.write_all(b"\n")?;self.metadata+=data.len() as u64+1;Ok(())
    }
    pub fn tensor(&mut self,at:Value,b:&MlxBuffer)->Result<()> {
        let data=b.as_logical_slice::<u8>()?;ensure!(!data.is_empty() && self.records<20000 &&
            self.payload.checked_add(data.len() as u64).is_some_and(|n|n<=PAYLOAD),"whole-case active payload quota");
        ensure!(b.byte_offset().checked_add(data.len() as u64).is_some_and(|n|n<=b.byte_len() as u64),"actual parent extent");
        let width=b.dtype().size_of();ensure!(b.shape().iter().try_fold(width,|n,&d|n.checked_mul(d))==Some(data.len()),"logical shape/bytes");
        match b.dtype() {
            DType::F32=>ensure!(data.chunks_exact(4).all(|x|f32::from_le_bytes(x.try_into().unwrap()).is_finite()),"nonfinite F32"),
            DType::F16=>ensure!(data.chunks_exact(2).all(|x|u16::from_le_bytes(x.try_into().unwrap())&0x7c00!=0x7c00),"nonfinite F16"),
            DType::BF16=>ensure!(data.chunks_exact(2).all(|x|u16::from_le_bytes(x.try_into().unwrap())&0x7f80!=0x7f80),"nonfinite BF16"),
            DType::U8=>{},_=>anyhow::bail!("unmodeled active role scalar"),
        }
        let name=format!("tensor-{:05}.bin",self.records);let mut f=OpenOptions::new().create_new(true).write(true).mode(0o600).open(self.root.join(&name))?;
        let mut full=Sha256::new();let mut pages=Vec::new();let mut offset=0;
        for page in data.chunks(1024*1024) {f.write_all(page)?;full.update(page);pages.push(json!({"offset":offset,"bytes":page.len(),"sha256":hex::encode(Sha256::digest(page))}));offset+=page.len();}
        self.event("tensor",json!({"at":at,"scalar":format!("{:?}",b.dtype()),"shape":b.shape(),"file":name,
            "logical_bytes":data.len(),"parent_identity":b.contents_ptr() as usize as u64,"parent_bytes":b.byte_len(),
            "parent_offset":b.byte_offset(),"pages":pages,"sha256":hex::encode(full.finalize()),
            "checked_boundary":"owning production call successful checked return; no new wait"}))?;
        self.payload+=data.len() as u64;self.records+=1;Ok(())
    }
    pub fn snapshot(&mut self,m:&crate::inference::models::gemma4::MlxModelWeights,lane:usize,phase:&str,mut sink:Sink<'_>)->Result<usize> {
        let hybrid=m.hybrid_kv.as_ref().context("actual Hybrid absent")?;
        ensure!(hybrid.len()==30 && m.kv_caches.len()==30,"actual selected30layer ownership");
        let pos=m.kv_caches[0].write_pos;ensure!((128..=148).contains(&pos),"bounded no-wrap transaction cursor");
        if let Sink::Store(_,manifest)=&sink {
            // Manifest is authoritative: the live banks must hold exactly its extent.
            let dense=m.dense_kvs.as_ref().context("T5 manifest requires the dense companion banks")?;
            ensure!(dense.len()==30 && dense.iter().all(|d|d.dtype==DType::F32),"T5 manifest dense companions are F32 per layer");
            let live:usize=(0..30).map(|l|if (l+1)%6==0 {2}else{8}).map(|h|h*5).sum();
            ensure!(live==*manifest,"live bank record count {live} differs from the source manifest {manifest}");
        }
        let mut cursors=Vec::new();
        for (layer,(c,cursor)) in hybrid.iter().zip(&m.kv_caches).enumerate() {
            ensure!(cursor.write_pos==pos && cursor.seq_len==pos && cursor.capacity>=pos && c.capacity>=pos &&
                c.bf16_xlen_k.is_none() && c.bf16_xlen_v.is_none(),"actual active cursors/profile/auxiliary roles");
            let (heads,hd)=if (layer+1)%6==0 {(2,512)}else{(8,256)};
            ensure!(c.is_sliding==((layer+1)%6!=0) && c.norms_per_pos==hd/256 && c.k.dtype()==DType::F16 && c.v_packed.dtype()==DType::U8 && c.v_norms.dtype()==DType::F32,"selected full Hybrid roles");
            cursors.push(json!({"layer":layer,"write_pos":cursor.write_pos,"seq_len":cursor.seq_len,"legacy_capacity":cursor.capacity,
                "actual_hybrid_capacity":c.capacity,"sliding":c.is_sliding}));
            for (b,role,width) in [(&c.k,"CacheKey",hd),(&c.v_packed,"CacheValuePacked",hd),(&c.v_norms,"CacheValueNorm",hd/256)] {
                self.heads(&mut sink,b,lane,phase,layer,role,heads,c.capacity,pos,width)?;
            }
            if let Some(dense)=m.dense_kvs.as_ref() {
                ensure!(dense.len()==30 && dense[layer].capacity>=pos,"dense companion role extent");
                for (b,role) in [(&dense[layer].k,"PrefillDenseKey"),(&dense[layer].v,"PrefillDenseValue")] {
                    self.heads(&mut sink,b,lane,phase,layer,role,heads,dense[layer].capacity,pos,hd)?;
                }
            }
        }
        self.event("cache_ownership",json!({"owner_epoch":lane,"phase":phase,"position":pos,"layers":cursors,
            "layout":"actual global Hybrid/dense companions, not mounted SlotAware or simultaneous B4",
            "lcp_snapshots_present":m.hybrid_kv_snapshot_for_lcp.is_some()||m.dense_kvs_snapshot_for_lcp.is_some(),
            "lcp_snapshot_scope":"not read by selected OptionC start0 or scalar continuation; no physical credit",
            "legacy_packed_payload_scope":"unused by selected Hybrid decoder; actual cursor metadata above"}))?;Ok(pos)
    }
    #[allow(clippy::too_many_arguments)]
    fn heads(&mut self,sink:&mut Sink<'_>,b:&MlxBuffer,lane:usize,phase:&str,layer:usize,role:&str,heads:usize,cap:usize,pos:usize,width:usize)->Result<()> {
        ensure!(head_role_shape(b.shape(),role,heads,cap,width),
            "actual head-major role shape: role={role} shape={:?} heads={heads} capacity={cap} width={width}",b.shape());
        for head in 0..heads {
            let offset=head.checked_mul(cap).and_then(|n|n.checked_mul(width)).and_then(|n|n.checked_mul(b.dtype().size_of())).context("head offset overflow")?;
            let view=b.slice_view(offset as u64,pos*width).with_shape(vec![1,pos,width])?;
            match sink {
                Sink::Journal=>self.tensor(json!({"owner_epoch":lane,"phase":phase,"layer":layer,"role":role,"head":head,"token_start":0,"token_count":pos,"source_parent_shape":b.shape()}),&view)?,
                Sink::Events=>{}
                Sink::Store(store,_)=>{
                    let (stage,segment)=store_role(role,head)?;
                    let mut at=crate::adr063_invariance::gemma_cache::at(lane,pos,layer,stage,head,false);at.segment=segment;
                    crate::adr063_invariance::capture::after_checked_drain(store,at,&view,&[1,pos,width],None,
                        "actual Gemma global-route checked return; cursor==boundary").map_err(anyhow::Error::msg)?;
                }
            }
        }Ok(())
    }
    pub fn finish(&mut self,status:&str)->Result<()> {self.event("end",json!({"status":status,"records":self.records,
        "payload_bytes":self.payload,"metadata_bytes_before_end":self.metadata,"payload_limit":PAYLOAD,"metadata_limit":METADATA,
        "oracle_qualified":false,"t5_transition_qualified":false,"whole_q1_qualified":false}))}
}
