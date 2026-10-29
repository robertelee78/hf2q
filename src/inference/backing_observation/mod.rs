//! Diagnostic actual-owner metadata, pending published native0.15.13 binding.
//! Not the model-free projection ledger and never physical admission.
mod qwen;
pub use qwen::{qwen_cache_roles,QwenCacheRoles};
use anyhow::{ensure, Context, Result};
use mlx_native::{gguf::{GgufFile,GgufMappedTensorSet},BufferBackingSnapshot,BufferProvenance,MlxBuffer,MlxDevice};
use std::{fs::File,os::{unix::fs::MetadataExt,fd::AsRawFd},sync::Arc};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct FileStat {
    pub device:u64,pub inode:u64,pub bytes:u64,
    pub mtime_seconds:i64,pub mtime_nanoseconds:i64,
    pub ctime_seconds:i64,pub ctime_nanoseconds:i64,
}
impl FileStat {
    pub fn read(file:&File)->Result<Self> {
        let m=file.metadata().context("stat actual GGUF owner FD")?;
        ensure!(m.is_file(),"GGUF owner is not a regular file");
        Ok(Self{device:m.dev(),inode:m.ino(),bytes:m.len(),mtime_seconds:m.mtime(),
            mtime_nanoseconds:m.mtime_nsec(),ctime_seconds:m.ctime(),ctime_nanoseconds:m.ctime_nsec()})
    }
}
#[derive(Clone,Debug)]
pub struct RoleSnapshot {
    pub role:String,
    pub backing:BufferBackingSnapshot,
    pub dtype:String,pub shape:Vec<usize>,
    /// Only existing KnownNative lineage, not physical ownership/coverage.
    pub native_origin_known:bool,
}
pub fn observe(role:String,buffer:&MlxBuffer)->Result<RoleSnapshot> {
    ensure!(!role.is_empty() && role.len()<=256,"bounded nonempty role required");
    ensure!(buffer.shape().len()<=16,"bounded tensor rank required");
    let backing=buffer.backing_snapshot()?;
    let native_origin_known=matches!(backing.origin,mlx_native::RootOriginSnapshot::KnownNative{..});
    Ok(RoleSnapshot{role,backing,native_origin_known,dtype:format!("{:?}",buffer.dtype()),shape:buffer.shape().to_vec()})
}
/// Same-FD source identity. Stat is not a full-content receipt or immutability proof.
/// The diagnostic caller must separately bind its retained full hash/loader recipe.
pub struct ObservedGgufOwner { file:Arc<File>,stat:FileStat,gguf:GgufFile }
impl ObservedGgufOwner {
    pub fn from_file(file:File)->Result<Self> {
        // Check the ACTUAL retained FD, not a caller-declared mode or pathname.
        let flags=unsafe { libc::fcntl(file.as_raw_fd(),libc::F_GETFL) };
        ensure!(flags>=0 && flags & libc::O_ACCMODE == libc::O_RDONLY,"readonly GGUF FD required");
        let stat=FileStat::read(&file)?;
        let gguf=GgufFile::from_file(file.try_clone()?)?;
        ensure!(FileStat::read(&file)?==stat,"GGUF FD changed during header parsing");
        Ok(Self{file:Arc::new(file),stat,gguf})
    }
    pub fn file_stat(&self)->FileStat { self.stat }
    /// Actual native mapped-set constructor, no payload read/copy. Maps the SAME
    /// segment topology as the existing native API; requires prior load admission.
    pub fn map_weights(&self,device:&MlxDevice)->Result<ObservedMappedWeights<'_>> {
        self.require_unchanged()?;
        let mapped=self.gguf.map_tensor_data(device)?;
        self.require_unchanged()?;
        Ok(ObservedMappedWeights{owner:self,mapped})
    }
    fn require_unchanged(&self)->Result<()> {
        ensure!(FileStat::read(&self.file)?==self.stat,"actual GGUF FD identity/stat changed"); Ok(())
    }
}
pub struct ObservedMappedWeights<'a> { owner:&'a ObservedGgufOwner,mapped:GgufMappedTensorSet<'a> }
/// Keeps the exact FD and actual buffer alive for subsequent range joins.
pub struct ObservedWeight { buffer:MlxBuffer,file:Arc<File>,stat:FileStat,tensor:String,logical_file_offset:u64 }
impl ObservedMappedWeights<'_> {
    pub fn load(&self,name:&str)->Result<ObservedWeight> {
        self.owner.require_unchanged()?;
        ensure!(name.len()<=249,"bounded tensor role name required");
        let info=self.owner.gguf.tensor_info(name).context("unknown mapped weight tensor")?;
        let logical_file_offset=self.owner.gguf.tensor_data_offset().checked_add(info.offset).context("tensor file offset overflow")?;
        let buffer=self.mapped.load_tensor(name)?;
        ensure!(buffer.data_byte_len()==info.byte_len,"mapped tensor logical extent mismatch");
        let out=ObservedWeight{buffer,file:self.owner.file.clone(),stat:self.owner.stat,tensor:name.into(),logical_file_offset};
        out.snapshot()?; // The actual-produced handle is checked before publication.
        Ok(out)
    }
}
#[derive(Clone,Debug)]
pub struct FileWeightSnapshot {
    pub role:RoleSnapshot,pub file:FileStat,pub logical_file_offset:u64,
    pub mapping_file_offset:u64,pub registered_file_offset:u64,
    /// Explicit registered final-page padding is not file contents or credit.
    pub registered_tail_padding_bytes:u64,
}
impl ObservedWeight {
    pub fn buffer(&self)->&MlxBuffer { &self.buffer }
    pub fn snapshot(&self)->Result<FileWeightSnapshot> {
        ensure!(FileStat::read(&self.file)?==self.stat,"weight owner FD changed");
        let role=observe(format!("weight/{}",self.tensor),&self.buffer)?;
        let (mapping_file_offset,registered_file_offset,registered_tail_padding_bytes)=join_file_range(&role.backing,self.logical_file_offset,self.stat.bytes)?;
        ensure!(FileStat::read(&self.file)?==self.stat,"weight owner FD changed during snapshot");
        Ok(FileWeightSnapshot{role,file:self.stat,logical_file_offset:self.logical_file_offset,
            mapping_file_offset,registered_file_offset,registered_tail_padding_bytes})
    }
}
fn join_file_range(b:&BufferBackingSnapshot,logical_file_offset:u64,file_bytes:u64)->Result<(u64,u64,u64)> {
    ensure!(b.provenance==BufferProvenance::NoCopy,"file join requires actual retained NoCopy backing");
    let n=b.no_copy.context("missing NoCopy range")?;
    let root_in_mapping=n.registered_base_address.checked_sub(n.mapping_base_address).context("registered range precedes mapping")?;
    ensure!(n.registered_bytes==b.metal_root_bytes,"registered parent extent mismatch");
    join_offsets(root_in_mapping,b.view_byte_offset,b.logical_bytes,n.mapping_bytes,
        n.registered_bytes,logical_file_offset,file_bytes)
}
fn join_offsets(root_in_mapping:usize,view_offset:u64,logical_bytes:usize,mapping_bytes:usize,
    registered_bytes:usize,logical_file_offset:u64,file_bytes:u64)->Result<(u64,u64,u64)> {
    let registered_file_offset=logical_file_offset.checked_sub(view_offset).context("file/root offset underflow")?;
    let mapping_file_offset=registered_file_offset.checked_sub(u64::try_from(root_in_mapping)?).context("file/mapping offset underflow")?;
    let logical_end=logical_file_offset.checked_add(u64::try_from(logical_bytes)?).context("logical file end overflow")?;
    let mapping_end=mapping_file_offset.checked_add(u64::try_from(mapping_bytes)?).context("mapping file end overflow")?;
    ensure!(logical_end<=file_bytes && mapping_end<=file_bytes,"file join exceeds actual FD extent");
    let root_end=registered_file_offset.checked_add(u64::try_from(registered_bytes)?).context("registered file end overflow")?;
    Ok((mapping_file_offset,registered_file_offset,root_end.saturating_sub(file_bytes)))
}
#[cfg(test)] mod tests;

#[cfg(test)] mod qwen_control;

// DeepSeek owns its private compact-anchor field traversal; no Qwen fallback.
pub use crate::inference::models::deepseek4::cache::backing_roles::{
    deepseek_cache_roles,deepseek_anchor_roles,deepseek_provided_owner_roles,
    Presence,FieldRole,DeepseekCacheRoles,ProvidedDeepseekOwners,DeepseekProvidedOwnerRoles,
};
#[cfg(test)]mod deepseek_control;
