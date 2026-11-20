//! C2-only acquisition: original four-lane/full-turn authority, never four×144.
use super::*;
use crate::serve::api::engine_qwen35::anchor_exactness_tests::frozen_script::{Corpus, OriginalCorpus, Script, Validated};
use std::sync::Arc;
thread_local! {
    static ROOT:RefCell<Option<Scope>>=const{RefCell::new(None)};
    static WORKER:RefCell<Option<(Scope,Arc<()>,u64)>>=const{RefCell::new(None)};
}
pub(crate) struct C2WorkerBinding {
    file:File, expected:FileIdentity, acquisition:Acquisition, opens:u64, witness:Arc<()>,
}
impl C2WorkerBinding {
    pub(crate) fn observed(&self,witness:&Arc<()>,device:&mlx_native::MlxDevice,model:u64)->Result<Acquisition> {
        ensure!(Arc::ptr_eq(witness,&self.witness),"different loaded model owning witness");
        ensure!(FileIdentity::observed(&self.file)?==self.expected,"loader FD changed before worker observation");
        // ACTIVE must initialize first, so WORKER drops its Scope before ACTIVE
        // is destroyed when this real Engine worker exits.
        ACTIVE.with(|_|{});
        WORKER.with(|slot| {
            let mut slot=slot.borrow_mut();
            if slot.is_none() {
                ACTIVE.with(|active|->Result<()> {
                    let mut active=active.borrow_mut();
                    ensure!(active.is_none(),"worker already owns another acquisition");
                    *active=Some(Active {file:self.file.try_clone()?,expected:self.expected.clone(),
                        opens:self.opens,nonce:self.acquisition.nonce.clone()});
                    Ok(())
                })?;
                *slot=Some((Scope {acquisition:self.acquisition.clone(),config_checked:std::cell::Cell::new(true),
                    _same_thread:std::marker::PhantomData},witness.clone(),model));
            }
            let (scope,owner,instance)=slot.as_ref().context("worker scope missing")?;
            ensure!(Arc::ptr_eq(owner,witness)&&*instance==model,"worker model owning allocation changed");
            scope.observed(device,model,1)
        })
    }
}
impl Scope {
    pub(crate) fn begin_c2(binding:&Binding, identity:&Identity, model:&Path, output:&Path,
        packet:&[u8], frozen_script:&[u8])->Result<(Self,Validated)> {
        identity.validate().map_err(anyhow::Error::msg)?;
        let validated=Script::validate_bytes(frozen_script,identity.family)?;
        anyhow::ensure!(hash(frozen_script)==identity.token_vectors_sha256,
            "identity must bind exact complete frozen-script bytes");
        Ok((Self::begin_bound(binding,identity,model,output,packet)?,validated))
    }
    pub(crate) fn begin_c2_capture(binding:&Binding,identity:&Identity,model:&Path,output:&Path,
        packet:&[u8],corpus:&[u8])->Result<(Self,OriginalCorpus)> {
        identity.validate().map_err(anyhow::Error::msg)?;
        let source:Corpus=serde_json::from_slice(corpus)?;
        let originals=source.validate(identity.family)?;
        anyhow::ensure!(hash(corpus)==identity.token_vectors_sha256,
            "canonical acquisition must bind exact complete original request corpus bytes");
        Ok((Self::begin_bound(binding,identity,model,output,packet)?,originals))
    }
    pub(crate) fn begin_c2_fresh_capture(binding:&Binding,identity:&Identity,model:&Path,output:&Path,
        packet:&[u8],inputs:&[u8])->Result<Self> {
        crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_reference::validate_current_identity(identity)?;
        let source:crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_reference::Inputs=serde_json::from_slice(inputs)?;
        source.validate(identity.family)?;
        ensure!(hash(inputs)==identity.token_vectors_sha256,"fresh acquisition identity must bind complete fixed inputs");
        Self::begin_bound(binding,identity,model,output,packet)
    }
    pub(crate) fn begin_c2_registered_replay(binding:&Binding,identity:&Identity,model:&Path,output:&Path,
        packet:&[u8],registration:&FileBinding)->Result<(Self,crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_reference::Validated)> {
        crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_reference::validate_current_identity(identity)?;
        let reference=crate::serve::api::engine_qwen35::anchor_exactness_tests::fresh_reference::load_registration(registration)?;
        ensure!(identity.token_vectors_sha256==registration.sha256 && identity.family==reference.capture.identity.family
            && identity.model_sha256==reference.capture.identity.model_sha256
            && identity.tokenizer_sha256==reference.capture.identity.tokenizer_sha256
            && identity.template_sha256==reference.capture.identity.template_sha256,
            "fresh replay changed independent registration/model/tokenizer/template binding");
        Ok((Self::begin_bound(binding,identity,model,output,packet)?,reference))
    }
    pub(crate) fn c2_hold(self)->Result<()> {
        ROOT.with(|slot| { let mut slot=slot.borrow_mut();
            ensure!(slot.is_none(),"C2 loader scope already active"); *slot=Some(self); Ok(()) })
    }
    pub(crate) fn c2_release()->Result<()> {
        ROOT.with(|slot| { let scope=slot.borrow_mut().take().context("C2 loader scope missing")?;
            scope.check_unchanged() })
    }
    pub(crate) fn c2_loaded_binding(identity:&Identity,recipe:&ConfigRecipe<'_>,witness:Arc<()>)->Result<C2WorkerBinding> {
        ROOT.with(|slot| {
            let slot=slot.borrow(); let scope=slot.as_ref().context("actual loader must run on bound acquisition thread")?;
            scope.check_config(identity,recipe)?;
            ACTIVE.with(|active| { let active=active.borrow();
                let active=active.as_ref().context("bound loader descriptor missing")?;
                ensure!(active.nonce==scope.acquisition.nonce&&active.opens>0
                    &&FileIdentity::observed(&active.file)?==active.expected,"actual loader did not consume unchanged bound FD");
                Ok(C2WorkerBinding {file:active.file.try_clone()?,expected:active.expected.clone(),
                    acquisition:scope.acquisition.clone(),opens:active.opens,witness})
            })
        })
    }
}

impl Acquisition {
    /// Narrow an observed scope to one actual, already-created trace store.
    /// The packet binds the scope root; a child cannot escape it or use a
    /// symlink spelling. Binary/model/FD/queue/nonce attestations stay intact.
    pub(crate) fn for_store_root(mut self, root: &Path) -> Result<Self> {
        ensure!(self.output_root.is_absolute()
            && self.output_root.canonicalize()? == self.output_root,
            "acquisition scope root is not canonical");
        ensure!(root.is_absolute() && root.canonicalize()? == root
            && root.is_dir() && root.starts_with(&self.output_root),
            "trace-store root must be an actual canonical descendant of its acquisition scope");
        self.output_root = root.to_path_buf();
        Ok(self)
    }
}
