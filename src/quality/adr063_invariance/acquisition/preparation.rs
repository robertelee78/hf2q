//! Prefix-only preparation guard cannot yield a comparable acquisition.
use super::*;
pub fn preparation_prefix_digest(
    family: Family,
    prompts: &[Vec<u32>],
    vocab: u32,
) -> Result<String> {
    ensure!(
        matches!(family, Family::Qwen38 | Family::Qwen36 | Family::Gemma4)
            && vocab > 0
            && prompts.len() == 4
            && prompts
                .iter()
                .all(|p| p.len() == 128 && p.iter().all(|&id| id < vocab)),
        "four exact128-token selected-family preparation prefixes required"
    );
    ensure!(
        prompts
            .iter()
            .enumerate()
            .all(|(i, p)| !prompts[..i].contains(p)),
        "preparation lane identities must differ"
    );
    let mut h = Sha256::new();
    h.update(b"hf2q.q1.preparation-prefix128.le-u32.v1\0");
    h.update(serde_json::to_vec(&family)?);
    h.update(vocab.to_le_bytes());
    for prefix in prompts {
        h.update((prefix.len() as u32).to_le_bytes());
        for id in prefix {
            h.update(id.to_le_bytes());
        }
    }
    Ok(hex::encode(h.finalize()))
}

fn checked_descriptor_count(observed: u64, required: u64) -> Result<u64> {
    ensure!(
        required > 0 && observed >= required,
        "actual family loader did not consume required verified descriptors"
    );
    Ok(observed)
}

impl Scope {
    pub fn begin_preparation(
        binding: &Binding,
        identity: &Identity,
        prompts: &[Vec<u32>],
        vocab: u32,
        model: &Path,
        output: &Path,
        packet: &[u8],
    ) -> Result<Self> {
        identity.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            preparation_prefix_digest(identity.family, prompts, vocab)?
                == identity.token_vectors_sha256,
            "actual preparation prefix digest mismatch"
        );
        let mut scope = Self::begin_bound(binding, identity, model, output, packet)?;
        scope.acquisition.provenance = "WorkloadPreparationNotComparable".into();
        Ok(scope)
    }
    pub fn check_preparation_loaded(&self, required_model_opens: u64) -> Result<u64> {
        ensure!(
            self.acquisition.provenance == "WorkloadPreparationNotComparable"
                && self.config_checked.get(),
            "preparation-only guard required"
        );
        self.check_unchanged()?;
        ACTIVE.with(|slot| -> Result<u64> {
            let slot = slot.borrow();
            let active = slot.as_ref().context("preparation disappeared")?;
            checked_descriptor_count(active.opens, required_model_opens)
        })
    }
}

#[cfg(test)]
mod controls {
    use super::*;
    #[test]
    fn q1_preparation_descriptor_requirement_is_positive_and_preserves_actual_count() {
        assert_eq!(checked_descriptor_count(2, 2).unwrap(), 2);
        assert_eq!(checked_descriptor_count(3, 2).unwrap(), 3);
        assert!(checked_descriptor_count(1, 2).is_err());
        assert!(checked_descriptor_count(0, 2).is_err());
        assert!(checked_descriptor_count(2, 0).is_err());
    }
    #[test]
    fn q1_preparation_prefix_digest_is_distinct_bounded_and_family_specific() {
        let p = (0..4).map(|n| vec![n; 128]).collect::<Vec<_>>();
        let q38 = preparation_prefix_digest(Family::Qwen38, &p, 32).unwrap();
        assert_ne!(
            q38,
            preparation_prefix_digest(Family::Qwen36, &p, 32).unwrap()
        );
        let mut wrong = p.clone();
        wrong[0].push(0);
        assert!(preparation_prefix_digest(Family::Qwen38, &wrong, 32).is_err());
        wrong = p.clone();
        wrong[1] = wrong[0].clone();
        assert!(preparation_prefix_digest(Family::Qwen38, &wrong, 32).is_err());
        wrong = p.clone();
        wrong[3][127] = 32;
        assert!(preparation_prefix_digest(Family::Qwen38, &wrong, 32).is_err());
        let w = Workload {
            family: Family::Qwen38,
            vocab_size: 32,
            lanes: p
                .into_iter()
                .map(|mut v| {
                    v.extend(0..16);
                    v
                })
                .collect(),
        };
        assert_ne!(q38, tokens_digest(&w).unwrap());
    }
}

impl Scope {
    /// Preparation-only runtime identity; cannot attest comparable acquisition.
    pub fn observed_preparation(&self, device:&mlx_native::MlxDevice, model_instance:u64,
        required_opens:u64)->Result<Acquisition> {
        use mlx_native::metal::foreign_types::ForeignType;
        let opens=self.check_preparation_loaded(required_opens)?;
        let mut observed=self.acquisition.clone();
        observed.observed_model_opens=opens;observed.required_model_opens=required_opens;
        observed.device_registry_id=device.registry_id();
        observed.queue_identity=device.metal_queue().as_ptr() as usize as u64;
        observed.model_instance=model_instance;
        ensure!(observed.device_registry_id>0 && observed.queue_identity>0 && model_instance>0,
            "missing preparation runtime identity");Ok(observed)
    }
}
