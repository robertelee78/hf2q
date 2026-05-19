//! ADR-033 §Pa exhaustive acceptance gate.
//!
//! For every entry in `data/apex-references/manifest.json` (21 entries
//! covering 6 model families × 3-6 tiers each), the test walks every
//! `(tensor_name, GgmlType)` line in the vendored mudler config and
//! asserts the algorithmic `ApexPolicy::target_for(tensor_name +
//! ".weight")` returns the same `GgmlType`.
//!
//! Per ADR-033 §Pa:
//!   > "For each supported MoE arch and each algorithmic tier:
//!   >  hf2q's `target_for` output ... matches `vendor/apex-quant/
//!   >  scripts/generate_config.sh --profile <tier> --layers <N>`
//!   >  output line-for-line."
//!
//! Equivalence: the vendored files ARE the canonical output of
//! `generate_config.sh` at the pinned mudler SHA
//! `63c5048b7dc9ff230f2397d7bc445ca28894b769`. A divergence means
//! either:
//!   (a) the Rust port at `rules.rs` drifted from mudler's bash, OR
//!   (b) the vendored config was hand-edited away from algorithmic.
//!
//! Either way the divergence is surfaced as a structured
//! `(entry, tensor_name, expected, actual)` diagnostic.
//!
//! Lives inside the apex module (not `tests/apex_rules.rs`) because
//! the full ggml_quants module tree isn't on the lib facade — the
//! mantra ("no shortcuts") favors running this test against the
//! production source directly rather than wiring 11 sibling modules
//! through `#[path = ...]` for narrow test-only visibility.

#[cfg(test)]
mod tests {
    use super::super::super::tensor_ref::{ArchName, SourceDtype, TensorRef};
    use super::super::fingerprint::{manifest_entries, vendor_config_content, ApexConfigRef};
    use super::super::mudler_config::MudlerConfig;
    use super::super::policy::ApexPolicy;
    use super::super::rules::ApexTier;

    /// Per ADR §Pa exit gate: pin the manifest entry count so an
    /// accidental drop of a family-tier surface in
    /// `data/apex-references/manifest.json` fails this test loudly.
    #[test]
    fn manifest_pins_at_21_entries() {
        let n = manifest_entries().len();
        assert_eq!(
            n, 21,
            "ADR-033 §Pa ships 21 manifest entries; got {n}"
        );
    }

    /// Every manifest entry must resolve to a parseable baked vendor
    /// config. Catches vendor-time mistakes (entry added without
    /// `include_str!` line) at test time rather than at runtime via
    /// `ApexError::FingerprintConfigMissing`.
    #[test]
    fn every_manifest_entry_has_baked_config() {
        for entry in manifest_entries() {
            let content = vendor_config_content(&entry.mudler_config_path)
                .unwrap_or_else(|| {
                    panic!(
                        "manifest entry {{family={f}, tier={t}, fp={fp}}} \
                         references config path `{p}` that is NOT in VENDOR_CONFIGS",
                        f = entry.model_id_pattern,
                        t = entry.tier,
                        fp = entry.fingerprint,
                        p = entry.mudler_config_path,
                    );
                });
            assert!(
                !content.is_empty(),
                "vendor config {} is empty",
                entry.mudler_config_path
            );
        }
    }

    /// Known-non-canonical vendored configs (i.e., NOT a plain
    /// `generate_config.sh --profile <tier> --layers <N>` output).
    /// Each entry was hand-generated with a NON-DEFAULT override flag
    /// (e.g. `--near-exp iq2_s`) and is intentionally divergent from
    /// the algorithmic generator's output. Verified at 2026-05-19 by
    /// direct `bash generate_config.sh` comparison.
    ///
    /// These configs are still useful via the §9 fingerprint override
    /// path (`MudlerConfig` overlays the algorithmic policy) — they
    /// just don't byte-match the algorithmic dispatch by design.
    ///
    /// **Composition note**: this list is closed at compile time. A
    /// future entry that diverges from algorithmic for unexplained
    /// reasons will fail the test loudly — `KNOWN_NON_CANONICAL` must
    /// be updated AT THE SAME COMMIT as the new vendored file, with a
    /// rationale comment. Per [[feedback-no-loop-suppression-2026-05-17]]
    /// this is the explicit-allowlist pattern, not a silent skip.
    const KNOWN_NON_CANONICAL: &[(&str, &str)] = &[(
        "vendor/apex-quant/configs/qwen35a3b_mini.txt",
        "Hand-generated with `--near-exp iq2_s` (non-default for `mini` tier; \
         bash default is Q3_K). Vendored 2026-05-18; rationale: operator wanted \
         a more aggressive NEAR band on qwen35a3b for memory-constrained M5 Max \
         inference. Reaches operators via the §9 fingerprint override; the \
         algorithmic generator with --quant apex-mini stays at canonical mudler \
         defaults (Q3_K).",
    )];

    /// **§Pa exhaustive acceptance gate.**
    ///
    /// For every (family × tier) manifest entry, build the
    /// algorithmic `ApexPolicy` and diff its per-tensor output
    /// against the literal vendored mudler config. Any divergence not
    /// in [`KNOWN_NON_CANONICAL`] fails the gate with a structured
    /// `(entry, tensor, expected, actual)` diagnostic.
    ///
    /// Two-step assertion:
    ///   (1) `KNOWN_NON_CANONICAL` entries DO diverge (catches a
    ///       silently-fixed vendored file — keeps the list accurate).
    ///   (2) Every OTHER manifest entry is line-for-line equal to the
    ///       algorithmic dispatch.
    #[test]
    fn target_for_matches_every_vendored_config_line_for_line() {
        let mut divergences: Vec<String> = Vec::new();
        let mut known_non_canonical_now_clean: Vec<String> = Vec::new();

        for entry in manifest_entries() {
            let mismatches = check_entry_against_config(entry);
            let is_known =
                KNOWN_NON_CANONICAL.iter().any(|(p, _)| *p == entry.mudler_config_path);

            if mismatches.is_empty() {
                if is_known {
                    // The vendored file used to diverge but now
                    // matches algorithmic. Update KNOWN_NON_CANONICAL.
                    known_non_canonical_now_clean.push(entry.mudler_config_path.clone());
                }
                continue;
            }
            if is_known {
                // Known divergence — skip silently (we already have a
                // reason on record).
                continue;
            }
            divergences.push(format!(
                "── {family} (fp={fp}, tier={tier}, config={path}) ──\n{detail}",
                family = entry.model_id_pattern,
                fp = &entry.fingerprint[..16],
                tier = entry.tier,
                path = entry.mudler_config_path,
                detail = mismatches.join("\n"),
            ));
        }

        if !known_non_canonical_now_clean.is_empty() {
            panic!(
                "KNOWN_NON_CANONICAL entries now match algorithmic — remove from \
                 the allowlist (and verify with `bash scripts/generate_config.sh`):\n  {}",
                known_non_canonical_now_clean.join("\n  "),
            );
        }

        if !divergences.is_empty() {
            panic!(
                "§Pa exhaustive acceptance gate FAILED — algorithmic ApexPolicy \
                 diverged from {n_diverged}/{n_total} vendored mudler configs:\n\n{joined}\n\n\
                 Either the Rust port at src/quantize/ggml_quants/apex/rules.rs \
                 drifted from mudler's bash at \
                 vendor/apex-quant/scripts/generate_config.sh, OR the vendored \
                 config was hand-edited (in which case add it to \
                 KNOWN_NON_CANONICAL with a rationale). Reproduce via: \
                 `cd vendor/apex-quant && bash scripts/generate_config.sh \
                 --profile <tier> --layers <N>`.",
                n_diverged = divergences.len(),
                n_total = manifest_entries().len(),
                joined = divergences.join("\n\n"),
            );
        }
    }

    /// Compare one manifest entry's vendored config to the algorithmic
    /// generator. Returns a list of human-readable mismatch lines
    /// (empty = line-for-line match).
    fn check_entry_against_config(entry: &ApexConfigRef) -> Vec<String> {
        let content = match vendor_config_content(&entry.mudler_config_path) {
            Some(c) => c,
            None => {
                return vec![format!(
                    "vendor config not baked: {}",
                    entry.mudler_config_path
                )]
            }
        };

        let mudler = match MudlerConfig::parse(content, "tests::acceptance:vendor_config") {
            Ok(m) => m,
            Err(e) => return vec![format!("MudlerConfig::parse failed: {e}")],
        };

        let tier = match ApexTier::from_cli_name(&entry.tier) {
            Some(t) => t,
            None => return vec![format!("unknown tier `{}` in manifest entry", entry.tier)],
        };
        let arch = match ArchName::from_label(&entry.arch) {
            Some(a) => a,
            None => return vec![format!("unknown arch `{}` in manifest entry", entry.arch)],
        };
        // n_layers = num_hidden_layers + mtp_num_hidden_layers.
        // MTP variants (e.g. carnice qwen3.6 MTP) carry an extra
        // speculative-decoding layer at index N (so blk.0..N inclusive,
        // total N+1 layers). mudler's `generate_config.sh` walks one
        // more block when `mtp_num_hidden_layers > 0`; the algorithmic
        // policy must see the same effective total for the EDGE/NEAR/
        // MID region boundaries to match.
        let n_hidden = match entry
            .expected_hparams
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
        {
            Some(n) => n as u32,
            None => {
                return vec![
                    "expected_hparams.num_hidden_layers missing or not u64".to_string(),
                ]
            }
        };
        let n_mtp = entry
            .expected_hparams
            .get("mtp_num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let n_layers = n_hidden + n_mtp;
        let n_expert = match entry
            .expected_hparams
            .get("num_experts")
            .and_then(|v| v.as_u64())
        {
            Some(n) => n as u32,
            None => {
                return vec!["expected_hparams.num_experts missing or not u64".to_string()]
            }
        };

        // I-tier policies need `new_with_imatrix`; non-I use `new`.
        // We're not threading actual imatrix data — the algorithmic
        // dispatch in `target_for` doesn't read imatrix; the I-tier
        // constructor gate just confirms the arch is in
        // `SUPPORTED_FOR_IMATRIX`.
        let policy_res = if tier.requires_imatrix() {
            ApexPolicy::new_with_imatrix(tier, arch, n_layers, n_expert)
        } else {
            ApexPolicy::new(tier, arch, n_layers, n_expert)
        };
        let policy = match policy_res {
            Ok(p) => p,
            Err(e) => return vec![format!("ApexPolicy construction failed: {e}")],
        };

        // Sort keys for deterministic diagnostic output.
        let mut keys: Vec<(&String, &super::super::super::ggml_type::GgmlType)> =
            mudler.map.iter().collect();
        keys.sort_by(|a, b| a.0.cmp(b.0));

        let mut mismatches = Vec::new();
        for (key, &expected_type) in &keys {
            // Mudler keys are GGUF tensor names WITHOUT `.weight`.
            // Canonical name appends it (matches stock
            // llama-quantize --tensor-type-file semantics).
            let canonical_name = format!("{key}.weight");
            let layer_index = parse_layer_index(&canonical_name);
            // **Placeholder shape**: `ApexPolicy::target_for` +
            // `classify_moe_tensor` are name- and layer-driven only —
            // the role classifier doesn't consult `tensor.shape`
            // (grep confirms zero `tensor.shape` hits in
            // `apex/arches.rs` + `apex/policy.rs` as of 2026-05-19).
            // If future ApexPolicy logic introduces shape-sensitive
            // role dispatch (per codex review on 25931974), this
            // test would mask the divergence — update both the
            // classifier AND every `let shape = ...` line in this
            // test at the same commit.
            let shape = [4096usize, 1];
            let tref = TensorRef {
                name: &canonical_name,
                shape: &shape,
                source_dtype: SourceDtype::BF16,
                arch,
                layer_index,
            };
            match policy.target_for(&tref) {
                Ok(actual) if actual == expected_type => continue,
                Ok(actual) => mismatches.push(format!(
                    "  • {tensor}: vendored = {exp}, algorithmic = {got}",
                    tensor = key,
                    exp = expected_type.name(),
                    got = actual.name(),
                )),
                Err(e) => mismatches.push(format!(
                    "  • {tensor}: vendored = {exp}, algorithmic raised {err}",
                    tensor = key,
                    exp = expected_type.name(),
                    err = e,
                )),
            }
        }
        mismatches
    }

    /// Extract `<i>` from a GGUF tensor name of the form
    /// `blk.<i>.<rest>`. Returns `None` for structural tensors.
    fn parse_layer_index(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("blk.")?;
        let dot = rest.find('.')?;
        rest[..dot].parse::<usize>().ok()
    }

    /// Pair every imatrix-gated tier with its non-I sibling. Returns
    /// `None` for tiers that have no I-variant (currently only `Mini`).
    fn i_sibling(base: ApexTier) -> Option<ApexTier> {
        match base {
            ApexTier::Quality => Some(ApexTier::IQuality),
            ApexTier::Balanced => Some(ApexTier::IBalanced),
            ApexTier::Compact => Some(ApexTier::ICompact),
            _ => None,
        }
    }

    /// **ADR-033 §P4b structural invariant.**
    ///
    /// Mudler treats the I-tier and non-I-tier as identical
    /// `tier_rules` 7-tuples (verified at `rules.rs:167-195`: each I
    /// variant shares a single match arm with its non-I sibling). This
    /// test pins that invariant so a future drift in `rules.rs`
    /// (e.g., someone splits the match arm to add an I-tier-only
    /// tweak) fails the gate loudly.
    ///
    /// Together with `p4b_i_tier_target_for_matches_non_i_tier_*`
    /// below, this gives source-level byte-cmp equivalence between
    /// `--quant apex-i-<tier>` and `--quant apex-<tier>` at the
    /// per-tensor target_for layer — which combined with §Pa's gate
    /// (non-I tier ≡ vendored mudler config) transitively proves
    /// I-tier ≡ vendored mudler config. The byte-cmp end-to-end on
    /// real models is operator-time but the source-level invariant
    /// holds at every commit.
    #[test]
    fn p4b_tier_rules_i_variant_equals_non_i_sibling() {
        use super::super::rules::tier_rules;
        for base in [ApexTier::Quality, ApexTier::Balanced, ApexTier::Compact] {
            let i = i_sibling(base).expect("non-Mini base tiers have an I sibling");
            assert_eq!(
                tier_rules(base),
                tier_rules(i),
                "§P4b: tier_rules({base:?}) must structurally equal tier_rules({i:?}) \
                 — mudler treats the I-prefix as `use imatrix at quantize time`, not a \
                 different per-tensor type table. If you've intentionally diverged the \
                 tables, update both ADR-033 §P4b and the comment at rules.rs:167-195 \
                 at the same commit."
            );
        }
    }

    /// **ADR-033 §P4b end-to-end policy gate.**
    ///
    /// For every gemma4/qwen35moe manifest entry whose tier is in
    /// `{Quality, Balanced, Compact}`, walk every tensor in the
    /// vendored mudler config and assert
    /// `ApexPolicy::new_with_imatrix(ITier, ...).target_for(tref)`
    /// equals
    /// `ApexPolicy::new(BaseTier, ...).target_for(tref)`.
    ///
    /// The acceptance chain:
    ///   §Pa: non-I-tier `target_for` ≡ vendored mudler config
    ///        (already proven by `target_for_matches_every_vendored_config_line_for_line`)
    ///   §P4b: I-tier `target_for` ≡ non-I-tier `target_for`
    ///        (this test)
    /// Therefore: I-tier `target_for` ≡ vendored mudler config.
    ///
    /// MiniMaxM2 is excluded because v1 routes its I-tier requests to
    /// `ApexError::ImatrixRequiresInference` (per ADR §"Acceptance
    /// criteria (overall)" #4 — "MiniMax-M2.7 is convert-only in v1").
    /// `Mini` is excluded because mudler doesn't ship an `i-mini`
    /// surface (see `ApexTier::requires_imatrix`).
    #[test]
    fn p4b_i_tier_target_for_matches_non_i_tier_for_every_manifest_entry() {
        let mut divergences: Vec<String> = Vec::new();
        let mut tested_entries: usize = 0;

        for entry in manifest_entries() {
            let arch = match ArchName::from_label(&entry.arch) {
                Some(a) => a,
                None => continue,
            };
            // Per ADR §"Acceptance criteria (overall)" #4: I-tier gate
            // applies to {gemma4, qwen35moe} only (the inference-supported
            // subset); MiniMax-M2.7 is convert-only in v1.
            if !matches!(arch, ArchName::Gemma4 | ArchName::Qwen35Moe) {
                continue;
            }
            let base_tier = match ApexTier::from_cli_name(&entry.tier) {
                Some(t) if !t.requires_imatrix() => t,
                _ => continue,
            };
            let i_tier = match i_sibling(base_tier) {
                Some(t) => t,
                None => continue,
            };

            let content = match vendor_config_content(&entry.mudler_config_path) {
                Some(c) => c,
                None => {
                    divergences.push(format!(
                        "vendor config not baked: {}",
                        entry.mudler_config_path
                    ));
                    continue;
                }
            };
            let mudler =
                match MudlerConfig::parse(content, "tests::acceptance:p4b_vendor_config") {
                    Ok(m) => m,
                    Err(e) => {
                        divergences.push(format!(
                            "── {family} (tier={tier}) MudlerConfig::parse failed: {e}",
                            family = entry.model_id_pattern,
                            tier = entry.tier,
                        ));
                        continue;
                    }
                };

            let n_hidden = match entry
                .expected_hparams
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
            {
                Some(n) => n as u32,
                None => {
                    divergences.push(format!(
                        "── {family} (tier={tier}) expected_hparams.num_hidden_layers missing",
                        family = entry.model_id_pattern,
                        tier = entry.tier,
                    ));
                    continue;
                }
            };
            let n_mtp = entry
                .expected_hparams
                .get("mtp_num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let n_layers = n_hidden + n_mtp;
            let n_expert = match entry
                .expected_hparams
                .get("num_experts")
                .and_then(|v| v.as_u64())
            {
                Some(n) => n as u32,
                None => {
                    divergences.push(format!(
                        "── {family} (tier={tier}) expected_hparams.num_experts missing",
                        family = entry.model_id_pattern,
                        tier = entry.tier,
                    ));
                    continue;
                }
            };

            let non_i = match ApexPolicy::new(base_tier, arch, n_layers, n_expert) {
                Ok(p) => p,
                Err(e) => {
                    divergences.push(format!(
                        "── {family} (tier={tier}) non-I ApexPolicy::new failed: {e}",
                        family = entry.model_id_pattern,
                        tier = entry.tier,
                    ));
                    continue;
                }
            };
            let i = match ApexPolicy::new_with_imatrix(i_tier, arch, n_layers, n_expert) {
                Ok(p) => p,
                Err(e) => {
                    divergences.push(format!(
                        "── {family} (tier=i-{tier}) I ApexPolicy::new_with_imatrix failed: {e}",
                        family = entry.model_id_pattern,
                        tier = entry.tier,
                    ));
                    continue;
                }
            };

            // Walk every tensor in the vendored config, sorted for
            // deterministic diagnostic output.
            let mut keys: Vec<(&String, &super::super::super::ggml_type::GgmlType)> =
                mudler.map.iter().collect();
            keys.sort_by(|a, b| a.0.cmp(b.0));

            let mut mismatches: Vec<String> = Vec::new();
            for (key, _expected) in &keys {
                let canonical_name = format!("{key}.weight");
                let layer_index = parse_layer_index(&canonical_name);
                // See `check_entry_against_config`'s shape-placeholder
                // note: target_for is shape-blind on apex paths.
                let shape = [4096usize, 1];
                let tref = TensorRef {
                    name: &canonical_name,
                    shape: &shape,
                    source_dtype: SourceDtype::BF16,
                    arch,
                    layer_index,
                };
                let non_i_out = non_i.target_for(&tref);
                let i_out = i.target_for(&tref);
                match (non_i_out, i_out) {
                    (Ok(a), Ok(b)) if a == b => {}
                    (Ok(a), Ok(b)) => mismatches.push(format!(
                        "  • {tensor}: non-I = {a}, I = {b}",
                        tensor = key,
                        a = a.name(),
                        b = b.name(),
                    )),
                    (Ok(a), Err(e)) => mismatches.push(format!(
                        "  • {tensor}: non-I = {a}, I raised {e}",
                        tensor = key,
                        a = a.name(),
                    )),
                    (Err(e), Ok(b)) => mismatches.push(format!(
                        "  • {tensor}: non-I raised {e}, I = {b}",
                        tensor = key,
                        b = b.name(),
                    )),
                    (Err(ea), Err(eb)) if format!("{ea}") == format!("{eb}") => {}
                    (Err(ea), Err(eb)) => mismatches.push(format!(
                        "  • {tensor}: non-I raised {ea}, I raised {eb}",
                        tensor = key,
                    )),
                }
            }

            if !mismatches.is_empty() {
                divergences.push(format!(
                    "── {family} (base={base}, I={iname}, fp={fp}) ──\n{detail}",
                    family = entry.model_id_pattern,
                    base = base_tier.cli_name(),
                    iname = i_tier.cli_name(),
                    fp = &entry.fingerprint[..16],
                    detail = mismatches.join("\n"),
                ));
            }
            tested_entries += 1;
        }

        assert!(
            tested_entries > 0,
            "§P4b gate must exercise at least one (arch, base_tier) pair — \
             manifest entries are filtered too aggressively or empty"
        );

        if !divergences.is_empty() {
            panic!(
                "§P4b I-tier acceptance gate FAILED — I-tier ApexPolicy diverged from \
                 its non-I sibling on {n_diverged} entries (tested {tested_entries} \
                 (arch × base_tier) pairs):\n\n{joined}\n\n\
                 Per ADR-033 §P4b mudler treats I and non-I as identical type-tables; \
                 a divergence here means rules.rs drifted from generate_config.sh or \
                 the I-tier construction path overrode something it shouldn't.",
                n_diverged = divergences.len(),
                joined = divergences.join("\n\n"),
            );
        }
    }
}
