//! Phase-specific REQUIRED domains, not proof that any domain was observed.
//! Preload upper bounds require independent review; Planned never supplies it.
use super::packet::Phase;

pub(super) fn describe(phase: Phase) -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::with_capacity(32);
    let proof = if phase == Phase::Preload {
        "ReviewedSourceUpperRequired_NotPlanned"
    } else {
        "ActualOwnerOrFullConservativeCoveredEnvelopeRequired"
    };
    match phase {
        Phase::Preload => {
            out.extend([
                (
                    "identity/source_binary_native_file_workload_policy",
                    "IndependentExactBindingsRequired",
                ),
                ("load/same_fd_header", "ActualReadonlyFDRequired"),
                ("weights/all_requested_parents", proof),
                ("weights/operating_working_set_and_inactive_residual", proof),
                ("load/cpu_staging_and_maximum_dirty_batch", proof),
            ]);
        }
        Phase::Load => {
            out.extend([
                ("load/same_fd_actual_tensor_ranges", proof),
                ("weights/all_mapped_parent_ids", proof),
                ("weights/expanded_f32_parent_ids", proof),
                ("load/cpu_staging_and_maximum_dirty_batch", proof),
            ]);
        }
        Phase::Provision | Phase::Forward | Phase::Drain => {
            out.extend([
                ("weights/all_requested_parents_still_retained", proof),
                ("weights/operating_working_set_and_inactive_residual", proof),
            ]);
        }
        Phase::Shutdown => {
            out.extend([
                ("worker/closed_actual_operations", proof),
                ("model/retained_or_released_observed_no_drop_refund", proof),
                ("load/same_fd_final_identity", proof),
            ]);
        }
    }
    if matches!(
        phase,
        Phase::Preload | Phase::Provision | Phase::Forward | Phase::Drain
    ) {
        for owner in [
            "loaded/current",
            "retained_slot/0",
            "retained_slot/1",
            "retained_slot/2",
            "retained_slot/3",
        ] {
            out.push((owner, proof));
        }
        out.extend([
            ("cache/current_and_pending_anchor_roles", proof),
            ("cache/live_logits_and_cursor_poison_graft_state", proof),
        ]);
    }
    if matches!(phase, Phase::Preload | Phase::Forward | Phase::Drain) {
        out.extend([
            ("tls/prefill_free_issued_residency_union", proof),
            ("tls/submission_input_free_issued_residency_union", proof),
            ("tls/decode_free_issued_residency_union", proof),
            ("forward/local_direct_and_native_internal_roots", proof),
            (
                "commands/queued_checked_and_completed_retained_resources",
                proof,
            ),
            ("commands/every_own_Result_and_moe_status", proof),
        ]);
    }
    if phase != Phase::Preload {
        out.push(("native/creation_attempt_history_NOT_live_peak", proof));
    }
    out.extend([
        ("native/whole_current_live_pending_scope", proof),
        (
            "native/role_partition_or_full_conservative_rejection",
            proof,
        ),
        ("cpu/request_channel_token_persist_capacities", proof),
        ("observer/bounded_metadata_and_io", proof),
        (
            "os/driver_competition_freshness_reaction",
            "IndependentFreshOperatingEvidenceRequired",
        ),
    ]);
    out
}
