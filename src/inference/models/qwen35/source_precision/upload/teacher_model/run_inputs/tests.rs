use std::cell::Cell;
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use mlx_native::MlxDevice;
use safetensors::Dtype;

use super::*;
use crate::inference::models::qwen35::source_precision::teacher_execution_plan::{
    preflight_qwen35_source_teacher_execution, Qwen35SourceTeacherRunLimitsV1,
};
use crate::inference::models::qwen35::source_precision::topology::admit_qwen35_bf16_topology;
use crate::inference::models::qwen35::source_precision::topology_tests::{fixture, open};
use crate::inference::models::qwen35::source_precision::upload::teacher_model::prepare_with_capacity_for_test;
use crate::inference::models::qwen35::source_precision::upload_plan::{
    QwenSourceMetalCapacityV1, QwenSourceMetalUploadLimits,
};
use crate::intelligence::calibration::prediction_plan_for_test_bound;
use crate::intelligence::exact_teacher::TeacherTargetArtifactLimits;

mod capacity;

fn target_limits() -> TeacherTargetArtifactLimits {
    TeacherTargetArtifactLimits {
        max_vocabulary_size: 8,
        max_prediction_rows: 3,
        max_target_bytes: 1024 * 1024,
        top_k: 4,
    }
}

fn run_limits() -> Qwen35SourceTeacherRunLimitsV1 {
    Qwen35SourceTeacherRunLimitsV1 {
        max_examples: 2,
        max_forward_calls: 34,
        max_input_tokens_processed: 64,
        max_output_head_evaluations: 34,
        max_cache_tokens: 47,
    }
}

fn upload_limits() -> QwenSourceMetalUploadLimits {
    QwenSourceMetalUploadLimits {
        max_output_tensors: 4_096,
        max_total_output_bytes: 128 * 1024 * 1024 * 1024,
        max_single_buffer_bytes: 8 * 1024 * 1024 * 1024,
        host_reserve_bytes: 0,
        metal_reserve_bytes: 0,
    }
}

fn preparation_policy() -> Qwen35SourceTeacherPreparationPolicyV1 {
    Qwen35SourceTeacherPreparationPolicyV1 {
        max_cpu_control_mirror_bytes: 1024 * 1024,
        unmeasured_runtime_reserve_bytes: 0,
    }
}

fn capacity() -> QwenSourceMetalCapacityV1 {
    QwenSourceMetalCapacityV1 {
        host_available_bytes: 256 * 1024 * 1024 * 1024,
        metal_recommended_working_set_bytes: 256 * 1024 * 1024 * 1024,
        metal_current_allocated_bytes: 0,
        metal_max_buffer_bytes: 16 * 1024 * 1024 * 1024,
    }
}

fn work() -> Result<StructurallyBoundQwen35SourceTeacherWorkV1> {
    let source_fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let plan = prediction_plan_for_test_bound(
        topology.source().clone(),
        topology.verified_source_manifest_sha256().into(),
    );
    preflight_qwen35_source_teacher_execution(topology, plan, target_limits(), run_limits())
}

fn private_entry_count(path: &Path) -> usize {
    std::fs::read_dir(path).unwrap().count()
}

fn private_target_path(path: &Path) -> PathBuf {
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|entry| {
            entry
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".hf2q-exact-teacher-")
        })
        .expect("private target must exist")
}

fn prepared_cache(
    teacher: &PreparedQwen35SourceTeacherV1,
    max_sequence_tokens: u32,
) -> Result<(PreparedQwen35BaseTextCacheV1, RuntimeCapacityRecheckV1)> {
    let recheck = runtime_capacity_recheck(teacher, capacity())?;
    let cache =
        prepare_qwen35_base_text_cache(&teacher.config, &teacher.device, max_sequence_tokens)?;
    Ok((cache, recheck))
}

#[test]
fn destination_collision_and_teacher_failure_precede_or_clean_all_metal_work() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let collision = temp.path().join("collision.bin");
    std::fs::write(&collision, b"competitor")?;
    let teacher_calls = Cell::new(0usize);
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &collision,
        preparation_policy(),
        |_, _| {
            teacher_calls.set(teacher_calls.get() + 1);
            Err(anyhow!("teacher allocation must not run"))
        },
        |_, _| panic!("cache allocation must not run"),
    );
    assert!(rejected.is_err());
    assert_eq!(teacher_calls.get(), 0);
    assert_eq!(std::fs::read(&collision)?, b"competitor");

    let failed = temp.path().join("teacher-failed.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &failed,
        preparation_policy(),
        |_, _| {
            assert_eq!(private_entry_count(temp.path()), 2);
            Err(anyhow!("injected teacher failure"))
        },
        |_, _| panic!("cache allocation must not run after teacher failure"),
    );
    let error = rejected.err().expect("teacher failure must reject");
    assert!(format!("{error:#}").contains("injected teacher failure"));
    assert_eq!(private_entry_count(temp.path()), 1);
    assert!(!failed.exists());
    Ok(())
}

#[test]
fn reservation_mutation_or_late_collision_rejects_before_cache_allocation() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    let mutated = temp.path().join("mutated.bin");
    let cache_calls = Cell::new(0usize);
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &mutated,
        preparation_policy(),
        |topology, limits| {
            let teacher = prepare_with_capacity_for_test(
                topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )?;
            let target = private_target_path(temp.path());
            OpenOptions::new()
                .write(true)
                .open(target)?
                .write_all_at(b"X", 0)?;
            Ok(teacher)
        },
        |teacher, max_sequence_tokens| {
            cache_calls.set(cache_calls.get() + 1);
            prepared_cache(teacher, max_sequence_tokens)
        },
    );
    let error = rejected.err().expect("magic mutation must reject");
    assert!(format!("{error:#}").contains("reservation magic differs"));
    assert_eq!(cache_calls.get(), 0);
    assert_eq!(private_entry_count(temp.path()), 0);
    assert!(!mutated.exists());

    let collision = temp.path().join("late-collision.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &collision,
        preparation_policy(),
        |topology, limits| {
            let teacher = prepare_with_capacity_for_test(
                topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )?;
            std::fs::write(&collision, b"competitor")?;
            Ok(teacher)
        },
        |_, _| panic!("cache allocation must not follow a late collision"),
    );
    let error = rejected.err().expect("late collision must reject");
    assert!(format!("{error:#}").contains("destination already exists"));
    assert_eq!(std::fs::read(&collision)?, b"competitor");
    assert_eq!(private_entry_count(temp.path()), 1);
    Ok(())
}

#[test]
fn cache_failure_drops_prepared_weights_and_private_target() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    let output = temp.path().join("cache-failed.bin");
    let cache_calls = Cell::new(0usize);
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &output,
        preparation_policy(),
        |topology, limits| {
            prepare_with_capacity_for_test(
                topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |_, _| {
            cache_calls.set(cache_calls.get() + 1);
            Err(anyhow!("injected cache failure"))
        },
    );
    let error = rejected.err().expect("cache failure must reject");
    assert!(format!("{error:#}").contains("injected cache failure"));
    assert_eq!(cache_calls.get(), 1);
    assert_eq!(private_entry_count(temp.path()), 0);
    assert!(!output.exists());
    Ok(())
}

#[test]
fn topology_live_config_and_cache_layout_substitution_fail_closed() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    let alternative_fixture = fixture(Dtype::BF16, |_, specs| specs.reverse());
    let alternative_topology = admit_qwen35_bf16_topology(open(&alternative_fixture)?)?;
    let bad_topology = temp.path().join("bad-topology.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &bad_topology,
        preparation_policy(),
        |_, limits| {
            prepare_with_capacity_for_test(
                alternative_topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |_, _| panic!("cache allocation must not follow topology substitution"),
    );
    assert!(rejected.is_err());
    assert!(!bad_topology.exists());
    assert_eq!(private_entry_count(temp.path()), 0);

    let bad_config = temp.path().join("bad-config.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &bad_config,
        preparation_policy(),
        |topology, limits| {
            let mut teacher = prepare_with_capacity_for_test(
                topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )?;
            teacher.config.rms_norm_eps = f32::from_bits(teacher.config.rms_norm_eps.to_bits() + 1);
            Ok(teacher)
        },
        |_, _| panic!("cache allocation must not follow config substitution"),
    );
    assert!(rejected.is_err());
    assert!(!bad_config.exists());
    assert_eq!(private_entry_count(temp.path()), 0);

    let bad_cache = temp.path().join("bad-cache-layout.bin");
    let rejected = prepare_run_inputs_with_for_test(
        work()?,
        &bad_cache,
        preparation_policy(),
        |topology, limits| {
            prepare_with_capacity_for_test(
                topology,
                &device,
                upload_limits(),
                limits,
                capacity(),
                |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
            )
        },
        |teacher, max_sequence_tokens| {
            let recheck = runtime_capacity_recheck(teacher, capacity())?;
            let mut swapped = teacher.config.clone();
            swapped.layer_types.reverse();
            let cache =
                prepare_qwen35_base_text_cache(&swapped, &teacher.device, max_sequence_tokens)?;
            Ok((cache, recheck))
        },
    );
    assert!(rejected.is_err());
    assert!(!bad_cache.exists());
    assert_eq!(private_entry_count(temp.path()), 0);
    Ok(())
}

#[test]
fn production_run_inputs_join_exact_work_teacher_cache_and_private_target() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let temp = tempfile::tempdir()?;
    let first_path = temp.path().join("first.bin");
    let mut first = prepare_qwen35_source_teacher_run_inputs(
        work()?,
        &first_path,
        &device,
        upload_limits(),
        preparation_policy(),
    )?;
    assert!(!first_path.exists());
    assert_eq!(private_entry_count(temp.path()), 1);
    assert_eq!(first.catalog_sha256().len(), 64);
    assert_eq!(first.receipt_sha256().len(), 64);
    assert_eq!(first.receipt_for_test().expected_work.max_cache_tokens, 47);
    assert_eq!(
        first.receipt_for_test().expected_work.prediction_row_count,
        3
    );
    assert_eq!(
        first.receipt_for_test().weight_precision,
        "source_bf16_controls_f32"
    );
    assert_eq!(
        first.receipt_for_test().cache_precision,
        "base_text_f32_one_sequence"
    );
    assert!(!first.receipt_for_test().q4_repack);
    assert!(!first.receipt_for_test().dwq);
    assert!(!first.receipt_for_test().tq);
    assert!(!first.receipt_for_test().mtp_executed);
    assert!(!first.receipt_for_test().graph_encoded);
    assert!(!first.receipt_for_test().submitted);
    assert!(!first.receipt_for_test().completed);
    assert!(!first.receipt_for_test().target_finished);
    assert!(!first.receipt_for_test().target_published);
    assert_eq!(
        catalog_sha256(first.receipt_for_test())?,
        first.catalog_sha256()
    );
    assert_eq!(
        receipt_sha256(first.receipt_for_test())?,
        first.receipt_sha256()
    );
    let mut visited = 0usize;
    first
        ._prediction_plan
        .visit_examples(|_, _, _, _| -> Result<()> {
            visited += 1;
            Ok(())
        })?;
    assert_eq!(visited, 2);
    let stable_catalog = first.catalog_sha256().to_owned();
    let process_receipt = first.receipt_sha256().to_owned();
    first
        .receipt
        .runtime_capacity_recheck
        .capacity
        .host_available_bytes += 1;
    assert_eq!(catalog_sha256(first.receipt_for_test())?, stable_catalog);
    assert_ne!(receipt_sha256(first.receipt_for_test())?, process_receipt);
    drop(first);
    assert_eq!(private_entry_count(temp.path()), 0);

    let second_path = temp.path().join("second.bin");
    let second = prepare_qwen35_source_teacher_run_inputs(
        work()?,
        &second_path,
        &device,
        upload_limits(),
        preparation_policy(),
    )?;
    assert_eq!(second.catalog_sha256(), stable_catalog);
    drop(second);
    assert_eq!(private_entry_count(temp.path()), 0);
    Ok(())
}
