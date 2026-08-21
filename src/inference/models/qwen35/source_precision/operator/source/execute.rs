//! Characterization and sealed AcceptanceHoldout execution entrypoints.

use super::*;

pub(crate) fn preflight_official_qwen38_source_teacher(
    request: &OfficialQwen38SourceTeacherRequestV1,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    let total_started = Instant::now();
    let profile = official_profile()?;
    let built = build_official_work(
        &request.model_dir,
        &request.output,
        &profile,
        OfficialPlanSelectionV1::Characterization(request.evaluation_split),
    )?;
    preflight_built_work(total_started, built, &request.output)
}

pub(crate) fn preflight_official_qwen38_acceptance_teacher(
    request: &OfficialQwen38AcceptanceTeacherRequestV1,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    let total_started = Instant::now();
    let profile = official_profile()?;
    let thresholds = official_acceptance_thresholds(&profile)?;
    let built = build_official_work(
        &request.model_dir,
        &request.output,
        &profile,
        OfficialPlanSelectionV1::Acceptance(thresholds),
    )?;
    preflight_built_work(total_started, built, &request.output)
}

fn preflight_built_work(
    total_started: Instant,
    mut built: OfficialWorkV1,
    output: &Path,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    built.work.preflight_target_destination(output)?;
    let device = MlxDevice::new().context("create official source-teacher Metal device")?;
    built.summary.metal_device = Some(device_summary(&device));
    let capacity_started = Instant::now();
    built.summary.capacity_preflight = Some(preflight_qwen35_source_teacher_run_inputs_capacity(
        &built.work,
        &device,
        built.upload_limits,
        built.preparation_policy,
    )?);
    built.summary.timings.capacity_preflight_ms = elapsed_ms(capacity_started.elapsed());
    built.summary.timings.total_ms = elapsed_ms(total_started.elapsed());
    Ok(built.summary)
}

pub(crate) fn run_official_qwen38_source_teacher(
    request: OfficialQwen38SourceTeacherRequestV1,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    let total_started = Instant::now();
    let profile = official_profile()?;
    let built = build_official_work(
        &request.model_dir,
        &request.output,
        &profile,
        OfficialPlanSelectionV1::Characterization(request.evaluation_split),
    )?;
    run_built_work(total_started, built, &request.output)
}

pub(crate) fn run_official_qwen38_acceptance_teacher(
    request: OfficialQwen38AcceptanceTeacherRequestV1,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    let total_started = Instant::now();
    let profile = official_profile()?;
    let thresholds = official_acceptance_thresholds(&profile)?;
    let built = build_official_work(
        &request.model_dir,
        &request.output,
        &profile,
        OfficialPlanSelectionV1::Acceptance(thresholds),
    )?;
    run_built_work(total_started, built, &request.output)
}

fn run_built_work(
    total_started: Instant,
    mut built: OfficialWorkV1,
    output: &Path,
) -> Result<OfficialQwen38SourceTeacherSummaryV1> {
    let device = MlxDevice::new().context("create official source-teacher Metal device")?;
    built.summary.metal_device = Some(device_summary(&device));
    let capacity_started = Instant::now();
    let capacity_preflight = preflight_qwen35_source_teacher_run_inputs_capacity(
        &built.work,
        &device,
        built.upload_limits,
        built.preparation_policy,
    )?;
    built.summary.timings.capacity_preflight_ms = elapsed_ms(capacity_started.elapsed());
    ensure!(
        capacity_preflight.eligible,
        "official source-teacher capacity preflight is not eligible"
    );
    built.summary.capacity_preflight = Some(capacity_preflight);
    let prepare_started = Instant::now();
    let inputs = prepare_qwen35_source_teacher_run_inputs(
        built.work,
        output,
        &device,
        built.upload_limits,
        built.preparation_policy,
    )?;
    built.summary.timings.prepare_weights_and_cache_ms =
        Some(elapsed_ms(prepare_started.elapsed()));
    let execution_started = Instant::now();
    let completed = run_qwen35_source_teacher(inputs)?;
    built.summary.timings.execute_and_publish_ms = Some(elapsed_ms(execution_started.elapsed()));
    built.summary.target_path = completed.path().to_path_buf();
    built.summary.executed = true;
    built.summary.target_artifact_sha256 = Some(completed.target_artifact_sha256().to_owned());
    built.summary.completion_receipt_sha256 =
        Some(completed.completion_receipt_sha256().to_owned());
    built.summary.structural_target_receipt = Some(completed.structural_target_receipt().clone());
    built.summary.timings.total_ms = elapsed_ms(total_started.elapsed());
    Ok(built.summary)
}
