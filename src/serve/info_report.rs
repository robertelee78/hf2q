//! Human-readable rendering for the static GGUF serving preflight.

use super::api::engine::EngineMode;
use super::info::StaticInspection;

pub(super) fn print_report(report: &StaticInspection) {
    let max_slots = match report.engine_mode {
        EngineMode::SerialFifo => 1,
        EngineMode::SlotAware { max_slots } => max_slots,
    };
    let scheduler = match report.engine_mode {
        EngineMode::SerialFifo => "fifo-serial",
        EngineMode::SlotAware { .. } => "inflight-batched",
    };
    let one_slot = report
        .kv_bytes_per_token
        .saturating_mul(u64::from(report.context.effective_tokens))
        .saturating_add(report.kv_fixed_bytes_per_slot);
    let worst_case = one_slot.saturating_mul(u64::from(max_slots));

    println!("Model Information");
    println!("Model: {}", report.model_path.display());
    println!("Model ID: {}", report.model_id);
    println!("Architecture: {} ({})", report.architecture, report.family);
    println!("Quantization: {}", report.quant);
    println!("Model file size: {}", format_bytes(report.file_bytes));
    println!(
        "GGUF directory: {} metadata entries, {} tensors; required names and nonzero shapes validated; encoded types parsed",
        report.metadata_count, report.tensor_count
    );
    println!(
        "Context: {} tokens per slot ({}; GGUF maximum {})",
        report.context.effective_tokens,
        report.context.origin.as_str(),
        report.context.declared_tokens
    );
    println!(
        "Scheduler: {scheduler}; max concurrent slots: {max_slots}; context is not divided by slot count"
    );
    match report.kv_budget.bytes {
        Some(bytes) => println!(
            "Shared KV-cache budget: {} ({})",
            format_bytes(bytes),
            report
                .kv_budget
                .origin
                .map(|origin| origin.as_str())
                .unwrap_or("default")
        ),
        None => println!("Shared KV-cache budget: no explicit ceiling"),
    }
    match report.kv_persist_dir.as_deref() {
        Some(path) => {
            println!("Persistent KV store: {}", path.display());
            match report.kv_persist_budget.bytes {
                Some(bytes) => println!(
                    "Persistent KV disk budget: {} ({})",
                    format_bytes(bytes),
                    report
                        .kv_persist_budget
                        .origin
                        .map(|origin| origin.as_str())
                        .unwrap_or("default")
                ),
                None => println!("Persistent KV disk budget: no explicit ceiling"),
            }
        }
        None => {
            println!("Persistent KV store: disabled (supply --kv-persist PATH to enable)");
            if let Some(bytes) = report.kv_persist_budget.bytes {
                println!(
                    "Persistent KV disk budget: {} ({}, inactive until a store path is supplied)",
                    format_bytes(bytes),
                    report
                        .kv_persist_budget
                        .origin
                        .map(|origin| origin.as_str())
                        .unwrap_or("default")
                );
            }
        }
    }
    if report.kv_bytes_per_token > 0 {
        println!(
            "Estimated KV cache: {}/token; {} for one full slot; {} for {max_slots} full slot(s) worst-case",
            format_bytes(report.kv_bytes_per_token),
            format_bytes(one_slot),
            format_bytes(worst_case)
        );
        println!(
            "Estimated model file + worst-case KV: {} (before runtime scratch and allocator overhead)",
            format_bytes(report.file_bytes.saturating_add(worst_case))
        );
    } else {
        println!("Estimated KV cache: unavailable for this rejected architecture");
    }
    if let Ok(hardware) = crate::core::hardware::HardwareProfiler::detect() {
        println!(
            "Host memory: {} total, {} currently available",
            format_bytes(hardware.total_memory_bytes),
            format_bytes(hardware.available_memory_bytes)
        );
        let planned_residency = report.file_bytes.saturating_add(worst_case);
        if report.kv_bytes_per_token > 0 && planned_residency > hardware.available_memory_bytes {
            println!(
                "Warning: model-file + worst-case KV estimate exceeds currently available host memory by {}, before runtime overhead",
                format_bytes(planned_residency.saturating_sub(hardware.available_memory_bytes))
            );
        }
    }
    if let Some(bytes) = report.kv_budget.bytes {
        if one_slot > bytes {
            println!(
                "Warning: one full-context slot is estimated to exceed the shared KV-cache budget"
            );
        } else if worst_case > bytes {
            println!(
                "Warning: all {max_slots} slots at full context are estimated to exceed the shared KV-cache budget; admission remains demand-based"
            );
        }
    }
    println!("Vision: {}", report.vision);
    if let Some(path) = report.projector.as_deref() {
        println!("Projector: {}", path.display());
    }
    if report.support.is_ok() {
        println!(
            "Static checks: family config, embedded tokenizer, serving chat contract, and tensor directory passed{}",
            if report.projector.is_some() {
                "; explicit projector binding passed"
            } else {
                ""
            }
        );
    } else {
        println!(
            "Static checks: GGUF header and tensor directory parsed; serving rejection follows"
        );
    }
    println!(
        "Validation: static preflight (tensor payloads not decoded or uploaded; Metal not initialized)"
    );
    match &report.support {
        Ok(_) if report.projector.is_some() => println!("Serve support: ready (text + vision)"),
        Ok(_) if report.vision.starts_with("supported by text model") => {
            println!("Serve support: ready (text-only; vision requires --mmproj)")
        }
        Ok(status) => println!("Serve support: {status}"),
        Err(reason) => println!("Serve support: rejected — {reason}"),
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[(&str, u64)] = &[
        ("TiB", 1u64 << 40),
        ("GiB", 1u64 << 30),
        ("MiB", 1u64 << 20),
        ("KiB", 1u64 << 10),
    ];
    for (unit, divisor) in UNITS {
        if bytes >= *divisor {
            return format!("{:.2} {unit}", bytes as f64 / *divisor as f64);
        }
    }
    format!("{bytes} B")
}
