//! The switch table and its answers. The joint step that keeps the stock
//! route on a host without the Metal tensor API is
//! `models::qwen35::route_host_tests`.

use super::*;

/// A host on which exactly `unavailable` cannot run.
fn host(unavailable: &[TensorKernelFamily]) -> HostRoutes {
    HostRoutes {
        device: "test device".to_owned(),
        registry_id: 1,
        unavailable: unavailable.to_vec(),
    }
}

/// The switches whose route needs a tensor-only family.
const TENSOR_ONLY: [&str; 9] = [
    "HF2Q_DFLASH2_DRAFT_STAGED",
    "HF2Q_REPACKED_GGUF_M32",
    "HF2Q_REPACKED_GGUF_M32_PROJ",
    "HF2Q_REPACKED_GGUF_M32_HEAD",
    "HF2Q_REPACKED_GGUF_ANY_WIDTH",
    "HF2Q_REPACKED_GGUF_SHARED_HEAD",
    "HF2Q_QWEN_PREFILL_PREPARED",
    "HF2Q_QWEN_MOE_STAGED_M8",
    "HF2Q_QWEN_MOE_LIVE_TILES",
];

/// What each tensor-only switch's own reader accepts as "on" on main.
/// `value` is what an operator might write; `on` is whether the reader
/// takes it.
const READER_VALUES: [(&str, bool, bool); 6] = [
    // value, on for an exact-`1` reader, on for a 1/on/true reader
    ("1", true, true),
    ("on", false, true),
    (" TRUE ", false, true),
    ("0", false, false),
    ("  ", false, false),
    ("yes", false, false),
];

#[test]
fn route_host_table_names_each_switch_once_and_only_probed_families() {
    for (index, row) in SWITCHES.iter().enumerate() {
        let name = row.name;
        assert!(name.starts_with("HF2Q_"), "{name}");
        assert_eq!(
            SWITCHES.iter().position(|other| other.name == name),
            Some(index),
            "duplicate {name}"
        );
        for family in row.families {
            assert!(
                FAMILIES.contains(family),
                "{name}: {family:?} is not probed"
            );
        }
        assert_eq!(
            TENSOR_ONLY.contains(&name),
            !row.families.is_empty(),
            "{name}: the list of tensor-only switches is out of date"
        );
    }
    for name in TENSOR_ONLY {
        assert!(SWITCHES.iter().any(|row| row.name == name), "{name}");
    }
}

/// The table's grammar is the grammar of the switch's reader, so a value
/// the reader treats as off is never refused as "on".
#[test]
fn route_host_grammar_is_each_readers_own() {
    use crate::inference::models::qwen35::joint_step_options::parse_switch;
    for (value, one, on_off_true) in READER_VALUES {
        assert_eq!(Grammar::One.is_on(value), one, "{value:?}");
        assert_eq!(Grammar::OnOffTrue.is_on(value), on_off_true, "{value:?}");
        // The reader of the prefill and MoE expert switches.
        assert_eq!(
            parse_switch("HF2Q_QWEN_PREFILL_PREPARED", Some(value)).unwrap_or(false),
            on_off_true,
            "{value:?}"
        );
    }
    let grammar = |name: &str| {
        SWITCHES
            .iter()
            .find(|row| row.name == name)
            .map(|row| row.grammar)
    };
    // Read with `== "1"`.
    for name in [
        "HF2Q_DFLASH2_DRAFT_AF4_G64",
        "HF2Q_DFLASH2_DRAFT_STAGED",
        "HF2Q_REPACKED_GGUF_M32",
        "HF2Q_REPACKED_GGUF_M32_PROJ",
        "HF2Q_REPACKED_GGUF_M32_HEAD",
        "HF2Q_REPACKED_GGUF_ANY_WIDTH",
        "HF2Q_REPACKED_GGUF_SHARED_HEAD",
        "HF2Q_QWEN_VERIFY_ATTN_TILE",
        "HF2Q_QWEN_VERIFY_MIXER_FUSED",
        "HF2Q_QWEN_VERIFY_REPLAY_REUSE",
        "HF2Q_DFLASH2_CONTEXT_FIRST",
        "HF2Q_DFLASH2_DRAFT_PREFETCH",
        "HF2Q_DFLASH2_GPU_DECISIONS",
    ] {
        assert_eq!(grammar(name), Some(Grammar::One), "{name}");
    }
    // Read through `parse_switch` (or the same three words).
    for name in [
        "HF2Q_DFLASH2_GPU_TARGET_TOKENS",
        "HF2Q_QWEN_PREFILL_PREPARED",
        "HF2Q_QWEN_PREFILL_FOLD_CUE",
        "HF2Q_QWEN_MOE_BATCHED",
        "HF2Q_QWEN_MOE_STAGED_M8",
        "HF2Q_QWEN_MOE_LIVE_TILES",
    ] {
        assert_eq!(grammar(name), Some(Grammar::OnOffTrue), "{name}");
    }
}

#[test]
fn route_host_with_the_tensor_api_runs_every_switch() {
    let host = host(&[]);
    for row in SWITCHES {
        assert_eq!(host.switch(row.name), Some(HostRoute::Runs), "{}", row.name);
    }
    assert_eq!(host.switch("HF2Q_NOT_A_ROUTE"), None);
    let all_on = |_: &str| Some("1".to_owned());
    host.require_explicit(&all_on).expect("every switch runs");
    assert!(
        host.summary()
            .contains("every fast-route kernel family runs"),
        "{}",
        host.summary()
    );
}

/// Without the tensor API every switch resolves off (a tensor-only family
/// of its route cannot run) or to a route of portable kernels.
#[test]
fn route_host_without_the_tensor_api_declines_exactly_the_tensor_only_switches() {
    let host = host(&FAMILIES);
    let cause = mlx_native::tensor_kernel_family_unavailable_reason();
    for row in SWITCHES {
        let name = row.name;
        let route = host.switch(name).expect("a table switch");
        match row.families.first() {
            None => {
                assert_eq!(route, HostRoute::Runs, "{name}");
                assert_eq!(route.reason(), None, "{name}");
            }
            Some(first) => {
                assert_eq!(route, HostRoute::Unavailable(*first), "{name}");
                let reason = route.reason().expect("a reason");
                assert!(reason.contains(first.name()), "{name}: {reason}");
                assert!(reason.contains(cause), "{name}: {reason}");
            }
        }
    }
    let summary = host.summary();
    assert!(summary.contains(cause), "{summary}");
    for name in TENSOR_ONLY {
        assert!(summary.contains(name), "{name}: {summary}");
    }
    assert!(!summary.contains("HF2Q_QWEN_MOE_BATCHED"), "{summary}");
}

/// The reason is the cause mlx-native reports for this process: the
/// disabling variable when it is set, the missing tensor API otherwise.
#[test]
fn route_host_reason_names_the_cause_of_this_process() {
    let disabled = std::env::var("MLX_NATIVE_DISABLE_METAL_TENSOR").as_deref() == Ok("1");
    let cause = mlx_native::tensor_kernel_family_unavailable_reason();
    assert_eq!(
        cause,
        if disabled {
            mlx_native::TENSOR_KERNELS_DISABLED
        } else {
            mlx_native::TENSOR_API_UNAVAILABLE
        }
    );
    let reason = HostRoute::Unavailable(TensorKernelFamily::RepackedGgufMm)
        .reason()
        .expect("a reason");
    assert_eq!(
        reason.contains("MLX_NATIVE_DISABLE_METAL_TENSOR"),
        disabled,
        "{reason}"
    );
}

#[test]
fn route_host_refuses_an_explicit_switch_it_cannot_run_and_names_it() {
    let host = host(&FAMILIES);
    let unset = |_: &str| None;
    host.require_explicit(&unset).expect("nothing is requested");
    // Off, blank and portable switches are not refused.
    let harmless = |name: &str| match name {
        "HF2Q_REPACKED_GGUF_M32" => Some("0".to_owned()),
        "HF2Q_QWEN_PREFILL_PREPARED" => Some("  ".to_owned()),
        "HF2Q_QWEN_MOE_BATCHED" | "HF2Q_QWEN_VERIFY_ATTN_TILE" => Some("1".to_owned()),
        _ => None,
    };
    host.require_explicit(&harmless)
        .expect("no tensor-only switch is on");
    for name in TENSOR_ONLY {
        let grammar = SWITCHES
            .iter()
            .find(|row| row.name == name)
            .expect("a table switch")
            .grammar;
        for (value, one, on_off_true) in READER_VALUES {
            let on = match grammar {
                Grammar::One => one,
                Grammar::OnOffTrue => on_off_true,
                Grammar::WaitLayers => value
                    .trim()
                    .parse::<usize>()
                    .is_ok_and(|layers| (2..=8).contains(&layers)),
            };
            let env = |asked: &str| (asked == name).then(|| value.to_owned());
            let result = host.require_explicit(&env);
            if !on {
                // The reader treats the value as off (or refuses it itself).
                result.unwrap_or_else(|error| panic!("{name}={value:?}: {error}"));
                continue;
            }
            let error = result
                .expect_err("an explicit switch the host cannot run")
                .to_string();
            assert!(error.starts_with(&format!("{name} is on, but ")), "{error}");
            assert!(
                error.contains(mlx_native::tensor_kernel_family_unavailable_reason()),
                "{error}"
            );
            assert!(error.contains("test device"), "{error}");
            assert!(error.contains(&format!("Unset {name}")), "{error}");
        }
    }
    // One missing family declines the switches that need it and no other.
    let host = self::host(&[TensorKernelFamily::MoeExpertTiles]);
    for row in SWITCHES {
        assert_eq!(
            host.switch(row.name) == Some(HostRoute::Runs),
            row.name != "HF2Q_QWEN_MOE_LIVE_TILES",
            "{}",
            row.name
        );
    }
}

/// The load-time rule: a probe that fails does not fail a load that selects
/// no tensor-only switch, and does fail one that selects one, naming it. A
/// probe that answers is held to the selected switches only.
#[test]
fn route_host_probe_failure_fails_only_a_load_that_selects_a_tensor_only_switch() {
    let broken = || -> Result<HostRoutes> { Err(anyhow::anyhow!("shader regression")) };
    let unset = |_: &str| None;
    require_selected(&unset, broken).expect("a stock load does not need the probe");
    // Portable switches, and a tensor-only switch its reader reads as off.
    let portable = |name: &str| match name {
        "HF2Q_QWEN_MOE_BATCHED" | "HF2Q_QWEN_VERIFY_ATTN_TILE" => Some("1".to_owned()),
        "HF2Q_REPACKED_GGUF_M32" => Some("on".to_owned()),
        _ => None,
    };
    require_selected(&portable, broken).expect("no tensor-only switch is selected");

    let selected = |name: &str| (name == "HF2Q_QWEN_MOE_STAGED_M8").then(|| "on".to_owned());
    let error = require_selected(&selected, broken).expect_err("a selected switch needs the probe");
    let text = format!("{error:#}");
    assert!(
        text.starts_with("HF2Q_QWEN_MOE_STAGED_M8 selected"),
        "{text}"
    );
    assert!(text.contains("shader regression"), "{text}");

    require_selected(&selected, || Ok(host(&[]))).expect("the host runs it");
    let error = require_selected(&selected, || Ok(host(&FAMILIES)))
        .expect_err("the host cannot run it")
        .to_string();
    assert!(
        error.starts_with("HF2Q_QWEN_MOE_STAGED_M8 is on, but "),
        "{error}"
    );
    require_selected(&portable, || Ok(host(&FAMILIES))).expect("portable switches run");
}

/// The admission points read the thread's host under test and the
/// process's otherwise; an unasked process admits, as before the probe
/// existed.
#[test]
fn route_host_admission_follows_the_asked_host() {
    {
        let _host = HostForTest::without_tensor_api();
        for family in FAMILIES {
            assert!(!runs(family), "{family:?}");
        }
        let nested = HostForTest::set(host(&[]));
        assert!(runs(TensorKernelFamily::RepackedGgufMm));
        drop(nested);
        assert!(!runs(TensorKernelFamily::RepackedGgufMm));
    }
    let process = HOST.get();
    for family in FAMILIES {
        assert_eq!(
            runs(family),
            process.is_none_or(|host| host.runs(family)),
            "{family:?}"
        );
    }
}

/// This host's own answers, from mlx-native: every family or none (the
/// tensor API is one capability). CI's hosted runners have none; the
/// summary this prints says which host ran.
#[test]
fn route_host_asks_this_device() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let host = HostRoutes::ask(&device, &mut registry).expect("ask this device");
    let again = HostRoutes::ask(&device, &mut registry).expect("ask again");
    assert_eq!(host, again);
    let missing = host.unavailable.len();
    assert!(
        missing == FAMILIES.len() || missing == 0,
        "{}: {missing} of {} tensor-only families unavailable",
        device.name(),
        FAMILIES.len()
    );
    for row in SWITCHES {
        let runs = host.switch(row.name) == Some(HostRoute::Runs);
        assert_eq!(
            runs,
            missing == 0 || row.families.is_empty(),
            "{}",
            row.name
        );
    }
    // A registry that compiles as a runtime compiler without the tensor
    // header answers as a host without the API on any device.
    let mut simulated = KernelRegistry::new();
    simulated
        .simulate_missing_tensor_header()
        .expect("simulate the missing header");
    let none = HostRoutes::ask(&device, &mut simulated).expect("ask the simulated host");
    assert_eq!(none.unavailable, FAMILIES.to_vec());
    eprintln!("{}", host.summary());
}
