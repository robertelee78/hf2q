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

#[test]
fn route_host_table_names_each_switch_once_and_only_probed_families() {
    for (index, (name, families)) in SWITCHES.iter().enumerate() {
        assert!(name.starts_with("HF2Q_"), "{name}");
        assert_eq!(
            SWITCHES.iter().position(|(other, _)| other == name),
            Some(index),
            "duplicate {name}"
        );
        for family in *families {
            assert!(
                FAMILIES.contains(family),
                "{name}: {family:?} is not probed"
            );
        }
        assert_eq!(
            TENSOR_ONLY.contains(name),
            !families.is_empty(),
            "{name}: the list of tensor-only switches is out of date"
        );
    }
    for name in TENSOR_ONLY {
        assert!(SWITCHES.iter().any(|(switch, _)| *switch == name), "{name}");
    }
}

#[test]
fn route_host_with_the_tensor_api_runs_every_switch() {
    let host = host(&[]);
    for (name, _) in SWITCHES {
        assert_eq!(host.switch(name), Some(HostRoute::Runs), "{name}");
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
    for (name, families) in SWITCHES {
        let route = host.switch(name).expect("a table switch");
        match families.first() {
            None => {
                assert_eq!(route, HostRoute::Runs, "{name}");
                assert_eq!(route.reason(), None, "{name}");
            }
            Some(first) => {
                assert_eq!(route, HostRoute::Unavailable(*first), "{name}");
                let reason = route.reason().expect("a reason");
                assert!(reason.contains(first.name()), "{name}: {reason}");
                assert!(
                    reason.contains(mlx_native::TENSOR_API_UNAVAILABLE),
                    "{name}: {reason}"
                );
            }
        }
    }
    let summary = host.summary();
    for name in TENSOR_ONLY {
        assert!(summary.contains(name), "{name}: {summary}");
    }
    assert!(!summary.contains("HF2Q_QWEN_MOE_BATCHED"), "{summary}");
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
        for value in ["1", "on", " TRUE "] {
            let env = |asked: &str| (asked == name).then(|| value.to_owned());
            let error = host
                .require_explicit(&env)
                .expect_err("an explicit switch the host cannot run")
                .to_string();
            assert!(error.starts_with(&format!("{name} is on, but ")), "{error}");
            assert!(
                error.contains(mlx_native::TENSOR_API_UNAVAILABLE),
                "{error}"
            );
            assert!(error.contains("test device"), "{error}");
            assert!(error.contains(&format!("Unset {name}")), "{error}");
        }
    }
    // One missing family declines the switches that need it and no other.
    let host = self::host(&[TensorKernelFamily::MoeExpertTiles]);
    for (name, _) in SWITCHES {
        assert_eq!(
            host.switch(name) == Some(HostRoute::Runs),
            *name != "HF2Q_QWEN_MOE_LIVE_TILES",
            "{name}"
        );
    }
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
    for (name, families) in SWITCHES {
        let runs = host.switch(name) == Some(HostRoute::Runs);
        assert_eq!(runs, missing == 0 || families.is_empty(), "{name}");
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
