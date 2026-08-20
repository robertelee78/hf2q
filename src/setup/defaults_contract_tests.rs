use super::schema::{ConfiguredScheduler, ConvertDefaultsV2, OperatorConfigV2, ServeDefaultsV2};

#[test]
fn operator_config_v2_is_canonical_strict_and_uses_the_guide_defaults() {
    let config = OperatorConfigV2::guide_defaults().expect("guide defaults are valid");
    assert_eq!(config.convert.quant, "q4_k_m");
    assert_eq!(config.serve.host, "127.0.0.1");
    assert_eq!(config.serve.port, 8081);
    assert_eq!(config.serve.scheduler, ConfiguredScheduler::InflightBatched);
    assert_eq!(config.serve.max_slots, 1);

    let bytes = config.to_canonical_bytes().expect("canonical config");
    assert_eq!(bytes, include_bytes!("testdata/config_v2.toml"));
    assert_eq!(OperatorConfigV2::parse(&bytes).unwrap(), config);

    let text = String::from_utf8(bytes).unwrap();
    for hostile in [
        text.replace("schema_version = 2", "schema_version = 3"),
        text.replace("package = \"hf2q\"", "package = \"other\""),
        text.replace("port = 8081", "port = 0"),
        text.replace("max_slots = 1", "max_slots = 0"),
        text.replace("host = \"127.0.0.1\"", "host = \"example.com\""),
        format!("{text}unknown = true\n"),
    ] {
        assert!(OperatorConfigV2::parse(hostile.as_bytes()).is_err());
    }
}

#[test]
fn operator_config_v2_rejects_incoherent_fifo_slots_and_invalid_quant() {
    assert!(OperatorConfigV2::new(
        ConvertDefaultsV2 {
            quant: "not-a-quant".to_owned(),
        },
        ServeDefaultsV2 {
            host: "127.0.0.1".to_owned(),
            port: 8081,
            scheduler: ConfiguredScheduler::InflightBatched,
            max_slots: 1,
        },
    )
    .is_err());

    assert!(OperatorConfigV2::new(
        ConvertDefaultsV2 {
            quant: "q4_k_m".to_owned(),
        },
        ServeDefaultsV2 {
            host: "127.0.0.1".to_owned(),
            port: 8081,
            scheduler: ConfiguredScheduler::FifoSerial,
            max_slots: 2,
        },
    )
    .is_err());
}

#[test]
fn absent_operator_config_preserves_existing_command_behavior_without_claiming_root_authority() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::TempDir::new().unwrap();
    let root = temp.path().canonicalize().unwrap().join("unmanaged-root");
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(super::load_operator_config(Some(&root)).unwrap().is_none());
    assert_eq!(
        std::fs::metadata(&root).unwrap().permissions().mode() & 0o777,
        0o755
    );
}

#[test]
fn selected_state_root_loads_the_exact_operator_config() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::TempDir::new().unwrap();
    let root = temp.path().canonicalize().unwrap().join("selected-state");
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(
        root.join("config.toml"),
        include_bytes!("testdata/config_v2.toml"),
    )
    .unwrap();
    std::fs::set_permissions(
        root.join("config.toml"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();

    let loaded = super::load_operator_config(Some(&root))
        .unwrap()
        .expect("selected config exists");
    assert_eq!(loaded, OperatorConfigV2::guide_defaults().unwrap());
}
