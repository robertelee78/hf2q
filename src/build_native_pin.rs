//! Shared build/test identity parser. Accepts source and Cargo-normalized manifests.
use toml::{Table, Value};
fn native_entry(key: &str, value: &Value) -> bool {
    key.replace('_', "-") == "mlx-native"
        || value
            .as_table()
            .and_then(|t| t.get("package"))
            .and_then(Value::as_str)
            == Some("mlx-native")
}
fn reject_native(table: Option<&Value>) -> Result<(), String> {
    if table
        .and_then(Value::as_table)
        .is_some_and(|t| t.iter().any(|(k, v)| native_entry(k, v)))
    {
        return Err("ambiguous target/alias/override native dependency".into());
    }
    Ok(())
}
pub(crate) fn exact_native_pin(manifest: &str) -> Result<String, String> {
    // Cargo validates TOML too. This parser also rejects duplicate keys/tables,
    // including quoted/escaped spellings which handwritten line parsing misses.
    let root: Table =
        toml::from_str(manifest).map_err(|e| format!("invalid Cargo manifest: {e}"))?;
    let dependencies = root
        .get("dependencies")
        .and_then(Value::as_table)
        .ok_or("missing root dependencies")?;
    let native = dependencies
        .get("mlx-native")
        .ok_or("missing exact mlx-native dependency")?;
    if dependencies
        .iter()
        .any(|(k, v)| k != "mlx-native" && native_entry(k, v))
    {
        return Err("aliased native dependency is ambiguous".into());
    }
    for section in ["dev-dependencies", "build-dependencies"] {
        reject_native(root.get(section))?;
    }
    if let Some(targets) = root.get("target").and_then(Value::as_table) {
        for target in targets.values().filter_map(Value::as_table) {
            for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
                reject_native(target.get(section))?;
            }
        }
    }
    if let Some(patches) = root.get("patch").and_then(Value::as_table) {
        for entries in patches.values() {
            reject_native(Some(entries))?;
        }
    }
    if root
        .get("replace")
        .and_then(Value::as_table)
        .is_some_and(|t| t.keys().any(|k| k.starts_with("mlx-native:")))
    {
        return Err("native replacement not a registry pin".into());
    }
    let version = match native {
        Value::String(v) => v.as_str(),
        Value::Table(t) => {
            if t.keys()
                .any(|k| !matches!(k.as_str(), "version" | "features" | "default-features"))
            {
                return Err("native dependency source must remain the exact registry pin".into());
            }
            if t.get("features").is_some_and(|v| {
                v.as_array()
                    .is_none_or(|a| a.iter().any(|x| x.as_str().is_none()))
            }) || t
                .get("default-features")
                .is_some_and(|v| v.as_bool().is_none())
            {
                return Err("malformed native feature declaration".into());
            }
            t.get("version")
                .and_then(Value::as_str)
                .ok_or("missing native table version")?
        }
        _ => return Err("invalid native dependency declaration".into()),
    };
    let exact = version
        .strip_prefix('=')
        .ok_or("native version must be exact")?;
    let parts: Vec<_> = exact.split('.').collect();
    if parts.len() != 3
        || parts.iter().any(|p| {
            p.is_empty()
                || !p.bytes().all(|b| b.is_ascii_digit())
                || (p.len() > 1 && p.starts_with('0'))
                || p.parse::<u64>().is_err()
        })
    {
        return Err("native version must be canonical numeric major.minor.patch".into());
    }
    Ok(exact.to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_pin_actual_source_normalized_package_and_next_version() {
        assert_eq!(
            exact_native_pin(include_str!("../Cargo.toml")).unwrap(),
            env!("HF2Q_MLX_NATIVE_VERSION")
        );
        // Entire Cargo-generated registry manifest, retained unchanged as an
        // historical fixture. Its0.15.1 is NOT a live producer/version whitelist.
        assert_eq!(
            exact_native_pin(include_str!(
                "../tests/fixtures/hf2q-0.1.24-normalized-cargo.toml"
            ))
            .unwrap(),
            "0.15.1"
        );
        assert_eq!(
            exact_native_pin("[dependencies.mlx-native]\nversion=\"=0.15.6\"\n").unwrap(),
            "0.15.6"
        );
    }
    #[test]
    fn native_pin_rejects_nonexact_and_malformed_versions() {
        for v in [
            "0.15.5",
            "^0.15.5",
            "=0.15",
            "=0.15.5-alpha",
            "=00.15.5",
            "=0.15.x",
        ] {
            for text in [
                format!("[dependencies]\nmlx-native=\"{v}\""),
                format!("[dependencies.mlx-native]\nversion=\"{v}\""),
            ] {
                assert!(exact_native_pin(&text).is_err());
            }
        }
    }
    #[test]
    fn native_pin_rejects_duplicate_target_alias_and_override_sources() {
        let base = "[dependencies]\nmlx-native=\"=0.15.5\"\n";
        for tail in [
            "mlx-native=\"=0.15.6\"",
            "[dependencies.mlx-native]\nversion=\"=0.15.5\"",
            "[target.'cfg(unix)'.dependencies]\nmlx-native=\"=0.15.5\"",
            "[dependencies.alias]\npackage=\"mlx-native\"\nversion=\"=0.15.5\"",
            "[patch.crates-io.mlx-native]\npath=\"/tmp/native\"",
        ] {
            assert!(exact_native_pin(&format!("{base}{tail}")).is_err());
        }
        for field in [
            "path=\"/tmp/native\"",
            "git=\"https://example.invalid/native\"",
            "workspace=true",
            "version=15",
        ] {
            assert!(exact_native_pin(&format!(
                "[dependencies.mlx-native]\nversion=\"=0.15.5\"\n{field}"
            ))
            .is_err());
        }
    }
    #[test]
    fn native_pin_comments_unrelated_and_source_table_forms() {
        assert!(
            exact_native_pin("# mlx-native=\"=0.15.5\"\n[dependencies]\nother=\"0.15.5\"").is_err()
        );
        for text in [
            "[dependencies]\nmlx-native=\"=0.15.5\" # pin",
            "[dependencies]\nmlx-native={version=\"=0.15.5\",default-features=false,features=[]}",
        ] {
            assert_eq!(exact_native_pin(text).unwrap(), "0.15.5");
        }
    }
}
