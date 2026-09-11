// Included in reader::tests so the existing focused conformance test filter
// also exercises metadata failures and their operator-visible diagnostics.
mod metadata_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct CapturedWarnings(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedWarnings {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn non_string_derivation_site_is_rejected_instead_of_treated_as_absent() {
        for value in [
            MetaValue::U32(1),
            MetaValue::F32(1.0),
            MetaValue::Bool(true),
        ] {
            let mut meta = base_meta();
            meta.push(("glp.derived_at", value));
            let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
            assert!(matches!(
                GlpVector::from_bytes(&bytes),
                Err(GlpError::Malformed(message))
                    if message == "glp.derived_at must be a string"
            ));
        }
    }

    #[test]
    fn site_transfer_warns_and_preserves_arbitrary_derivation_labels() {
        for declared in [
            None,
            Some("residual_stream_post_layer"),
            Some("ffn_out_pre_residual"),
            Some("residual_stream_post_layer:stream_zero"),
        ] {
            let mut meta = base_meta();
            if let Some(site) = declared {
                meta.push(("glp.derived_at", MetaValue::Str(site.into())));
            }
            let bytes = build_gguf(&meta, &[("direction.3", vec![1.0])]);
            let output = CapturedWarnings::default();
            let writer = output.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::WARN)
                .without_time()
                .with_ansi(false)
                .with_writer(move || writer.clone())
                .finish();
            let vector = tracing::subscriber::with_default(subscriber, || {
                GlpVector::from_bytes(&bytes).unwrap()
            });
            assert_eq!(vector.derived_at.as_deref(), declared);
            assert_eq!(vector.hook_point, GlpHookPoint::ResidualStreamPostLayer);
            let warning = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
            if let Some(site) = declared.filter(|site| *site != vector.hook_point.as_str()) {
                assert!(warning.contains("WARN"), "{warning}");
                assert!(warning.contains("different site"), "{warning}");
                assert!(warning.contains(site), "{warning}");
                assert!(warning.contains(vector.hook_point.as_str()), "{warning}");
            } else {
                assert!(warning.is_empty(), "unexpected warning: {warning}");
            }
        }
    }
}
