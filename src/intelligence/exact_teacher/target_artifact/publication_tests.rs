use std::io::{Read, Seek, SeekFrom, Write};

use super::*;

#[test]
fn no_replace_rename_rejects_a_collision_created_after_the_last_precheck() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("target.bin");
    let mut retained = RetainedTargetTemp::create(&output).unwrap();
    retained.as_file_mut().write_all(b"teacher").unwrap();

    let competing_output = output.clone();
    let result = retained.publish_noclobber_with_before_rename_for_test(
        7,
        |file| {
            file.seek(SeekFrom::Start(0)).unwrap();
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, b"teacher");
            Ok(())
        },
        move || std::fs::write(&competing_output, b"competitor").unwrap(),
    );

    assert!(result.is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"competitor");
    let entries = std::fs::read_dir(directory.path())
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(entries.len(), 1, "the private temporary must be removed");
    assert_eq!(entries[0].path(), output);
}
