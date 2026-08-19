use super::*;

#[test]
fn executable_text_and_modern_entry_point_are_exact() {
    let missing_main = fixture(FixtureOptions {
        include_main: false,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &missing_main,
        &manifest(missing_main.len() as u64, "14.0", false),
    );

    let duplicate_main = fixture(FixtureOptions {
        extra_commands: vec![main_command(1, 0)],
        ..FixtureOptions::default()
    });
    assert_invalid(
        &duplicate_main,
        &manifest(duplicate_main.len() as u64, "14.0", false),
    );

    let custom_stack = fixture(FixtureOptions {
        main_stack_size: 4096,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &custom_stack,
        &manifest(custom_stack.len() as u64, "14.0", false),
    );

    let mut zero_entry = fixture(FixtureOptions::default());
    let main = command_offset(&zero_entry, LC_MAIN);
    set_u64(&mut zero_entry, main + 8, 0);
    assert_invalid(
        &zero_entry,
        &manifest(zero_entry.len() as u64, "14.0", false),
    );

    let mut entry_in_text_padding = fixture(FixtureOptions::default());
    let main = command_offset(&entry_in_text_padding, LC_MAIN);
    let entry = u64::from_le_bytes(
        entry_in_text_padding[main + 8..main + 16]
            .try_into()
            .unwrap(),
    );
    set_u64(&mut entry_in_text_padding, main + 8, entry + 16);
    assert_invalid(
        &entry_in_text_padding,
        &manifest(entry_in_text_padding.len() as u64, "14.0", false),
    );

    let mut unaligned_entry = fixture(FixtureOptions::default());
    let main = command_offset(&unaligned_entry, LC_MAIN);
    let entry = u64::from_le_bytes(unaligned_entry[main + 8..main + 16].try_into().unwrap());
    set_u64(&mut unaligned_entry, main + 8, entry + 2);
    assert_invalid(
        &unaligned_entry,
        &manifest(unaligned_entry.len() as u64, "14.0", false),
    );

    let mut detached_text = fixture(FixtureOptions::default());
    let text = command_offset(&detached_text, LC_SEGMENT_64);
    set_u64(&mut detached_text, text + 40, 1);
    assert_invalid(
        &detached_text,
        &manifest(detached_text.len() as u64, "14.0", false),
    );

    let mut short_text = fixture(FixtureOptions::default());
    let text = command_offset(&short_text, LC_SEGMENT_64);
    set_u64(&mut short_text, text + 48, 31);
    assert_invalid(
        &short_text,
        &manifest(short_text.len() as u64, "14.0", false),
    );

    let mut non_instruction_text = fixture(FixtureOptions::default());
    let text = command_offset(&non_instruction_text, LC_SEGMENT_64);
    set_u32(&mut non_instruction_text, text + 72 + 64, 0);
    assert_invalid(
        &non_instruction_text,
        &manifest(non_instruction_text.len() as u64, "14.0", false),
    );

    let mut extra_text_attribute = fixture(FixtureOptions::default());
    let text = command_offset(&extra_text_attribute, LC_SEGMENT_64);
    set_u32(&mut extra_text_attribute, text + 72 + 64, 0x8200_0400);
    assert_invalid(
        &extra_text_attribute,
        &manifest(extra_text_attribute.len() as u64, "14.0", false),
    );

    let mut reserved_text_field = fixture(FixtureOptions::default());
    let text = command_offset(&reserved_text_field, LC_SEGMENT_64);
    set_u32(&mut reserved_text_field, text + 72 + 68, 1);
    assert_invalid(
        &reserved_text_field,
        &manifest(reserved_text_field.len() as u64, "14.0", false),
    );

    let mut incongruent_text_mapping = fixture(FixtureOptions::default());
    let text = command_offset(&incongruent_text_mapping, LC_SEGMENT_64);
    let address = u64::from_le_bytes(
        incongruent_text_mapping[text + 72 + 32..text + 72 + 40]
            .try_into()
            .unwrap(),
    );
    set_u64(&mut incongruent_text_mapping, text + 72 + 32, address + 16);
    assert_invalid(
        &incongruent_text_mapping,
        &manifest(incongruent_text_mapping.len() as u64, "14.0", false),
    );

    let mut overlapping_segment_vm = fixture(FixtureOptions::default());
    let text = command_offset(&overlapping_segment_vm, LC_SEGMENT_64);
    let linkedit = text + segment_command_size(&overlapping_segment_vm);
    let text_vm = u64::from_le_bytes(
        overlapping_segment_vm[text + 24..text + 32]
            .try_into()
            .unwrap(),
    );
    set_u64(&mut overlapping_segment_vm, linkedit + 24, text_vm);
    assert_invalid(
        &overlapping_segment_vm,
        &manifest(overlapping_segment_vm.len() as u64, "14.0", false),
    );

    let mut oversized_segment_file = fixture(FixtureOptions::default());
    let linkedit = command_offset(&oversized_segment_file, LC_SEGMENT_64)
        + segment_command_size(&oversized_segment_file);
    set_u64(&mut oversized_segment_file, linkedit + 32, 1);
    assert_invalid(
        &oversized_segment_file,
        &manifest(oversized_segment_file.len() as u64, "14.0", false),
    );

    let mut unknown_segment_flag = fixture(FixtureOptions::default());
    let text = command_offset(&unknown_segment_flag, LC_SEGMENT_64);
    set_u32(&mut unknown_segment_flag, text + 68, 1);
    assert_invalid(
        &unknown_segment_flag,
        &manifest(unknown_segment_flag.len() as u64, "14.0", false),
    );

    let overlapping_sections = fixture(FixtureOptions {
        extra_commands: vec![data_segment_with_overlapping_zerofill()],
        ..FixtureOptions::default()
    });
    assert_invalid(
        &overlapping_sections,
        &manifest(overlapping_sections.len() as u64, "14.0", false),
    );

    let mut relocated_text = fixture(FixtureOptions::default());
    let text = command_offset(&relocated_text, LC_SEGMENT_64);
    set_u32(&mut relocated_text, text + 72 + 56, 1);
    set_u32(&mut relocated_text, text + 72 + 60, 1);
    assert_invalid(
        &relocated_text,
        &manifest(relocated_text.len() as u64, "14.0", false),
    );
}

#[test]
fn load_command_strings_cannot_alias_fixed_fields() {
    let mut dylib_alias = fixed_command(LC_LOAD_DYLIB, &[0; 24]);
    set_u32(&mut dylib_alias, 8, 8);
    let bytes = fixture(FixtureOptions {
        extra_commands: vec![dylib_alias],
        ..FixtureOptions::default()
    });
    assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", false));

    let mut dylinker_alias = fixed_command(LC_LOAD_DYLINKER, &[0; 16]);
    set_u32(&mut dylinker_alias, 8, 8);
    let bytes = fixture(FixtureOptions {
        extra_commands: vec![dylinker_alias],
        ..FixtureOptions::default()
    });
    assert_invalid(&bytes, &manifest(bytes.len() as u64, "14.0", false));
}

#[test]
fn code_signature_and_load_command_bounds_are_exact() {
    let missing_signature = fixture(FixtureOptions {
        include_code_signature: false,
        ..FixtureOptions::default()
    });
    assert_invalid(
        &missing_signature,
        &manifest(missing_signature.len() as u64, "14.0", false),
    );

    let mut trailing = fixture(FixtureOptions::default());
    trailing.push(0);
    let linkedit = command_offset(&trailing, LC_SEGMENT_64) + segment_command_size(&trailing);
    let old_size = u64::from_le_bytes(trailing[linkedit + 48..linkedit + 56].try_into().unwrap());
    set_u64(&mut trailing, linkedit + 48, old_size + 1);
    assert_invalid(&trailing, &manifest(trailing.len() as u64, "14.0", false));

    let mut malformed_size = fixture(FixtureOptions::default());
    malformed_size[32 + 4..32 + 8].copy_from_slice(&7_u32.to_le_bytes());
    assert_invalid(
        &malformed_size,
        &manifest(malformed_size.len() as u64, "14.0", false),
    );

    let truncated = &fixture(FixtureOptions::default())[..40];
    assert!(matches!(
        verify_bytes(truncated, &manifest(truncated.len() as u64, "14.0", false)),
        Err(MachOError::Invalid | MachOError::Read)
    ));

    let unknown = fixture(FixtureOptions {
        extra_commands: vec![fixed_command(0x7fff_fff0, &[])],
        ..FixtureOptions::default()
    });
    assert_invalid(&unknown, &manifest(unknown.len() as u64, "14.0", false));
}

#[test]
fn supported_linkedit_metadata_is_bounded_and_disjoint_from_the_signature() {
    let bytes = fixture(FixtureOptions::default());
    verify_bytes(&bytes, &manifest(bytes.len() as u64, "14.0", false))
        .expect("canonical linkedit metadata");

    let mut overlaps_signature = fixture(FixtureOptions::default());
    let signature = command_offset(&overlaps_signature, LC_CODE_SIGNATURE);
    let signature_offset = u32::from_le_bytes(
        overlaps_signature[signature + 8..signature + 12]
            .try_into()
            .unwrap(),
    );
    let function_starts = command_offset(&overlaps_signature, LC_FUNCTION_STARTS);
    set_u32(
        &mut overlaps_signature,
        function_starts + 8,
        signature_offset,
    );
    set_u32(&mut overlaps_signature, function_starts + 12, 1);
    assert_invalid(
        &overlaps_signature,
        &manifest(overlaps_signature.len() as u64, "14.0", false),
    );

    let mut overlapping_payloads = fixture(FixtureOptions::default());
    let dyld = command_offset(&overlapping_payloads, LC_DYLD_INFO_ONLY);
    let dyld_offset = u32::from_le_bytes(
        overlapping_payloads[dyld + 8..dyld + 12]
            .try_into()
            .unwrap(),
    );
    let function_starts = command_offset(&overlapping_payloads, LC_FUNCTION_STARTS);
    set_u32(&mut overlapping_payloads, function_starts + 8, dyld_offset);
    assert_invalid(
        &overlapping_payloads,
        &manifest(overlapping_payloads.len() as u64, "14.0", false),
    );
}

#[test]
fn required_metadata_commands_and_symbol_partition_are_exact() {
    for kind in [
        LC_SYMTAB,
        LC_DYSYMTAB,
        LC_DYLD_INFO_ONLY,
        LC_UUID,
        LC_FUNCTION_STARTS,
        LC_DATA_IN_CODE,
    ] {
        let missing = fixture(FixtureOptions {
            omitted_metadata_command: Some(kind),
            ..FixtureOptions::default()
        });
        assert_invalid(&missing, &manifest(missing.len() as u64, "14.0", false));
    }

    let mut bad_partition = fixture(FixtureOptions::default());
    let dysymtab = command_offset(&bad_partition, LC_DYSYMTAB);
    set_u32(&mut bad_partition, dysymtab + 28, 2);
    assert_invalid(
        &bad_partition,
        &manifest(bad_partition.len() as u64, "14.0", false),
    );

    let mut malformed_data_in_code = fixture(FixtureOptions::default());
    let data_in_code = command_offset(&malformed_data_in_code, LC_DATA_IN_CODE);
    set_u32(&mut malformed_data_in_code, data_in_code + 12, 1);
    assert_invalid(
        &malformed_data_in_code,
        &manifest(malformed_data_in_code.len() as u64, "14.0", false),
    );

    let mut empty_function_starts = fixture(FixtureOptions::default());
    let function_starts = command_offset(&empty_function_starts, LC_FUNCTION_STARTS);
    set_u32(&mut empty_function_starts, function_starts + 12, 0);
    assert_invalid(
        &empty_function_starts,
        &manifest(empty_function_starts.len() as u64, "14.0", false),
    );

    let mut zero_uuid = fixture(FixtureOptions::default());
    let uuid = command_offset(&zero_uuid, LC_UUID);
    zero_uuid[uuid + 8..uuid + 24].fill(0);
    assert_invalid(&zero_uuid, &manifest(zero_uuid.len() as u64, "14.0", false));
}
