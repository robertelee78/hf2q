use super::*;

pub(super) fn assert_invalid(bytes: &[u8], manifest: &ReleaseManifestV1) {
    assert!(verify_bytes(bytes, manifest).is_err());
}

pub(super) fn fixture(options: FixtureOptions) -> Vec<u8> {
    let mut commands = vec![
        segment(
            "__TEXT",
            options.segment_max_protection,
            options.segment_initial_protection,
        ),
        segment("__LINKEDIT", 1, 1),
    ];
    for (kind, command) in [
        (LC_DYLD_INFO_ONLY, dyld_info_command()),
        (LC_SYMTAB, symtab_command()),
        (LC_DYSYMTAB, fixed_command(LC_DYSYMTAB, &[0; 72])),
        (LC_UUID, uuid_command()),
        (
            LC_FUNCTION_STARTS,
            linkedit_data_command(LC_FUNCTION_STARTS),
        ),
        (LC_DATA_IN_CODE, linkedit_data_command(LC_DATA_IN_CODE)),
    ] {
        if options.omitted_metadata_command != Some(kind) {
            commands.push(command);
        }
    }
    if options.include_build_version {
        commands.push(build_version(options.minimum_macos, options.sdk_macos));
    }
    commands.push(string_command(LC_LOAD_DYLINKER, options.dylinker));
    commands.push(dylib_command(options.dylib));
    commands.extend(options.extra_commands);
    if options.include_main {
        commands.push(main_command(0, options.main_stack_size));
    }
    if options.include_code_signature {
        commands.push(fixed_command(LC_CODE_SIGNATURE, &[0; 8]));
    }

    let command_bytes = commands.iter().map(Vec::len).sum::<usize>();
    let code_length = 32;
    let text_section_length = 16;
    let linkedit_payload_length = 40;
    let signature_length = usize::from(options.include_code_signature) * 16;
    let code_offset = 32 + command_bytes;
    let linkedit_offset = code_offset + code_length;
    let signature_offset = linkedit_offset + linkedit_payload_length;
    let file_length = signature_offset + signature_length;
    commands[0][48..56].copy_from_slice(&(linkedit_offset as u64).to_le_bytes());
    set_u64(
        &mut commands[0],
        72 + 32,
        0x1_0000_0000 + code_offset as u64,
    );
    set_u64(&mut commands[0], 72 + 40, text_section_length);
    set_u32(&mut commands[0], 72 + 48, code_offset as u32);
    commands[1][40..48].copy_from_slice(&(linkedit_offset as u64).to_le_bytes());
    commands[1][48..56]
        .copy_from_slice(&((linkedit_payload_length + signature_length) as u64).to_le_bytes());
    if let Some(dyld) = commands
        .iter_mut()
        .find(|command| command_kind(command) == LC_DYLD_INFO_ONLY)
    {
        set_u32(dyld, 8, linkedit_offset as u32);
        set_u32(dyld, 12, 8);
    }
    if let Some(symtab) = commands
        .iter_mut()
        .find(|command| command_kind(command) == LC_SYMTAB)
    {
        set_u32(symtab, 8, (linkedit_offset + 16) as u32);
        set_u32(symtab, 12, 1);
        set_u32(symtab, 16, (linkedit_offset + 32) as u32);
        set_u32(symtab, 20, 8);
    }
    if let Some(dysymtab) = commands
        .iter_mut()
        .find(|command| command_kind(command) == LC_DYSYMTAB)
    {
        set_u32(dysymtab, 24, 0);
        set_u32(dysymtab, 28, 1);
    }
    if let Some(function_starts) = commands
        .iter_mut()
        .find(|command| command_kind(command) == LC_FUNCTION_STARTS)
    {
        set_u32(function_starts, 8, (linkedit_offset + 8) as u32);
        set_u32(function_starts, 12, 8);
    }
    if let Some(data_in_code) = commands
        .iter_mut()
        .find(|command| command_kind(command) == LC_DATA_IN_CODE)
    {
        set_u32(data_in_code, 8, (linkedit_offset + 16) as u32);
        set_u32(data_in_code, 12, 0);
    }
    if options.include_main {
        let main = commands
            .iter_mut()
            .find(|command| u32::from_le_bytes(command[0..4].try_into().unwrap()) == LC_MAIN)
            .expect("main command");
        set_u64(main, 8, code_offset as u64);
    }
    if options.include_code_signature {
        let signature = commands.last_mut().expect("signature command");
        signature[8..12].copy_from_slice(&(signature_offset as u32).to_le_bytes());
        signature[12..16].copy_from_slice(&(signature_length as u32).to_le_bytes());
    }

    let mut bytes = Vec::with_capacity(file_length);
    push_u32(&mut bytes, 0xfeedfacf);
    push_u32(&mut bytes, 0x0100_000c);
    push_u32(&mut bytes, options.cpu_subtype);
    push_u32(&mut bytes, options.file_type);
    push_u32(&mut bytes, commands.len() as u32);
    push_u32(&mut bytes, command_bytes as u32);
    push_u32(&mut bytes, options.header_flags);
    push_u32(&mut bytes, 0);
    for command in commands {
        bytes.extend_from_slice(&command);
    }
    bytes.resize(file_length, 0xa5);
    bytes
}

pub(super) fn segment(name: &str, maximum_protection: u32, initial_protection: u32) -> Vec<u8> {
    let section_count = usize::from(name == "__TEXT");
    let mut command = vec![0_u8; 72 + section_count * 80];
    set_u32(&mut command, 0, LC_SEGMENT_64);
    let command_len = command.len() as u32;
    set_u32(&mut command, 4, command_len);
    command[8..8 + name.len()].copy_from_slice(name.as_bytes());
    let vm_address = if name == "__LINKEDIT" {
        0x1_0001_0000
    } else {
        0x1_0000_0000
    };
    set_u64(&mut command, 24, vm_address);
    set_u64(&mut command, 32, 0x4_000);
    set_u64(&mut command, 40, 0);
    set_u32(&mut command, 56, maximum_protection);
    set_u32(&mut command, 60, initial_protection);
    set_u32(&mut command, 64, section_count as u32);
    if section_count == 1 {
        command[72..78].copy_from_slice(b"__text");
        command[88..94].copy_from_slice(b"__TEXT");
        set_u32(&mut command, 72 + 52, 3);
        set_u32(&mut command, 72 + 64, 0x8000_0400);
    }
    command
}

pub(super) fn data_segment_with_overlapping_zerofill() -> Vec<u8> {
    let mut command = vec![0_u8; 72 + 2 * 80];
    set_u32(&mut command, 0, LC_SEGMENT_64);
    let command_len = command.len() as u32;
    set_u32(&mut command, 4, command_len);
    command[8..14].copy_from_slice(b"__DATA");
    set_u64(&mut command, 24, 0x1_0002_0000);
    set_u64(&mut command, 32, 0x1_000);
    set_u32(&mut command, 56, 3);
    set_u32(&mut command, 60, 3);
    set_u32(&mut command, 64, 2);
    for (index, section_name) in ["__bss", "__common"].into_iter().enumerate() {
        let start = 72 + index * 80;
        command[start..start + section_name.len()].copy_from_slice(section_name.as_bytes());
        command[start + 16..start + 22].copy_from_slice(b"__DATA");
        set_u64(&mut command, start + 32, 0x1_0002_0100);
        set_u64(&mut command, start + 40, 0x100);
        set_u32(&mut command, start + 52, 4);
        set_u32(&mut command, start + 64, 1);
    }
    command
}

pub(super) fn build_version(minimum_macos: u32, sdk_macos: u32) -> Vec<u8> {
    let mut command = fixed_command(LC_BUILD_VERSION, &[0; 16]);
    set_u32(&mut command, 8, 1);
    set_u32(&mut command, 12, minimum_macos);
    set_u32(&mut command, 16, sdk_macos);
    set_u32(&mut command, 20, 0);
    command
}

pub(super) fn main_command(entry_offset: u64, stack_size: u64) -> Vec<u8> {
    let mut command = fixed_command(LC_MAIN, &[0; 16]);
    set_u64(&mut command, 8, entry_offset);
    set_u64(&mut command, 16, stack_size);
    command
}

pub(super) fn dyld_info_command() -> Vec<u8> {
    fixed_command(LC_DYLD_INFO_ONLY, &[0; 40])
}

pub(super) fn symtab_command() -> Vec<u8> {
    fixed_command(LC_SYMTAB, &[0; 16])
}

pub(super) fn uuid_command() -> Vec<u8> {
    fixed_command(LC_UUID, &[0xa5; 16])
}

pub(super) fn linkedit_data_command(kind: u32) -> Vec<u8> {
    fixed_command(kind, &[0; 8])
}

pub(super) fn dylib_command(path: &str) -> Vec<u8> {
    dylib_command_kind(LC_LOAD_DYLIB, path)
}

pub(super) fn dylib_command_kind(kind: u32, path: &str) -> Vec<u8> {
    let mut command = string_command_with_offset(kind, path, 24);
    set_u32(&mut command, 12, 0);
    set_u32(&mut command, 16, 0x1_0000);
    set_u32(&mut command, 20, 0x1_0000);
    command
}

pub(super) fn string_command(kind: u32, value: &str) -> Vec<u8> {
    string_command_with_offset(kind, value, 12)
}

pub(super) fn string_command_with_offset(kind: u32, value: &str, offset: usize) -> Vec<u8> {
    let size = (offset + value.len() + 1).next_multiple_of(8);
    let mut command = vec![0_u8; size];
    set_u32(&mut command, 0, kind);
    set_u32(&mut command, 4, size as u32);
    set_u32(&mut command, 8, offset as u32);
    command[offset..offset + value.len()].copy_from_slice(value.as_bytes());
    command
}

pub(super) fn fixed_command(kind: u32, body: &[u8]) -> Vec<u8> {
    let mut command = vec![0_u8; 8 + body.len()];
    let command_len = command.len() as u32;
    set_u32(&mut command, 0, kind);
    set_u32(&mut command, 4, command_len);
    command[8..].copy_from_slice(body);
    command
}

pub(super) fn command_offset(bytes: &[u8], wanted: u32) -> usize {
    let count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let mut offset = 32;
    for _ in 0..count {
        let kind = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        if kind == wanted {
            return offset;
        }
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        offset += size as usize;
    }
    panic!("fixture command {wanted:#x} is absent");
}

pub(super) fn segment_command_size(bytes: &[u8]) -> usize {
    u32::from_le_bytes(bytes[36..40].try_into().unwrap()) as usize
}

pub(super) fn command_kind(command: &[u8]) -> u32 {
    u32::from_le_bytes(command[0..4].try_into().unwrap())
}

pub(super) fn manifest(
    binary_size: u64,
    minimum_macos: &str,
    non_system_dependency: bool,
) -> ReleaseManifestV1 {
    let dependencies = if non_system_dependency {
        json!([{"consumer":"bin/hf2q","install_name":"@rpath/libmlx.dylib"}])
    } else {
        json!([])
    };
    let bytes = serde_json::to_vec(&json!({
        "kind": "hf2q.release-manifest",
        "schema_version": 1,
        "package": "hf2q",
        "version": "0.2.0",
        "target": "aarch64-apple-darwin",
        "minimum_macos": minimum_macos,
        "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "channel": "stable",
        "code_signing": {
            "team_id": "A1B2C3D4E5",
            "identifier": "us.hf2q.cli",
            "certificate_common_name": "Developer ID Application: hf2q (A1B2C3D4E5)"
        },
        "compatibility": {
            "minimum_installer_protocol": 1,
            "minimum_updater_protocol": 1,
            "launcher_registry_schema": 1
        },
        "files": [
            {"path":"bin/hf2q","type":"regular","size":binary_size,"mode":"0755","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
            {"path":"share/doc/hf2q/README.md","type":"regular","size":1,"mode":"0644","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
            {"path":"share/licenses/hf2q/LICENSE-APACHE","type":"regular","size":1,"mode":"0644","sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}
        ],
        "non_system_dynamic_dependencies": dependencies
    }))
    .expect("manifest JSON");
    ReleaseManifestV1::parse_and_validate(&bytes).expect("valid test manifest")
}

pub(super) fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn set_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn set_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
