#!/usr/bin/env bash

# Emit the exact non-system Mach-O runtime closure for one executable.  A
# launcher digest alone is insufficient for dynamically linked developer
# baselines: the launcher can remain byte-identical while sibling engine
# dylibs change.  Rows are stable, sorted, and include the resolved path,
# bytes, full file snapshot, and SHA-256 of every admitted image.

hf2q_macos_runtime_manifest() {
    local executable=$1 executable_dir current dependency candidate resolved
    local index=0
    local -a queue=()
    local -a rows=()
    # Bash 3.2 with `set -u` treats expansion of an empty local array as an
    # unbound variable. Absolute runtime paths cannot equal this sentinel.
    local -a seen=("__hf2q_runtime_identity_sentinel__")

    [[ "$executable" == /* && -x "$executable" ]] || {
        echo "runtime-identity executable must be an absolute executable path: $executable" >&2
        return 1
    }
    for command in awk otool realpath shasum stat sort; do
        command -v "$command" >/dev/null || {
            echo "missing runtime-identity command: $command" >&2
            return 1
        }
    done

    executable=$(realpath "$executable") || return 1
    executable_dir=$(dirname "$executable")
    queue+=("$executable")
    while ((index < ${#queue[@]})); do
        current=${queue[$index]}
        index=$((index + 1))
        if printf '%s\n' "${seen[@]}" | awk -v target="$current" '$0 == target { found = 1 } END { exit !found }'; then
            continue
        fi
        seen+=("$current")
        [[ -f "$current" && -r "$current" ]] || {
            echo "runtime-identity image disappeared: $current" >&2
            return 1
        }
        case "$current" in
            *$'\t'*|*$'\n'*)
                echo "runtime-identity rejects control characters in paths" >&2
                return 1
                ;;
        esac
        rows+=("$(shasum -a 256 "$current" | awk '{print $1}')"$'\t'"$(stat -f '%d:%i:%z:%m:%c' "$current")"$'\t'"$current")

        while IFS= read -r dependency; do
            case "$dependency" in
                /System/Library/*|/usr/lib/*)
                    continue
                    ;;
                @rpath/*)
                    candidate="$executable_dir/${dependency#@rpath/}"
                    ;;
                @loader_path/*)
                    candidate="$(dirname "$current")/${dependency#@loader_path/}"
                    ;;
                @executable_path/*)
                    candidate="$executable_dir/${dependency#@executable_path/}"
                    ;;
                /*)
                    candidate=$dependency
                    ;;
                *)
                    echo "unsupported Mach-O dependency path: $dependency" >&2
                    return 1
                    ;;
            esac
            [[ -e "$candidate" ]] || {
                echo "runtime-identity dependency is unresolved: $dependency from $current" >&2
                return 1
            }
            resolved=$(realpath "$candidate") || return 1
            case "$resolved" in
                /System/Library/*|/usr/lib/*)
                    continue
                    ;;
            esac
            queue+=("$resolved")
        done < <(otool -L "$current" | awk 'NR > 1 { print $1 }')
    done

    ((${#rows[@]} > 0)) || {
        echo "runtime-identity closure is empty" >&2
        return 1
    }
    printf '%s\n' "${rows[@]}" | LC_ALL=C sort
}

hf2q_macos_verify_runtime_manifest() {
    local executable=$1 expected=$2 actual
    actual=$(hf2q_macos_runtime_manifest "$executable") || return 1
    [[ "$actual" == "$expected" ]] || {
        echo "Mach-O runtime closure changed during the measured run" >&2
        return 1
    }
}
