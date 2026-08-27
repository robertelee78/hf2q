#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/macos_runtime_identity.sh
source "$root_dir/scripts/macos_runtime_identity.sh"

fixture_dir=$(mktemp -d)
trap 'rm -rf "$fixture_dir"' EXIT
mkdir -p "$fixture_dir/bin" "$fixture_dir/tools"
printf '%s\n' launcher >"$fixture_dir/bin/server"
printf '%s\n' engine-v1 >"$fixture_dir/bin/libengine-v1.dylib"
printf '%s\n' engine-v2 >"$fixture_dir/bin/libengine-v2.dylib"
chmod 755 "$fixture_dir/bin/server"
ln -s libengine-v1.dylib "$fixture_dir/bin/libengine.dylib"

cat >"$fixture_dir/tools/otool" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == -L ]]
printf '%s:\n' "$2"
if [[ "$(basename "$2")" == server ]]; then
    printf '\t@rpath/libengine.dylib (compatibility version 0.0.0, current version 0.0.0)\n'
    printf '\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n'
fi
EOF
chmod 755 "$fixture_dir/tools/otool"
export PATH="$fixture_dir/tools:$PATH"

manifest=$(hf2q_macos_runtime_manifest "$fixture_dir/bin/server")
[[ "$(printf '%s\n' "$manifest" | wc -l | tr -d ' ')" == 2 ]]
grep -Fq "$fixture_dir/bin/server" <<<"$manifest"
grep -Fq "$fixture_dir/bin/libengine-v1.dylib" <<<"$manifest"
hf2q_macos_verify_runtime_manifest "$fixture_dir/bin/server" "$manifest"

# The launcher is unchanged, but changing one engine image must invalidate the
# recorded closure.
printf '%s\n' mutated-engine >"$fixture_dir/bin/libengine-v1.dylib"
if hf2q_macos_verify_runtime_manifest "$fixture_dir/bin/server" "$manifest" \
  >/dev/null 2>&1; then
    echo 'runtime manifest accepted a mutated dylib' >&2
    exit 1
fi

# Restore the bytes, then retarget the admitted symlink. Resolution identity is
# part of the closure, so this must also fail even though the launcher is still
# byte-identical.
printf '%s\n' engine-v1 >"$fixture_dir/bin/libengine-v1.dylib"
rm "$fixture_dir/bin/libengine.dylib"
ln -s libengine-v2.dylib "$fixture_dir/bin/libengine.dylib"
if hf2q_macos_verify_runtime_manifest "$fixture_dir/bin/server" "$manifest" \
  >/dev/null 2>&1; then
    echo 'runtime manifest accepted a retargeted dylib symlink' >&2
    exit 1
fi

echo 'macOS runtime identity mutation contract passed'
