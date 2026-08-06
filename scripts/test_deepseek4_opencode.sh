#!/usr/bin/env bash
set -euo pipefail

# Real OpenCode acceptance gate for an already-running DeepSeek hf2q server.
# This is intentionally not an HTTP-only protocol probe: OpenCode must inspect,
# edit, and test a disposable Rust project, then continue the same session and
# make a second verified change.

BASE_URL=${BASE_URL:-http://127.0.0.1:8080/v1}
MODEL_ID=${MODEL_ID:-}
PROVIDER_ID=${PROVIDER_ID:-hf2q-deepseek-acceptance}
OPENCODE=${OPENCODE:-/Users/robert.lee/.opencode/bin/opencode}
KEEP_WORK_DIR=${KEEP_WORK_DIR:-0}
TURN_TIMEOUT_SECONDS=${TURN_TIMEOUT_SECONDS:-600}
TOOL_TIMEOUT_SECONDS=${TOOL_TIMEOUT_SECONDS:-180}
SANDBOX_EXEC=${SANDBOX_EXEC:-/usr/bin/sandbox-exec}
TIMEOUT_BIN=${TIMEOUT_BIN:-}

for command in basename cargo cat chmod cmp cp curl dirname env git jq kill ls mkdir mktemp nc realpath rg rm \
  rustc sed sleep xcode-select xcrun; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
if [[ ! -x "$OPENCODE" ]]; then
  echo "OpenCode executable not found: $OPENCODE" >&2
  exit 2
fi
if [[ ! -x "$SANDBOX_EXEC" ]]; then
  echo "macOS sandbox executable not found: $SANDBOX_EXEC" >&2
  exit 2
fi
if [[ -z "$TIMEOUT_BIN" ]]; then
  if command -v gtimeout >/dev/null; then
    TIMEOUT_BIN=$(command -v gtimeout)
  elif command -v timeout >/dev/null; then
    TIMEOUT_BIN=$(command -v timeout)
  else
    echo "missing timeout/gtimeout watchdog" >&2
    exit 2
  fi
fi
if [[ ! -x "$TIMEOUT_BIN" ]]; then
  echo "timeout watchdog is not executable: $TIMEOUT_BIN" >&2
  exit 2
fi
TIMEOUT_BIN=$(realpath "$TIMEOUT_BIN")
if [[ "$KEEP_WORK_DIR" != "0" && "$KEEP_WORK_DIR" != "1" ]]; then
  echo "KEEP_WORK_DIR must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$TURN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TURN_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$TOOL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOOL_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if (( TOOL_TIMEOUT_SECONDS >= TURN_TIMEOUT_SECONDS )); then
  echo "TOOL_TIMEOUT_SECONDS must be less than TURN_TIMEOUT_SECONDS" >&2
  exit 2
fi
if [[ ! "$BASE_URL" =~ ^http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1$ ]]; then
  echo "BASE_URL must be a loopback endpoint of the form http://127.0.0.1:PORT/v1" >&2
  exit 2
fi
SERVER_PORT=${BASH_REMATCH[1]}
if (( SERVER_PORT > 65535 )); then
  echo "BASE_URL port is out of range: $SERVER_PORT" >&2
  exit 2
fi

# Pin OpenCode's cargo calls to the already-installed compiler instead of
# letting rustup populate the isolated HOME (1+ GiB and non-hermetic).
host_home=$(cd "${HOME:?HOME must be set}" && pwd -P)
opencode_dir=$(cd "$(dirname "$OPENCODE")" && pwd -P)
OPENCODE="$opencode_dir/$(basename "$OPENCODE")"
opencode_root=$(cd "$opencode_dir/.." && pwd -P)
rust_sysroot=$(rustc --print sysroot)
rust_sysroot=$(cd "$rust_sysroot" && pwd -P)
rust_toolchain_bin="$rust_sysroot/bin"
sdk_root=$(xcrun --show-sdk-path)
sdk_root=$(cd "$sdk_root" && pwd -P)
developer_dir=$(xcode-select -p)
developer_dir=$(cd "$developer_dir" && pwd -P)
linker_bin="$developer_dir/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
for executable in "$rust_toolchain_bin/cargo" "$rust_toolchain_bin/rustc" "$linker_bin"; do
  if [[ ! -x "$executable" ]]; then
    echo "installed Rust toolchain executable is missing: $executable" >&2
    exit 2
  fi
done
case "$OPENCODE" in
  "$host_home"/*) ;;
  *) echo "OpenCode must be installed below the current HOME for the acceptance sandbox" >&2; exit 2 ;;
esac
case "$rust_sysroot" in
  "$host_home"/*) ;;
  *) echo "Rust sysroot must be installed below the current HOME for the acceptance sandbox" >&2; exit 2 ;;
esac

models_json=$(curl --fail-with-body --silent --show-error \
  --connect-timeout 5 --max-time 10 "$BASE_URL/models")
if [[ -z "$MODEL_ID" ]]; then
  loaded_deepseek_count=$(jq '[.data[] | select(.loaded == true and .arch == "deepseek4")] | length' \
    <<<"$models_json")
  if (( loaded_deepseek_count != 1 )); then
    echo "hf2q gate requires exactly one loaded deepseek4 model; found $loaded_deepseek_count" >&2
    exit 1
  fi
  MODEL_ID=$(jq -r '.data[] | select(.loaded == true and .arch == "deepseek4") | .id' \
    <<<"$models_json")
else
  selected_model_count=$(jq --arg id "$MODEL_ID" \
    '[.data[] | select(.id == $id and .loaded == true and .arch == "deepseek4")] | length' \
    <<<"$models_json")
  if (( selected_model_count != 1 )); then
    echo "MODEL_ID=$MODEL_ID is not exactly one loaded deepseek4 model" >&2
    exit 1
  fi
fi
if [[ -z "$MODEL_ID" ]]; then
  echo "hf2q /v1/models returned an empty loaded DeepSeek model ID" >&2
  exit 1
fi

work_dir=$(mktemp -d -t hf2q-deepseek-opencode.XXXXXX)
work_dir=$(cd "$work_dir" && pwd -P)
project_dir="$work_dir/project"
runtime_dir="$work_dir/runtime"
capture_dir="$work_dir/capture"
events_first="$capture_dir/opencode-first.jsonl"
events_second="$capture_dir/opencode-second.jsonl"
stderr_first="$capture_dir/opencode-first.stderr"
stderr_second="$capture_dir/opencode-second.stderr"
initial_test_log="$capture_dir/initial-test.log"
sandbox_test_log="$capture_dir/sandbox-test.log"
final_test_log="$capture_dir/final-test.log"
before_source="$capture_dir/lib.rs.before"
oracle_source="$capture_dir/oracle.rs"
verifier_probe_source="$capture_dir/verifier-probe.rs"
opencode_sandbox_profile="$work_dir/opencode.sb"
code_sandbox_profile="$work_dir/code.sb"
sandbox_shell="$runtime_dir/bin/hf2q-sandbox-shell"
cargo_home="$project_dir/.cargo-home"
run_succeeded=0
sentinel_pid=
cleanup() {
  if [[ -n "$sentinel_pid" ]]; then
    /bin/kill "$sentinel_pid" >/dev/null 2>&1 || true
    wait "$sentinel_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$KEEP_WORK_DIR" == "1" || "$run_succeeded" != "1" ]]; then
    echo "OpenCode acceptance workspace retained at $work_dir" >&2
  else
    rm -rf "$work_dir"
  fi
}
trap cleanup EXIT

mkdir -p "$project_dir/src" "$project_dir/.tmp" "$cargo_home" "$capture_dir" "$runtime_dir/bin" \
  "$runtime_dir/home" "$runtime_dir/config" \
  "$runtime_dir/data" "$runtime_dir/cache" "$runtime_dir/state" "$runtime_dir/tmp"
git -C "$project_dir" init --quiet

cat >"$project_dir/Cargo.toml" <<'EOF'
[package]
name = "hf2q-opencode-acceptance"
version = "0.1.0"
edition = "2021"

[lib]
path = "src/lib.rs"
EOF

cat >"$project_dir/src/lib.rs" <<'EOF'
/// Return a valid index for a non-empty slice, or `None` for an empty slice.
pub fn bounded_index(index: usize, len: usize) -> Option<usize> {
    if len == 0 {
        None
    } else {
        Some(index.min(len))
    }
}

#[cfg(test)]
mod tests {
    use super::bounded_index;

    #[test]
    fn high_index_is_clamped() {
        assert_eq!(bounded_index(99, 3), Some(2));
    }

    #[test]
    fn empty_slice_has_no_index() {
        assert_eq!(bounded_index(7, 0), None);
    }
}
EOF
cp "$project_dir/src/lib.rs" "$before_source"

cat >"$oracle_source" <<EOF
#[path = "$project_dir/src/lib.rs"]
mod candidate;

fn main() {
    assert_eq!(candidate::bounded_index(99, 3), Some(2));
    assert_eq!(candidate::bounded_index(7, 0), None);
    assert_eq!(candidate::bounded_index(1, 3), Some(1));
}
EOF

cat >"$verifier_probe_source" <<EOF
fn main() {
    assert!(std::env::var_os("HF2Q_ACCEPTANCE_SECRET_SENTINEL").is_none());
    assert!(std::fs::File::create("$runtime_dir/verifier-escape").is_err());
    assert!(std::net::TcpStream::connect("1.1.1.1:80").is_err());
}
EOF

cat >"$opencode_sandbox_profile" <<EOF
(version 1)
(allow default)

; OpenCode is trusted orchestration code, but it may only mutate the
; disposable crate and its isolated runtime state.
(deny file-write*
  (require-not
    (require-any
      (literal (param "PROJECT"))
      (subpath (param "PROJECT"))
      (literal (param "RUNTIME"))
      (subpath (param "RUNTIME"))
      (literal "/dev/null")
      (literal "/dev/dtracehelper"))))

(deny process-exec
  (require-not
    (require-any
      (literal (param "OPENCODE"))
      (literal (param "TIMEOUT"))
      (subpath (param "PROJECT"))
      (subpath (param "RUNTIME"))
      (subpath (param "RUST"))
      (subpath (param "SDK"))
      (subpath (param "DEVELOPER"))
      (subpath "/System")
      (subpath "/usr/bin")
      (subpath "/bin")
      (subpath "/Library/Developer/CommandLineTools"))))

; The command-shell wrapper drops the orchestration sandbox and immediately
; applies code.sb. The timeout watchdog also drops the orchestration sandbox
; so it can terminate that deliberately distinct sandbox on timeout.
(allow process-exec
  (literal (param "SANDBOX_EXEC"))
  (with no-sandbox))
(allow process-exec
  (literal (param "TIMEOUT"))
  (with no-sandbox))

; OpenCode may talk only to the already-running loopback hf2q server.
(deny network-inbound
  (require-not (local ip)))
(deny network-outbound
  (require-not (remote ip "localhost:$SERVER_PORT")))

(deny signal
  (require-not (require-any (target self) (target same-sandbox))))
(deny process-info*
  (require-not (require-any (target self) (target same-sandbox))))
EOF

cat >"$code_sandbox_profile" <<EOF
(version 1)
(allow default)

; Every model-requested shell command and every compiled test process inherits
; this narrower policy. They cannot inspect OpenCode state, the immutable
; oracle/capture files, the caller's home, or private system configuration.
(deny file-read-data file-read-xattr
  (require-not
    (require-any
      (literal "/")
      (literal "/Users")
      (literal (param "HOST_HOME"))
      (literal (param "OPENCODE_ROOT"))
      (literal (param "OPENCODE_DIR"))
      (literal (param "OPENCODE"))
      (literal "$host_home/.rustup")
      (literal "$host_home/.rustup/toolchains")
      (literal "/private")
      (literal "/private/var")
      (literal "/private/var/folders")
      (literal (param "PROJECT"))
      (subpath (param "PROJECT"))
      (subpath (param "RUST"))
      (subpath (param "SDK"))
      (subpath (param "DEVELOPER"))
      (subpath "/System")
      (subpath "/usr/lib")
      (subpath "/usr/share")
      (subpath "/usr/bin")
      (subpath "/bin")
      (subpath "/Library/Developer/CommandLineTools")
      (subpath "/private/var/db/dyld")
      (literal "/private/etc/localtime")
      (literal "/private/etc/ssl/openssl.cnf")
      (literal "/dev")
      (literal "/dev/null")
      (literal "/dev/random")
      (literal "/dev/urandom")
      (literal "/dev/zero")
      (literal "/dev/tty")
      (literal "/dev/dtracehelper"))))

; Ancestor directory data is needed for path traversal to the pinned Rust
; toolchain, but directory enumeration is not. Deny listings explicitly.
(deny file-read-data
  (require-any
    (literal "/Users")
    (literal (param "HOST_HOME"))))

; Writes are confined to the disposable crate.
(deny file-write*
  (require-not
    (require-any
      (literal (param "PROJECT"))
      (subpath (param "PROJECT"))
      (literal "/dev/null")
      (literal "/dev/dtracehelper"))))

; Cargo, rustc, the linker, and test binaries can spawn only inside this same
; non-escalating policy.
(deny process-exec
  (require-not
    (require-any
      (subpath (param "PROJECT"))
      (subpath (param "RUST"))
      (subpath (param "SDK"))
      (subpath (param "DEVELOPER"))
      (subpath "/System")
      (subpath "/usr/bin")
      (subpath "/bin")
      (subpath "/Library/Developer/CommandLineTools"))))
(deny process-exec
  (require-any
    (literal "/usr/bin/pbcopy")
    (literal "/usr/bin/pbpaste")
    (literal "/usr/bin/security")))

; Model-authored code does not need a network listener or client.
(deny network-inbound)
(deny network-outbound)

; Deny direct access to credential and clipboard brokers. Keep unrelated Mach
; services available because the pinned compiler/runtime require system IPC.
(deny mach-lookup
  (require-any
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.pboard")
    (global-name-regex #"^com\\.apple\\.security")
    (global-name-regex #"^com\\.apple\\.pasteboard")
    (global-name-regex #"^com\\.apple\\..*pasteboard")))
(deny appleevent-send)

; Model-authored code may introspect/signal itself, never unrelated processes.
(deny signal
  (require-not (require-any (target self) (target same-sandbox))))
(deny process-info*
  (require-not (require-any (target self) (target same-sandbox))))
EOF

cat >"$sandbox_shell" <<EOF
#!/bin/bash
set -euo pipefail

if (( \$# != 2 )) || [[ \$1 != "-c" ]]; then
  echo "hf2q OpenCode gate: shell requires exactly -c COMMAND" >&2
  exit 126
fi
command=\$2
case "\$command" in
  cargo\ test*|cargo\ check*|cargo\ fmt*|git\ diff*|git\ status*) ;;
  *)
    echo "hf2q OpenCode gate: command is outside the acceptance allowlist" >&2
    exit 126
    ;;
esac

# rustdoc and rustc create transient files below TMPDIR. Keep model-authored
# commands inside the writable disposable project rather than widening access
# to OpenCode's isolated runtime state.
export TMPDIR="$project_dir/.tmp"

exec "$TIMEOUT_BIN" --kill-after=5 "${TOOL_TIMEOUT_SECONDS}s" \
  "$SANDBOX_EXEC" \
  -D "HOST_HOME=$host_home" \
  -D "OPENCODE_ROOT=$opencode_root" \
  -D "OPENCODE_DIR=$opencode_dir" \
  -D "OPENCODE=$OPENCODE" \
  -D "PROJECT=$project_dir" \
  -D "RUNTIME=$runtime_dir" \
  -D "RUST=$rust_sysroot" \
  -D "SDK=$sdk_root" \
  -D "DEVELOPER=$developer_dir" \
  -f "$code_sandbox_profile" \
  /bin/bash --noprofile --norc -c "\$command"
EOF
chmod 0555 "$sandbox_shell"

selected_model="$PROVIDER_ID/$MODEL_ID"
opencode_config=$(jq -cn \
  --arg provider "$PROVIDER_ID" --arg model "$MODEL_ID" \
  --arg selected "$selected_model" --arg base_url "$BASE_URL" \
  --arg shell "$sandbox_shell" '{
    model: $selected,
    small_model: $selected,
    shell: $shell,
    share: "disabled",
    permission: {
      "*": "deny",
      read: "allow",
      edit: "allow",
      glob: "allow",
      grep: "allow",
      list: "allow",
      todowrite: "allow",
      bash: {
        "*": "deny",
        "cargo test*": "allow",
        "cargo check*": "allow",
        "cargo fmt*": "allow",
        "git diff*": "allow",
        "git status*": "allow"
      },
      external_directory: "deny",
      webfetch: "deny",
      websearch: "deny",
      task: "deny",
      skill: "deny",
      question: "deny"
    },
    provider: {
      ($provider): {
        npm: "@ai-sdk/openai-compatible",
        name: "hf2q DeepSeek acceptance",
        options: {baseURL: $base_url, apiKey: "local"},
        models: {
          ($model): {
            name: "DeepSeek V4 Flash via hf2q",
            tool_call: true,
            temperature: true,
            limit: {context: 131072, output: 8192}
          }
        }
      }
    }
  }')

seatbelt_params=(
  -D "HOST_HOME=$host_home"
  -D "OPENCODE_ROOT=$opencode_root"
  -D "OPENCODE_DIR=$opencode_dir"
  -D "OPENCODE=$OPENCODE"
  -D "PROJECT=$project_dir"
  -D "RUNTIME=$runtime_dir"
  -D "RUST=$rust_sysroot"
  -D "SDK=$sdk_root"
  -D "DEVELOPER=$developer_dir"
  -D "SANDBOX_EXEC=$SANDBOX_EXEC"
  -D "TIMEOUT=$TIMEOUT_BIN"
)
opencode_seatbelt_args=("${seatbelt_params[@]}" -f "$opencode_sandbox_profile")
code_seatbelt_args=("${seatbelt_params[@]}" -f "$code_sandbox_profile")

run_code_sandboxed_with_timeout() {
  local timeout_seconds=$1
  shift
  (
    cd "$project_dir"
    env -i \
    HOME="$runtime_dir/home" \
    XDG_CONFIG_HOME="$runtime_dir/config" \
    XDG_DATA_HOME="$runtime_dir/data" \
    XDG_CACHE_HOME="$runtime_dir/cache" \
    XDG_STATE_HOME="$runtime_dir/state" \
    TMPDIR="$project_dir/.tmp" \
    CARGO_HOME="$cargo_home" \
    CARGO_NET_OFFLINE=true \
    CARGO_TARGET_DIR="$project_dir/target" \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$linker_bin" \
    DEVELOPER_DIR="$developer_dir" \
    SDKROOT="$sdk_root" \
    RUSTC="$rust_toolchain_bin/rustc" \
    PATH="$rust_toolchain_bin:/usr/bin:/bin" \
    LC_ALL=C \
    NO_COLOR=1 \
      "$TIMEOUT_BIN" --kill-after=5 "${timeout_seconds}s" \
      "$SANDBOX_EXEC" "${code_seatbelt_args[@]}" "$@"
  )
}

run_code_sandboxed() {
  run_code_sandboxed_with_timeout "$TOOL_TIMEOUT_SECONDS" "$@"
}

run_opencode_sandboxed() {
  (
    cd "$project_dir"
    env -i \
    HOME="$runtime_dir/home" \
    XDG_CONFIG_HOME="$runtime_dir/config" \
    XDG_DATA_HOME="$runtime_dir/data" \
    XDG_CACHE_HOME="$runtime_dir/cache" \
    XDG_STATE_HOME="$runtime_dir/state" \
    TMPDIR="$runtime_dir/tmp" \
    CARGO_HOME="$cargo_home" \
    CARGO_NET_OFFLINE=true \
    CARGO_TARGET_DIR="$project_dir/target" \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$linker_bin" \
    DEVELOPER_DIR="$developer_dir" \
    SDKROOT="$sdk_root" \
    RUSTC="$rust_toolchain_bin/rustc" \
    PATH="$rust_toolchain_bin:/usr/bin:/bin" \
    LC_ALL=C \
    NO_COLOR=1 \
      "$SANDBOX_EXEC" "${opencode_seatbelt_args[@]}" "$@"
  )
}

run_opencode() {
  (
    cd "$project_dir"
    env -i \
    HOME="$runtime_dir/home" \
    XDG_CONFIG_HOME="$runtime_dir/config" \
    XDG_DATA_HOME="$runtime_dir/data" \
    XDG_CACHE_HOME="$runtime_dir/cache" \
    XDG_STATE_HOME="$runtime_dir/state" \
    TMPDIR="$runtime_dir/tmp" \
    CARGO_HOME="$cargo_home" \
    CARGO_NET_OFFLINE=true \
    CARGO_TARGET_DIR="$project_dir/target" \
    CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER="$linker_bin" \
    DEVELOPER_DIR="$developer_dir" \
    SDKROOT="$sdk_root" \
    RUSTC="$rust_toolchain_bin/rustc" \
    PATH="$rust_toolchain_bin:/usr/bin:/bin" \
    LC_ALL=C \
    NO_COLOR=1 \
    OPENCODE_CONFIG_CONTENT="$opencode_config" \
      "$TIMEOUT_BIN" --kill-after=10 "${TURN_TIMEOUT_SECONDS}s" \
      "$SANDBOX_EXEC" "${opencode_seatbelt_args[@]}" "$OPENCODE" "$@"
  )
}

run_code_cargo() {
  run_code_sandboxed "$rust_toolchain_bin/cargo" "$@"
}

run_oracle() {
  local binary rc
  binary=$(mktemp "$project_dir/.hf2q-oracle-bin.XXXXXX") || return 1
  rm -f "$binary"
  if ! run_code_sandboxed "$rust_toolchain_bin/rustc" \
    --edition 2021 -C "linker=$linker_bin" -o "$binary" - <"$oracle_source"; then
    rm -f "$binary"
    return 1
  fi
  if run_code_sandboxed "$binary"; then
    rc=0
  else
    rc=$?
  fi
  rm -f "$binary"
  return "$rc"
}

run_verifier_probe() {
  local binary rc
  binary=$(mktemp "$project_dir/.hf2q-verifier-probe-bin.XXXXXX") || return 1
  rm -f "$binary"
  export HF2Q_ACCEPTANCE_SECRET_SENTINEL='must-not-cross-verifier-boundary'
  if ! run_code_sandboxed "$rust_toolchain_bin/rustc" \
    --edition 2021 -C "linker=$linker_bin" -o "$binary" - <"$verifier_probe_source"; then
    rc=1
  elif run_code_sandboxed "$binary"; then
    rc=0
  else
    rc=$?
  fi
  unset HF2Q_ACCEPTANCE_SECRET_SENTINEL
  rm -f "$binary"
  return "$rc"
}

# Fail closed if a future profile edit widens the untrusted-code envelope.
if run_code_sandboxed /bin/cat /etc/hosts >/dev/null 2>&1; then
  echo "OpenCode gate failed: sandbox can read host data outside the fixture" >&2
  exit 1
fi
if run_code_sandboxed /bin/ls "$host_home" >/dev/null 2>&1 ||
  run_code_sandboxed /bin/ls /Users >/dev/null 2>&1; then
  echo "OpenCode gate failed: sandbox can enumerate host user directories" >&2
  exit 1
fi
export HF2Q_ACCEPTANCE_SECRET_SENTINEL='must-not-cross-env-i-boundary'
if run_code_sandboxed /usr/bin/env |
  rg -q '^HF2Q_ACCEPTANCE_SECRET_SENTINEL='; then
  echo "OpenCode gate failed: parent environment leaked into model-authored code" >&2
  exit 1
fi
unset HF2Q_ACCEPTANCE_SECRET_SENTINEL
if run_code_sandboxed /usr/bin/pbpaste >/dev/null 2>&1 ||
  run_code_sandboxed /usr/bin/pbcopy </dev/null >/dev/null 2>&1 ||
  run_code_sandboxed /usr/bin/security list-keychains >/dev/null 2>&1; then
  echo "OpenCode gate failed: sandbox can invoke a clipboard or keychain client" >&2
  exit 1
fi
if run_code_sandboxed /usr/bin/nc -G 2 -z 1.1.1.1 80 >/dev/null 2>&1; then
  echo "OpenCode gate failed: sandbox can reach a non-loopback network peer" >&2
  exit 1
fi
/bin/sleep 60 &
sentinel_pid=$!
if ! /bin/kill -0 "$sentinel_pid" >/dev/null 2>&1; then
  echo "OpenCode gate failed: could not establish same-user signal sentinel" >&2
  exit 1
fi
if run_code_sandboxed /bin/kill -0 "$sentinel_pid" >/dev/null 2>&1; then
  echo "OpenCode gate failed: sandbox can signal an unrelated process" >&2
  exit 1
fi
/bin/kill "$sentinel_pid" >/dev/null 2>&1 || true
wait "$sentinel_pid" >/dev/null 2>&1 || true
sentinel_pid=
if run_code_sandboxed /usr/bin/touch "$runtime_dir/code-escape" >/dev/null 2>&1; then
  echo "OpenCode gate failed: model-authored code can alter OpenCode runtime state" >&2
  exit 1
fi
if run_opencode_sandboxed "$sandbox_shell" -c \
  'git status --short; /bin/cat /etc/hosts' >/dev/null 2>&1; then
  echo "OpenCode gate failed: shell-command suffix escaped the code sandbox" >&2
  exit 1
fi
tool_timeout_started=$SECONDS
set +e
run_opencode_sandboxed "$TIMEOUT_BIN" --kill-after=1 1 \
  "$SANDBOX_EXEC" "${code_seatbelt_args[@]}" /bin/sleep 60 \
  >/dev/null 2>&1
tool_timeout_status=$?
set -e
tool_timeout_elapsed=$((SECONDS - tool_timeout_started))
if (( tool_timeout_status != 124 || tool_timeout_elapsed > 5 )); then
  echo "OpenCode gate failed: cross-sandbox tool watchdog did not terminate its child" >&2
  echo "status=$tool_timeout_status elapsed_seconds=$tool_timeout_elapsed" >&2
  exit 1
fi
verifier_timeout_started=$SECONDS
set +e
run_code_sandboxed_with_timeout 1 /bin/sleep 60 >/dev/null 2>&1
verifier_timeout_status=$?
set -e
verifier_timeout_elapsed=$((SECONDS - verifier_timeout_started))
if (( verifier_timeout_status != 124 || verifier_timeout_elapsed > 5 )); then
  echo "OpenCode gate failed: verifier watchdog did not terminate its child" >&2
  echo "status=$verifier_timeout_status elapsed_seconds=$verifier_timeout_elapsed" >&2
  exit 1
fi
if ! run_opencode_sandboxed /usr/bin/curl --fail --silent --show-error \
  --connect-timeout 5 --max-time 10 "$BASE_URL/models" >/dev/null; then
  echo "OpenCode gate failed: sandbox cannot reach the loopback hf2q server" >&2
  exit 1
fi
if ! run_opencode_sandboxed "$sandbox_shell" -c \
  "cargo test --no-run --quiet --manifest-path $project_dir/Cargo.toml" \
  >"$sandbox_test_log" 2>&1; then
  echo "OpenCode gate failed: pinned offline Rust toolchain cannot compile in the sandbox" >&2
  sed -n '1,160p' "$sandbox_test_log" >&2
  exit 1
fi
if ! run_verifier_probe >>"$sandbox_test_log" 2>&1; then
  echo "OpenCode gate failed: verifier sandbox is not fail-closed" >&2
  sed -n '1,200p' "$sandbox_test_log" >&2
  exit 1
fi
if run_code_cargo test --quiet --manifest-path "$project_dir/Cargo.toml" \
  >"$initial_test_log" 2>&1; then
  echo "OpenCode gate fixture is invalid: initial tests unexpectedly passed" >&2
  exit 1
fi

assert_event_stream() {
  local events=$1
  if [[ ! -s "$events" ]] || ! jq -e -s 'all(.[]; type == "object")' "$events" >/dev/null; then
    echo "OpenCode gate failed: malformed or empty JSON event stream: $events" >&2
    exit 1
  fi
  if jq -e -s 'any(.[]; .type == "error")' "$events" >/dev/null; then
    echo "OpenCode gate failed: session emitted an error event" >&2
    jq -s '[.[] | select(.type == "error")]' "$events" >&2
    exit 1
  fi
  if ! jq -e -s 'any(.[]; .type == "step_finish")' "$events" >/dev/null; then
    echo "OpenCode gate failed: session emitted no completed model step" >&2
    exit 1
  fi
}

assert_completed_tool() {
  local events=$1
  local expression=$2
  local description=$3
  if ! jq -e -s --argjson names "$expression" '
    any(.[].part?;
      .type == "tool"
      and .state.status == "completed"
      and (.tool as $tool | ($names | index($tool)) != null)
    )
  ' "$events" >/dev/null; then
    echo "OpenCode gate failed: no completed $description tool call" >&2
    jq -s '[.[] | select(.type == "tool_use") | {tool: .part.tool, state: .part.state.status}]' \
      "$events" >&2
    exit 1
  fi
}

first_prompt='Inspect this Rust crate. Use the read tool to understand the failing test, fix the production bug with an edit tool, and use the bash tool to run cargo test. Do not stop until the tests pass.'
if ! run_opencode run --format json --pure --agent build \
  --title hf2q-deepseek-acceptance --dir "$project_dir" \
  --model "$selected_model" "$first_prompt" \
  >"$events_first" 2>"$stderr_first"; then
  echo "OpenCode gate failed: first coding turn exited unsuccessfully" >&2
  sed -n '1,160p' "$stderr_first" >&2
  sed -n '1,80p' "$events_first" >&2
  exit 1
fi
assert_event_stream "$events_first"
assert_completed_tool "$events_first" '["read"]' "read"
assert_completed_tool "$events_first" '["edit", "write", "apply_patch"]' "source mutation"
assert_completed_tool "$events_first" '["bash"]' "bash"

session_id=$(jq -r -s '
  [.[].sessionID | select(type == "string" and length > 0)]
  | unique
  | if length == 1 then .[0] else empty end
' "$events_first")
if [[ -z "$session_id" ]]; then
  echo "OpenCode gate failed: first event stream did not contain one session ID" >&2
  exit 1
fi
if cmp -s "$before_source" "$project_dir/src/lib.rs"; then
  echo "OpenCode gate failed: first turn did not modify src/lib.rs" >&2
  exit 1
fi
if ! run_verifier_probe >"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: post-model verifier sandbox is not fail-closed" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi
if ! run_oracle >"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: immutable behavior oracle rejected the first-turn fix" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi
if ! run_code_cargo test --quiet --manifest-path "$project_dir/Cargo.toml" \
  >>"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: first-turn project tests still fail" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi

second_prompt='Continue this same coding session. Add a regression test named in_range_index_is_unchanged for bounded_index(1, 3), then use bash to run cargo test again. Do not stop until it passes.'
if ! run_opencode run --format json --pure --agent build \
  --dir "$project_dir" --session "$session_id" --model "$selected_model" \
  "$second_prompt" >"$events_second" 2>"$stderr_second"; then
  echo "OpenCode gate failed: continued coding turn exited unsuccessfully" >&2
  sed -n '1,160p' "$stderr_second" >&2
  sed -n '1,80p' "$events_second" >&2
  exit 1
fi
assert_event_stream "$events_second"
assert_completed_tool "$events_second" '["edit", "write", "apply_patch"]' "continued source mutation"
assert_completed_tool "$events_second" '["bash"]' "continued bash"
if ! jq -e -s --arg session "$session_id" \
  'all(.[]; .sessionID == $session)' "$events_second" >/dev/null; then
  echo "OpenCode gate failed: continuation did not preserve session $session_id" >&2
  exit 1
fi
if ! run_oracle >"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: immutable behavior oracle rejected the continued-turn code" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi
if ! test_list_output=$(run_code_cargo test --manifest-path "$project_dir/Cargo.toml" -- --list 2>&1); then
  echo "OpenCode gate failed: could not enumerate continued-turn Rust tests" >&2
  printf '%s\n' "$test_list_output" >&2
  exit 1
fi
if ! rg -q '^tests::in_range_index_is_unchanged: test$' <<<"$test_list_output"; then
  echo "OpenCode gate failed: the named regression is not a discoverable Rust test" >&2
  printf '%s\n' "$test_list_output" >&2
  exit 1
fi
if ! run_code_cargo test --manifest-path "$project_dir/Cargo.toml" \
  tests::in_range_index_is_unchanged -- --exact >"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: the named regression test failed" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi
if ! rg -q 'test result: ok\. 1 passed; 0 failed' "$final_test_log"; then
  echo "OpenCode gate failed: the named regression did not execute exactly once" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi
if ! run_code_cargo test --quiet --manifest-path "$project_dir/Cargo.toml" \
  >>"$final_test_log" 2>&1; then
  echo "OpenCode gate failed: continued-turn project tests fail" >&2
  sed -n '1,160p' "$final_test_log" >&2
  exit 1
fi

first_tools=$(jq -s '[.[] | select(.type == "tool_use" and .part.state.status == "completed")] | length' "$events_first")
second_tools=$(jq -s '[.[] | select(.type == "tool_use" and .part.state.status == "completed")] | length' "$events_second")
opencode_version=$("$OPENCODE" --version)
session_deleted=false
if [[ "$KEEP_WORK_DIR" == "0" ]]; then
  if ! run_opencode session delete "$session_id" --pure >/dev/null; then
    echo "OpenCode gate failed: could not delete isolated session $session_id" >&2
    exit 1
  fi
  if ! session_list_output=$(run_opencode session list --pure --format json); then
    echo "OpenCode gate failed: could not list isolated sessions after deletion" >&2
    exit 1
  fi
  # OpenCode 1.18.14 intentionally prints no bytes for an empty session store.
  # Normalize only that successful case; command failures and malformed JSON
  # remain fatal instead of being mistaken for absence.
  if [[ -z "$session_list_output" ]]; then
    session_list_output='[]'
  fi
  if ! jq -e '
    type == "array"
    and all(.[]; type == "object" and (.id | type == "string"))
  ' <<<"$session_list_output" >/dev/null; then
    echo "OpenCode gate failed: malformed session-list JSON after deletion" >&2
    printf '%s\n' "$session_list_output" >&2
    exit 1
  fi
  if jq -e --arg session "$session_id" \
    'any(.[]; .id == $session)' <<<"$session_list_output" >/dev/null; then
    echo "OpenCode gate failed: isolated session $session_id remains after deletion" >&2
    exit 1
  fi
  session_deleted=true
fi
run_succeeded=1
jq -n --arg status pass --arg opencode_version "$opencode_version" \
  --arg session_id "$session_id" --arg model "$selected_model" \
  --argjson session_deleted "$session_deleted" \
  --argjson first_turn_completed_tools "$first_tools" \
  --argjson second_turn_completed_tools "$second_tools" '{
    status: $status,
    opencode_version: $opencode_version,
    model: $model,
    session_id: $session_id,
    first_turn_completed_tools: $first_turn_completed_tools,
    second_turn_completed_tools: $second_turn_completed_tools,
    source_changed: true,
    immutable_behavior_oracle_passed: true,
    named_regression_executed: true,
    tests_passed_after_each_turn: true,
    continued_same_session: true,
    isolated_session_deleted: $session_deleted
  }'
