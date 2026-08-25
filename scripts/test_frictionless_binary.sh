#!/usr/bin/env bash
# Exact-binary, no-network smoke for the frictionless repository UX.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <hf2q-binary> <temporary-root-prefix>" >&2
  exit 2
fi

binary=$1
root_prefix=$2
test -x "$binary"
binary_dir=$(cd "$(dirname "$binary")" && pwd -P)
binary="$binary_dir/$(basename "$binary")"
test_root=$(mktemp -d "${root_prefix}.XXXXXX")
trap 'rm -rf -- "$test_root"' EXIT
mkdir -p "$test_root/home" "$test_root/data" "$test_root/cache" \
  "$test_root/state" "$test_root/server-state"
printf 'this = [is not toml\n' > "$test_root/state/config.toml"

run_isolated() {
  HOME="$test_root/home" \
  XDG_DATA_HOME="$test_root/data" \
  XDG_CACHE_HOME="$test_root/cache" \
  HF_HOME="$test_root/cache/huggingface" \
  HF_HUB_OFFLINE=1 \
    "$binary" "$@"
}

run_isolated --state-root "$test_root/state" serve list \
  > "$test_root/serve.list"
run_isolated --state-root "$test_root/state" chat list \
  > "$test_root/chat.list"
cmp "$test_root/serve.list" "$test_root/chat.list"
test ! -e "$test_root/data/hf2q/models"

# A redirected bare invocation remains a clean Clap diagnostic: no graphics
# or wordmark leak into either output stream.
set +e
run_isolated > "$test_root/bare.stdout" 2> "$test_root/bare.stderr"
bare_redirected_exit=$?
set -e
test "$bare_redirected_exit" -eq 2
test ! -s "$test_root/bare.stdout"
grep -Fq 'Usage: hf2q' "$test_root/bare.stderr"
! grep -Fq 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/bare.stderr"

# Exercise the exact installed artifact through a real macOS PTY. The global
# rabbit banner is scrollback-safe and must never enter the alternate screen;
# every structured/noninteractive suppression branch stays banner-free.
pty_run() {
  local output=$1
  shift
  /usr/bin/script -q "$output" /usr/bin/env \
    -u CI \
    -u CMUX_SURFACE_ID \
    -u KITTY_WINDOW_ID \
    -u GHOSTTY_RESOURCES_DIR \
    -u LC_TERMINAL \
    -u TMUX \
    HOME="$test_root/home" \
    XDG_DATA_HOME="$test_root/data" \
    XDG_CACHE_HOME="$test_root/cache" \
    HF_HOME="$test_root/cache/huggingface" \
    HF_HUB_OFFLINE=1 \
    TERM=xterm-256color \
    TERM_PROGRAM=Apple_Terminal \
    COLUMNS=100 \
    "$@" >/dev/null
}

pty_run "$test_root/banner.pty" \
  "$binary" --terminal-graphics ansi --state-root "$test_root/state" serve list
test "$(grep -aFc 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/banner.pty")" -eq 1
grep -aFq $'\033[38;2;' "$test_root/banner.pty"
! grep -aFq $'\033[?1049h' "$test_root/banner.pty"
! grep -aFq $'\033[?1049l' "$test_root/banner.pty"

# Bare `hf2q` is the interactive landing surface, not an explicit help
# protocol request. It must brand the command overview before Clap exits.
# Pin each operator-supported terminal identity instead of inheriting the
# outer test runner's multiplexer hints.
set +e
pty_run "$test_root/bare.pty" "$binary"
bare_exit=$?
set -e
test "$bare_exit" -eq 2
test "$(grep -aFc 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/bare.pty")" -eq 1
grep -aFq $'\033[38;2;' "$test_root/bare.pty"
grep -aFq 'Usage: hf2q' "$test_root/bare.pty"
! grep -aFq $'\033[?1049h' "$test_root/bare.pty"
! grep -aFq $'\033[?1049l' "$test_root/bare.pty"

set +e
pty_run "$test_root/bare-alacritty.pty" /usr/bin/env TERM_PROGRAM=Alacritty \
  "$binary"
bare_alacritty_exit=$?
set -e
test "$bare_alacritty_exit" -eq 2
test "$(grep -aFc 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/bare-alacritty.pty")" -eq 1
grep -aFq $'\033[38;2;' "$test_root/bare-alacritty.pty"
! grep -aFq $'\033[?1049h' "$test_root/bare-alacritty.pty"
! grep -aFq $'\033[?1049l' "$test_root/bare-alacritty.pty"

set +e
pty_run "$test_root/bare-cmux.pty" /usr/bin/env CMUX_SURFACE_ID=hf2q-test \
  "$binary"
bare_cmux_exit=$?
set -e
test "$bare_cmux_exit" -eq 2
test "$(grep -aFc 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/bare-cmux.pty")" -eq 1
/usr/bin/perl -0777 -ne \
  'while (/\e_G[^;]*;([A-Za-z0-9+\/=]+)\e\\/g) { print $1 }' \
  "$test_root/bare-cmux.pty" | base64 -D > "$test_root/bare-cmux-rabbit.png"
test "$(shasum -a 256 "$test_root/bare-cmux-rabbit.png" | awk '{print $1}')" = \
  fe8cc15cc2693c38ab8510724566a22455b2d33bc7332229deedb88bc5e28aad
! grep -aFq $'\033[?1049h' "$test_root/bare-cmux.pty"
! grep -aFq $'\033[?1049l' "$test_root/bare-cmux.pty"

set +e
pty_run "$test_root/bare-ci.pty" /usr/bin/env CI=1 "$binary"
bare_ci_exit=$?
set -e
test "$bare_ci_exit" -eq 2
grep -aFq 'Usage: hf2q' "$test_root/bare-ci.pty"
! grep -aFq 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/bare-ci.pty"

pty_run "$test_root/kitty.pty" \
  "$binary" --terminal-graphics kitty --state-root "$test_root/state" serve list
/usr/bin/perl -0777 -ne \
  'while (/\e_G[^;]*;([A-Za-z0-9+\/=]+)\e\\/g) { print $1 }' \
  "$test_root/kitty.pty" | base64 -D > "$test_root/rabbit.png"
test "$(shasum -a 256 "$test_root/rabbit.png" | awk '{print $1}')" = \
  fe8cc15cc2693c38ab8510724566a22455b2d33bc7332229deedb88bc5e28aad
! grep -aFq $'\033[?1049h' "$test_root/kitty.pty"
! grep -aFq $'\033[?1049l' "$test_root/kitty.pty"

for suppressed in off json ci dumb help version completions; do
  case "$suppressed" in
    off)
      pty_run "$test_root/$suppressed.pty" \
        "$binary" --terminal-graphics off --state-root "$test_root/state" serve list
      ;;
    json)
      pty_run "$test_root/$suppressed.pty" \
        "$binary" --log-format json --state-root "$test_root/state" serve list
      ;;
    ci)
      pty_run "$test_root/$suppressed.pty" /usr/bin/env CI=1 \
        "$binary" --state-root "$test_root/state" serve list
      ;;
    dumb)
      pty_run "$test_root/$suppressed.pty" /usr/bin/env TERM=dumb \
        "$binary" --state-root "$test_root/state" serve list
      ;;
    help)
      pty_run "$test_root/$suppressed.pty" "$binary" --help
      ;;
    version)
      pty_run "$test_root/$suppressed.pty" "$binary" --version
      ;;
    completions)
      pty_run "$test_root/$suppressed.pty" \
        "$binary" completions --shell bash
      ;;
  esac
  ! grep -aFq 'Hugging Face → native GGUF → Apple Silicon' \
    "$test_root/$suppressed.pty"
done

HF2Q_PTY_BIN="$binary" HF2Q_PTY_STDOUT="$test_root/redirected.stdout" \
  pty_run "$test_root/stdout-redirected.pty" /bin/sh -c \
  'exec "$HF2Q_PTY_BIN" serve list >"$HF2Q_PTY_STDOUT"'
! grep -aFq 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/stdout-redirected.pty"

HF2Q_PTY_BIN="$binary" HF2Q_PTY_STDERR="$test_root/redirected.stderr" \
  pty_run "$test_root/stderr-redirected.pty" /bin/sh -c \
  'exec "$HF2Q_PTY_BIN" serve list 2>"$HF2Q_PTY_STDERR"'
! grep -aFq 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/stderr-redirected.pty"
! grep -aFq 'Hugging Face → native GGUF → Apple Silicon' \
  "$test_root/redirected.stderr"

# A real model-less direct server exercises the exact installed artifact's
# startup row, post-poll HTTP readiness transition, alternate-screen
# dashboard, SIGINT drain, and one-time terminal restoration. No model bytes,
# network, or Metal device are needed for this lifecycle proof.
HF2Q_PTY_BIN="$binary" \
HF2Q_PTY_CAPTURE="$test_root/dashboard.pty" \
HF2Q_PTY_PIDFILE="$test_root/dashboard.pid" \
HF2Q_PTY_STATE="$test_root/server-state" \
  pty_run "$test_root/dashboard.pty" /usr/bin/env \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    ALL_PROXY=http://127.0.0.1:9 NO_PROXY= /bin/sh -c '
    set -eu
    "$HF2Q_PTY_BIN" --terminal-graphics off --state-root "$HF2Q_PTY_STATE" \
      serve --host 127.0.0.1 --port 0 --operator-ui dashboard &
    server_pid=$!
    printf "%s\n" "$server_pid" >"$HF2Q_PTY_PIDFILE"
    trap '\''kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true'\'' EXIT HUP TERM
    ready=0
    attempt=0
    escape=$(printf "\033")
    while [ "$attempt" -lt 100 ]; do
      if grep -aFq "${escape}[?1049h" "$HF2Q_PTY_CAPTURE" &&
         grep -aFq "● ready" "$HF2Q_PTY_CAPTURE"; then
        ready=1
        break
      fi
      attempt=$((attempt + 1))
      /bin/sleep 0.05
    done
    test "$ready" -eq 1
    timeout_marker="$HF2Q_PTY_PIDFILE.timeout"
    (
      /bin/sleep 5
      if kill -0 "$server_pid" 2>/dev/null; then
        : >"$timeout_marker"
        kill -TERM "$server_pid" 2>/dev/null || true
      fi
    ) &
    watchdog_pid=$!
    kill -INT "$server_pid"
    server_status=0
    wait "$server_pid" || server_status=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    test ! -e "$timeout_marker"
    test "$server_status" -eq 0
    trap - EXIT HUP TERM
  '
test "$(grep -aFc $'\033[?1049h' "$test_root/dashboard.pty")" -eq 1
test "$(grep -aFc $'\033[?1049l' "$test_root/dashboard.pty")" -eq 1
grep -aFq 'no model preloaded' "$test_root/dashboard.pty"
! grep -aFq 'model prepared' "$test_root/dashboard.pty"
grep -aFq 'listener bound' "$test_root/dashboard.pty"
grep -aFq 'starting HTTP service' "$test_root/dashboard.pty"
grep -aFq '● ready' "$test_root/dashboard.pty"
dashboard_pid=$(cat "$test_root/dashboard.pid")
if kill -0 "$dashboard_pid" 2>/dev/null; then
  echo "dashboard server survived SIGINT lifecycle smoke: $dashboard_pid" >&2
  exit 1
fi

# The API-triggered normal shutdown path has a separate restoration contract.
HF2Q_PTY_BIN="$binary" \
HF2Q_PTY_CAPTURE="$test_root/dashboard-shutdown.pty" \
HF2Q_PTY_PIDFILE="$test_root/dashboard-shutdown.pid" \
HF2Q_PTY_STATE="$test_root/server-state" \
  pty_run "$test_root/dashboard-shutdown.pty" /usr/bin/env \
    HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
    ALL_PROXY=http://127.0.0.1:9 NO_PROXY= /bin/sh -c '
    set -eu
    "$HF2Q_PTY_BIN" --terminal-graphics off --state-root "$HF2Q_PTY_STATE" \
      serve --host 127.0.0.1 --port 0 --operator-ui dashboard &
    server_pid=$!
    printf "%s\n" "$server_pid" >"$HF2Q_PTY_PIDFILE"
    trap '\''kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true'\'' EXIT HUP TERM
    ready=0
    attempt=0
    escape=$(printf "\033")
    while [ "$attempt" -lt 100 ]; do
      if grep -aFq "$escape[?1049h" "$HF2Q_PTY_CAPTURE" &&
         grep -aFq "● ready" "$HF2Q_PTY_CAPTURE"; then
        ready=1
        break
      fi
      attempt=$((attempt + 1))
      /bin/sleep 0.05
    done
    test "$ready" -eq 1
    port=$(/usr/bin/perl -0777 -ne \
      '\''while (/http:\/\/127\.0\.0\.1:(\d+)/g) { $port = $1 } END { print $port // "" }'\'' \
      "$HF2Q_PTY_CAPTURE")
    test -n "$port"
    timeout_marker="$HF2Q_PTY_PIDFILE.timeout"
    (
      /bin/sleep 5
      if kill -0 "$server_pid" 2>/dev/null; then
        : >"$timeout_marker"
        kill -TERM "$server_pid" 2>/dev/null || true
      fi
    ) &
    watchdog_pid=$!
    /usr/bin/curl --noproxy "*" -fsS -X POST "http://127.0.0.1:$port/shutdown" \
      >/dev/null || true
    server_status=0
    wait "$server_pid" || server_status=$?
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
    test ! -e "$timeout_marker"
    test "$server_status" -eq 0
    trap - EXIT HUP TERM
  '
test "$(grep -aFc $'\033[?1049h' "$test_root/dashboard-shutdown.pty")" -eq 1
test "$(grep -aFc $'\033[?1049l' "$test_root/dashboard-shutdown.pty")" -eq 1
grep -aFq '● ready' "$test_root/dashboard-shutdown.pty"
shutdown_pid=$(cat "$test_root/dashboard-shutdown.pid")
if kill -0 "$shutdown_pid" 2>/dev/null; then
  echo "dashboard server survived API shutdown lifecycle smoke: $shutdown_pid" >&2
  exit 1
fi

# Owned chat preparation is scrollback-only, even when offline resolution
# fails. It must never take alternate-screen ownership from the chat client.
pty_run "$test_root/chat-startup-failure.pty" \
  "$binary" --terminal-graphics off --state-root "$test_root/server-state" \
    chat owner/model:Q4_K_M || true
! grep -aFq $'\033[?1049h' "$test_root/chat-startup-failure.pty"
! grep -aFq $'\033[?1049l' "$test_root/chat-startup-failure.pty"
grep -aEq 'Inspecting local (model stores|stores for owner/model)|chat-started hf2q serve exited' \
  "$test_root/chat-startup-failure.pty"

for command in serve chat convert; do
  run_isolated "$command" owner/model:Q4_K_M --help >/dev/null
  run_isolated "$command" owner/model --help >/dev/null
done
test ! -e "$test_root/data/hf2q/models"
