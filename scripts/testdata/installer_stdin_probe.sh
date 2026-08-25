#!/bin/sh
set -eu

probe_log=${HF2Q_STDIN_PROBE_LOG:?HF2Q_STDIN_PROBE_LOG is required}
probe_version=${HF2Q_STDIN_PROBE_VERSION:?HF2Q_STDIN_PROBE_VERSION is required}

if IFS= read -r captured; then
  printf 'captured:%s:%s\n' "${1:-none}" "$captured" >>"$probe_log"
  exit 90
fi
printf 'eof:%s\n' "${1:-none}" >>"$probe_log"

case "${1:-}" in
  --version)
    printf 'hf2q %s\n' "$probe_version"
    ;;
  __standalone-install)
    shift
    install_dir=
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --install-dir)
          install_dir=${2:?missing --install-dir value}
          shift 2
          ;;
        *)
          shift
          ;;
      esac
    done
    [ -n "$install_dir" ]
    mkdir -p "$install_dir"
    cp "$0" "$install_dir/hf2q"
    chmod 0555 "$install_dir/hf2q"
    ;;
  completions)
    ;;
  *)
    exit 91
    ;;
esac
