#!/usr/bin/env bash
# release_gate_preflight.sh — fail fast when the host cannot run the
# cross-family qualification gate cleanly.
#
# The gate's memory guard requires zero swap movement while the DeepSeek
# decode-cohort benchmark holds ~115 GiB resident, and its runtime guard
# requires no existing hf2q/llama processes. Both conditions historically
# surfaced 20-40 minutes into a run (v0.1.21: stray servers failed the run
# at minute 1; swapout growth during model load killed it at minute 20,
# after the host had been in active use). This preflight checks the same
# conditions up front so a dirty host fails in seconds with an actionable
# message instead of after a partial run.
#
# Contract: run the full gate on a freshly rebooted, otherwise-idle machine.
set -euo pipefail

fail() {
  echo "preflight: $*" >&2
  exit 1
}

# 1. No existing model runtimes (the gate's own first check, moved here).
for name in hf2q llama-server llama-cli; do
  if /usr/bin/pgrep -x "$name" >/dev/null 2>&1; then
    fail "existing $name runtime present; stop it before the gate"
  fi
done

# 2. Zero swap in use. The decode-cohort memory guard requires flat
#    swapouts from the setup baseline; any swap debt left over from prior
#    work makes mid-run growth likely. A reboot is the deterministic reset.
swap_used=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
if awk -v s="$swap_used" 'BEGIN { exit !(s + 0 > 0.5) }'; then
  fail "swap in use (${swap_used}M); reboot before running the gate"
fi

# 3. Enough free memory for the ~115 GiB DeepSeek decode-cohort benchmark
#    plus system overhead on a 128 GiB host, with pressure nominal.
free_line=$(memory_pressure -Q 2>/dev/null | tail -1)
free_pct=${free_line##*: }
free_pct=${free_pct%\%}
[[ "$free_pct" =~ ^[0-9]+$ ]] || fail "cannot read system-wide free memory percentage"
(( free_pct >= 90 )) || fail "free memory ${free_pct}% (< 90%); close applications or reboot"

# 4. Thermal nominal, via the gate's own sourced guard and probe (the
#    guard performs no work on import; the probe compiles once).
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
thermal_read_state || fail "cannot read thermal state"
case "$THERMAL_STATE" in
  nominal | fair) ;;
  *) fail "thermal state '$THERMAL_STATE'; let the machine cool before the gate" ;;
esac

echo "preflight: host clean (no runtimes, ${swap_used}M swap, ${free_pct}% free, thermal $THERMAL_STATE)"
