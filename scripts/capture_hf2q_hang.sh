#!/usr/bin/env bash
# capture_hf2q_hang.sh — run this while hf2q appears frozen.
#
# Records, without sudo and without stopping anything:
#   - every thread stack of every running hf2q process (sample, 10 s)
#   - thermal and power state, memory pressure, and the Apple GPU state
#   - the server's own health, runtime, and metrics endpoints
# and writes them to one folder in $HOME. Attach that folder to the issue.
#
# Usage: scripts/capture_hf2q_hang.sh [port ...]   (default port 8081)
set -uo pipefail

out="$HOME/hf2q-hang-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
ports=("$@")
[[ ${#ports[@]} -gt 0 ]] || ports=(8081)

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  sw_vers
  sysctl -n machdep.cpu.brand_string hw.memsize
  uptime
} >"$out/system.txt" 2>&1

pmset -g therm >"$out/thermal.txt" 2>&1
pmset -g >"$out/power.txt" 2>&1
memory_pressure -Q >"$out/memory-pressure.txt" 2>&1
vm_stat >"$out/vm_stat.txt" 2>&1
sysctl vm.swapusage >"$out/swap.txt" 2>&1
ioreg -r -c AGXAccelerator -d 1 >"$out/gpu-ioreg.txt" 2>&1
ps -axo pid,ppid,%cpu,%mem,rss,etime,state,command | grep -E '[h]f2q' >"$out/processes.txt" 2>&1

pids=$(pgrep -x hf2q || true)
if [[ -z "$pids" ]]; then
  echo "no hf2q process is running" | tee "$out/NO-HF2Q-PROCESS.txt"
fi
for pid in $pids; do
  ps -M -p "$pid" >"$out/threads-$pid.txt" 2>&1
  sample "$pid" 10 -mayDie -file "$out/sample-$pid.txt" >/dev/null 2>&1 &
done

for port in "${ports[@]}"; do
  for path in readyz health hf2q/v1/runtime metrics; do
    curl -s -m 5 "http://127.0.0.1:$port/$path" \
      >"$out/endpoint-$port-${path//\//_}.txt" 2>&1 || true
  done
done
wait

# A second GPU snapshot shows whether the GPU is still busy or idle.
ioreg -r -c AGXAccelerator -d 1 >"$out/gpu-ioreg-after.txt" 2>&1

echo "hang evidence written to $out"
echo "the key file is sample-<pid>.txt: look for waitUntilCompleted (GPU never finished) vs a lock or channel wait"
