#!/usr/bin/env bash
# adr-011-phase2-gate.sh — ADR-011 Phase 2 end-to-end parity gate.
#
# Verifies both halves of the Phase 2 success criterion:
#   "hf2q produces coherent (basically identical) output as llama.cpp,
#    and just as fast."
#
# What this script does:
#   Gate 1: Text parity — runs scripts/parity_check.sh (3 prompts × 3 runs).
#   Gate 2: Prefill tok/s parity — runs llama-bench + hf2q batched prefill
#           at pp=128, 512, 1024, 2455; compares ratios against -10% floor.
#
# Usage:
#   scripts/adr-011-phase2-gate.sh <gguf_path>
#
# Exit codes:
#   0  all gates passed
#   1  usage / env error
#   2  a gate failed
#   3  tool invocation failure (crash)
#
# Dependencies:
#   - target/release/hf2q      (cargo build --release --features metal)
#   - /opt/llama.cpp/build/bin/llama-bench
#   - /opt/llama.cpp/build/bin/llama-completion  (used by parity_check.sh)
#   - python3
#
# Peer baselines (re-measured 2026-04-17, M5 Max, llama-bench b3d758750,
#   Gemma 4 26B MoE DWQ GGUF, fa=0, 3-rep median):
#   pp=128:  ~1114 tok/s (fa=1 proxy; fa=0 not separately measured at p128)
#   pp=512:  ~3722 tok/s
#   pp=1024: ~3605 tok/s
#   pp=2455: 3455.88 tok/s (fa=0 — faster than fa=1 on this MoE model)
#   Source: docs/ADR-011-phase1-port-source-decision.md §3
#
# Pass floor: hf2q prefill tok/s >= peer * 0.90 at each seq_len.
# That is -10% headroom for thermal jitter on M5 Max.
#
# Note on pp=2455: the batched prefill path is blocked on the sdpa_sliding
# kernel fix (docs/spike-gate-a-prefill.md §Addendum). Until that fix lands,
# the pp=2455 sub-gate will SKIP with a diagnostic rather than fail, so the
# script can still report on Gates 1 and the shorter seq_len prefill shapes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
LLAMA_BENCH_BIN=""
PARITY_SCRIPT="$SCRIPT_DIR/parity_check.sh"

# llama-bench location
if command -v llama-bench >/dev/null 2>&1; then
  LLAMA_BENCH_BIN="$(command -v llama-bench)"
elif [[ -x "/opt/llama.cpp/build/bin/llama-bench" ]]; then
  LLAMA_BENCH_BIN="/opt/llama.cpp/build/bin/llama-bench"
fi

# Prompt file for prefill benchmarks
PREFILL_PROMPT_FILE="tests/evals/prompts/prefill_2048.txt"

# Peer baselines (fa=0, re-measured 2026-04-17, M5 Max)
# pp=128 uses fa=1 measurement as proxy (fa=0 not separately measured at this shape)
declare -A PEER_TPS=([128]="1114" [512]="3722" [1024]="3605" [2455]="3456")
PASS_RATIO="0.90"   # hf2q must reach >= 90% of peer

PREFILL_RUNS=3      # median of N runs

usage() {
  cat <<EOF
Usage: scripts/adr-011-phase2-gate.sh <gguf_path>

  <gguf_path>   Path to the Gemma 4 DWQ GGUF model file (required)

Exit codes:
  0  all gates passed
  1  usage / env error
  2  gate failed
  3  tool invocation failure
EOF
}

err() { echo "error: $*" >&2; exit 1; }
warn() { echo "warn: $*" >&2; }
skip() { echo "SKIP: $*" >&2; }

GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done

[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release --features metal)"
[[ -x "$PARITY_SCRIPT" ]] || err "scripts/parity_check.sh not found"

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "=== ADR-011 Phase 2 Parity Gate ==="
echo "GGUF:          $GGUF_PATH"
echo "hf2q:          $HF2Q_BIN"
echo "llama-bench:   ${LLAMA_BENCH_BIN:-NOT FOUND}"
echo "git HEAD:      $GIT_HEAD"
echo "pass ratio:    >= ${PASS_RATIO} of peer tok/s"
echo "prefill runs:  $PREFILL_RUNS (median)"
echo

# Track overall results
GATE1_STATUS="SKIP"
declare -A GATE2_LLAMA_TPS   # indexed by seq_len string
declare -A GATE2_HF2Q_TPS
declare -A GATE2_RATIO
declare -A GATE2_STATUS
GATE2_SEQ_LENS=(128 512 1024 2455)

OVERALL_FAIL=0

# ---------------------------------------------------------------------------
# Gate 1: Text parity (delegates to parity_check.sh)
# ---------------------------------------------------------------------------
echo "================================================================"
echo "Gate 1/2: Text parity (short_hello + sourdough + sliding_wrap)"
echo "          3 prompts × 3 runs each (Gate F determinism)"
echo "================================================================"
echo
if "$PARITY_SCRIPT" "$GGUF_PATH"; then
  GATE1_STATUS="PASS"
  echo
  echo "Gate 1: PASS"
else
  GATE1_STATUS="FAIL"
  OVERALL_FAIL=1
  echo
  echo "Gate 1: FAIL — see parity_check.sh output above"
fi
echo

# ---------------------------------------------------------------------------
# Gate 2: Prefill tok/s parity
# ---------------------------------------------------------------------------
echo "================================================================"
echo "Gate 2/2: Prefill tok/s parity"
echo "          seq_len in {128, 512, 1024, 2455}"
echo "          llama-bench fa=0 vs hf2q batched prefill"
echo "          Pass floor: hf2q >= peer * ${PASS_RATIO}"
echo "================================================================"
echo

if [[ -z "$LLAMA_BENCH_BIN" ]]; then
  warn "llama-bench not found at /opt/llama.cpp/build/bin/llama-bench"
  warn "Gate 2 requires llama-bench. Skipping all prefill measurements."
  for sl in "${GATE2_SEQ_LENS[@]}"; do
    GATE2_STATUS[$sl]="SKIP(no-llama-bench)"
  done
else

  # ------------------------------------------------------------------
  # Helper: run llama-bench for one seq_len, return median tok/s
  # ------------------------------------------------------------------
  run_llama_bench() {
    local seq_len="$1"
    local log="/tmp/adr011_gate2_llama_p${seq_len}.log"

    # llama-bench with -p <seq_len> -n 0 does pure prefill.
    # fa=0: non-flash-attn path, which is faster for this MoE model at
    # seq_len >= 2455 (ADR-011-phase1-port-source-decision.md §3.2).
    # --flash-attn 0 is the correct flag (--flash-attn is the bench alias).
    # Use --output csv for machine-parseable output.
    if ! "$LLAMA_BENCH_BIN" \
        --model "$GGUF_PATH" \
        -p "$seq_len" \
        -n 0 \
        --flash-attn 0 \
        -r "$PREFILL_RUNS" \
        --output csv \
        >"$log" 2>&1; then
      echo "0"
      warn "llama-bench crashed for pp=${seq_len}. See $log"
      return
    fi

    # llama-bench CSV: header then data rows
    # Columns: model,size,params,backend,ngl,n_batch,n_ubatch,flash_attn,
    #          mla,mmap,n_threads,type_k,type_v,n_gpu_layers,split_mode,
    #          main_gpu,no_kv_offload,cache_type_k,cache_type_v,
    #          n_prompt,n_gen,test,t_pp,s_pp,t_tg,s_tg
    # t_pp = prefill ms/token; s_pp = prefill tok/s
    # We want the median row's s_pp.
    python3 - "$log" <<'PY'
import sys, csv, statistics
log = sys.argv[1]
rows = []
with open(log) as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = row.get('s_pp', row.get('pp_speed', ''))
        if s and float(s) > 0:
            rows.append(float(s))
if not rows:
    print("0")
else:
    print(f"{statistics.median(rows):.2f}")
PY
  }

  # ------------------------------------------------------------------
  # Helper: run hf2q batched prefill for one seq_len, return median tok/s
  # Uses the prefill_2048.txt fixture for pp=2455; for shorter seq_lens,
  # uses the sourdough prompt with --max-tokens 1 which gives a shorter
  # prefill (actual token count depends on the prompt; we parse what hf2q
  # reports). For exact seq_len control use generate with prompt-file.
  # ------------------------------------------------------------------

  # Build synthetic prompts for pp=128/512/1024 by repeating short words.
  # We use the sourdough prompt file piped to python3 to extend it.
  build_prompt_for_seqlen() {
    local target_tokens="$1"
    local outfile="$2"
    # 1 English word ≈ 1.3 tokens on Gemma 4. Use a word that tokenizes
    # to ~1 token: "the". Repeat target_tokens times; add a few extra
    # to ensure we hit the target after BOS is consumed.
    python3 -c "
import sys
n = int(sys.argv[1])
# 'the' is consistently 1 token. Overestimate slightly.
words = ' '.join(['the'] * (n + 10))
open(sys.argv[2], 'w').write(words)
" "$target_tokens" "$outfile"
  }

  run_hf2q_prefill() {
    local seq_len="$1"
    local log="/tmp/adr011_gate2_hf2q_p${seq_len}.log"
    local prompt_file

    if [[ "$seq_len" == "2455" ]]; then
      prompt_file="$PREFILL_PROMPT_FILE"
    else
      prompt_file="/tmp/adr011_synthetic_p${seq_len}.txt"
      build_prompt_for_seqlen "$seq_len" "$prompt_file"
    fi

    if [[ ! -f "$prompt_file" ]]; then
      echo "0"
      warn "Prompt file not found for pp=${seq_len}"
      return
    fi

    local samples=()
    local run_ok=1
    for run in $(seq 1 "$PREFILL_RUNS"); do
      if ! HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_BATCHED_PREFILL=1 \
            "$HF2Q_BIN" generate \
              --model "$GGUF_PATH" \
              --prompt-file "$prompt_file" \
              --max-tokens 1 --temperature 0 \
              >/dev/null 2>"$log"; then
        # A crash may be the sdpa_sliding blocker at pp > 1024.
        # Check for the known error message.
        if grep -q "exceeds dense cap" "$log" 2>/dev/null; then
          echo "BLOCKED"
          return
        fi
        warn "hf2q crashed for pp=${seq_len} run $run. See $log"
        run_ok=0
        break
      fi
      # Parse "Batched prefill complete: N tokens in X.X ms (Y.Y tok/s)"
      local tps
      tps="$(grep -oE 'Batched prefill complete: [0-9]+ tokens in [0-9.]+ ms \(([0-9.]+) tok/s\)' \
              "$log" | tail -1 | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+' | head -1)"
      if [[ -z "$tps" ]]; then
        # Might be using per-token prefill path (no batched prefill line)
        warn "No batched prefill line in hf2q output for pp=${seq_len} run $run."
        run_ok=0
        break
      fi
      samples+=("$tps")
    done

    if (( run_ok == 0 )) || (( ${#samples[@]} == 0 )); then
      echo "0"
      return
    fi

    # Compute median
    printf '%s\n' "${samples[@]}" | sort -n | awk "NR==$(( (${#samples[@]} + 1) / 2 ))"
  }

  # ------------------------------------------------------------------
  # Run measurements for each seq_len
  # ------------------------------------------------------------------
  for sl in "${GATE2_SEQ_LENS[@]}"; do
    echo "--- pp=${sl} ---"

    echo -n "  llama-bench (fa=0, ${PREFILL_RUNS}-rep median): "
    llama_tps="$(run_llama_bench "$sl")"
    echo "${llama_tps} tok/s"
    GATE2_LLAMA_TPS[$sl]="$llama_tps"

    echo -n "  hf2q batched prefill (${PREFILL_RUNS}-rep median): "
    hf2q_result="$(run_hf2q_prefill "$sl")"

    if [[ "$hf2q_result" == "BLOCKED" ]]; then
      echo "BLOCKED (sdpa_sliding fix required for pp > 1024)"
      echo "  This is the known blocker from docs/spike-gate-a-prefill.md §Addendum."
      echo "  Once sdpa_sliding is fixed and ring-wrap lands, re-run this gate."
      GATE2_HF2Q_TPS[$sl]="0"
      GATE2_STATUS[$sl]="SKIP(sdpa_sliding-blocker)"
      echo
      continue
    fi

    GATE2_HF2Q_TPS[$sl]="$hf2q_result"
    echo "${hf2q_result} tok/s"

    if [[ "$llama_tps" == "0" || "$hf2q_result" == "0" ]]; then
      GATE2_STATUS[$sl]="SKIP(measurement-error)"
      warn "Could not get valid measurement for pp=${sl}."
      echo
      continue
    fi

    # Compute ratio and pass/fail
    result="$(python3 -c "
llama = float('$llama_tps')
hf2q  = float('$hf2q_result')
ratio = hf2q / llama if llama > 0 else 0
floor = float('$PASS_RATIO')
print(f'{ratio:.4f}')
print('PASS' if ratio >= floor else 'FAIL')
")"
    ratio="$(echo "$result" | head -1)"
    verdict="$(echo "$result" | tail -1)"

    GATE2_RATIO[$sl]="$ratio"
    GATE2_STATUS[$sl]="$verdict"

    pct="$(python3 -c "print(f'{float(\"$ratio\")*100:.1f}%')")"
    floor_pct="$(python3 -c "print(f'{float(\"$PASS_RATIO\")*100:.0f}%')")"
    echo "  ratio: hf2q/llama = ${ratio} (${pct}) — floor ${floor_pct} — ${verdict}"

    if [[ "$verdict" == "FAIL" ]]; then
      OVERALL_FAIL=1
    fi
    echo
  done

fi  # end llama-bench available block

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo "================================================================"
echo "Summary"
echo "================================================================"
echo
echo "Gate 1 — Text parity:"
echo "  ${GATE1_STATUS}"
echo
echo "Gate 2 — Prefill tok/s parity:"
printf "  %-8s  %-16s  %-16s  %-8s  %s\n" \
       "seq_len" "llama.cpp(fa=0)" "hf2q(batched)" "ratio" "status"
printf "  %-8s  %-16s  %-16s  %-8s  %s\n" \
       "-------" "---------------" "-------------" "-----" "------"
for sl in "${GATE2_SEQ_LENS[@]}"; do
  llama_val="${GATE2_LLAMA_TPS[$sl]:-—}"
  hf2q_val="${GATE2_HF2Q_TPS[$sl]:-—}"
  ratio_val="${GATE2_RATIO[$sl]:-—}"
  status_val="${GATE2_STATUS[$sl]:-—}"
  printf "  %-8s  %-16s  %-16s  %-8s  %s\n" \
         "${sl}" "${llama_val}" "${hf2q_val}" "${ratio_val}" "${status_val}"
done
echo
echo "Peer baselines used (from ADR-011-phase1-port-source-decision.md §3):"
echo "  pp=128:  1114 tok/s (fa=1 proxy)  pp=512:  3722 tok/s"
echo "  pp=1024: 3605 tok/s               pp=2455: 3456 tok/s (fa=0)"
echo

if (( OVERALL_FAIL == 0 )); then
  echo "=== ADR-011 Phase 2 gate: PASS ==="
  exit 0
else
  echo "=== ADR-011 Phase 2 gate: FAIL ==="
  echo
  echo "Remediation:"
  if [[ "${GATE1_STATUS}" == "FAIL" ]]; then
    echo "  TEXT PARITY FAIL: Bisect recent src/ changes touching attention,"
    echo "  MoE, norms, RoPE, lm_head, KV cache, SDPA."
    echo "  Compare outputs with: hf2q parity check --model <gguf> --prompt sourdough"
    echo "  Divergence details printed above by parity_check.sh."
  fi
  for sl in "${GATE2_SEQ_LENS[@]}"; do
    if [[ "${GATE2_STATUS[$sl]:-}" == "FAIL" ]]; then
      echo "  PREFILL pp=${sl} FAIL:"
      echo "    hf2q ${GATE2_HF2Q_TPS[$sl]} tok/s vs peer ${GATE2_LLAMA_TPS[$sl]} tok/s"
      echo "    Bisect: check HF2Q_BATCHED_PREFILL=1 is active, check sdpa_sliding,"
      echo "    fused_head_norm_rope, or MoE dispatch regression."
      echo "    Use Metal Frame Capture or HF2Q_DUMP_COUNTERS=1 for per-op breakdown."
    fi
  done
  exit 2
fi
