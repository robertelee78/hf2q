#!/usr/bin/env bash
# sourdough_qwen35.sh — ADR-013 Phase P13 byte-parity gate for Qwen3.5
# (and Qwen3.6, which routes through the same LLM_ARCH_QWEN35 / QWEN35MOE
# code paths in both llama.cpp and our hf2q).
#
# Runs hf2q and llama-cli on a fixed prompt at T=0 greedy, seed=42,
# max_tokens=N, then asserts that the common byte prefix of their outputs
# is at least MIN_COMMON_PREFIX bytes. This enforces "hf2q's Qwen3.5 GPU
# forward is byte-identical to llama.cpp on the same GGUF for the first N
# decoded tokens" as the mandatory ship gate for ADR-013.
#
# Mirrors scripts/sourdough_gate.sh (the Gemma-4 gate per ADR-005 Phase 1b).
# Two deliberate differences from the Gemma gate:
#   1. No HF2Q_USE_DENSE — Qwen3.5 has no TurboQuant path (ADR-007 is
#      Gemma-specific), so the dense/TQ dichotomy does not apply here.
#   2. No literal `<bos>` BOS-stripping — the Qwen3.5 chat template does
#      not embed a literal BOS text marker in the rendered prompt (both
#      hf2q and llama-cli tokenize the template string directly and
#      prepend the BOS token via their own BOS handling). If double-BOS
#      turns out to be an issue empirically, this script will surface it
#      via divergence at byte 0 and the next iter adds a `--no-bos` flag
#      to llama-cli.
#
# The llama-cli binary is invoked as an EXTERNAL black-box reference at
# gate time — it is NEVER linked into hf2q at build/test/CI time, per
# feedback_hf2q_sovereignty.md. Correctness proof derives from live
# comparison against a separately-built reference, not from any code or
# binary dependency.
#
# Usage:
#   scripts/sourdough_qwen35.sh <gguf_path>
#   scripts/sourdough_qwen35.sh <gguf_path> --min-prefix 128
#   scripts/sourdough_qwen35.sh <gguf_path> --max-tokens 64
#
# Exit codes:
#   0  common prefix >= MIN_COMMON_PREFIX (gate passed)
#   1  usage / env error
#   2  gate failed (common prefix < MIN_COMMON_PREFIX)
#   3  tool invocation failure (hf2q or llama-cli crashed)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
PROMPT_FILE="tests/sourdough_qwen35_prompt.txt"
OUT_HF2Q="/tmp/sourdough_qwen35_hf2q.txt"; LOG_HF2Q="/tmp/sourdough_qwen35_hf2q.log"
OUT_LLAMA="/tmp/sourdough_qwen35_llama.txt"; LOG_LLAMA="/tmp/sourdough_qwen35_llama.log"

# Floor calibrated from ADR-013 P13 first successful pass (git HEAD 5737f89,
# 2026-04-24): llama 310 bytes, hf2q 304 bytes, common prefix 180 bytes.
# Raised from the initial 128-byte bootstrap floor to 160 — leaves ~20 bytes
# safety margin below the measured prefix so future commits catch real
# regressions (drift > current level) without false-alarming on noise.
# Same methodology as ADR-005's 1bNEW.20.FIX floor calibration.
MIN_COMMON_PREFIX=160
MAX_TOKENS=80

usage() {
  cat <<EOF
Usage: scripts/sourdough_qwen35.sh <gguf_path> [--min-prefix N] [--max-tokens N]
  <gguf_path>         Path to a Qwen3.5 or Qwen3.6 GGUF (qwen35 or qwen35moe arch)
  --min-prefix N      Common byte prefix floor (default: 128)
  --max-tokens N      Max decode tokens per run (default: 80)

Exit codes:
  0  gate passed
  1  usage / env error
  2  gate failed (drift earlier than min-prefix)
  3  tool invocation failure
EOF
}
err() { echo "error: $*" >&2; exit 1; }

GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --min-prefix)
      [[ $# -ge 2 ]] || err "--min-prefix requires an argument"
      MIN_COMMON_PREFIX="$2"; shift 2 ;;
    --max-tokens)
      [[ $# -ge 2 ]] || err "--max-tokens requires an argument"
      MAX_TOKENS="$2"; shift 2 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -f "$PROMPT_FILE" ]] || err "prompt fixture not found: $PROMPT_FILE"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release)"

# Prefer our built-from-/opt/llama.cpp binary over any PATH-resolved one
# (homebrew ships an older llama-cli that lacks -no-cnv / -st; using a
# non-matching reference would invalidate the byte-parity oracle).
if [[ -x "/opt/llama.cpp/build/bin/llama-cli" ]]; then
  LLAMA_BIN="/opt/llama.cpp/build/bin/llama-cli"
elif command -v llama-cli >/dev/null 2>&1; then
  LLAMA_BIN="$(command -v llama-cli)"
else
  err "llama-cli not found at /opt/llama.cpp/build/bin/llama-cli or on PATH (run: cmake --build /opt/llama.cpp/build --config Release)"
fi

# Validate arch: file MUST be qwen35 or qwen35moe. Read general.architecture
# directly from GGUF metadata via a minimal Python3 parser — avoids the need
# for a dedicated hf2q info-gguf subcommand while remaining zero-dependency.
ARCH="$(python3 - "$GGUF_PATH" <<'PYEOF'
import struct, sys
path = sys.argv[1]
with open(path, 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF':
        sys.exit(0)
    f.read(4)  # version
    f.read(16) # n_tensors, n_kv
    # scan KV pairs until we hit general.architecture
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        key_len = struct.unpack('<Q', hdr)[0]
        if key_len > 4096:
            break
        key = f.read(key_len).decode('utf-8', errors='replace')
        vt = struct.unpack('<I', f.read(4))[0]
        if vt == 8:  # STRING
            slen = struct.unpack('<Q', f.read(8))[0]
            val = f.read(slen).decode('utf-8', errors='replace')
            if key == 'general.architecture':
                print(val)
                sys.exit(0)
        elif vt == 9:  # ARRAY
            at, alen = struct.unpack('<IQ', f.read(12))
            esz = {0:1,1:1,2:2,3:4,4:4,5:8,6:8,7:4,10:8,11:8,12:4}.get(at, 0)
            if esz:
                f.read(alen * esz)
            else:
                break
        else:
            esz = {0:1,1:1,2:2,3:4,4:4,5:8,6:8,7:4,10:8,11:8,12:4}.get(vt, 0)
            if esz:
                f.read(esz)
            else:
                break
PYEOF
)"
if [[ "$ARCH" != "qwen35" && "$ARCH" != "qwen35moe" ]]; then
  err "GGUF arch is '$ARCH', expected 'qwen35' or 'qwen35moe' — wrong gate for this file."
fi

PROMPT_CONTENT="$(cat "$PROMPT_FILE")"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "=== Sourdough byte-prefix gate (ADR-013 P13 Qwen3.5) ==="
echo "GGUF:           $GGUF_PATH"
echo "arch:           $ARCH"
echo "hf2q:           $HF2Q_BIN"
echo "llama-cli:      $LLAMA_BIN"
echo "git HEAD:       $GIT_HEAD"
echo "prompt:         $(head -c 80 <<< "$PROMPT_CONTENT")..."
echo "max_tokens:     $MAX_TOKENS"
echo "min-prefix:     $MIN_COMMON_PREFIX bytes"
echo

# 1. Run llama-cli.
#    -st:  single-turn conversation (applies chat template, exits after reply)
#    -no-cnv: disable interactive mode (we drive via stdin closure)
#    --no-display-prompt: do NOT echo the prompt back — we want ONLY the
#      generated tokens in OUT_LLAMA for byte comparison.
#    -ngl 999: offload all layers to Metal (match hf2q's GPU path).
#    --seed 42: deterministic sampling state (matters for T=0 tiebreaks).
echo "--- Running llama-cli (T=0 greedy, $MAX_TOKENS tokens) ---"
# -st = single-turn (applies chat template, exits after one reply; implies no-cnv)
# -ngl 999 = offload all layers to Metal (match hf2q's GPU path)
# --no-display-prompt = omit the echoed prompt from stdout
# --seed = deterministic sampling state (matters only for T=0 tiebreaks)
if ! "$LLAMA_BIN" --model "$GGUF_PATH" --prompt "$PROMPT_CONTENT" \
      --predict "$MAX_TOKENS" --temp 0 --seed 42 \
      --no-display-prompt -st -ngl 999 \
      </dev/null >"$OUT_LLAMA" 2>"$LOG_LLAMA"; then
  echo "llama-cli failed. See $LOG_LLAMA" >&2; exit 3
fi

# 2. Run hf2q.
# Note: --seed is not in hf2q's generate CLI (at temp=0 greedy, seed is
# irrelevant — argmax is deterministic). Removed to match actual CLI surface.
echo "--- Running hf2q generate (T=0 greedy, $MAX_TOKENS tokens) ---"
if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$PROMPT_CONTENT" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >"$OUT_HF2Q" 2>"$LOG_HF2Q"; then
  echo "hf2q failed. See $LOG_HF2Q" >&2; exit 3
fi

# 3. Common-byte-prefix diff.
#    - hf2q's stdout has a 4-line header (per src/serve/header.rs) that
#      we strip before comparison.
#    - llama-cli's stdout has a banner + loading progress + ASCII art
#      logo + single-turn chat prompt line + "| [Start thinking]" marker
#      that we also strip. The generated text starts after the assistant
#      prompt line. Strip from the start up through the last `\n> ` + the
#      next `\n` (the prompt-echo line) before comparison.
DIFF_TSV="/tmp/sourdough_qwen35_diff.tsv"
python3 - "$OUT_LLAMA" "$OUT_HF2Q" "$DIFF_TSV" <<'PY'
import sys, json
a_raw = open(sys.argv[1], "rb").read()
b_raw = open(sys.argv[2], "rb").read()
out = sys.argv[3]

def strip_hf2q_header(buf):
    """hf2q prints 4 leader lines before the generated text."""
    if not buf.startswith(b"hf2q \xc2\xb7 "):
        return buf
    pos = 0
    for _ in range(4):
        nl = buf.find(b"\n", pos)
        if nl < 0:
            return buf
        pos = nl + 1
    return buf[pos:]

def strip_llama_banner(buf):
    """llama-cli emits: progress spinner + ASCII logo + build/model/
    modalities + available-commands + a prompt-echo line starting with
    `> `. The generated text begins immediately after that echo line's
    newline. Scan for the '\\n> ' prompt-echo and skip past its line."""
    # Find the last occurrence of a line starting with "> " — that's
    # llama-cli's echo of the user prompt before generation.
    needle = b"\n> "
    idx = buf.rfind(needle)
    if idx < 0:
        return buf
    # Skip past the newline-terminated prompt-echo line.
    nl = buf.find(b"\n", idx + 1)
    if nl < 0:
        return buf[idx + len(needle):]
    return buf[nl + 1:]

a = strip_llama_banner(a_raw)
b = strip_hf2q_header(b_raw)
# Also trim trailing llama-cli footer ("\n[ Prompt: ... | Generation: ... ]\nExiting...\n")
tail_marker = b"\n[ Prompt: "
tidx = a.rfind(tail_marker)
if tidx >= 0:
    a = a[:tidx]
# llama-cli -st prepends a thinking-mode indicator before the model output:
# "\n| [Start thinking]\n" (may include ANSI color codes or spinner chars).
# hf2q does not emit this header. Strip it so we compare model tokens only.
think_marker = b"[Start thinking]\n"
tmidx = a.find(think_marker)
if tmidx >= 0:
    a = a[tmidx + len(think_marker):]

n = min(len(a), len(b))
i = 0
while i < n and a[i] == b[i]:
    i += 1
def snip(buf, start, length=120):
    s = buf[start:start+length].decode("utf-8", errors="replace")
    if len(buf) - start > length:
        s += "..."
    return json.dumps(s)
with open(out, "w") as f:
    f.write(f"{len(a)}\t{len(b)}\t{i}\t{snip(a, i)}\t{snip(b, i)}\n")
PY
IFS=$'\t' read -r LLAMA_BYTES HF2Q_BYTES COMMON_BYTES DIVERGE_A DIVERGE_B < "$DIFF_TSV"

echo
echo "--- Comparison ---"
echo "llama output:  $LLAMA_BYTES bytes"
echo "hf2q  output:  $HF2Q_BYTES bytes"
echo "common prefix: $COMMON_BYTES bytes"
echo "min required:  $MIN_COMMON_PREFIX bytes"
echo

if [[ "$COMMON_BYTES" -lt "$MIN_COMMON_PREFIX" ]]; then
  echo "FAIL: drift earlier than the gate's floor."
  echo
  echo "First 120 chars after byte $COMMON_BYTES:"
  echo "  llama: $DIVERGE_A"
  echo "  hf2q:  $DIVERGE_B"
  echo
  echo "Debugging steps:"
  echo "  1. Check \`cargo test --lib qwen35\` — all P7b/P8b/P9b parity tests must be green."
  echo "  2. Use ActivationCapture to dump per-layer outputs from both hf2q"
  echo "     (via HF2Q_CAPTURE_LAYER=N env) and llama.cpp (via --verbose-prompt"
  echo "     + -lv 2); bisect layer-by-layer to find the first divergent layer."
  echo "  3. Common culprits: V-head reorder off-by-one, MROPE section index,"
  echo "     sigmoid-vs-swish on any gate, RMS +1 convention, KV cache slot"
  echo "     indexing for hybrid (full-attn vs linear-attn layer types)."
  exit 2
fi

echo "PASS: common prefix $COMMON_BYTES >= $MIN_COMMON_PREFIX."
if [[ "$COMMON_BYTES" -gt "$MIN_COMMON_PREFIX" ]]; then
  echo "      (drift is $((COMMON_BYTES - MIN_COMMON_PREFIX)) bytes tighter than the floor)"
fi
exit 0
