# Operator runbook: refreshing `tests/evals/reference/*_llama.txt`

**Author:** ADR-005 iter-219b 2026-05-01
**Trigger:** parity_check.sh Gates C + E (vs-llama on long prompts) FAIL with low common-prefix despite hf2q being deterministic.
**Pre-conditions:** system-quiet window (no parallel-session llama-cli or hf2q-bench holding the M5 Max GPU; check `ps -A -o %cpu,etime,comm | grep -iE "llama|hf2q"` returns nothing >5% CPU).

## Context

The `tests/evals/reference/*_llama.txt` references were captured at llama_cpp_commit `b3d758750a` 2026-04-16. Subsequent ADR-013 / ADR-017 kernel changes drift hf2q's argmax-token selection on long prompts. When the cross-ADR work converges, today's hf2q output may match neither the frozen Apr 16 llama nor today's llama on the same prompt; the canonical tie-breaker is to refresh the llama side at today's hardware to anchor the parity-gate against TODAY's peer-equivalent reference.

## Steps

### 1. Confirm system-quiet

```sh
ps -A -o pid,%cpu,etime,comm | grep -iE "llama|hf2q.*generate" | head -5
```

Expected: empty output, or only short-running processes. If any long-running (`etime > 30:00`) llama-cli is at >50% CPU, **STOP** and wait — competing on the GPU produces noise.

### 2. Locate the reference llama-cli binary

The locked `b3d758750a` commit binary path priority (per `scripts/sourdough_qwen35.sh`):

```sh
LLAMA_BIN=""
[[ -x "/opt/llama.cpp/build/bin/llama-cli" ]] && LLAMA_BIN="/opt/llama.cpp/build/bin/llama-cli"
[[ -z "$LLAMA_BIN" ]] && LLAMA_BIN="$(command -v llama-cli)"
echo "Using $LLAMA_BIN"
```

If `/opt/llama.cpp/build/bin/llama-cli` is absent (typical post `git fetch`), build from the locked commit:

```sh
cd /opt/llama.cpp
git stash --include-untracked      # save any in-flight work
git checkout b3d758750a            # locked commit per MANIFEST.json
mkdir -p build && cd build
cmake -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -j$(sysctl -n hw.ncpu)
ls bin/llama-cli   # should now exist
cd /opt/hf2q
git -C /opt/llama.cpp checkout -    # return to the previous branch (don't leave llama.cpp on detached HEAD long-term)
```

### 3. Run llama-cli on each prompt

Settings per `MANIFEST.json`: `temperature=0`, `seed_llama=42`, `top_p=1.0`, `repetition_penalty=1.0`.

```sh
GGUF=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
LLAMA_BIN=/opt/llama.cpp/build/bin/llama-cli   # adjust per step 2

for p in short_hello sourdough sliding_wrap; do
  case "$p" in
    short_hello)  N=50 ;;
    sourdough)    N=1000 ;;
    sliding_wrap) N=500 ;;
  esac
  PROMPT_FILE=/opt/hf2q/tests/evals/prompts/${p}.txt
  OUT=/tmp/${p}_llama_refresh.txt

  "$LLAMA_BIN" \
      --model "$GGUF" \
      --file "$PROMPT_FILE" \
      --temp 0 --seed 42 \
      --predict $N \
      --no-conversation \
      --simple-io \
      2>/dev/null > "$OUT"

  echo "$p captured: $(wc -c < "$OUT") bytes"
done
```

The `--no-conversation` flag is critical — homebrew-default llama-cli starts in interactive `>` prompt mode, which never produces a non-interactive completion. The `--simple-io` flag suppresses ANSI control sequences from stdout.

### 4. Sanity-check coherence

```sh
for p in short_hello sourdough sliding_wrap; do
  echo "=== $p (first 100 bytes) ==="
  head -c 100 /tmp/${p}_llama_refresh.txt
  echo
done
```

Expected: each output reads as coherent English completing the prompt. Reject any output that:
- Starts with `<|channel>thought` or `<|tool_call>` (chat-template fallback / tool-mode regression).
- Is empty (binary error, model failed to load).
- Contains repeated single tokens (degenerate decode).

### 5. Compare to hf2q (today's deterministic output)

```sh
for p in sourdough sliding_wrap; do
  python3 - <<EOF
with open('/tmp/${p}_llama_refresh.txt','rb') as f: llama = f.read()
with open('/opt/hf2q/tests/evals/reference/${p}_hf2q.txt','rb') as f: hf2q = f.read()
n = sum(1 for a,b in zip(hf2q, llama) if a == b)
m = 0
for a,b in zip(hf2q, llama):
    if a != b: break
    m += 1
print(f'${p}: hf2q={len(hf2q)} llama={len(llama)} byte_eq_count={n} common_prefix={m}')
EOF
done
```

`common_prefix` is the canonical parity-gate metric.

### 6. Decide: refresh fixtures vs adjust thresholds

Per the existing thresholds in `parity_check.sh`:

- `sourdough` requires `common_prefix >= 3094` (Gate C).
- `sliding_wrap` requires `common_prefix >= 700` (Gate C).
- `short_hello` requires `common_prefix >= 29` (already passes — no refresh needed).

**Option A** — common_prefix meets the threshold against today's freshly-captured llama: copy `/tmp/${p}_llama_refresh.txt` into `tests/evals/reference/${p}_llama.txt`, update `MANIFEST.json` `llama_cpp_commit` field to today's `b3d758750a` short-hash (or whatever locked commit was built), bump `generated` field to today's date.

**Option B** — common_prefix is below threshold even against today's freshly-captured llama: there's a real cross-ADR drift in hf2q's argmax-token selection. Document the divergence point and the divergent token in MANIFEST's `note` field. Operator decision: lower the threshold (acknowledge cross-ADR drift) OR file an investigation issue against the upstream ADR (013/017) responsible for the drift.

### 7. Re-run release-check to verify

```sh
cd /opt/hf2q
GGUF=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
scripts/release-check.sh "$GGUF"
```

Expected post-refresh: 6/6 parity checks PASS. If not, return to step 6 Option B.

## Receipts to capture (operator)

When the refresh lands, the commit message MUST include:
- `llama_cpp_commit` short-hash that produced the new fixtures.
- `hardware`: M5 Max | M5 Pro | etc. (re-measurement is hardware-anchored per `project_end_gate_reality_check`).
- `system_quiet_evidence`: `ps -A | grep -iE "llama|hf2q" | head -3` output proving no contention.
- The `common_prefix` value for each prompt and its relation to the threshold.
- Per-prompt diff summary: where today's hf2q vs today's llama first diverge (byte offset + 50-byte snippet around the divergence).

These receipts are the operator's audit trail for the parity-gate adjustment. Future iterations can bisect ADR-005 closure quality by reading them.
