# Operator runbook: refreshing `tests/evals/reference/*_llama.txt`

**Author:** ADR-005 iter-220 2026-05-01 (corrected from iter-219b draft)
**Trigger:** parity_check.sh Gates C + E (vs-llama on long prompts) FAIL with low common-prefix despite hf2q being deterministic.
**Pre-conditions:** system-quiet window (no parallel-session llama-cli, llama-completion, or hf2q-bench holding the M5 Max GPU; check `ps -A -o %cpu,etime,comm | grep -iE "llama|hf2q"` returns nothing >5% CPU).

**iter-220 correction (2026-05-01):** the iter-219b draft of this runbook used `llama-cli --no-conversation` which the locked commit `b3d758750a` REJECTS at startup with "--no-conversation is not supported by llama-cli, please use llama-completion instead" (and falls back to interactive mode that hangs forever). The CANONICAL invocation is `llama-completion` (raw-completion mode) with the chat template manually applied to the `--prompt` string. The frozen Apr 16 fixtures used this same invocation; the iter-219b draft mis-named the binary based on `scripts/sourdough_qwen35.sh` which is QWEN-specific (Qwen3.5/3.6 chat template inlines cleanly with `llama-cli -st`, but Gemma 4's `<|channel>thought\n<channel|>` template tags leak verbatim under `-st`).

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

### 3. Run llama-completion on each prompt (with manual chat template)

Settings per `MANIFEST.json`: `temperature=0`, `seed_llama=42`, `top_p=1.0`, `repetition_penalty=1.0`.

The CLI fallback chat template hf2q applies (see `src/serve/mod.rs::FALLBACK_GEMMA4_CHAT_TEMPLATE`) is:

```
<bos><|turn>user
{USER_PROMPT}<turn|>
<|turn>model
<|channel>thought
<channel|>
```

For peer parity, llama-completion must receive the same pre-formatted prompt (raw-completion mode does not auto-apply a chat template). Step 2 builds `llama-completion` via `cmake --build . --config Release --target llama-completion`.

```sh
GGUF=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf
LLAMA_BIN=/opt/llama.cpp/build/bin/llama-completion   # NOT llama-cli

apply_template() {
  local user_prompt="$1"
  printf '<bos><|turn>user\n%s<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' "$user_prompt"
}

for p in short_hello sourdough sliding_wrap; do
  case "$p" in
    short_hello)  N=50 ;;
    sourdough)    N=1000 ;;
    sliding_wrap) N=500 ;;
  esac
  USER_PROMPT=$(cat /opt/hf2q/tests/evals/prompts/${p}.txt)
  TEMPLATED=$(apply_template "$USER_PROMPT")
  OUT=/tmp/${p}_llama_refresh.txt

  "$LLAMA_BIN" --model "$GGUF" --prompt "$TEMPLATED" \
      -n $N --temp 0 --seed 42 \
      --no-display-prompt -ngl 999 \
      </dev/null >"$OUT" 2>/dev/null

  echo "$p captured: $(wc -c < "$OUT") bytes"
done
```

Why `llama-completion` (not `llama-cli`):
- The locked commit `b3d758750a` `llama-cli` binary explicitly rejects `--no-conversation` and `-no-cnv` and emits "please use llama-completion instead". Without `-no-cnv` it forces interactive mode and hangs on stdin.
- `llama-cli -st` (single-turn) IS supported but auto-applies the chat template — for Gemma 4 GGUFs without an embedded `tokenizer.chat_template`, the auto-template includes `<|channel>thought\n<channel|>` tags that leak into output verbatim.
- `llama-completion` is raw-completion mode (no template auto-applied), so we manually pre-format the prompt to match what hf2q's tokenizer sees.

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

Per the live thresholds in `parity_check.sh` (anchored 2026-05-01):

- `short_hello` requires `common_prefix >= 29` (full answer match).
- `sourdough` requires `common_prefix >= 179` (long-prompt regression-detector floor).
- `sliding_wrap` requires `common_prefix >= 108` (long-prompt regression-detector floor).

**Option A** — common_prefix meets the threshold against today's freshly-captured llama: copy `/tmp/${p}_llama_refresh.txt` into `tests/evals/reference/${p}_llama.txt`, update `MANIFEST.json` `llama_cpp_commit` field to today's `b3d758750a` short-hash (or whatever locked commit was built), bump `generated` field to today's date.

**Option B** — common_prefix is below threshold even against today's freshly-captured llama: there's a real cross-ADR drift in hf2q's argmax-token selection. Document the divergence point and the divergent token in MANIFEST's `divergence_note` field. Operator decision: lower the threshold (acknowledge cross-implementation drift) OR file an investigation issue against ADR-015 (mlx-native kernel parity) — note that ADR-013/017 ARE the upstream ADRs but the per-token argmax math lives in mlx-native (sibling repo at /opt/mlx-native).

**iter-220 precedent (2026-05-01):** post 14 days of ADR-013 P16 + ADR-017 phase work, sourdough common_prefix dropped from 3094 → 179 and sliding_wrap dropped from 700 → 108 even after fresh-llama capture at the same locked commit. Both outputs verified coherent on both sides (just argmax-flip on word choice — `flavor and texture are superior` vs `result is a superior flavor and texture`; `"calculating" to "processing"` vs `physical movement to electrical movement`). Floors anchored to today's measurement; any future hf2q kernel change that further reduces common_prefix trips the regression gate.

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
