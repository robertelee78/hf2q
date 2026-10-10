#!/usr/bin/env bash
# ADR-059 — ISOLATED dose-response experiment: does a donor-derived KV
# graft produce a measurable behavioral effect, alone, as a function of
# dose? One variable (graft n_slots at 0/64/128/256); everything else
# identical. No GCD, no other controls, no combined arms.
#
# Pre-registered metrics (declared before running):
#   1. french_rate — fraction of the 12 English-question responses that
#      contain French (diacritics or >=2 French function words). The
#      donor bank is a French-persona prefill; baseline expectation ~0.
#   2. capability — GSM8K exact-match on the 15-probe set (cost side).
#
# A dose-response trend (french_rate rising with n_slots) is the signal
# that separates a real effect from noise. Flat-at-zero at every dose is
# the falsifier for donor banks on this surface.
#
# Usage: bash scripts/graft_probe/dose_response.sh [MODEL] [PORT]
set -euo pipefail

MODEL="${1:-/opt/hf2q/models/qwen-family-mtp-qualified/qwen35-dense/Qwen3.5-4B-Q4_K_M.gguf}"
PORT="${2:-8396}"
BIN="$(cd "$(dirname "$0")/../.." && pwd)/target/release/hf2q"
WORK="$(mktemp -d /tmp/hf2q-graft-dose.XXXXXX)"
PROBES="$(cd "$(dirname "$0")" && pwd)"
SRV_PID=""

stop_server() {
    if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
}
trap stop_server EXIT

run_arm() {
    local arm="$1"
    shift
    local log="$WORK/serve-$arm.log"
    echo "── arm $arm: booting on :$PORT"
    HF2Q_TQ_KV=1 HF2Q_QWEN_SPECULATION=off "$BIN" serve "$MODEL" \
        --port "$PORT" --quiet --scheduler fifo-serial \
        --default-thinking-token-budget 0 "$@" >"$log" 2>&1 &
    SRV_PID=$!
    local i
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        kill -0 "$SRV_PID" 2>/dev/null || { echo "FAIL: serve $arm exited" >&2; tail -5 "$log" >&2; exit 1; }
        sleep 2
    done
    local model_id
    model_id="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
    for probe_file in probes_english probes_capability; do
        while IFS= read -r line; do
            python3 - "$PORT" "$model_id" "$arm" "$probe_file" "$line" "$WORK" <<'PY'
import json, sys, urllib.request
port, model_id, arm, probe_file, line, work = sys.argv[1:7]
probe = json.loads(line)
body = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": probe["prompt"]}],
    "temperature": 0, "max_tokens": 256, "seed": 42,
    "stream": False, "hf2q_enable_thinking": False,
})
req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    data=body.encode(), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.load(resp)
except urllib.error.HTTPError as error:
    result = {"http_error": error.code}
choice = (result.get("choices") or [{}])[0]
message = choice.get("message", {})
record = {
    "arm": arm, "probe_file": probe_file, "probe_id": probe.get("id"),
    "expected": probe.get("answer"),
    "content": message.get("content") or "",
    "finish_reason": choice.get("finish_reason"),
    "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
    "http_error": result.get("http_error"),
}
with open("%s/results-%s.jsonl" % (work, arm), "a") as f:
    f.write(json.dumps(record) + "\n")
PY
        done <"$PROBES/$probe_file.jsonl"
        echo "   arm $arm: $probe_file done"
    done
    stop_server
    sleep 2
}

echo "dose-response workdir: $WORK"
run_arm dose0
run_arm dose64 --kv-graft /tmp/q35-4b-fr-64.graft.gguf
run_arm dose128 --kv-graft /tmp/q35-4b-fr-128.graft.gguf
run_arm dose256 --kv-graft /tmp/q35-4b-fr-256.graft.gguf

python3 - "$WORK" <<'PY'
import json, os, re, sys

work = sys.argv[1]
ARMS = ["dose0", "dose64", "dose128", "dose256"]

FRENCH_WORDS = {"le", "la", "les", "un", "une", "des", "est", "et", "pour",
                "avec", "je", "tu", "vous", "nous", "dans", "sur", "pas",
                "réponse", "question", "voici", "car", "mais", "ou", "où"}
DIACRITICS = set("éèàçùêôîâëïüœÉÈÀÇÙÊÔÎÂËÏÜŒ")

def is_french(text):
    if any(ch in DIACRITICS for ch in text):
        return True
    words = re.findall(r"[a-zA-Zà-ÿÀ-Ý']+", text.lower())
    hits = sum(1 for w in words if w in FRENCH_WORDS)
    return hits >= 2

def norm(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None

print("arm      french_rate   capability   median tokens")
rows_by_arm = {}
for arm in ARMS:
    path = os.path.join(work, "results-%s.jsonl" % arm)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows_by_arm[arm] = rows
    eng = [r for r in rows if r["probe_file"] == "probes_english"]
    cap = [r for r in rows if r["probe_file"] == "probes_capability"]
    french = sum(1 for r in eng if is_french(r.get("content") or ""))
    correct = 0
    for r in cap:
        content = r.get("content") or ""
        got = None
        numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
        got = numbers[-1] if numbers else None
        correct += int(norm(got) is not None and norm(got) == norm(r["expected"]))
    toks = sorted(r.get("completion_tokens") or 0 for r in rows)
    median = toks[len(toks)//2] if toks else 0
    print("%-8s %d/%d           %d/%d          %s" % (arm, french, len(eng), correct, len(cap), median))

manifest = {
    "experiment": "adr-059-isolated-dose-response",
    "design": "one variable: graft n_slots (donor French-persona bank); no GCD, no combined arms",
    "preregistered_metrics": ["french_rate (probes_english)", "capability exact-match (probes_capability)"],
    "arms": {},
}
for arm in ARMS:
    eng = [r for r in rows_by_arm[arm] if r["probe_file"] == "probes_english"]
    cap = [r for r in rows_by_arm[arm] if r["probe_file"] == "probes_capability"]
    french = sum(1 for r in eng if is_french(r.get("content") or ""))
    correct = 0
    for r in cap:
        content = r.get("content") or ""
        numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
        got = numbers[-1] if numbers else None
        correct += int(norm(got) is not None and norm(got) == norm(r["expected"]))
    manifest["arms"][arm] = {"french_rate": french, "english_n": len(eng),
                             "capability": correct, "capability_n": len(cap)}
path = os.path.join(work, "dose-manifest.json")
json.dump(manifest, open(path, "w"), indent=2)
print("manifest: %s" % path)
PY
