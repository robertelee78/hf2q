#!/usr/bin/env bash
# ADR-063 Q4-GEMMA-COLD G-S3 (design §7.1): the no-boundary cold batch on the
# per-sequence cold attention (this checkout) against the block-diagonal
# route (the base commit), same harness, same artifact, quiet host.
#
#   HF2Q_GEMMA4_GGUF=/path/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
#   CARGO_TARGET_DIR=/path/target \
#   scripts/q4_gemma_cold_gs3.sh OUT_DIR [BASE_COMMIT]
#
# GPU lane only: run while holding /Users/robert/.local/state/hf2q/v0130/HOST-LOCK.
# The harness (`gemma_cold_gs3_multi_seq_timing`) is copied unchanged into a
# detached worktree of BASE_COMMIT (default: merge-base with origin/main), so
# each binary times its own `forward_prefill_batched_multi_seq`. Arms run in
# ABBA order with HF2Q_GEMMA_COLD_GS3_REPEATS=5 each (10 transactions per arm
# and set). Gate: per set, candidate median <= 1.05 x base median.
set -euo pipefail

out=${1:?usage: q4_gemma_cold_gs3.sh OUT_DIR [BASE_COMMIT]}
: "${HF2Q_GEMMA4_GGUF:?HF2Q_GEMMA4_GGUF must name the official Gemma 4 GGUF}"
: "${CARGO_TARGET_DIR:?CARGO_TARGET_DIR must be set}"
lock=/Users/robert/.local/state/hf2q/v0130/HOST-LOCK
[[ -e $lock ]] || { echo "G-S3 needs the GPU lane: create $lock first" >&2; exit 2; }

root=$(git rev-parse --show-toplevel)
candidate=$(git -C "$root" rev-parse HEAD)
base=${2:-$(git -C "$root" merge-base HEAD origin/main)}
[[ $base != "$candidate" ]] || { echo "base equals candidate ($base)" >&2; exit 2; }
mkdir -p "$out"
out=$(cd "$out" && pwd)
src="$out/base-src"
harness=src/serve/api/engine_gemma_cold_gs3_timing.rs

cleanup() { git -C "$root" worktree remove --force "$src" 2>/dev/null || true; }
trap cleanup EXIT
git -C "$root" worktree add --detach "$src" "$base" >/dev/null
/bin/cp "$root/$harness" "$src/$harness"
printf '\n#[cfg(test)]\n#[path = "engine_gemma_cold_gs3_timing.rs"]\nmod gemma_cold_gs3_timing;\n' \
  >>"$src/src/serve/api/engine.rs"

build() { # dir target -> test executable path
  (cd "$1" && GIT_COMMIT_SHA=$(git rev-parse HEAD) RUSTC_WRAPPER= CARGO_INCREMENTAL=0 \
    CARGO_TARGET_DIR="$2" cargo test --release --locked --bin hf2q --no-run \
    --message-format=json 2>>"$out/build.log") |
    python3 -c 'import json,sys
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "hf2q" and m["profile"]["test"]:
        print(m["executable"])' | tail -n 1
}
cand_bin=$(build "$root" "$CARGO_TARGET_DIR")
base_bin=$(build "$src" "$CARGO_TARGET_DIR-gs3-base")
[[ -x $cand_bin && -x $base_bin ]] || { echo "build failed; see $out/build.log" >&2; exit 1; }

test_name=serve::api::engine::gemma_cold_gs3_timing::gemma_cold_gs3_multi_seq_timing
run() { # arm bin index
  HF2Q_GEMMA_COLD_GS3_REPEATS=5 "$2" "$test_name" --exact --ignored --nocapture \
    --test-threads=1 >"$out/$3-$1.log" 2>&1
}
run candidate "$cand_bin" 1
run base "$base_bin" 2
run base "$base_bin" 3
run candidate "$cand_bin" 4

native=$(awk '/^name = "mlx-native"/{getline; print $3}' "$root/Cargo.lock" | tr -d '"')
python3 - "$out" "$candidate" "$base" "$cand_bin" "$base_bin" "$HF2Q_GEMMA4_GGUF" "$native" <<'PY'
import glob, hashlib, json, os, statistics, sys
out, candidate, base, cand_bin, base_bin, gguf, native = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()
records = {"candidate": {}, "base": {}}
for log in sorted(glob.glob(os.path.join(out, "[0-9]-*.log"))):
    arm = os.path.basename(log)[2:-4]
    for line in open(log):
        if line.startswith("GEMMA_COLD_GS3_RECORD "):
            r = json.loads(line.split(" ", 1)[1])
            records[arm].setdefault(r["set"], []).append(r["ms"])
sets = sorted(set(records["candidate"]) | set(records["base"]))
summary = {"schema": "hf2q.q4-gemma-cold.g-s3.v1", "candidate_commit": candidate,
           "base_commit": base, "candidate_binary_sha256": sha(cand_bin),
           "base_binary_sha256": sha(base_bin), "artifact": gguf,
           "artifact_sha256": sha(gguf), "native_version": native, "bound": 1.05, "sets": {}}
ok = True
for name in sets:
    c, b = records["candidate"].get(name, []), records["base"].get(name, [])
    if len(c) < 10 or len(b) < 10:
        ok = False
        summary["sets"][name] = {"error": f"records candidate={len(c)} base={len(b)} (< 10)"}
        continue
    ratio = statistics.median(c) / statistics.median(b)
    passed = ratio <= 1.05
    ok &= passed
    summary["sets"][name] = {"candidate_ms": c, "base_ms": b,
                             "candidate_median_ms": statistics.median(c),
                             "base_median_ms": statistics.median(b),
                             "ratio": ratio, "pass": passed}
summary["verdict"] = "pass" if ok and sets else "fail"
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
print(json.dumps({k: v.get("ratio", v) for k, v in summary["sets"].items()}), summary["verdict"])
sys.exit(0 if summary["verdict"] == "pass" else 1)
PY
