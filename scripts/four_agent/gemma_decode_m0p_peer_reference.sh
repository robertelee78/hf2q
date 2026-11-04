#!/bin/bash
# Q4-GEMMA-DECODE M0-P (ADR-063 design §7.3): the peer's backend-op perf
# mode at Gemma 4 decode shapes, as a reference only (never a gate).
#
#   build  — copy the reference source tree into a separate directory, apply
#            the test-only case-list patch there, record its diff, and build
#            `test-backend-ops` in its own build directory. The reference
#            checkout and its serving build directory are never modified.
#   run    — run `test-backend-ops perf` for MUL_MAT, MUL_MAT_ID and
#            FLASH_ATTN_EXT with the Gemma case list. Needs the root's GPU
#            lane and HOST-LOCK (held by the caller), quiet host.
#
# Usage: gemma_decode_m0p_peer_reference.sh build|run [OUT_DIR]
#
# Environment:
#   PEER_REF_SRC  reference checkout (default
#                 /opt/hf2q-worktrees/adr063-peer-reference-e86f5fd1)
#   M0P_ROOT      work root (default
#                 /Users/robert/.local/state/hf2q/ext/peer-ref-m0p-e86f5fd1)
set -euo pipefail

MODE="${1:?usage: $0 build|run [OUT_DIR]}"
PEER_REF_SRC="${PEER_REF_SRC:-/opt/hf2q-worktrees/adr063-peer-reference-e86f5fd1}"
M0P_ROOT="${M0P_ROOT:-/Users/robert/.local/state/hf2q/ext/peer-ref-m0p-e86f5fd1}"
SRC="$M0P_ROOT/src"
BUILD="$M0P_ROOT/build-m0p"
CASES_FILE="tests/test-backend-ops.cpp"

build() {
  mkdir -p "$M0P_ROOT"
  # Bulk copy with the real binary (agent-tool coreutils builtins must not
  # carry tree copies); build directories are excluded.
  /usr/bin/rsync -a --delete --exclude '/build*' "$PEER_REF_SRC/" "$SRC/"
  /bin/cp "$SRC/$CASES_FILE" "$M0P_ROOT/test-backend-ops.orig.cpp"
  python3 - "$SRC/$CASES_FILE" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
anchor = "static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n"
assert text.count(anchor) == 1, "perf case-list anchor not found exactly once"
cases = r'''
    // hf2q Q4-GEMMA-DECODE M0-P: Gemma 4 decode shapes only (HF2Q_M0P_GEMMA=1).
    if (getenv("HF2Q_M0P_GEMMA") != nullptr) {
        struct dense_shape { ggml_type type; int64_t n; int64_t k; };
        const dense_shape dense[] = {
            {GGML_TYPE_Q4_K, 4096, 2816}, {GGML_TYPE_Q4_K, 8192, 2816},
            {GGML_TYPE_Q4_K, 2048, 2816}, {GGML_TYPE_Q4_K, 1024, 2816},
            {GGML_TYPE_Q6_K, 2048, 2816}, {GGML_TYPE_Q4_K, 2816, 4096},
            {GGML_TYPE_Q4_K, 2816, 8192}, {GGML_TYPE_Q4_K, 2112, 2816},
            {GGML_TYPE_Q5_0, 2816, 2112}, {GGML_TYPE_Q8_0, 2816, 2112},
            {GGML_TYPE_Q6_K, 262144, 2816},
        };
        for (const auto & s : dense) {
            for (int64_t w : {1, 2, 3, 4, 5, 6, 7, 8}) {
                test_cases.emplace_back(new test_mul_mat(s.type, GGML_TYPE_F32, s.n, w, s.k, {1, 1}, {1, 1}));
            }
        }
        for (int64_t n_tokens : {1, 4, 8, 16}) {
            // expert gate_up: broadcast token rows, 128 experts, top-8.
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 128, 8, true, 1408, n_tokens, 2816));
            // expert down: one input row per (token, expert) pair.
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q5_0, GGML_TYPE_F32, 128, 8, false, 2816, n_tokens, 704));
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 128, 8, false, 2816, n_tokens, 704));
        }
        // decode attention, the peer's serving F16 K/V: sliding (8 KV heads,
        // 16 query heads, d256) and global (2 KV heads, 16 query heads, d512).
        for (int64_t kv : {512, 1024, 2048, 4096}) {
            for (int64_t nb : {1, 4}) {
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 8, {2, 1}, kv, nb, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
                test_cases.emplace_back(new test_flash_attn_ext(512, 512, 2, {8, 1}, kv, nb, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
            }
            // four independent sequences, one query each
            test_cases.emplace_back(new test_flash_attn_ext(256, 256, 8, {2, 4}, kv, 1, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
            test_cases.emplace_back(new test_flash_attn_ext(512, 512, 2, {8, 4}, kv, 1, true, false, 0.0f, 0.0f, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
        }
        return test_cases;
    }
'''
open(path, "w").write(text.replace(anchor, anchor + cases))
PY
  # Keep the test-only patch with its diff (diff exits 1 on differences).
  /usr/bin/diff -u "$M0P_ROOT/test-backend-ops.orig.cpp" "$SRC/$CASES_FILE" \
    > "$M0P_ROOT/gemma-m0p-cases.patch" || [ $? -eq 1 ]
  shasum -a 256 "$M0P_ROOT/gemma-m0p-cases.patch" > "$M0P_ROOT/gemma-m0p-cases.patch.sha256"
  # Same release flags as the serving build, separate directory.
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON -DGGML_ACCELERATE=ON
  cmake --build "$BUILD" --target test-backend-ops -j 8
  shasum -a 256 "$BUILD/bin/test-backend-ops" | tee "$M0P_ROOT/test-backend-ops.sha256"
}

run() {
  local out="${1:?run needs OUT_DIR}"
  mkdir -p "$out"
  local bin="$BUILD/bin/test-backend-ops"
  [ -x "$bin" ] || { echo "build first: $bin missing" >&2; exit 2; }
  {
    echo "{"
    echo "  \"schema\": \"hf2q.q4-gemma-decode.m0p.identity.v1\","
    echo "  \"reference_source\": \"$PEER_REF_SRC\","
    echo "  \"binary_sha256\": \"$(shasum -a 256 "$bin" | cut -d' ' -f1)\","
    echo "  \"patch_sha256\": \"$(cut -d' ' -f1 < "$M0P_ROOT/gemma-m0p-cases.patch.sha256")\","
    echo "  \"macos_build\": \"$(sw_vers -buildVersion)\","
    echo "  \"hw_model\": \"$(sysctl -n hw.model)\","
    echo "  \"host_lock_holder\": \"$(cat /Users/robert/.local/state/hf2q/v0130/HOST-LOCK 2>/dev/null || true)\","
    echo "  \"kv_representation\": \"peer F16 K and F16 V (hf2q: F16 K, byte-packed TQ-HB V)\""
    echo "}"
  } > "$out/identity.json"
  /bin/cp "$M0P_ROOT/gemma-m0p-cases.patch" "$out/"
  for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT; do
    HF2Q_M0P_GEMMA=1 "$bin" perf -b MTL0 -o "$op" 2>&1 | tee "$out/perf-$op.txt"
  done
}

case "$MODE" in
  build) build ;;
  run) run "${2:-}" ;;
  *) echo "usage: $0 build|run [OUT_DIR]" >&2; exit 2 ;;
esac
