#!/bin/bash
# ADR-029 iter-151: AOT-compile + AIR-diff our PORT_NWG32 vs peer's flash_attn_ext_vec_f16_dk256_dv256.
# Unblocked by operator installing Metal Toolchain 2026-05-12 → metal-objdump now available.
#
# Apple Silicon doesn't allow per-dispatch GPU timing (iter-143). The remaining
# attribution path is to compare the compiler's IR output. This script does that
# at the AIR (LLVM bitcode) layer.
#
# Limitation: runtime PSO instantiation with function constants differs from AOT
# without. This diffs the TEMPLATE shape. Different function-constant-resolved
# instances may produce different IR; closer-to-runtime requires capturing the
# actual runtime PSO (Metal `MTLBinaryArchive`-based) which is multi-iter work.

set -euo pipefail

WORK=/tmp/adr029_air
mkdir -p "$WORK"
cd "$WORK"

OUR_SHADER=/opt/mlx-native/src/shaders/flash_attn_vec_peer_port_f16_nwg32.metal
PEER_SHADER=/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal

echo "[iter-151] compiling our PORT_NWG32 shader to AIR..."
xcrun -sdk macosx metal -c "$OUR_SHADER" -o our_port_nwg32.air 2>&1 | tail -3

echo "[iter-151] compiling peer's library to AIR (~2-3s)..."
xcrun -sdk macosx metal -c -DGGML_METAL_USE_BF16 \
    -I /opt/llama.cpp/ggml/src \
    -I /opt/llama.cpp/ggml/src/ggml-metal \
    "$PEER_SHADER" -o peer_full.air 2>&1 | tail -3

echo ""
echo "[iter-151] disassembling our kernel..."
xcrun metal-objdump -d our_port_nwg32.air > our_port_nwg32.ll 2>&1
wc -l our_port_nwg32.ll

echo "[iter-151] disassembling peer's flash_attn_ext_vec_f16_dk256_dv256..."
# Find the peer template instantiation we care about
xcrun metal-objdump -d peer_full.air > peer_full.ll 2>&1
echo "Peer .ll size:"
wc -l peer_full.ll

echo ""
echo "[iter-151] instruction-count summary:"
echo "OUR kernel air intrinsics:"
grep -oE "air\.[a-z_.]+" our_port_nwg32.ll | sort | uniq -c | sort -rn | head -15
echo ""
echo "PEER flash_attn intrinsics (in flash_attn_ext_vec section):"
awk '/^define.*kernel_flash_attn_ext_vec_f16_dk256_dv256/,/^}/' peer_full.ll | grep -oE "air\.[a-z_.]+" | sort | uniq -c | sort -rn | head -15

echo ""
echo "Artifacts:"
echo "  our_port_nwg32.air      ($(stat -f%z our_port_nwg32.air) bytes)"
echo "  our_port_nwg32.ll       ($(wc -l < our_port_nwg32.ll) lines)"
echo "  peer_full.air           ($(stat -f%z peer_full.air) bytes)"
echo "  peer_full.ll            ($(wc -l < peer_full.ll) lines)"
