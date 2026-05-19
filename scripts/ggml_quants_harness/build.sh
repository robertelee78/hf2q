#!/usr/bin/env bash
# Build the ADR-033 P0 reference-fixture generator against llama.cpp's
# pre-built dylibs at /opt/llama.cpp/build/bin/.
#
# llama.cpp pin: data/llama_cpp_pin.txt
# Required built: cd /opt/llama.cpp && cmake -B build -DGGML_NATIVE=ON && cmake --build build
# (already built on the operator's machine; sanity-checked by ls of libggml*.dylib)

set -euo pipefail

LLAMA=${LLAMA_CPP_DIR:-/opt/llama.cpp}
HERE=$(cd "$(dirname "$0")" && pwd)

[ -f "$LLAMA/ggml/include/ggml.h" ] || { echo "missing $LLAMA/ggml/include/ggml.h"; exit 1; }
[ -f "$LLAMA/build/bin/libggml.dylib" ] || { echo "missing $LLAMA/build/bin/libggml.dylib — build llama.cpp first"; exit 1; }

clang -O2 -std=c11 -Wall -Wextra \
    -I"$LLAMA/ggml/include" \
    "$HERE/gen.c" \
    -L"$LLAMA/build/bin" \
    -lggml -lggml-cpu -lggml-base -lm \
    -Wl,-rpath,"$LLAMA/build/bin" \
    -o "$HERE/gen"

echo "built: $HERE/gen"
