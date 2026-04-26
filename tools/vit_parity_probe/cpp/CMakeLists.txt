# peer_dumper — llama.cpp ViT reference dumper for ADR-005 parity probe.
#
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/opt/homebrew
#   cmake --build build -j
#
# Output: build/peer_dumper
#
# Links against the Homebrew-installed libmtmd / libllama / libggml*.
# /opt/llama.cpp itself is read-only-reference per project policy and is
# NOT touched by this build — we use only the public headers + libs that
# `brew install llama.cpp` provides at /opt/homebrew/{include,lib}.

cmake_minimum_required(VERSION 3.20)
project(peer_dumper CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to /opt/homebrew if the caller didn't pin a prefix.
if(NOT DEFINED HOMEBREW_PREFIX)
    set(HOMEBREW_PREFIX "/opt/homebrew")
endif()

add_executable(peer_dumper peer_dumper.cpp)

target_include_directories(peer_dumper PRIVATE
    "${HOMEBREW_PREFIX}/include"
)

# Resolve dylibs by absolute path; we don't want CMake to substitute a
# different libmtmd if a stray copy lives under /usr/local.
find_library(LIB_MTMD       mtmd       PATHS "${HOMEBREW_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(LIB_LLAMA      llama      PATHS "${HOMEBREW_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(LIB_GGML       ggml       PATHS "${HOMEBREW_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(LIB_GGML_BASE  ggml-base  PATHS "${HOMEBREW_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

target_link_libraries(peer_dumper PRIVATE
    ${LIB_MTMD}
    ${LIB_LLAMA}
    ${LIB_GGML}
    ${LIB_GGML_BASE}
)

# Set rpath so the binary finds the Homebrew dylibs at runtime without
# the user needing to set DYLD_LIBRARY_PATH.
set_target_properties(peer_dumper PROPERTIES
    INSTALL_RPATH "${HOMEBREW_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
