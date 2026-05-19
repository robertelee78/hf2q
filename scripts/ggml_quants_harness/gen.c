// ADR-033 P0 reference-fixture generator.
//
// Wraps llama.cpp's public `ggml_quantize_chunk` to emit byte-identical
// reference output for the per-`GgmlType` byte-cmp tests at
// `tests/fixtures/ggml_quants/<type>_<n>_<variant>_{input,expected}.bin`.
//
// Pinned: llama.cpp @ /opt/llama.cpp (`data/llama_cpp_pin.txt`).
// Build:  scripts/ggml_quants_harness/build.sh
// Use:    scripts/ggml_quants_harness/generate_all.sh
//
// Deterministic PRNG: mulberry32 (small, easy to mirror in Rust).
// Input bytes (F32, little-endian native, IEEE-754) are written alongside
// the expected output so the Rust test does NOT have to re-derive input
// from a seed — it reads the input bytes and feeds them to hf2q's
// quantizer, then compares the result to the expected bytes byte-for-byte.

#include "ggml.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t mulberry32_step(uint32_t *state) {
    *state += 0x6D2B79F5u;
    uint32_t t = *state;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    return t ^ (t >> 14);
}

// Fill [-1.0, 1.0] uniform F32 with deterministic seed.
static void make_input(float *out, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t u = mulberry32_step(&state);
        out[i] = ((float)u / (float)0xFFFFFFFFu) * 2.0f - 1.0f;
    }
}

// Fill imatrix with positive importance weights (abs(uniform) + 1e-3) from seed.
static void make_imatrix(float *out, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t u = mulberry32_step(&state);
        float v = ((float)u / (float)0xFFFFFFFFu) * 2.0f - 1.0f;
        out[i] = fabsf(v) + 1e-3f;
    }
}

static enum ggml_type parse_type(const char *s) {
    if (!strcmp(s, "q4_0"))   return GGML_TYPE_Q4_0;
    if (!strcmp(s, "q4_1"))   return GGML_TYPE_Q4_1;
    if (!strcmp(s, "q5_0"))   return GGML_TYPE_Q5_0;
    if (!strcmp(s, "q5_1"))   return GGML_TYPE_Q5_1;
    if (!strcmp(s, "q8_0"))   return GGML_TYPE_Q8_0;
    if (!strcmp(s, "iq4_nl")) return GGML_TYPE_IQ4_NL;
    if (!strcmp(s, "q2_k"))   return GGML_TYPE_Q2_K;
    if (!strcmp(s, "q3_k"))   return GGML_TYPE_Q3_K;
    if (!strcmp(s, "q4_k"))   return GGML_TYPE_Q4_K;
    if (!strcmp(s, "q5_k"))   return GGML_TYPE_Q5_K;
    if (!strcmp(s, "q6_k"))   return GGML_TYPE_Q6_K;
    fprintf(stderr, "unknown type: %s\n", s);
    exit(2);
}

static int write_all(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    size_t w = fwrite(data, 1, bytes, f);
    if (w != bytes) { fprintf(stderr, "short write: %s (%zu/%zu)\n", path, w, bytes); fclose(f); return 1; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr,
            "usage: %s <type> <n_per_row> <n_rows> <input_seed> <imatrix_seed|none> <out_input> <out_expected>\n"
            "  <type>: q4_0 q4_1 q5_0 q5_1 q8_0 iq4_nl q2_k q3_k q4_k q5_k q6_k\n",
            argv[0]);
        return 2;
    }

    enum ggml_type ttype = parse_type(argv[1]);
    int64_t n_per_row    = atoll(argv[2]);
    int64_t n_rows       = atoll(argv[3]);
    uint32_t input_seed  = (uint32_t)strtoul(argv[4], NULL, 10);
    const char *imseed_s = argv[5];
    const char *out_in   = argv[6];
    const char *out_ex   = argv[7];

    int64_t blck = ggml_blck_size(ttype);
    if (n_per_row % blck != 0) {
        fprintf(stderr, "n_per_row %lld not a multiple of block size %lld for type %s\n",
                (long long)n_per_row, (long long)blck, argv[1]);
        return 2;
    }

    size_t n_elements = (size_t)(n_per_row * n_rows);
    float *input = (float *)malloc(n_elements * sizeof(float));
    if (!input) { perror("malloc input"); return 1; }
    make_input(input, n_elements, input_seed);

    float *imatrix = NULL;
    int has_imatrix = strcmp(imseed_s, "none") != 0;
    if (has_imatrix) {
        uint32_t im_seed = (uint32_t)strtoul(imseed_s, NULL, 10);
        imatrix = (float *)malloc((size_t)n_per_row * sizeof(float));
        if (!imatrix) { perror("malloc imatrix"); return 1; }
        make_imatrix(imatrix, (size_t)n_per_row, im_seed);
    } else if (ggml_quantize_requires_imatrix(ttype)) {
        fprintf(stderr, "type %s requires imatrix; pass a numeric imatrix seed\n", argv[1]);
        return 2;
    }

    size_t row_size = ggml_row_size(ttype, n_per_row);
    size_t out_size = row_size * (size_t)n_rows;
    void *output = malloc(out_size);
    if (!output) { perror("malloc output"); return 1; }

    ggml_quantize_init(ttype);
    size_t actual = ggml_quantize_chunk(ttype, input, output, 0, n_rows, n_per_row, imatrix);
    if (actual != out_size) {
        fprintf(stderr, "ggml_quantize_chunk size mismatch: got %zu expected %zu\n", actual, out_size);
        return 1;
    }

    if (write_all(out_in, input,  n_elements * sizeof(float))) return 1;
    if (write_all(out_ex, output, out_size))                   return 1;

    fprintf(stderr,
        "ok: type=%s n_per_row=%lld n_rows=%lld input_seed=%u imatrix_seed=%s "
        "input_bytes=%zu expected_bytes=%zu\n",
        argv[1], (long long)n_per_row, (long long)n_rows, input_seed, imseed_s,
        n_elements * sizeof(float), out_size);

    free(input); free(output); if (imatrix) free(imatrix);
    return 0;
}
