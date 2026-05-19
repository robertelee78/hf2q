# `ggml_quants` reference fixtures (ADR-033 P0)

Byte-cmp ground truth for the per-`GgmlType` Rust quantizer ports under `src/quantize/ggml_quants/`. Each fixture pair `<type>_<n_per_row>_<variant>_{input,expected}.bin` is the F32 input bytes (little-endian, IEEE-754 native) plus the bytes emitted by `ggml_quantize_chunk` at the pinned llama.cpp SHA. The Rust port's `Quantizer::quantize(read_floats(input), n_per_row, imatrix)` MUST return exactly the bytes in `expected.bin`.

## Pins

- llama.cpp: `data/llama_cpp_pin.txt` (currently `c779f619802c310798ca8c89695cec7dcfe38a99`)
- Build flags: `GGML_NATIVE=ON` on `aarch64-apple-darwin` (NEON enabled — see ADR-033 §P0 acceptance finding I)
- PRNG: mulberry32 with input_seed=1, imatrix_seed=2 (constants; changing them is an ADR amendment because all fixtures regenerate)

## Fixture set (v1)

11 types × 2 variants × 2 rows × `n_per_row` size matching the canonical block boundary:

| Type | block size | n_per_row | n_rows | bytes/expected | imatrix variant |
|---|---|---|---|---|---|
| Q4_0 | 18 | 64 | 2 | 72 | noim, im |
| Q4_1 | 20 | 64 | 2 | 80 | noim, im |
| Q5_0 | 22 | 64 | 2 | 88 | noim, im |
| Q5_1 | 24 | 64 | 2 | 96 | noim, im |
| Q8_0 | 34 | 64 | 2 | 136 | noim, im |
| IQ4_NL | 18 | 64 | 2 | 72 | noim, im |
| Q2_K | 84 | 512 | 2 | 336 | noim, im |
| Q3_K | 110 | 512 | 2 | 440 | noim, im |
| Q4_K | 144 | 512 | 2 | 576 | noim, im |
| Q5_K | 176 | 512 | 2 | 704 | noim, im |
| Q6_K | 210 | 512 | 2 | 840 | noim, im |

## Regenerate

```bash
cd /opt/llama.cpp && git checkout $(cat /opt/hf2q/data/llama_cpp_pin.txt) && cmake -B build -DGGML_NATIVE=ON && cmake --build build
cd /opt/hf2q && scripts/ggml_quants_harness/build.sh
scripts/ggml_quants_harness/generate_all.sh
```

The harness source is `scripts/ggml_quants_harness/gen.c`; it links `libggml.dylib + libggml-cpu.dylib + libggml-base.dylib` and wraps the public `ggml_quantize_chunk` (`/opt/llama.cpp/ggml/include/ggml.h:2764`).

## NEON-vs-scalar caveat

Per ADR-033 §P0 acceptance finding I and the module-doc note at `src/quantize/k_quant.rs:9-18`, llama.cpp's `make_qkx2_quants` has a NEON-vs-scalar argument-order divergence. These fixtures are generated NEON-on. If you rebuild the harness `x86_64-pc-linux-gnu` (scalar path) and the bytes differ, hf2q's ports are matched against the NEON variant. Cross-arch divergence (if any) is recorded in this README.

**Cross-arch byte-identity status (as of P0 ship):** TBD — to be filled when x86 cross-check runs.

## Format

- `*_input.bin`: `n_per_row × n_rows` little-endian F32 floats. PRNG is mulberry32 seeded with the constant `input_seed=1` filling `[-1.0, 1.0]` uniform. Imatrix variants additionally use `imatrix_seed=2` for the per-column importance vector (`abs(mulberry32) + 1e-3`, length `n_per_row`).
- `*_expected.bin`: opaque bytes emitted by `ggml_quantize_chunk` for that type/imatrix/input combination. The Rust test reads them whole and `cmp`s against `Quantizer::quantize`'s output.

Mulberry32 step (mirror this exactly in the Rust test if regenerating in-place):

```c
static uint32_t mulberry32_step(uint32_t *state) {
    *state += 0x6D2B79F5u;
    uint32_t t = *state;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + (t ^ (t >> 7)) * (t | 61u);
    return t ^ (t >> 14);
}
```
