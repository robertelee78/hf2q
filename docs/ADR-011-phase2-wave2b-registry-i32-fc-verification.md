# Wave 2B — KernelRegistry i32 Function-Constant Support: Verification

**Status**: Complete  
**Date**: 2026-04-17  
**Agent**: 2B (registry-i32-fc), CFA swarm `swarm-1776516482254-ft5mwj`  
**File modified**: `/opt/mlx-native/src/kernel_registry.rs`

---

## API Shape

```rust
pub fn get_pipeline_with_constants(
    &mut self,
    name: &str,
    device: &metal::DeviceRef,
    bool_constants: &[(usize, bool)],
    int_constants: &[(usize, i32)],
) -> Result<&ComputePipelineState>
```

`bool_constants` maps to `[[function_constant(index)]]` bool declarations in
the MSL shader.  `int_constants` maps to `[[function_constant(index)]]`
`constant int` (int32_t) declarations.  Both slices may be empty.

`get_pipeline_with_bool_constants` is preserved as a thin wrapper that calls
`get_pipeline_with_constants` with `int_constants: &[]`.  Its signature is
unchanged; all existing callers continue to compile without modification.

---

## Cache Key Format

```
<kernel_name>|<bool_index>:b<0|1>|...|<int_index>:i<value>|...
```

Examples:

| Call | Cache key |
|------|-----------|
| `get_pipeline_with_constants("steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16", dev, &[(200,true),(201,false),(300,false),(301,true)], &[])` | `steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16\|200:b1\|201:b0\|300:b0\|301:b1` |
| `get_pipeline_with_constants("flash_attn_prefill_bf16_d512", dev, &[(200,true),(201,true),(300,false),(301,false)], &[(322,8)])` | `flash_attn_prefill_bf16_d512\|200:b1\|201:b1\|300:b0\|301:b0\|322:i8` |
| `get_pipeline_with_bool_constants("flash_attn_prefill_bf16_d256", dev, &[(200,true),(201,false)])` | `flash_attn_prefill_bf16_d256\|200:b1\|201:b0` |

The 'b' and 'i' type markers prevent collision between a bool index N value 1
and an int index N value 1 — they are distinct entries in the key space.

---

## Implementation Details

`get_pipeline_with_constants` calls
`metal::FunctionConstantValues::set_constant_value_at_index` once per bool
and once per int constant:

- Bool constants: pointer to `u8` (0 or 1), `MTLDataType::Bool` (= 53).
- i32 constants: pointer to `i32`, `MTLDataType::Int` (= 29).

This matches the pattern used by llama.cpp's host-side dispatch for
`FC_flash_attn_ext_nsg` (`ggml-metal-ops.cpp:2807`, `ggml-metal.metal:5735`):
the shader declares `constant int32_t FC_flash_attn_ext_nsg [[function_constant(FC_FLASH_ATTN_EXT + 22)]]`
and the host sets it via `MTLFunctionConstantValues` before pipeline creation.

For Wave 2C's D=512 dispatcher the call will be:

```rust
registry.get_pipeline_with_constants(
    kernel_name,
    device.metal_device(),
    &[
        (200, align_q),   // bool: align_Q
        (201, align_k),   // bool: align_K
        (300, has_mask),  // bool: has_mask
        (301, do_causal), // bool: do_causal
    ],
    &[(322, nsg as i32)], // i32: NSG (FC index = FC_FLASH_ATTN_EXT + 22 = 300 + 22 = 322)
)
```

---

## Backward Compatibility

`get_pipeline_with_bool_constants` is now a one-line wrapper:

```rust
pub fn get_pipeline_with_bool_constants(
    &mut self,
    name: &str,
    device: &metal::DeviceRef,
    bool_constants: &[(usize, bool)],
) -> Result<&ComputePipelineState> {
    self.get_pipeline_with_constants(name, device, bool_constants, &[])
}
```

The cache-key format for pure-bool calls has changed from `"<name>|<idx>:<0|1>"`
to `"<name>|<idx>:b<0|1>"`.  This is a **cold-cache-only format change** — the
cache is in-process and is never persisted to disk, so no existing cached entry
is ever misread.  All callers that go through the wrapper receive the new key
format automatically.

---

## Test Output

```
test kernel_registry::tests::test_int_fc_distinct_pipelines_and_bool_compat ... ok

test result: ok. 74 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
```

New test `test_int_fc_distinct_pipelines_and_bool_compat` verifies:

1. Compiling with int FC index 100, value 4 produces a cached pipeline.
2. Compiling with int FC index 100, value 8 produces a **distinct** cached
   pipeline (different pointer, cache count increments by 1).
3. Re-requesting index 100 value 4 is a cache hit (no new entry, same pointer).
4. `get_pipeline_with_bool_constants` with an empty slice still succeeds and
   inserts exactly one new cache entry (backward-compat wrapper path).

No net-new clippy warnings on `kernel_registry.rs`.

---

## Feeds Into

Wave 2C (D=512 dispatcher) will call `get_pipeline_with_constants` with 4 bool
constants (align_Q, align_K, has_mask, do_causal) and 1 i32 constant (NSG=8)
in a single specialisation call, enabling the full NSG-parametric pipeline cache
without separate method flavors.
