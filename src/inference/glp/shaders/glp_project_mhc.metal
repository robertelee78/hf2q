// glp_project_mhc.metal — GLP projection for multi-hyper-connection (mHC)
// state: `[rows, hc, hidden]` F32. One projection per (row, stream) slice.
//
// One threadgroup per (row, stream) pair; the threadgroup is sized to the
// Metal-safe 256 threads and strides the hidden dimension (4096). The
// dot-product reduction is thread-safe: each thread accumulates its strided
// elements, then one thread reduce.

#include <metal_stdlib>
using namespace metal;

struct GlpProjectMhcParams {
    uint rows;
    uint hc;
    uint hidden;
    uint per_stream_directions;
    float alpha;
    float d_norm_sq;
};

kernel void glp_project_mhc_f32(
    constant GlpProjectMhcParams& params [[buffer(0)]],
    device const float*           direction [[buffer(1)]],
    device float*                 state [[buffer(2)]],
    uint                          tid_in_tg [[thread_position_in_threadgroup]],
    uint                          tg_size [[threads_per_threadgroup]],
    uint                          tg_id [[threadgroup_position_in_grid]]
) {
    const uint row = tg_id / params.hc;
    const uint stream = tg_id - row * params.hc;
    if (row >= params.rows) {
        return;
    }
    device float* slice = state + (row * params.hc + stream) * params.hidden;
    device const float* d = direction + (params.per_stream_directions ? stream * params.hidden : 0);

    // accumulate strided elements
    float local_dot = 0.0f;
    for (uint col = tid_in_tg; col < params.hidden; col += tg_size) {
        local_dot += slice[col] * d[col];
    }

    // threadgroup tree reduction on the 256-thread window
    threadgroup float partial[256];
    partial[tid_in_tg] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid_in_tg < stride) {
            partial[tid_in_tg] += partial[tid_in_tg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = partial[0];
    const float scale = params.alpha * dot / params.d_norm_sq;
    for (uint col = tid_in_tg; col < params.hidden; col += tg_size) {
        slice[col] -= scale * d[col];
    }
}
