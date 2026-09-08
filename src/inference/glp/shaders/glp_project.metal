// glp_project.metal — per-row projection of activations off a direction.
//
// For each row m of a `[M, N]` row-major f32 activation matrix:
//
//   row_m <- row_m - alpha * (row_m . d) * d / ‖d‖²
//
// where `d` is the GLP direction for the layer. The host passes ‖d‖² so
// the kernel does not require a normalized vector. One thread per element;
// the per-row dot is cooperatively reduced inside a threadgroup of 256,
// then every thread applies the same scale-subtract to its own element.
// (One threadgroup per row is NOT assumed: the grid is flat over M*N and
// the threadgroup map may straddle rows; the reduction is therefore
// restricted to the row-local segment via the row's own thread ids.)
//
// NOTE: this kernel is correct when each threadgroup covers exactly one
// full row (tg_size == n). The host must dispatch with
// threads_per_threadgroup = n (hidden width), which for our families is
// 2048 (Qwen) — the Metal maximum threadgroup size. Rows are processed
// one per threadgroup.

#include <metal_stdlib>
using namespace metal;

struct GlpProjectParams {
    uint m;
    uint n;
    float alpha;
    float d_norm_sq;
};

kernel void glp_project_f32(
    constant GlpProjectParams& params [[buffer(0)]],
    device const float*          direction [[buffer(1)]],
    device float*                hidden [[buffer(2)]],
    uint                         tid_in_tg [[thread_position_in_threadgroup]],
    uint                         tg_size [[threads_per_threadgroup]],
    uint                         tg_id [[threadgroup_position_in_grid]]
) {
    const uint row = tg_id;  // one threadgroup per row
    if (row >= params.m) {
        return;
    }
    device float* row_ptr = hidden + row * params.n;
    // accumulate strided elements
    float local_dot = 0.0f;
    for (uint col = tid_in_tg; col < params.n; col += tg_size) {
        local_dot += row_ptr[col] * direction[col];
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
    float scale = params.alpha * dot / params.d_norm_sq;
    for (uint col = tid_in_tg; col < params.n; col += tg_size) {
        row_ptr[col] -= scale * direction[col];
    }
}
