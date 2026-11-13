#include <metal_stdlib>
using namespace metal;

struct DraftRepetitionPrior {
    uint active;
    uint penalties[4];
    uint counts[4];
    uint history[4][64];
};
struct Geometry {
    uint lanes;
    uint rows_per_lane;
    uint vocabulary;
};

kernel void hf2q_dflash2_proposal_repetition(
    device float *logits [[buffer(0)]],
    constant DraftRepetitionPrior &prior [[buffer(1)]],
    constant Geometry &geometry [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
    const uint slot = position.x;
    const uint row = position.y + 1;
    const uint lane = position.z;
    if (lane >= geometry.lanes || row >= geometry.rows_per_lane ||
        !(prior.active & (1u << lane)) || slot >= prior.counts[lane]) return;
    const uint token = prior.history[lane][slot];
    // One writer per (lane, row, token), including repeated history tokens.
    for (uint earlier = 0; earlier < slot; ++earlier) {
        if (prior.history[lane][earlier] == token) return;
    }
    const uint index = (lane * geometry.rows_per_lane + row) * geometry.vocabulary + token;
    const float value = logits[index];
    const float penalty = as_type<float>(prior.penalties[lane]);
    logits[index] = value > 0.0f ? value / penalty : value * penalty;
}
