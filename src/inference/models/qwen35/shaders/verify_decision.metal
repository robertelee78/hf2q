// verify_decision.metal: the history-token side of a DFlash2 verify row's
// greedy decision (HF2Q_DFLASH2_GPU_DECISIONS).
//
// A host-sampled verify row is decided by a repetition penalty over the
// lane's recent tokens followed by an argmax. The penalty touches at most
// `slots` logits of the row, so the row's decision needs only
//   - the logits of those tokens (gathered here, penalised on the host with
//     the sampler's own arithmetic), and
//   - the largest logit among all other tokens (the stock `argmax_f32`
//     kernel over the row once the history tokens are masked here).
// No arithmetic on a logit happens on the GPU.
//
// Row table, four uints per packed row:
//   [0] packed row of its window's first row (whose input is the lane's
//       pending token; the inputs of the following rows are the drafts)
//   [1] the row's index r in its window
//   [2] element offset of the lane's base history in `base`
//   [3] the base history's token count n0, or 0xffffffff when the row has no
//       decision (nothing is gathered or masked for it)
// The row's history is the last `slots` tokens of
//   base[0..n0] ++ row_tokens[start + 1 ..= start + r],
// oldest first: the sampler's repetition window after the window's first r
// drafts were accepted.

#include <metal_stdlib>
using namespace metal;

struct QwenVerifyDecisionParams {
    uint rows;
    uint vocab;
    uint slots;
    uint row_token_count;
};

static inline uint qwen_verify_history_token(
    device const uint *table,
    device const uint *base,
    device const uint *row_tokens,
    uint row,
    uint slot,
    uint slots,
    uint row_token_count) {
    const uint start = table[4 * row];
    const uint r = table[4 * row + 1];
    const uint offset = table[4 * row + 2];
    const uint n0 = table[4 * row + 3];
    if (n0 == 0xffffffffu) return 0xffffffffu;
    const uint total = n0 + r;
    const uint first = total > slots ? total - slots : 0u;
    const uint element = first + slot;
    if (element >= total) return 0xffffffffu;
    if (element < n0) return base[offset + element];
    const uint tile = start + 1u + (element - n0);
    return tile < row_token_count ? row_tokens[tile] : 0xffffffffu;
}

// Grid (slots, rows), one thread per history slot of a row.
kernel void qwen_verify_decision_gather(
    device const float *logits [[buffer(0)]],
    device const uint *table [[buffer(1)]],
    device const uint *base [[buffer(2)]],
    device const uint *row_tokens [[buffer(3)]],
    device uint *history_tokens [[buffer(4)]],
    device float *history_logits [[buffer(5)]],
    constant QwenVerifyDecisionParams &params [[buffer(6)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint slot = position.x;
    const uint row = position.y;
    if (slot >= params.slots || row >= params.rows) return;
    const uint token = qwen_verify_history_token(
        table, base, row_tokens, row, slot, params.slots, params.row_token_count);
    float value = 0.0f;
    if (token < params.vocab) value = logits[row * params.vocab + token];
    history_tokens[row * params.slots + slot] = token;
    history_logits[row * params.slots + slot] = value;
}

// Grid (slots, rows). Runs behind a barrier after the gather: every history
// token's logit becomes the lowest finite float, so the row's argmax is the
// argmax over the tokens outside the history. Duplicate tokens write the
// same value.
kernel void qwen_verify_decision_mask(
    device float *logits [[buffer(0)]],
    device const uint *history_tokens [[buffer(1)]],
    constant QwenVerifyDecisionParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint slot = position.x;
    const uint row = position.y;
    if (slot >= params.slots || row >= params.rows) return;
    const uint token = history_tokens[row * params.slots + slot];
    if (token < params.vocab)
        logits[row * params.vocab + token] = as_type<float>(0xff7fffffu);
}
