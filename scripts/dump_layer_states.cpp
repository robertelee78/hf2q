// dump_layer_states.cpp — ADR-009 Phase 3A: dump per-layer hidden states.
//
// Uses llama.cpp's eval callback to capture "l_out" tensors (end-of-layer
// hidden states) at a specific decode position.
//
// Build:
//   cd /opt/llama.cpp/build
//   g++ -std=c++17 -O2 -I../include -I../ggml/include \
//       /opt/hf2q/scripts/dump_layer_states.cpp \
//       -L./src -L./ggml/src -lllama -lggml -lggml-base -lggml-metal \
//       -framework Foundation -framework Metal -framework MetalKit \
//       -framework Accelerate \
//       -o /opt/hf2q/scripts/dump_layer_states
//
// Usage:
//   scripts/dump_layer_states <gguf_path> <rendered_prompt_file> <target_decode_token> <output_dir>

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

struct dump_state {
    int target_pos;       // seq_pos at which to dump
    int current_pos;      // current seq_pos being evaluated
    std::string out_dir;
    bool active;          // only dump when active
};

static dump_state g_dump;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // yes, we want to observe all tensors

    if (!g_dump.active) return true;

    const char * name = ggml_get_name(t);
    if (!name) return true;

    // We want "l_out", "attn_out", and "kqv_out" tensors
    // kqv_out is the raw SDPA output before O-proj in llama.cpp
    bool is_l_out = (strncmp(name, "l_out", 5) == 0);
    bool is_attn_out = (strncmp(name, "attn_out", 8) == 0);
    bool is_kqv_out = (strncmp(name, "kqv_out", 7) == 0);
    bool is_kqv = (!is_kqv_out && strncmp(name, "kqv", 3) == 0 && (name[3] == '-' || name[3] == '\0'));
    bool is_qcur_pos = (strncmp(name, "Qcur_pos", 8) == 0);
    bool is_kcur_pos = (strncmp(name, "Kcur_pos", 8) == 0);
    if (!is_l_out && !is_attn_out && !is_kqv_out && !is_kqv
        && !is_qcur_pos && !is_kcur_pos) return true;

    // Extract layer number from name: "l_out-0", "attn_out-0", etc.
    int layer = -1;
    const char * dash = strrchr(name, '-');
    if (dash) {
        layer = atoi(dash + 1);
    }
    const char * prefix = is_l_out ? "l_out"
        : is_attn_out ? "attn_out"
        : is_kqv_out ? "kqv_out"
        : is_kqv ? "kqv"
        : is_qcur_pos ? "q_normed"
        : is_kcur_pos ? "k_normed"
        : "unknown";

    // Get tensor data
    int64_t n_elements = ggml_nelements(t);
    size_t n_bytes = n_elements * sizeof(float);

    // Read tensor data to CPU
    std::vector<float> data(n_elements);
    ggml_backend_tensor_get(t, data.data(), 0, n_bytes);

    // Write to file
    char path[512];
    snprintf(path, sizeof(path), "%s/llama_%s_layer%02d_pos%d.bin",
        g_dump.out_dir.c_str(), prefix, layer, g_dump.current_pos);
    FILE * f = fopen(path, "wb");
    if (f) {
        fwrite(data.data(), sizeof(float), n_elements, f);
        fclose(f);
        fprintf(stderr, "[DUMP] %s: %lld f32 -> %s\n", name, (long long)n_elements, path);
    }

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <gguf> <prompt_file> <target_decode_token> <output_dir>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt_file = argv[2];
    int target_decode_token = atoi(argv[3]);
    const char * out_dir = argv[4];

    // Read prompt
    std::ifstream pf(prompt_file);
    std::string prompt((std::istreambuf_iterator<char>(pf)),
                        std::istreambuf_iterator<char>());
    fprintf(stderr, "Prompt: %zu bytes\n", prompt.size());

    // Init llama
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    auto * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;

    auto * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    // Set eval callback
    g_dump.out_dir = out_dir;
    g_dump.active = false;
    g_dump.target_pos = -1;

    // llama_set_eval_callback is not directly available in the public API.
    // Instead, use the context params cb_eval.
    // Since we already created the context, we need to recreate with callback.
    llama_free(ctx);

    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;
    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context with callback\n");
        return 1;
    }

    // Tokenize
    const auto * vocab = llama_model_get_vocab(model);
    int n_prompt_max = prompt.size() + 256;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                   tokens.data(), n_prompt_max, true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "Tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tokens);
    fprintf(stderr, "Tokens: %d\n", n_tokens);

    // Prefill
    fprintf(stderr, "Prefilling %d tokens...\n", n_tokens);
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prefill failed\n");
        return 1;
    }

    // Decode loop
    int n_decoded = 0;
    llama_token prev_token = -1;
    for (int i = 0; i < target_decode_token + 5; i++) {
        // Sample greedy
        auto * logits = llama_get_logits_ith(ctx, -1);
        llama_token best = 0;
        float best_logit = logits[0];
        int n_vocab = llama_vocab_n_tokens(vocab);
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best = v;
            }
        }

        // Check if we should activate dump for next eval
        int seq_pos = n_tokens + i; // position of the next token to be generated
        if (i == target_decode_token - 1) {
            // The NEXT decode call will generate the target token
            g_dump.active = true;
            g_dump.current_pos = n_tokens + i;
            fprintf(stderr, "Activating dump at decode step %d (seq_pos=%d)\n", i+1, n_tokens + i + 1);
        } else {
            g_dump.active = false;
        }

        // Prepare next token
        llama_batch next_batch = llama_batch_get_one(&best, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            fprintf(stderr, "Decode failed at step %d\n", i);
            return 1;
        }

        // Also dump logits at target position
        if (i == target_decode_token) {
            auto * target_logits = llama_get_logits_ith(ctx, -1);
            char lpath[512];
            snprintf(lpath, sizeof(lpath), "%s/llama_logits_pos%d.bin", out_dir, n_tokens + i);
            FILE * f = fopen(lpath, "wb");
            if (f) {
                fwrite(target_logits, sizeof(float), n_vocab, f);
                fclose(f);
                fprintf(stderr, "[DUMP] logits (%d f32) -> %s\n", n_vocab, lpath);
            }

            // Print top-10
            std::vector<std::pair<int, float>> indexed;
            for (int v = 0; v < n_vocab; v++) {
                indexed.push_back({v, target_logits[v]});
            }
            std::sort(indexed.begin(), indexed.end(),
                [](auto & a, auto & b) { return a.second > b.second; });
            fprintf(stderr, "Top-10 logits at seq_pos=%d:\n", n_tokens + i);
            for (int k = 0; k < 10; k++) {
                fprintf(stderr, "  tok=%6d logit=%.6f\n", indexed[k].first, indexed[k].second);
            }

            g_dump.active = false;
        }

        prev_token = best;
        n_decoded++;
    }

    fprintf(stderr, "Decoded %d tokens\n", n_decoded);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
