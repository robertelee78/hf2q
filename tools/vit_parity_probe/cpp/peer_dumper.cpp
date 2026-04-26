// peer_dumper.cpp — llama.cpp ViT reference dumper for hf2q parity probe.
//
// ADR-005 Phase 2c iter 124 (W55).
//
// Usage:
//   peer_dumper -m <text-gguf> --mmproj <mmproj-gguf> --image <png>
//               --dump-dir <out>
//
// This binary loads a Gemma 4 model + mmproj via libmtmd, registers an
// eval-callback on the ViT scheduler, and runs the standard
// mtmd_tokenize → mtmd_encode_chunk path on a real PNG image.  Every
// tensor produced by the ViT graph fires the callback; we capture the
// ones whose names are in our parity-stage allowlist and write them to
// disk in the same on-disk format as hf2q's `vit_dump.rs`:
//
//   <name>.bin   — raw F32 LE, contiguous
//   <name>.json  — { "name": ..., "dtype": "f32",
//                    "shape": [...], "n_elements": ... }
//
// Dequantisation: every captured tensor is converted to F32 if it's not
// already F32. Production gemma4v ViT activations are F32 throughout
// (BF16 K cast inside attention is internal), so this should be a
// no-op on real captures, but we guard anyway.
//
// Stages dumped (mapped to hf2q's vit_dump.rs taxonomy):
//
//   hf2q                    | llama.cpp tensor name
//   ────────────────────── ─|──────────────────────
//   00_preprocess           | inp_raw_scaled
//   01_patch_embd           | inp
//   02_pos_embd             | pos_embd
//   03_block_NN             | layer_out-NN  (NN = layer index)
//   30_final_pool           | pooled
//   31_pool_sqrt_scale      | (folded into pooled — gemma4v's pooled
//                              callback fires AFTER ggml_scale)
//   32_std_bias_scale       | std_scaled
//   33_projector            | projected
//   34_post_proj_rms        | projected_normed
//
// Note: `pooled` in gemma4v.cpp captures the tensor AFTER both pool +
// sqrt(n_embd) scale. There's no separate `30_final_pool` checkpoint
// pre-scale on llama.cpp's side. So our `30_final_pool` and
// `31_pool_sqrt_scale` are aliases of the same `pooled` capture; the
// diff binary will compare both hf2q stages against the same llama.cpp
// reference, and divergence between them in hf2q indicates the scale
// itself is broken.
//
// ──────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace fs = std::filesystem;

// ──────────────────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────────────────

struct args {
    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string dump_dir;
};

static int parse_args(int argc, char ** argv, args & out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "ERR: %s requires an argument\n", flag);
                exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (a == "-m"        || a == "--model")    out.model_path  = need_next("-m");
        else if (a == "--mmproj")                       out.mmproj_path = need_next("--mmproj");
        else if (a == "--image")                        out.image_path  = need_next("--image");
        else if (a == "--dump-dir")                     out.dump_dir    = need_next("--dump-dir");
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s -m <text-gguf> --mmproj <mmproj-gguf> "
                   "--image <png> --dump-dir <out>\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "ERR: unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (out.model_path.empty() || out.mmproj_path.empty() ||
        out.image_path.empty() || out.dump_dir.empty()) {
        fprintf(stderr, "ERR: -m, --mmproj, --image, --dump-dir all required\n");
        return 2;
    }
    return 0;
}

// ──────────────────────────────────────────────────────────────────────
// Tensor capture
// ──────────────────────────────────────────────────────────────────────

struct capture_state {
    fs::path dump_dir;
    // Map from llama.cpp tensor name → hf2q stage name.
    // Layer-output entries are matched by prefix and constructed at
    // capture time (e.g. "layer_out-12" → "03_block_12").
    std::unordered_map<std::string, std::string> name_map;
    // Set of stage names already written so we don't double-write if
    // the scheduler revisits a node.
    std::unordered_set<std::string> written;
};

// Convert raw bytes from `tensor` (after ggml_backend_tensor_get) to a
// `std::vector<float>` in the buffer's natural element order. ggml's
// `ne[0]` is the fastest-varying dim, so a contiguous F32 tensor of
// shape `{ne0, ne1, ...}` is laid out exactly like a row-major
// `[ne_last, ..., ne1, ne0]` in programmer's notation.
//
// We support the dtypes that actually appear in gemma4v ViT activations:
//   - F32: passthrough.
//   - F16 / BF16: cast to F32.
//   - quantised: NOT supported here — these are weight tensors, not
//     activations, so they shouldn't appear in cb_eval anyway.
static std::vector<float> tensor_to_f32(const ggml_tensor * t) {
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float> out(n, 0.0f);
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> raw(nb);
    if (ggml_backend_buffer_is_host(t->buffer)) {
        // CPU-resident tensor (e.g. inputs). Copy directly.
        std::memcpy(raw.data(), t->data, nb);
    } else {
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
    }
    switch (t->type) {
        case GGML_TYPE_F32: {
            std::memcpy(out.data(), raw.data(), n * sizeof(float));
            break;
        }
        case GGML_TYPE_F16: {
            const ggml_fp16_t * src = (const ggml_fp16_t *) raw.data();
            for (size_t i = 0; i < n; ++i) {
                out[i] = ggml_fp16_to_fp32(src[i]);
            }
            break;
        }
        case GGML_TYPE_BF16: {
            const ggml_bf16_t * src = (const ggml_bf16_t *) raw.data();
            for (size_t i = 0; i < n; ++i) {
                out[i] = ggml_bf16_to_fp32(src[i]);
            }
            break;
        }
        default: {
            fprintf(stderr,
                "WARN: unsupported tensor dtype for parity dump: %s "
                "(name=%s) — skipping\n",
                ggml_type_name(t->type), ggml_get_name(t));
            return {};
        }
    }
    return out;
}

static std::string ne_string(const ggml_tensor * t) {
    char buf[128];
    int n = ggml_n_dims(t);
    if (n <= 0) n = 1;
    int written = 0;
    for (int i = 0; i < n; ++i) {
        int rc = snprintf(buf + written, sizeof(buf) - written,
                          "%s%lld", i == 0 ? "" : ",",
                          (long long) t->ne[i]);
        if (rc < 0 || (size_t)(written + rc) >= sizeof(buf)) break;
        written += rc;
    }
    return std::string(buf, written);
}

static void write_dump(const fs::path & dir, const std::string & stage_name,
                       const ggml_tensor * t) {
    const auto data = tensor_to_f32(t);
    if (data.empty()) {
        return;  // unsupported dtype warning already printed
    }

    // Build the shape string honouring ggml's ne ordering: ne[0] is the
    // fastest-varying. We emit it in big-end-first programmer order
    // (i.e. `[ne_last, ..., ne1, ne0]`) so the on-disk shape matches
    // the row-major reading convention used by Python/numpy/Rust
    // consumers. This matches hf2q's `[n_patches, hidden]` row-major
    // layout for stages where ggml uses `{hidden, n_patches}` ne.
    int n_dims = ggml_n_dims(t);
    if (n_dims <= 0) n_dims = 1;
    std::string shape_str;
    for (int i = n_dims - 1; i >= 0; --i) {
        if (!shape_str.empty()) shape_str += ",";
        shape_str += std::to_string((long long) t->ne[i]);
    }

    fs::path bin_path  = dir / (stage_name + ".bin");
    fs::path json_path = dir / (stage_name + ".json");

    {
        std::ofstream f(bin_path, std::ios::binary);
        f.write((const char *) data.data(), data.size() * sizeof(float));
    }
    {
        std::ofstream f(json_path);
        f << "{\"name\":\"" << stage_name
          << "\",\"dtype\":\"f32\""
          << ",\"shape\":[" << shape_str << "]"
          << ",\"n_elements\":" << data.size()
          << ",\"src_name\":\"" << ggml_get_name(t) << "\""
          << ",\"src_ne\":\"" << ne_string(t) << "\""
          << ",\"src_dtype\":\"" << ggml_type_name(t->type) << "\""
          << "}\n";
    }

    fprintf(stderr,
            "[peer_dumper] wrote stage=%-22s src=%-22s ne=%s dtype=%s n=%zu\n",
            stage_name.c_str(), ggml_get_name(t),
            ne_string(t).c_str(), ggml_type_name(t->type), data.size());
}

// Match ggml tensor name to a hf2q stage name. Returns "" when not in
// the parity-stage allowlist.
static std::string map_to_stage(const std::string & ggml_name,
                                const capture_state & st) {
    auto direct = st.name_map.find(ggml_name);
    if (direct != st.name_map.end()) {
        return direct->second;
    }
    // Layer outputs: "layer_out-NN" → "03_block_NN" (zero-padded to 2
    // digits to match hf2q's `format!("03_block_{:02}", idx)`).
    static const char prefix[] = "layer_out-";
    if (ggml_name.rfind(prefix, 0) == 0) {
        const std::string idx_str = ggml_name.substr(sizeof(prefix) - 1);
        if (!idx_str.empty()) {
            char buf[32];
            // idx_str is a base-10 integer in the source.
            int idx = std::atoi(idx_str.c_str());
            std::snprintf(buf, sizeof(buf), "03_block_%02d", idx);
            return std::string(buf);
        }
    }
    return {};
}

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (ask) {
        // Tell the scheduler we're interested in EVERY node so the
        // post-eval pass fires for everything; we filter inside the
        // post-eval branch below. The cost is one ggml_backend_tensor_get
        // per node — negligible on a 27-block ViT for one image.
        return true;
    }
    if (!t || !t->name[0]) return true;
    const std::string ggml_name(t->name);
    const std::string stage = map_to_stage(ggml_name, *st);
    if (stage.empty()) return true;
    if (st->written.count(stage)) return true;  // already captured
    write_dump(st->dump_dir, stage, t);
    st->written.insert(stage);
    return true;
}

// ──────────────────────────────────────────────────────────────────────
// PNG loader (uses stb_image via mtmd-helper)
// ──────────────────────────────────────────────────────────────────────

static mtmd_bitmap * load_image_bitmap(const std::string & path) {
    // Use libmtmd's helper which wraps stb_image; the helper accepts a
    // file path and returns a fully-formed mtmd_bitmap (RGB8).
    return mtmd_helper_bitmap_init_from_file(nullptr, path.c_str());
}

// ──────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    args A;
    if (parse_args(argc, argv, A) != 0) return 2;
    fs::create_directories(A.dump_dir);

    capture_state st;
    st.dump_dir = A.dump_dir;
    st.name_map = {
        { "inp_raw_scaled", "00_preprocess"      },
        { "inp",            "01_patch_embd"      },
        { "pos_embd",       "02_pos_embd"        },
        // 03_block_NN handled by prefix match in map_to_stage().
        { "pooled",         "30_final_pool"      },  // post-pool + sqrt
        { "std_scaled",     "32_std_bias_scale"  },
        { "projected",      "33_projector"       },
        { "projected_normed","34_post_proj_rms"  },
    };

    // ── Load the text model so libmtmd can probe vocab/template etc.
    // We never decode any text — we only need a valid llama_model
    // handle for `mtmd_init_from_file`.
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // Tiny CPU footprint; we never decode text.
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;
    llama_model * model = llama_model_load_from_file(A.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "ERR: llama_model_load_from_file failed for %s\n",
                A.model_path.c_str());
        return 1;
    }

    // ── Construct mtmd context with our cb_eval hook.
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu          = true;
    cp.print_timings    = false;
    cp.n_threads        = 1;
    cp.warmup           = false;
    cp.cb_eval          = eval_callback;
    cp.cb_eval_user_data = &st;
    mtmd_context * ctx = mtmd_init_from_file(A.mmproj_path.c_str(), model, cp);
    if (!ctx) {
        fprintf(stderr, "ERR: mtmd_init_from_file failed for %s\n",
                A.mmproj_path.c_str());
        llama_model_free(model);
        return 1;
    }
    if (!mtmd_support_vision(ctx)) {
        fprintf(stderr, "ERR: mmproj does not support vision\n");
        mtmd_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // ── Load the image into a bitmap.
    mtmd_bitmap * bmp = load_image_bitmap(A.image_path);
    if (!bmp) {
        fprintf(stderr, "ERR: load_image_bitmap failed for %s\n",
                A.image_path.c_str());
        mtmd_free(ctx);
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr,
            "[peer_dumper] loaded image %s (%ux%u)\n",
            A.image_path.c_str(),
            mtmd_bitmap_get_nx(bmp), mtmd_bitmap_get_ny(bmp));

    // ── Tokenize a single image marker prompt so libmtmd produces an
    // image chunk we can encode. The text content is meaningless — we
    // never decode it, only the image chunk matters.
    const std::string prompt = std::string(mtmd_default_marker());
    mtmd_input_text it{};
    it.text         = prompt.c_str();
    it.add_special  = false;
    it.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bmp };
    int rc = mtmd_tokenize(ctx, chunks, &it, bitmaps, 1);
    if (rc != 0) {
        fprintf(stderr, "ERR: mtmd_tokenize rc=%d\n", rc);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // ── Find the image chunk and encode it. cb_eval fires on every
    // named ViT node and writes our parity dumps.
    bool encoded = false;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            int erc = mtmd_encode_chunk(ctx, c);
            if (erc != 0) {
                fprintf(stderr, "ERR: mtmd_encode_chunk rc=%d\n", erc);
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                mtmd_free(ctx);
                llama_model_free(model);
                return 1;
            }
            encoded = true;
            break;
        }
    }
    if (!encoded) {
        fprintf(stderr, "ERR: no image chunk produced by mtmd_tokenize\n");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        mtmd_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr,
            "[peer_dumper] encoded successfully; %zu stages written to %s\n",
            st.written.size(), A.dump_dir.c_str());

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
