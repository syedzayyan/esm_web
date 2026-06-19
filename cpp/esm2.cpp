#include "esm2.h"
#include "ggml_common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#if defined(__APPLE__) && !defined(GGML_NO_METAL)
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>

struct esm2_layer {
    ggml_tensor * attn_norm_w;
    ggml_tensor * attn_norm_b;

    ggml_tensor * wq;
    ggml_tensor * bq;
    ggml_tensor * wk;
    ggml_tensor * bk;
    ggml_tensor * wv;
    ggml_tensor * bv;

    ggml_tensor * wo;
    ggml_tensor * bo;

    ggml_tensor * ffn_norm_w;
    ggml_tensor * ffn_norm_b;

    ggml_tensor * ffn_up_w;
    ggml_tensor * ffn_up_b;
    ggml_tensor * ffn_down_w;
    ggml_tensor * ffn_down_b;
};

struct esm2_model {
    // hyperparameters
    int32_t n_vocab = 0;
    int32_t n_embd = 0;
    int32_t n_layer = 0;
    int32_t n_head = 0;
    int32_t n_ff = 0;
    float eps = 1e-5f;
    int32_t mask_token_id = -1;
    bool token_dropout = false;

    ggml_tensor * tok_embd = nullptr;
    std::vector<esm2_layer> layers;
    ggml_tensor * output_norm_w = nullptr;
    ggml_tensor * output_norm_b = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~esm2_model() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

esm2_model * esm2_load(const std::string & path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "esm2_load: failed to open '%s'\n", path.c_str());
        return nullptr;
    }

    auto model = std::make_unique<esm2_model>();

    auto get_u32 = [&](const char * key) -> int32_t {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) throw std::runtime_error(std::string("missing key: ") + key);
        return (int32_t) gguf_get_val_u32(gguf_ctx, idx);
    };
    auto get_f32 = [&](const char * key) -> float {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) throw std::runtime_error(std::string("missing key: ") + key);
        return gguf_get_val_f32(gguf_ctx, idx);
    };

    try {
        model->n_embd = get_u32("esm2.hidden_size");
        model->n_layer = get_u32("esm2.num_layers");
        model->n_head = get_u32("esm2.num_heads");
        model->n_ff = get_u32("esm2.intermediate_size");
        model->n_vocab = get_u32("esm2.vocab_size");
        model->eps = get_f32("esm2.layer_norm_eps");

        int64_t mask_idx = gguf_find_key(gguf_ctx, "esm2.mask_token_id");
        if (mask_idx >= 0) model->mask_token_id = (int32_t) gguf_get_val_u32(gguf_ctx, mask_idx);

        int64_t dropout_idx = gguf_find_key(gguf_ctx, "esm2.token_dropout");
        if (dropout_idx >= 0) model->token_dropout = gguf_get_val_bool(gguf_ctx, dropout_idx);
    } catch (const std::exception & e) {
        fprintf(stderr, "esm2_load: %s\n", e.what());
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return nullptr;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);

    ggml_init_params ctx_params = {
        /*.mem_size   =*/ (size_t) (n_tensors + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    model->ctx = ggml_init(ctx_params);

    std::map<std::string, ggml_tensor *> tensors;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * src = ggml_get_tensor(meta_ctx, name);
        ggml_tensor * cur = ggml_dup_tensor(model->ctx, src);
        ggml_set_name(cur, name);
        tensors[name] = cur;
    }

    // Prefer Metal on Apple Silicon; fall back to CPU everywhere else.
#if defined(__APPLE__) && !defined(GGML_NO_METAL)
    model->backend = ggml_backend_metal_init();
    if (!model->backend) {
        fprintf(stderr, "esm2: Metal init failed, falling back to CPU\n");
    }
#endif
    if (!model->backend) {
        model->backend = ggml_backend_cpu_init();
    }

#if defined(__EMSCRIPTEN__)
    // Single-threaded build: no -pthread (avoids the COOP/COEP header
    // requirements that real Wasm threads need, which static hosting like
    // GitHub Pages / `npx serve` doesn't set).
    ggml_backend_cpu_set_n_threads(model->backend, 1);
#endif

    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "esm2_load: failed to reopen '%s'\n", path.c_str());
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return nullptr;
    }

    const size_t data_offset = gguf_get_data_offset(gguf_ctx);
    std::vector<uint8_t> read_buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * cur = tensors.at(name);
        const size_t offset = gguf_get_tensor_offset(gguf_ctx, i);
        const size_t nbytes = ggml_nbytes(cur);

        read_buf.resize(nbytes);
        fseek(f, (long) (data_offset + offset), SEEK_SET);
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "esm2_load: short read for tensor '%s'\n", name);
            fclose(f);
            gguf_free(gguf_ctx);
            ggml_free(meta_ctx);
            return nullptr;
        }
        ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
    }
    fclose(f);

    auto get_tensor = [&](const std::string & name) -> ggml_tensor * {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return it->second;
    };

    try {
        model->tok_embd = get_tensor("tok_embd.weight");

        model->layers.resize(model->n_layer);
        for (int i = 0; i < model->n_layer; i++) {
            auto & layer = model->layers[i];
            const std::string p = "layers." + std::to_string(i) + ".";

            layer.attn_norm_w = get_tensor(p + "attn_norm.weight");
            layer.attn_norm_b = get_tensor(p + "attn_norm.bias");

            layer.wq = get_tensor(p + "attn_q.weight");
            layer.bq = get_tensor(p + "attn_q.bias");
            layer.wk = get_tensor(p + "attn_k.weight");
            layer.bk = get_tensor(p + "attn_k.bias");
            layer.wv = get_tensor(p + "attn_v.weight");
            layer.bv = get_tensor(p + "attn_v.bias");

            layer.wo = get_tensor(p + "attn_output.weight");
            layer.bo = get_tensor(p + "attn_output.bias");

            layer.ffn_norm_w = get_tensor(p + "ffn_norm.weight");
            layer.ffn_norm_b = get_tensor(p + "ffn_norm.bias");

            layer.ffn_up_w = get_tensor(p + "ffn_up.weight");
            layer.ffn_up_b = get_tensor(p + "ffn_up.bias");
            layer.ffn_down_w = get_tensor(p + "ffn_down.weight");
            layer.ffn_down_b = get_tensor(p + "ffn_down.bias");
        }

        model->output_norm_w = get_tensor("output_norm.weight");
        model->output_norm_b = get_tensor("output_norm.bias");
    } catch (const std::exception & e) {
        fprintf(stderr, "esm2_load: %s\n", e.what());
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return nullptr;
    }

    gguf_free(gguf_ctx);
    ggml_free(meta_ctx);

    return model.release();
}

void esm2_free(esm2_model * model) {
    delete model;
}

int esm2_n_embd(const esm2_model * model) {
    return model->n_embd;
}


std::vector<float> esm2_eval(esm2_model * model, const std::vector<int32_t> & tokens) {
    const int n_tokens = (int) tokens.size();
    const int n_embd = model->n_embd;
    const int n_head = model->n_head;
    const int head_dim = n_embd / n_head;
    const float eps = model->eps;

    const size_t buf_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    ggml_init_params gparams = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(gparams);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // per-token embedding scale, replicating HuggingFace EsmEmbeddings'
    // token_dropout rescale (zeroes <mask> positions and rescales the rest
    // so their expected magnitude matches training-time masking ratios)
    ggml_tensor * inp_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_tokens);
    ggml_set_name(inp_scale, "inp_scale");
    ggml_set_input(inp_scale);

    // word embeddings, no position/type embeddings (rotary attention instead)
    ggml_tensor * cur = ggml_get_rows(ctx, model->tok_embd, inp_tokens); // [n_embd, n_tokens]
    cur = ggml_mul(ctx, cur, inp_scale);

    const float kq_scale = 1.0f / sqrtf((float) head_dim);

    for (int il = 0; il < model->n_layer; il++) {
        const esm2_layer & layer = model->layers[il];

        // pre-norm self-attention block
        ggml_tensor * attn_in = layer_norm(ctx, cur, layer.attn_norm_w, layer.attn_norm_b, eps);

        ggml_tensor * q = linear(ctx, layer.wq, layer.bq, attn_in);
        ggml_tensor * k = linear(ctx, layer.wk, layer.bk, attn_in);
        ggml_tensor * v = linear(ctx, layer.wv, layer.bv, attn_in);

        q = ggml_reshape_3d(ctx, q, head_dim, n_head, n_tokens);
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, n_tokens);
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, n_tokens);

        // rotary position embeddings (NeoX/rotate-half style, base 10000)
        q = ggml_rope_ext(ctx, q, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                           10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                           10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]
        ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, n_tokens, n_head]

        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp); // [n_tokens(k), n_tokens(q), n_head]
        kq = ggml_scale(ctx, kq, kq_scale);
        kq = ggml_soft_max(ctx, kq);

        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n_tokens, head_dim, n_head]
        ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq); // [head_dim, n_tokens(q), n_head]

        ggml_tensor * merged = ggml_permute(ctx, kqv, 0, 2, 1, 3); // [head_dim, n_head, n_tokens]
        merged = ggml_cont(ctx, merged);
        merged = ggml_reshape_2d(ctx, merged, n_embd, n_tokens);

        ggml_tensor * attn_out = linear(ctx, layer.wo, layer.bo, merged);
        cur = ggml_add(ctx, cur, attn_out);

        // pre-norm feed-forward block
        ggml_tensor * ffn_in = layer_norm(ctx, cur, layer.ffn_norm_w, layer.ffn_norm_b, eps);

        ggml_tensor * ff = linear(ctx, layer.ffn_up_w, layer.ffn_up_b, ffn_in);
        ff = ggml_gelu_erf(ctx, ff);
        ff = linear(ctx, layer.ffn_down_w, layer.ffn_down_b, ff);

        cur = ggml_add(ctx, cur, ff);
    }

    cur = layer_norm(ctx, cur, model->output_norm_w, model->output_norm_b, eps);
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; i++) pos[i] = i;

    std::vector<float> scale(n_tokens, 1.0f);
    if (model->token_dropout) {
        const float mask_ratio_train = 0.15f * 0.8f;
        int n_mask = 0;
        for (int32_t t : tokens) {
            if (t == model->mask_token_id) n_mask++;
        }
        const float mask_ratio_observed = (float) n_mask / (float) n_tokens;
        const float s = (1.0f - mask_ratio_train) / (1.0f - mask_ratio_observed);
        for (int i = 0; i < n_tokens; i++) {
            scale[i] = (tokens[i] == model->mask_token_id) ? 0.0f : s;
        }
    }

    ggml_backend_tensor_set(inp_tokens, tokens.data(), 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_scale, scale.data(), 0, n_tokens * sizeof(float));

    ggml_backend_graph_compute(model->backend, gf);

    std::vector<float> result((size_t) n_embd * n_tokens);
    ggml_backend_tensor_get(cur, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    return result;
}
