#include "esmfold_trunk.h"
#include "ggml_common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ── per-block weight pointers ──────────────────────────────────────────────
struct trunk_block {
    // sequence attention
    ggml_tensor * seq_ln_w, * seq_ln_b;
    ggml_tensor * seq_attn_qkv;          // [3*c_s, c_s] no bias
    ggml_tensor * seq_attn_g_w, * seq_attn_g_b;
    ggml_tensor * seq_attn_o_w, * seq_attn_o_b;
    // sequence MLP
    ggml_tensor * mlp_seq_ln_w, * mlp_seq_ln_b;
    ggml_tensor * mlp_seq_fc1_w, * mlp_seq_fc1_b;
    ggml_tensor * mlp_seq_fc2_w, * mlp_seq_fc2_b;
    // pair-to-sequence
    ggml_tensor * p2s_ln_w, * p2s_ln_b;
    ggml_tensor * p2s_linear_w;          // [seq_heads, c_z] no bias
    // sequence-to-pair
    ggml_tensor * s2p_ln_w, * s2p_ln_b;
    ggml_tensor * s2p_proj_w, * s2p_proj_b; // [2*inner, c_s]
    ggml_tensor * s2p_o_w, * s2p_o_b;
    // triangle multiplicative update (outgoing)
    ggml_tensor * tmo_ln_in_w, * tmo_ln_in_b;
    ggml_tensor * tmo_ln_out_w, * tmo_ln_out_b;
    ggml_tensor * tmo_ap_w, * tmo_ap_b;
    ggml_tensor * tmo_ag_w, * tmo_ag_b;
    ggml_tensor * tmo_bp_w, * tmo_bp_b;
    ggml_tensor * tmo_bg_w, * tmo_bg_b;
    ggml_tensor * tmo_z_w, * tmo_z_b;
    ggml_tensor * tmo_g_w, * tmo_g_b;
    // triangle multiplicative update (incoming)
    ggml_tensor * tmi_ln_in_w, * tmi_ln_in_b;
    ggml_tensor * tmi_ln_out_w, * tmi_ln_out_b;
    ggml_tensor * tmi_ap_w, * tmi_ap_b;
    ggml_tensor * tmi_ag_w, * tmi_ag_b;
    ggml_tensor * tmi_bp_w, * tmi_bp_b;
    ggml_tensor * tmi_bg_w, * tmi_bg_b;
    ggml_tensor * tmi_z_w, * tmi_z_b;
    ggml_tensor * tmi_g_w, * tmi_g_b;
    // triangle attention (starting)
    ggml_tensor * tas_ln_w, * tas_ln_b;
    ggml_tensor * tas_bias_w;           // [pair_heads, c_z] no bias
    ggml_tensor * tas_q_w, * tas_k_w, * tas_v_w;   // no bias
    ggml_tensor * tas_g_w, * tas_g_b;
    ggml_tensor * tas_o_w, * tas_o_b;
    // triangle attention (ending)
    ggml_tensor * tae_ln_w, * tae_ln_b;
    ggml_tensor * tae_bias_w;
    ggml_tensor * tae_q_w, * tae_k_w, * tae_v_w;
    ggml_tensor * tae_g_w, * tae_g_b;
    ggml_tensor * tae_o_w, * tae_o_b;
    // pair MLP
    ggml_tensor * mlp_pair_ln_w, * mlp_pair_ln_b;
    ggml_tensor * mlp_pair_fc1_w, * mlp_pair_fc1_b;
    ggml_tensor * mlp_pair_fc2_w, * mlp_pair_fc2_b;
};

struct esmfold_trunk_model {
    int32_t num_blocks  = 0;
    int32_t c_s         = 0;  // sequence_state_dim
    int32_t c_z         = 0;  // pairwise_state_dim
    int32_t seq_hw      = 0;  // sequence_head_width
    int32_t pair_hw     = 0;  // pairwise_head_width
    int32_t pos_bins    = 0;  // position_bins
    float   eps         = 1e-5f;

    // top-level weights
    ggml_tensor * relpos_w       = nullptr; // [66, c_z]
    ggml_tensor * recycle_s_w    = nullptr;
    ggml_tensor * recycle_s_b    = nullptr;
    ggml_tensor * recycle_z_w    = nullptr;
    ggml_tensor * recycle_z_b    = nullptr;

    std::vector<trunk_block> blocks;

    ggml_context      * ctx     = nullptr;
    ggml_backend_t      backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~esmfold_trunk_model() {
        if (buffer)  ggml_backend_buffer_free(buffer);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// ── load ──────────────────────────────────────────────────────────────────
esmfold_trunk_model * esmfold_trunk_load(const std::string & path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { true, &meta_ctx };
    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gp);
    if (!gguf_ctx) { fprintf(stderr, "esmfold_trunk_load: cannot open '%s'\n", path.c_str()); return nullptr; }

    auto model = std::make_unique<esmfold_trunk_model>();

    auto get_u32 = [&](const char * k) -> int32_t {
        int64_t idx = gguf_find_key(gguf_ctx, k);
        if (idx < 0) throw std::runtime_error(std::string("missing key: ") + k);
        return (int32_t) gguf_get_val_u32(gguf_ctx, idx);
    };
    auto get_f32 = [&](const char * k) -> float {
        int64_t idx = gguf_find_key(gguf_ctx, k);
        if (idx < 0) throw std::runtime_error(std::string("missing key: ") + k);
        return gguf_get_val_f32(gguf_ctx, idx);
    };

    try {
        model->num_blocks = get_u32("esmfold.num_blocks");
        model->c_s        = get_u32("esmfold.seq_state_dim");
        model->c_z        = get_u32("esmfold.pair_state_dim");
        model->seq_hw     = get_u32("esmfold.seq_head_width");
        model->pair_hw    = get_u32("esmfold.pair_head_width");
        model->pos_bins   = get_u32("esmfold.position_bins");
        model->eps        = get_f32("esmfold.layer_norm_eps");
    } catch (const std::exception & e) {
        fprintf(stderr, "esmfold_trunk_load: %s\n", e.what());
        gguf_free(gguf_ctx); ggml_free(meta_ctx); return nullptr;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    ggml_init_params ctx_params = {
        (size_t)(n_tensors + 1) * ggml_tensor_overhead(), nullptr, true
    };
    model->ctx = ggml_init(ctx_params);

    std::map<std::string, ggml_tensor*> tensors;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * src = ggml_get_tensor(meta_ctx, name);
        ggml_tensor * cur = ggml_dup_tensor(model->ctx, src);
        ggml_set_name(cur, name);
        tensors[name] = cur;
    }

    model->backend = ggml_backend_cpu_init();
    model->buffer  = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "esmfold_trunk_load: cannot reopen '%s'\n", path.c_str());
              gguf_free(gguf_ctx); ggml_free(meta_ctx); return nullptr; }
    const size_t data_off = gguf_get_data_offset(gguf_ctx);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * cur = tensors.at(name);
        size_t off = gguf_get_tensor_offset(gguf_ctx, i);
        size_t nb  = ggml_nbytes(cur);
        buf.resize(nb);
        fseek(f, (long)(data_off + off), SEEK_SET);
        if (fread(buf.data(), 1, nb, f) != nb) {
            fprintf(stderr, "esmfold_trunk_load: short read '%s'\n", name);
            fclose(f); gguf_free(gguf_ctx); ggml_free(meta_ctx); return nullptr;
        }
        ggml_backend_tensor_set(cur, buf.data(), 0, nb);
    }
    fclose(f);
    gguf_free(gguf_ctx); ggml_free(meta_ctx);

    auto gt = [&](const std::string & name) -> ggml_tensor * {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    };

    try {
        model->relpos_w    = gt("relpos.weight");
        model->recycle_s_w = gt("recycle_s_norm.weight");
        model->recycle_s_b = gt("recycle_s_norm.bias");
        model->recycle_z_w = gt("recycle_z_norm.weight");
        model->recycle_z_b = gt("recycle_z_norm.bias");

        model->blocks.resize(model->num_blocks);
        for (int i = 0; i < model->num_blocks; i++) {
            trunk_block & bl = model->blocks[i];
            const std::string g = "layers." + std::to_string(i);
            bl.seq_ln_w       = gt(g+".seq_ln.weight");
            bl.seq_ln_b       = gt(g+".seq_ln.bias");
            bl.seq_attn_qkv   = gt(g+".seq_attn.proj.weight");
            bl.seq_attn_g_w   = gt(g+".seq_attn.g_proj.weight");
            bl.seq_attn_g_b   = gt(g+".seq_attn.g_proj.bias");
            bl.seq_attn_o_w   = gt(g+".seq_attn.o_proj.weight");
            bl.seq_attn_o_b   = gt(g+".seq_attn.o_proj.bias");
            bl.mlp_seq_ln_w   = gt(g+".mlp_seq.ln.weight");
            bl.mlp_seq_ln_b   = gt(g+".mlp_seq.ln.bias");
            bl.mlp_seq_fc1_w  = gt(g+".mlp_seq.fc1.weight");
            bl.mlp_seq_fc1_b  = gt(g+".mlp_seq.fc1.bias");
            bl.mlp_seq_fc2_w  = gt(g+".mlp_seq.fc2.weight");
            bl.mlp_seq_fc2_b  = gt(g+".mlp_seq.fc2.bias");
            bl.p2s_ln_w       = gt(g+".p2s.ln.weight");
            bl.p2s_ln_b       = gt(g+".p2s.ln.bias");
            bl.p2s_linear_w   = gt(g+".p2s.linear.weight");
            bl.s2p_ln_w       = gt(g+".s2p.ln.weight");
            bl.s2p_ln_b       = gt(g+".s2p.ln.bias");
            bl.s2p_proj_w     = gt(g+".s2p.proj.weight");
            bl.s2p_proj_b     = gt(g+".s2p.proj.bias");
            bl.s2p_o_w        = gt(g+".s2p.o_proj.weight");
            bl.s2p_o_b        = gt(g+".s2p.o_proj.bias");
            bl.tmo_ln_in_w    = gt(g+".tri_mul_out.ln_in.weight");
            bl.tmo_ln_in_b    = gt(g+".tri_mul_out.ln_in.bias");
            bl.tmo_ln_out_w   = gt(g+".tri_mul_out.ln_out.weight");
            bl.tmo_ln_out_b   = gt(g+".tri_mul_out.ln_out.bias");
            bl.tmo_ap_w       = gt(g+".tri_mul_out.a_p.weight");
            bl.tmo_ap_b       = gt(g+".tri_mul_out.a_p.bias");
            bl.tmo_ag_w       = gt(g+".tri_mul_out.a_g.weight");
            bl.tmo_ag_b       = gt(g+".tri_mul_out.a_g.bias");
            bl.tmo_bp_w       = gt(g+".tri_mul_out.b_p.weight");
            bl.tmo_bp_b       = gt(g+".tri_mul_out.b_p.bias");
            bl.tmo_bg_w       = gt(g+".tri_mul_out.b_g.weight");
            bl.tmo_bg_b       = gt(g+".tri_mul_out.b_g.bias");
            bl.tmo_z_w        = gt(g+".tri_mul_out.linear_z.weight");
            bl.tmo_z_b        = gt(g+".tri_mul_out.linear_z.bias");
            bl.tmo_g_w        = gt(g+".tri_mul_out.linear_g.weight");
            bl.tmo_g_b        = gt(g+".tri_mul_out.linear_g.bias");
            bl.tmi_ln_in_w    = gt(g+".tri_mul_in.ln_in.weight");
            bl.tmi_ln_in_b    = gt(g+".tri_mul_in.ln_in.bias");
            bl.tmi_ln_out_w   = gt(g+".tri_mul_in.ln_out.weight");
            bl.tmi_ln_out_b   = gt(g+".tri_mul_in.ln_out.bias");
            bl.tmi_ap_w       = gt(g+".tri_mul_in.a_p.weight");
            bl.tmi_ap_b       = gt(g+".tri_mul_in.a_p.bias");
            bl.tmi_ag_w       = gt(g+".tri_mul_in.a_g.weight");
            bl.tmi_ag_b       = gt(g+".tri_mul_in.a_g.bias");
            bl.tmi_bp_w       = gt(g+".tri_mul_in.b_p.weight");
            bl.tmi_bp_b       = gt(g+".tri_mul_in.b_p.bias");
            bl.tmi_bg_w       = gt(g+".tri_mul_in.b_g.weight");
            bl.tmi_bg_b       = gt(g+".tri_mul_in.b_g.bias");
            bl.tmi_z_w        = gt(g+".tri_mul_in.linear_z.weight");
            bl.tmi_z_b        = gt(g+".tri_mul_in.linear_z.bias");
            bl.tmi_g_w        = gt(g+".tri_mul_in.linear_g.weight");
            bl.tmi_g_b        = gt(g+".tri_mul_in.linear_g.bias");
            bl.tas_ln_w       = gt(g+".tri_att_start.ln.weight");
            bl.tas_ln_b       = gt(g+".tri_att_start.ln.bias");
            bl.tas_bias_w     = gt(g+".tri_att_start.bias_proj.weight");
            bl.tas_q_w        = gt(g+".tri_att_start.mha.q.weight");
            bl.tas_k_w        = gt(g+".tri_att_start.mha.k.weight");
            bl.tas_v_w        = gt(g+".tri_att_start.mha.v.weight");
            bl.tas_g_w        = gt(g+".tri_att_start.mha.g.weight");
            bl.tas_g_b        = gt(g+".tri_att_start.mha.g.bias");
            bl.tas_o_w        = gt(g+".tri_att_start.mha.o.weight");
            bl.tas_o_b        = gt(g+".tri_att_start.mha.o.bias");
            bl.tae_ln_w       = gt(g+".tri_att_end.ln.weight");
            bl.tae_ln_b       = gt(g+".tri_att_end.ln.bias");
            bl.tae_bias_w     = gt(g+".tri_att_end.bias_proj.weight");
            bl.tae_q_w        = gt(g+".tri_att_end.mha.q.weight");
            bl.tae_k_w        = gt(g+".tri_att_end.mha.k.weight");
            bl.tae_v_w        = gt(g+".tri_att_end.mha.v.weight");
            bl.tae_g_w        = gt(g+".tri_att_end.mha.g.weight");
            bl.tae_g_b        = gt(g+".tri_att_end.mha.g.bias");
            bl.tae_o_w        = gt(g+".tri_att_end.mha.o.weight");
            bl.tae_o_b        = gt(g+".tri_att_end.mha.o.bias");
            bl.mlp_pair_ln_w  = gt(g+".mlp_pair.ln.weight");
            bl.mlp_pair_ln_b  = gt(g+".mlp_pair.ln.bias");
            bl.mlp_pair_fc1_w = gt(g+".mlp_pair.fc1.weight");
            bl.mlp_pair_fc1_b = gt(g+".mlp_pair.fc1.bias");
            bl.mlp_pair_fc2_w = gt(g+".mlp_pair.fc2.weight");
            bl.mlp_pair_fc2_b = gt(g+".mlp_pair.fc2.bias");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "esmfold_trunk_load: %s\n", e.what());
        return nullptr;
    }

    return model.release();
}

void esmfold_trunk_free(esmfold_trunk_model * model) { delete model; }
int  esmfold_trunk_c_s(const esmfold_trunk_model * m) { return m->c_s; }
int  esmfold_trunk_c_z(const esmfold_trunk_model * m) { return m->c_z; }

// ── graph sub-operations ───────────────────────────────────────────────────

// Gated sequence self-attention with external pair bias.
// s   : [c_s, L]     (ggml convention: ne[0]=c_s, ne[1]=L)
// z   : [c_z, L, L]
// bias_proj_w : [seq_heads, c_z]  ->  pair bias [seq_heads, L, L]
// Returns: [c_s, L]
static ggml_tensor * seq_attention(
        ggml_context * ctx,
        ggml_tensor  * s,
        ggml_tensor  * z,
        ggml_tensor  * ln_w, ggml_tensor * ln_b,
        ggml_tensor  * qkv_w,                      // [3*c_s, c_s]
        ggml_tensor  * g_w,  ggml_tensor * g_b,
        ggml_tensor  * o_w,  ggml_tensor * o_b,
        ggml_tensor  * p2s_ln_w, ggml_tensor * p2s_ln_b,
        ggml_tensor  * p2s_w,                      // [seq_heads, c_z]
        float eps, int c_s, int seq_hw) {
    const int n_head = c_s / seq_hw;
    const int L = (int)s->ne[1];
    const float scale = 1.0f / sqrtf((float)seq_hw);

    // pair-to-sequence bias: LayerNorm(z) @ p2s_w^T -> [seq_heads, L, L]
    // z shape: [c_z, L, L]  (ne[0]=c_z, ne[1]=L, ne[2]=L)
    ggml_tensor * zn = layer_norm(ctx, z, p2s_ln_w, p2s_ln_b, eps);
    // zn: [c_z, L, L];  p2s_w: [seq_heads, c_z]
    // mul_mat(p2s_w, zn) contracts ne[0]=c_z -> result [seq_heads, L, L]
    ggml_tensor * bias = ggml_mul_mat(ctx, p2s_w, zn); // [seq_heads, L, L]

    // sequence LN + QKV projection
    ggml_tensor * sn = layer_norm(ctx, s, ln_w, ln_b, eps); // [c_s, L]
    ggml_tensor * qkv = ggml_mul_mat(ctx, qkv_w, sn);       // [3*c_s, L]
    // split into Q, K, V each [c_s, L] via views
    ggml_tensor * q = ggml_view_2d(ctx, qkv, c_s, L, qkv->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx, qkv, c_s, L, qkv->nb[1], (size_t)c_s * sizeof(float));
    ggml_tensor * v = ggml_view_2d(ctx, qkv, c_s, L, qkv->nb[1], (size_t)2*c_s * sizeof(float));

    // reshape to [seq_hw, n_head, L] then permute to [seq_hw, L, n_head]
    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), seq_hw, n_head, L);
    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), seq_hw, n_head, L);
    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), seq_hw, n_head, L);
    // permute(0,2,1,3) in 3d: (seq_hw, n_head, L) -> (seq_hw, L, n_head)
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [seq_hw, L, n_head]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // attention: scores = Q^T K / sqrt(hw)  ->  [L_k, L_q, n_head]
    ggml_tensor * kc = ggml_cont(ctx, k);
    ggml_tensor * scores = ggml_mul_mat(ctx, kc, q); // [L, L, n_head]
    scores = ggml_scale(ctx, scores, scale);
    // add pair bias: bias is [seq_heads, L, L], scores is [L, L, n_head]
    // bias needs permute to [L, L, seq_heads] = same axes but reordered
    ggml_tensor * bias_t = ggml_cont(ctx, ggml_permute(ctx, bias, 1, 2, 0, 3)); // [L, L, seq_heads]
    scores = ggml_add(ctx, scores, bias_t);
    scores = ggml_soft_max(ctx, scores); // [L, L, n_head]

    // context = scores @ V:  V is [seq_hw, L, n_head], transpose for mul_mat
    ggml_tensor * vc = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [L, seq_hw, n_head]
    ggml_tensor * attn_out = ggml_mul_mat(ctx, vc, scores);               // [seq_hw, L, n_head]
    // permute back to [seq_hw, n_head, L] then reshape to [c_s, L]
    attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3); // [seq_hw, n_head, L]
    attn_out = ggml_cont(ctx, attn_out);
    attn_out = ggml_reshape_2d(ctx, attn_out, c_s, L);  // [c_s, L]

    // gating: sigmoid(g_proj(s)) * attn_out
    ggml_tensor * gate = ggml_sigmoid(ctx, linear(ctx, g_w, g_b, s)); // [c_s, L]
    attn_out = ggml_mul(ctx, gate, attn_out);

    // output projection
    return linear(ctx, o_w, o_b, attn_out); // [c_s, L]
}

// ResidueMLP: x + fc2(ReLU(fc1(LayerNorm(x))))
static ggml_tensor * residue_mlp(
        ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * ln_w, ggml_tensor * ln_b,
        ggml_tensor * fc1_w, ggml_tensor * fc1_b,
        ggml_tensor * fc2_w, ggml_tensor * fc2_b,
        float eps) {
    ggml_tensor * h = layer_norm(ctx, x, ln_w, ln_b, eps);
    h = linear(ctx, fc1_w, fc1_b, h);
    h = ggml_relu(ctx, h);
    h = linear(ctx, fc2_w, fc2_b, h);
    return ggml_add(ctx, x, h);
}

// SequenceToPair: outer-product + difference of projected sequence state -> pair delta.
// s    : [c_s, L]
// Returns: [c_z, L, L]
static ggml_tensor * sequence_to_pair(
        ggml_context * ctx, ggml_tensor * s,
        ggml_tensor * ln_w, ggml_tensor * ln_b,
        ggml_tensor * proj_w, ggml_tensor * proj_b, // [2*inner, c_s]
        ggml_tensor * o_w,    ggml_tensor * o_b,    // [c_z, 2*inner]
        float eps, int /*c_s*/, int c_z, int L) {
    const int inner = c_z / 2;

    ggml_tensor * sn = layer_norm(ctx, s, ln_w, ln_b, eps); // [c_s, L]
    ggml_tensor * qk = linear(ctx, proj_w, proj_b, sn);      // [2*inner, L]
    // split: q = qk[0:inner,:], k = qk[inner:2*inner,:]
    ggml_tensor * q = ggml_view_2d(ctx, qk, inner, L, qk->nb[1], 0);
    ggml_tensor * k = ggml_view_2d(ctx, qk, inner, L, qk->nb[1], (size_t)inner * sizeof(float));
    q = ggml_cont(ctx, q); // [inner, L]
    k = ggml_cont(ctx, k); // [inner, L]

    // prod[j,i,c] = q[c,i] * k[c,j]  via batched outer product.
    // Treat `inner` as batch (ne2). Reshape q,k [inner,L] -> [1,L,inner], then
    // mul_mat(qb,kb) contracts ne0=1 per batch -> [L,L,inner].
    ggml_tensor * qt = ggml_cont(ctx, ggml_transpose(ctx, q)); // [L, inner]
    ggml_tensor * kt = ggml_cont(ctx, ggml_transpose(ctx, k)); // [L, inner]
    ggml_tensor * qb = ggml_reshape_3d(ctx, qt, 1, L, inner);  // [1, L, inner]
    ggml_tensor * kb = ggml_reshape_3d(ctx, kt, 1, L, inner);  // [1, L, inner]

    ggml_tensor * prod = ggml_mul_mat(ctx, qb, kb); // [L(j), L(i), inner]
    // diff[j,i,c] = q[c,i] - k[c,j]
    ggml_tensor * q_rep = ggml_repeat(ctx, qb, prod);
    ggml_tensor * k_rep = ggml_repeat(ctx, ggml_reshape_3d(ctx, kt, L, 1, inner), prod);
    ggml_tensor * diff  = ggml_sub(ctx, q_rep, k_rep); // [L,L,inner]

    // cat([prod, diff], dim=ne2) -> [L, L, 2*inner]
    ggml_tensor * cat = ggml_concat(ctx, prod, diff, 2); // [L, L, 2*inner]

    // o_proj: [c_z, 2*inner] x [L, L, 2*inner] -> [c_z, L, L]
    // mul_mat(o_w, cat): o_w=[c_z, 2*inner], cat=[2*inner, L, L] -> [c_z, L, L] ✓
    // but cat has ne0=L, ne1=L, ne2=2*inner - need to permute so ne0=2*inner
    ggml_tensor * cat_p = ggml_cont(ctx, ggml_permute(ctx, cat, 2, 0, 1, 3)); // [2*inner, L, L]
    return linear(ctx, o_w, o_b, cat_p); // [c_z, L, L]
}

// TriangleMultiplicativeUpdate (outgoing or incoming).
// z : [c_z, L, L]   returns [c_z, L, L]
// outgoing=true:  result[c,i,j] = sum_k  a[c,i,k] * b[c,j,k]
// outgoing=false: result[c,i,j] = sum_k  a[c,k,i] * b[c,k,j]
static ggml_tensor * triangle_mul(
        ggml_context * ctx, ggml_tensor * z, bool outgoing,
        ggml_tensor * ln_in_w, ggml_tensor * ln_in_b,
        ggml_tensor * ln_out_w, ggml_tensor * ln_out_b,
        ggml_tensor * ap_w, ggml_tensor * ap_b,
        ggml_tensor * ag_w, ggml_tensor * ag_b,
        ggml_tensor * bp_w, ggml_tensor * bp_b,
        ggml_tensor * bg_w, ggml_tensor * bg_b,
        ggml_tensor * lz_w, ggml_tensor * lz_b,
        ggml_tensor * lg_w, ggml_tensor * lg_b,
        float eps, int /*c_z*/, int /*L*/) {
    ggml_tensor * zn = layer_norm(ctx, z, ln_in_w, ln_in_b, eps);
    ggml_tensor * a = ggml_mul(ctx, ggml_sigmoid(ctx, linear(ctx, ag_w, ag_b, zn)),
                                    linear(ctx, ap_w, ap_b, zn)); // [c_z, L, L]
    ggml_tensor * b = ggml_mul(ctx, ggml_sigmoid(ctx, linear(ctx, bg_w, bg_b, zn)),
                                    linear(ctx, bp_w, bp_b, zn)); // [c_z, L, L]

    // Batched matmul over c_z channels.
    // a,b shapes: ne0=c_z, ne1=L, ne2=L  (ggml stores [c_z, L(col), L(row)])
    // Interpreting: a[c, j, i] in memory order = a_{i,j,c} conceptually.
    // We want:
    //   outgoing: x[c,i,j] = sum_k a[c,i,k] * b[c,j,k]  = A_i dot B_j over k (per channel c)
    //   incoming: x[c,i,j] = sum_k a[c,k,i] * b[c,k,j]  = A^T_i dot B^T_j
    //
    // ggml tensor layout: ne[0]=c_z (fastest), ne[1]=L, ne[2]=L.
    // To treat c_z as the batch dimension, permute to [L, L, c_z]:
    ggml_tensor * ap = ggml_cont(ctx, ggml_permute(ctx, a, 1, 2, 0, 3)); // [L, L, c_z]
    ggml_tensor * bp = ggml_cont(ctx, ggml_permute(ctx, b, 1, 2, 0, 3)); // [L, L, c_z]

    // Now ap[row=ne1, col=ne0, batch=ne2] = [L, L, c_z]
    // For outgoing: result[i,j,c] = sum_k ap[i,k,c] * bp[j,k,c]
    //   = mul_mat over batch c_z: ap treated as [L_k, L_i, c_z], bp as [L_k, L_j, c_z]
    //   ggml_mul_mat(A, B): A=[ne0_A=L, ne1_A=L, ne2=c_z], B=[ne0_B=L, ne1_B=L, ne2=c_z]
    //   contracts ne0: result[ne1_A, ne1_B, c_z] = [L_i, L_j, c_z]
    // For incoming: swap indices: result[i,j,c] = sum_k ap[k,i,c]*bp[k,j,c]
    //   = mul_mat(ap^T, bp) where transpose swaps ne0 and ne1
    //   We need A=[L_i, L_k, c_z] = ap transposed per-batch:
    //   ggml_permute(ap, 1, 0, 2, 3) -> [L, L, c_z] with axes swapped
    ggml_tensor * x;
    if (outgoing) {
        x = ggml_mul_mat(ctx, ap, bp); // [L_i, L_j, c_z]
    } else {
        ggml_tensor * ap_t = ggml_cont(ctx, ggml_permute(ctx, ap, 1, 0, 2, 3)); // [L,L,c_z] swapped
        x = ggml_mul_mat(ctx, ap_t, bp); // [L_i, L_j, c_z]
    }
    // x: [L, L, c_z]; permute back to [c_z, L, L]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [c_z, L, L]

    x = layer_norm(ctx, x, ln_out_w, ln_out_b, eps);
    x = linear(ctx, lz_w, lz_b, x);
    ggml_tensor * gate = ggml_sigmoid(ctx, linear(ctx, lg_w, lg_b, zn));
    return ggml_mul(ctx, x, gate); // [c_z, L, L]
}

// TriangleAttention (starting or ending node).
// z : [c_z, L, L]   returns [c_z, L, L]
// starting=true:  attention along ne1 (rows), with ne2 (cols) as batch
// starting=false: transpose ne1/ne2, attend, transpose back
static ggml_tensor * triangle_att(
        ggml_context * ctx, ggml_tensor * z, bool starting,
        ggml_tensor * ln_w, ggml_tensor * ln_b,
        ggml_tensor * bias_w,                        // [pair_heads, c_z]
        ggml_tensor * q_w, ggml_tensor * k_w, ggml_tensor * v_w, // [c_z, c_z] no bias
        ggml_tensor * g_w, ggml_tensor * g_b,
        ggml_tensor * o_w, ggml_tensor * o_b,
        float eps, int c_z, int pair_hw, int L) {
    const int n_head = c_z / pair_hw;
    const float scale = 1.0f / sqrtf((float)pair_hw);

    // For ending-node attention, swap the two L dimensions before processing.
    // z ne: [c_z, L, L]  (ne0=c_z, ne1=L(j), ne2=L(i))
    ggml_tensor * zi = z;
    if (!starting) {
        // swap ne1 and ne2
        zi = ggml_cont(ctx, ggml_permute(ctx, z, 0, 2, 1, 3)); // [c_z, L(i), L(j)]
    }

    ggml_tensor * zn = layer_norm(ctx, zi, ln_w, ln_b, eps); // [c_z, L, L]

    // Per-head bias: linear(zn) [pair_heads, L, L]
    ggml_tensor * tri_bias = ggml_mul_mat(ctx, bias_w, zn); // [pair_heads, L, L]

    // Q, K, V projections: each [c_z, L, L]
    ggml_tensor * q = ggml_mul_mat(ctx, q_w, zn); // [c_z, L, L]
    ggml_tensor * k = ggml_mul_mat(ctx, k_w, zn);
    ggml_tensor * v = ggml_mul_mat(ctx, v_w, zn);

    // For each row i (ne2 dimension), perform multi-head attention over the L(j) columns.
    // Reshape: treat ne2=L as batch.
    // q,k,v: [c_z, L, L] -> reshape to [pair_hw, n_head, L, L]
    q = ggml_reshape_4d(ctx, ggml_cont(ctx, q), pair_hw, n_head, L, L);
    k = ggml_reshape_4d(ctx, ggml_cont(ctx, k), pair_hw, n_head, L, L);
    v = ggml_reshape_4d(ctx, ggml_cont(ctx, v), pair_hw, n_head, L, L);
    // permute to [pair_hw, L(j-attend), n_head, L(i-batch)]
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [pair_hw, L, n_head, L_batch]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // scores = K^T Q * scale: [L_j, L_j, n_head, L_batch]
    ggml_tensor * kc = ggml_cont(ctx, k);
    ggml_tensor * scores = ggml_mul_mat(ctx, kc, q); // [L, L, n_head, L_batch]
    scores = ggml_scale(ctx, scores, scale);

    // tri_bias: [pair_heads, L, L] -> [n_head, L, L] (same thing)
    // Broadcast across L_batch (ne3): reshape to [L, L, n_head, 1] then repeat to [L,L,n_head,L_batch]
    ggml_tensor * tb = ggml_cont(ctx, ggml_permute(ctx, tri_bias, 1, 2, 0, 3)); // [L,L,n_head]
    tb = ggml_reshape_4d(ctx, tb, L, L, n_head, 1);
    tb = ggml_repeat(ctx, tb, scores); // [L, L, n_head, L_batch]
    scores = ggml_add(ctx, scores, tb);
    scores = ggml_soft_max(ctx, scores); // [L, L, n_head, L_batch]

    // context = scores @ V
    ggml_tensor * vc = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [L, pair_hw, n_head, L_batch]
    ggml_tensor * out = ggml_mul_mat(ctx, vc, scores);                    // [pair_hw, L, n_head, L_batch]
    // permute back: [pair_hw, n_head, L, L_batch]
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont(ctx, out);
    out = ggml_reshape_3d(ctx, out, c_z, L, L); // [c_z, L, L]

    // gating: sigmoid(g(zi)) * out
    ggml_tensor * gate = ggml_sigmoid(ctx, linear(ctx, g_w, g_b, zi)); // [c_z, L, L]
    out = ggml_mul(ctx, gate, out);

    out = linear(ctx, o_w, o_b, out); // [c_z, L, L]

    if (!starting) {
        // swap back
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    }
    return out;
}

// ── eval ──────────────────────────────────────────────────────────────────
bool esmfold_trunk_eval(
        esmfold_trunk_model       * model,
        const std::vector<float>  & s_s_0_data,
        const std::vector<int32_t>& residx_data,
        std::vector<float>        & s_s_out,
        std::vector<float>        & s_z_out,
        int n_blocks_override) {
    const int L    = (int)residx_data.size();
    const int c_s  = model->c_s;
    const int c_z  = model->c_z;
    const int nb   = (n_blocks_override < 0) ? model->num_blocks : n_blocks_override;
    const int bins = model->pos_bins;     // 32
    const float eps = model->eps;

    assert((int)s_s_0_data.size() == L * c_s);

    // Graph memory: 1024 tensors overhead is sufficient for the intermediate ops.
    // Each block creates ~200 tensor nodes; 48 blocks = ~9600; overhead * 12000 is safe.
    const size_t buf_size = (size_t)12000 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params gp = { buf_size, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph  * gf  = ggml_new_graph_custom(ctx, 32768, false);

    // ── inputs ──────────────────────────────────────────────────────────
    ggml_tensor * inp_s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c_s, L);
    ggml_set_name(inp_s, "inp_s"); ggml_set_input(inp_s);

    ggml_tensor * inp_residx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(inp_residx, "inp_residx"); ggml_set_input(inp_residx);

    // ── recycle bias (num_recycles=0: recycle_norm(zeros) = bias) ───────
    // LayerNorm(0) = (0 - 0) / sqrt(0+eps) * weight + bias = bias  (weight * 0 = 0)
    // So recycle_s contribution = recycle_s_norm.bias  (broadcast over L)
    ggml_tensor * s = ggml_add(ctx, inp_s,
                         ggml_repeat(ctx,
                           ggml_reshape_2d(ctx, model->recycle_s_b, c_s, 1),
                           inp_s)); // [c_s, L]

    // ── relative position embedding -> initial pair state ───────────────
    // diff[i,j] = clamp(residx[i] - residx[j], -bins, bins) + bins + 1
    // Using ggml arithmetic on the 1D residx to form the [L,L] diff matrix:
    ggml_tensor * ridx = ggml_reshape_2d(ctx, inp_residx, L, 1); // [L, 1] as float? no, i32
    // Cast to f32 for arithmetic
    ggml_tensor * ridx_f = ggml_cont(ctx, ggml_cast(ctx, ridx, GGML_TYPE_F32)); // [L, 1]
    // row_i - col_j: repeat ridx_f to [L,L] (column) minus transpose [1,L] -> [L,L]
    ggml_tensor * ri = ggml_repeat(ctx, ridx_f,
                         ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, L)); // [L(i), L(j)]
    ggml_tensor * rj = ggml_repeat(ctx,
                         ggml_reshape_2d(ctx, inp_residx, 1, L), // this is I32; need f32
                         ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, L));
    // rj via f32: reshape ridx_f (1D L) to [1, L] and repeat
    ggml_tensor * ridx_f_row = ggml_cont(ctx, ggml_reshape_2d(ctx,
                         ggml_cont(ctx, ggml_cast(ctx, inp_residx, GGML_TYPE_F32)), 1, L));
    rj = ggml_repeat(ctx, ridx_f_row, ri); // [L, L]
    // diff = ri - rj, clamped, shifted
    ggml_tensor * diff_f = ggml_sub(ctx, ri, rj);
    diff_f = ggml_clamp(ctx, diff_f, (float)-bins, (float)bins);
    diff_f = ggml_add_inplace(ctx, diff_f,
                 ggml_new_f32(ctx, (float)(bins + 1)));
    // convert to i32 for embedding lookup
    ggml_tensor * diff_i = ggml_cast(ctx, diff_f, GGML_TYPE_I32);  // [L, L]
    // flatten to 1D for get_rows, then reshape
    ggml_tensor * diff_1d = ggml_reshape_1d(ctx, diff_i, L * L);
    ggml_tensor * relpos_flat = ggml_get_rows(ctx, model->relpos_w, diff_1d); // [c_z, L*L]
    ggml_tensor * relpos = ggml_reshape_3d(ctx, relpos_flat, c_z, L, L); // [c_z, L, L]

    // z = recycle_z_norm.bias (broadcast) + relpos
    ggml_tensor * z = ggml_add(ctx,
                         ggml_repeat(ctx,
                           ggml_reshape_3d(ctx, model->recycle_z_b, c_z, 1, 1),
                           relpos),
                         relpos); // [c_z, L, L]

    // ── 48x TriangularSelfAttentionBlock ────────────────────────────────
    for (int il = 0; il < nb; il++) {
        const trunk_block & bl = model->blocks[il];
        const int seq_hw  = model->seq_hw;
        const int pair_hw = model->pair_hw;

        // 1. Sequence attention with pair bias (residual)
        ggml_tensor * ds = seq_attention(ctx, s, z,
            bl.seq_ln_w, bl.seq_ln_b,
            bl.seq_attn_qkv, bl.seq_attn_g_w, bl.seq_attn_g_b,
            bl.seq_attn_o_w, bl.seq_attn_o_b,
            bl.p2s_ln_w, bl.p2s_ln_b, bl.p2s_linear_w,
            eps, c_s, seq_hw);
        s = ggml_add(ctx, s, ds);

        // 2. Sequence MLP (residual built in)
        s = residue_mlp(ctx, s,
            bl.mlp_seq_ln_w, bl.mlp_seq_ln_b,
            bl.mlp_seq_fc1_w, bl.mlp_seq_fc1_b,
            bl.mlp_seq_fc2_w, bl.mlp_seq_fc2_b, eps);

        // 3. Sequence-to-pair update (residual)
        z = ggml_add(ctx, z, sequence_to_pair(ctx, s,
            bl.s2p_ln_w, bl.s2p_ln_b,
            bl.s2p_proj_w, bl.s2p_proj_b,
            bl.s2p_o_w, bl.s2p_o_b,
            eps, c_s, c_z, L));

        // 4. Triangle multiplicative update outgoing (residual)
        z = ggml_add(ctx, z, triangle_mul(ctx, z, true,
            bl.tmo_ln_in_w, bl.tmo_ln_in_b, bl.tmo_ln_out_w, bl.tmo_ln_out_b,
            bl.tmo_ap_w, bl.tmo_ap_b, bl.tmo_ag_w, bl.tmo_ag_b,
            bl.tmo_bp_w, bl.tmo_bp_b, bl.tmo_bg_w, bl.tmo_bg_b,
            bl.tmo_z_w, bl.tmo_z_b, bl.tmo_g_w, bl.tmo_g_b,
            eps, c_z, L));

        // 5. Triangle multiplicative update incoming (residual)
        z = ggml_add(ctx, z, triangle_mul(ctx, z, false,
            bl.tmi_ln_in_w, bl.tmi_ln_in_b, bl.tmi_ln_out_w, bl.tmi_ln_out_b,
            bl.tmi_ap_w, bl.tmi_ap_b, bl.tmi_ag_w, bl.tmi_ag_b,
            bl.tmi_bp_w, bl.tmi_bp_b, bl.tmi_bg_w, bl.tmi_bg_b,
            bl.tmi_z_w, bl.tmi_z_b, bl.tmi_g_w, bl.tmi_g_b,
            eps, c_z, L));

        // 6. Triangle attention starting (residual)
        z = ggml_add(ctx, z, triangle_att(ctx, z, true,
            bl.tas_ln_w, bl.tas_ln_b, bl.tas_bias_w,
            bl.tas_q_w, bl.tas_k_w, bl.tas_v_w,
            bl.tas_g_w, bl.tas_g_b, bl.tas_o_w, bl.tas_o_b,
            eps, c_z, pair_hw, L));

        // 7. Triangle attention ending (residual)
        z = ggml_add(ctx, z, triangle_att(ctx, z, false,
            bl.tae_ln_w, bl.tae_ln_b, bl.tae_bias_w,
            bl.tae_q_w, bl.tae_k_w, bl.tae_v_w,
            bl.tae_g_w, bl.tae_g_b, bl.tae_o_w, bl.tae_o_b,
            eps, c_z, pair_hw, L));

        // 8. Pair MLP (residual built in)
        z = residue_mlp(ctx, z,
            bl.mlp_pair_ln_w, bl.mlp_pair_ln_b,
            bl.mlp_pair_fc1_w, bl.mlp_pair_fc1_b,
            bl.mlp_pair_fc2_w, bl.mlp_pair_fc2_b, eps);
    }

    ggml_set_output(s);
    ggml_set_output(z);
    ggml_build_forward_expand(gf, s);
    ggml_build_forward_expand(gf, z);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(galloc, gf);

    // set inputs
    ggml_backend_tensor_set(inp_s,      s_s_0_data.data(),  0, (size_t)L*c_s*sizeof(float));
    ggml_backend_tensor_set(inp_residx, residx_data.data(), 0, (size_t)L*sizeof(int32_t));

    ggml_backend_graph_compute(model->backend, gf);

    s_s_out.resize((size_t)L * c_s);
    s_z_out.resize((size_t)L * L * c_z);
    ggml_backend_tensor_get(s, s_s_out.data(), 0, s_s_out.size() * sizeof(float));
    ggml_backend_tensor_get(z, s_z_out.data(), 0, s_z_out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}
