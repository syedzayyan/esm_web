#!/usr/bin/env python3
"""Extract ESMFold folding-trunk weights from facebook/esmfold_v1 into a GGUF.

Usage:
    python3 cpp/convert_esmfold_trunk.py [output.gguf]

Downloads facebook/esmfold_v1 on first run (several GB, cached by HF).
Only the trunk slice (~650 MB fp32) is written to the GGUF; the ESM2-3B
backbone and structure-module weights are skipped.
"""
import sys
from pathlib import Path

import numpy as np
import gguf
import torch
from transformers import EsmForProteinFolding

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chp/esmfold/trunk-f32.gguf")
out_path.parent.mkdir(parents=True, exist_ok=True)

print("loading facebook/esmfold_v1 …")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model.eval()

trunk = model.trunk
cfg = model.config.esmfold_config.trunk
eps = 1e-5  # EsmFold uses default LN eps

writer = gguf.GGUFWriter(str(out_path), "esmfold_trunk")
writer.add_uint32("esmfold.num_blocks",      cfg.num_blocks)
writer.add_uint32("esmfold.seq_state_dim",   cfg.sequence_state_dim)
writer.add_uint32("esmfold.pair_state_dim",  cfg.pairwise_state_dim)
writer.add_uint32("esmfold.seq_head_width",  cfg.sequence_head_width)
writer.add_uint32("esmfold.pair_head_width", cfg.pairwise_head_width)
writer.add_uint32("esmfold.position_bins",   cfg.position_bins)
writer.add_float32("esmfold.layer_norm_eps", eps)

sd = {k: v.float().numpy() for k, v in trunk.state_dict().items()}

def add(gguf_name, hf_key):
    arr = np.ascontiguousarray(sd[hf_key])
    writer.add_tensor(gguf_name, arr)

add("relpos.weight",        "pairwise_positional_embedding.embedding.weight")
add("recycle_s_norm.weight","recycle_s_norm.weight")
add("recycle_s_norm.bias",  "recycle_s_norm.bias")
add("recycle_z_norm.weight","recycle_z_norm.weight")
add("recycle_z_norm.bias",  "recycle_z_norm.bias")

for i in range(cfg.num_blocks):
    p = f"blocks.{i}"
    g = f"layers.{i}"

    add(f"{g}.seq_ln.weight",          f"{p}.layernorm_1.weight")
    add(f"{g}.seq_ln.bias",            f"{p}.layernorm_1.bias")
    add(f"{g}.seq_attn.proj.weight",   f"{p}.seq_attention.proj.weight")   # [3072,1024]
    add(f"{g}.seq_attn.g_proj.weight", f"{p}.seq_attention.g_proj.weight")
    add(f"{g}.seq_attn.g_proj.bias",   f"{p}.seq_attention.g_proj.bias")
    add(f"{g}.seq_attn.o_proj.weight", f"{p}.seq_attention.o_proj.weight")
    add(f"{g}.seq_attn.o_proj.bias",   f"{p}.seq_attention.o_proj.bias")

    add(f"{g}.mlp_seq.ln.weight",      f"{p}.mlp_seq.mlp.0.weight")
    add(f"{g}.mlp_seq.ln.bias",        f"{p}.mlp_seq.mlp.0.bias")
    add(f"{g}.mlp_seq.fc1.weight",     f"{p}.mlp_seq.mlp.1.weight")
    add(f"{g}.mlp_seq.fc1.bias",       f"{p}.mlp_seq.mlp.1.bias")
    add(f"{g}.mlp_seq.fc2.weight",     f"{p}.mlp_seq.mlp.3.weight")
    add(f"{g}.mlp_seq.fc2.bias",       f"{p}.mlp_seq.mlp.3.bias")

    add(f"{g}.p2s.ln.weight",          f"{p}.pair_to_sequence.layernorm.weight")
    add(f"{g}.p2s.ln.bias",            f"{p}.pair_to_sequence.layernorm.bias")
    add(f"{g}.p2s.linear.weight",      f"{p}.pair_to_sequence.linear.weight")   # [32,128] no bias

    add(f"{g}.s2p.ln.weight",          f"{p}.sequence_to_pair.layernorm.weight")
    add(f"{g}.s2p.ln.bias",            f"{p}.sequence_to_pair.layernorm.bias")
    add(f"{g}.s2p.proj.weight",        f"{p}.sequence_to_pair.proj.weight")
    add(f"{g}.s2p.proj.bias",          f"{p}.sequence_to_pair.proj.bias")
    add(f"{g}.s2p.o_proj.weight",      f"{p}.sequence_to_pair.o_proj.weight")
    add(f"{g}.s2p.o_proj.bias",        f"{p}.sequence_to_pair.o_proj.bias")

    for tag, hf in [("tri_mul_out", f"{p}.tri_mul_out"), ("tri_mul_in", f"{p}.tri_mul_in")]:
        g2 = f"{g}.{tag}"
        add(f"{g2}.ln_in.weight",  f"{hf}.layer_norm_in.weight")
        add(f"{g2}.ln_in.bias",    f"{hf}.layer_norm_in.bias")
        add(f"{g2}.ln_out.weight", f"{hf}.layer_norm_out.weight")
        add(f"{g2}.ln_out.bias",   f"{hf}.layer_norm_out.bias")
        add(f"{g2}.a_p.weight",    f"{hf}.linear_a_p.weight")
        add(f"{g2}.a_p.bias",      f"{hf}.linear_a_p.bias")
        add(f"{g2}.a_g.weight",    f"{hf}.linear_a_g.weight")
        add(f"{g2}.a_g.bias",      f"{hf}.linear_a_g.bias")
        add(f"{g2}.b_p.weight",    f"{hf}.linear_b_p.weight")
        add(f"{g2}.b_p.bias",      f"{hf}.linear_b_p.bias")
        add(f"{g2}.b_g.weight",    f"{hf}.linear_b_g.weight")
        add(f"{g2}.b_g.bias",      f"{hf}.linear_b_g.bias")
        add(f"{g2}.linear_z.weight", f"{hf}.linear_z.weight")
        add(f"{g2}.linear_z.bias",   f"{hf}.linear_z.bias")
        add(f"{g2}.linear_g.weight", f"{hf}.linear_g.weight")
        add(f"{g2}.linear_g.bias",   f"{hf}.linear_g.bias")

    for tag, hf in [("tri_att_start", f"{p}.tri_att_start"), ("tri_att_end", f"{p}.tri_att_end")]:
        g2 = f"{g}.{tag}"
        add(f"{g2}.ln.weight",       f"{hf}.layer_norm.weight")
        add(f"{g2}.ln.bias",         f"{hf}.layer_norm.bias")
        add(f"{g2}.bias_proj.weight",f"{hf}.linear.weight")         # [4,128] no bias
        add(f"{g2}.mha.q.weight",    f"{hf}.mha.linear_q.weight")  # no bias
        add(f"{g2}.mha.k.weight",    f"{hf}.mha.linear_k.weight")  # no bias
        add(f"{g2}.mha.v.weight",    f"{hf}.mha.linear_v.weight")  # no bias
        add(f"{g2}.mha.g.weight",    f"{hf}.mha.linear_g.weight")
        add(f"{g2}.mha.g.bias",      f"{hf}.mha.linear_g.bias")
        add(f"{g2}.mha.o.weight",    f"{hf}.mha.linear_o.weight")
        add(f"{g2}.mha.o.bias",      f"{hf}.mha.linear_o.bias")

    add(f"{g}.mlp_pair.ln.weight",   f"{p}.mlp_pair.mlp.0.weight")
    add(f"{g}.mlp_pair.ln.bias",     f"{p}.mlp_pair.mlp.0.bias")
    add(f"{g}.mlp_pair.fc1.weight",  f"{p}.mlp_pair.mlp.1.weight")
    add(f"{g}.mlp_pair.fc1.bias",    f"{p}.mlp_pair.mlp.1.bias")
    add(f"{g}.mlp_pair.fc2.weight",  f"{p}.mlp_pair.mlp.3.weight")
    add(f"{g}.mlp_pair.fc2.bias",    f"{p}.mlp_pair.mlp.3.bias")

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print(f"wrote {out_path}")
