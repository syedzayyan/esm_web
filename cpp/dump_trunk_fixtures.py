#!/usr/bin/env python3
"""Run facebook/esmfold_v1 with num_recycles=0 and dump verification fixtures.

Usage:
    python3 cpp/dump_trunk_fixtures.py [output_dir]

Requires facebook/esmfold_v1 to be cached (run convert_esmfold_trunk.py first).
Writes to chp/esmfold/ (or output_dir):
  fixtures.npz  - s_s_0, residx, mask, s_s_final, s_z_final,
                  block0_s_in, block0_z_in, block0_s_out, block0_z_out
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import EsmForProteinFolding, EsmTokenizer

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chp/esmfold")
out_dir.mkdir(parents=True, exist_ok=True)

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"  # short for fast fixture generation

print("loading facebook/esmfold_v1 …")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model.eval()
tok = EsmTokenizer.from_pretrained("facebook/esmfold_v1")

inputs = tok([SEQ], return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"]
B, L = input_ids.shape

with torch.no_grad():
    # --- replicate EsmForProteinFolding.forward up to the trunk call ---
    cfg = model.config.esmfold_config
    aa = input_ids
    attention_mask = torch.ones_like(aa)
    position_ids = torch.arange(L).unsqueeze(0)

    esmaa = model.af2_idx_to_esm_idx(aa, attention_mask)
    esm_s = model.compute_language_model_representations(esmaa)
    esm_s = esm_s.to(model.esm_s_combine.dtype)
    esm_s = esm_s.detach()
    esm_s = (model.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
    s_s_0 = model.esm_s_mlp(esm_s)                    # [1, L, 1024]
    if cfg.embed_aa:
        s_s_0 = s_s_0 + model.embedding(aa)
    s_z_0 = s_s_0.new_zeros(1, L, L, cfg.trunk.pairwise_state_dim)  # [1,L,L,128]

    # --- hook block 0 I/O ---
    trunk = model.trunk
    block0_s_in = block0_z_in = block0_s_out = block0_z_out = None

    original_block0_forward = trunk.blocks[0].forward
    def hooked_block0(seq_state, pairwise_state, **kw):
        global block0_s_in, block0_z_in, block0_s_out, block0_z_out
        block0_s_in = seq_state.detach().cpu().float().numpy()
        block0_z_in = pairwise_state.detach().cpu().float().numpy()
        s_out, z_out = original_block0_forward(seq_state, pairwise_state, **kw)
        block0_s_out = s_out.detach().cpu().float().numpy()
        block0_z_out = z_out.detach().cpu().float().numpy()
        return s_out, z_out
    trunk.blocks[0].forward = hooked_block0

    # run trunk with num_recycles=0 (single pass, no structure-module dependency)
    structure = trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=0)

arrays = {
    "s_s_0":        s_s_0.detach().cpu().float().numpy()[0],
    "residx":       position_ids.float().numpy()[0],
    "mask":         attention_mask.float().numpy()[0],
    "s_s_final":    structure["s_s"].detach().cpu().float().numpy()[0],
    "s_z_final":    structure["s_z"].detach().cpu().float().numpy()[0],
    "block0_s_in":  block0_s_in[0],
    "block0_z_in":  block0_z_in[0],
    "block0_s_out": block0_s_out[0],
    "block0_z_out": block0_z_out[0],
}

np.savez_compressed(out_dir / "fixtures.npz", **arrays)

# Also write a plain .bin sidecar readable by test_trunk without libzip:
# [4B n_arrays] [foreach: 64B name, 4B ndim, ndim*4B shape, float32 data]
import struct
bin_path = out_dir / "fixtures.bin"
with open(bin_path, "wb") as fout:
    fout.write(struct.pack("<i", len(arrays)))
    for name, arr in arrays.items():
        arr_f = arr.astype(np.float32)
        name_b = name.encode()[:63].ljust(64, b"\x00")
        fout.write(name_b)
        fout.write(struct.pack("<i", arr_f.ndim))
        fout.write(struct.pack(f"<{arr_f.ndim}i", *arr_f.shape))
        fout.write(arr_f.tobytes())

print(f"wrote {out_dir}/fixtures.npz + fixtures.bin  (L={L}, seq={SEQ})")
