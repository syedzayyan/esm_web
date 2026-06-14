#!/usr/bin/env python3
"""Convert a HuggingFace ESM2 checkpoint (config.json + model.safetensors)
into a GGUF file consumable by cpp/esm2.cpp.

Usage:
    python3 convert_esm2.py <checkpoint_dir> [output.gguf]

<checkpoint_dir> must contain config.json and model.safetensors.
"""
import json
import sys
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ckpt_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt_dir / "esm2-f32.gguf"

    with open(ckpt_dir / "config.json") as f:
        config = json.load(f)

    tensors = load_file(ckpt_dir / "model.safetensors")

    n_layer = config["num_hidden_layers"]
    n_head = config["num_attention_heads"]
    n_embd = config["hidden_size"]
    n_ff = config["intermediate_size"]
    n_vocab = config["vocab_size"]
    eps = config["layer_norm_eps"]
    mask_token_id = config.get("mask_token_id", 32)
    token_dropout = config.get("token_dropout", False)

    writer = gguf.GGUFWriter(out_path, "esm2")
    writer.add_uint32("esm2.hidden_size", n_embd)
    writer.add_uint32("esm2.num_layers", n_layer)
    writer.add_uint32("esm2.num_heads", n_head)
    writer.add_uint32("esm2.intermediate_size", n_ff)
    writer.add_uint32("esm2.vocab_size", n_vocab)
    writer.add_float32("esm2.layer_norm_eps", eps)
    writer.add_uint32("esm2.mask_token_id", mask_token_id)
    writer.add_bool("esm2.token_dropout", token_dropout)

    def add(name, src):
        arr = np.ascontiguousarray(tensors[src].astype(np.float32))
        writer.add_tensor(name, arr)

    add("tok_embd.weight", "esm.embeddings.word_embeddings.weight")

    for i in range(n_layer):
        p = f"esm.encoder.layer.{i}"
        add(f"layers.{i}.attn_norm.weight", f"{p}.attention.LayerNorm.weight")
        add(f"layers.{i}.attn_norm.bias", f"{p}.attention.LayerNorm.bias")

        add(f"layers.{i}.attn_q.weight", f"{p}.attention.self.query.weight")
        add(f"layers.{i}.attn_q.bias", f"{p}.attention.self.query.bias")
        add(f"layers.{i}.attn_k.weight", f"{p}.attention.self.key.weight")
        add(f"layers.{i}.attn_k.bias", f"{p}.attention.self.key.bias")
        add(f"layers.{i}.attn_v.weight", f"{p}.attention.self.value.weight")
        add(f"layers.{i}.attn_v.bias", f"{p}.attention.self.value.bias")

        add(f"layers.{i}.attn_output.weight", f"{p}.attention.output.dense.weight")
        add(f"layers.{i}.attn_output.bias", f"{p}.attention.output.dense.bias")

        add(f"layers.{i}.ffn_norm.weight", f"{p}.LayerNorm.weight")
        add(f"layers.{i}.ffn_norm.bias", f"{p}.LayerNorm.bias")

        add(f"layers.{i}.ffn_up.weight", f"{p}.intermediate.dense.weight")
        add(f"layers.{i}.ffn_up.bias", f"{p}.intermediate.dense.bias")
        add(f"layers.{i}.ffn_down.weight", f"{p}.output.dense.weight")
        add(f"layers.{i}.ffn_down.bias", f"{p}.output.dense.bias")

    add("output_norm.weight", "esm.encoder.emb_layer_norm_after.weight")
    add("output_norm.bias", "esm.encoder.emb_layer_norm_after.bias")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
