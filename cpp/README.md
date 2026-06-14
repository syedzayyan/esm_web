# ESM2 in C / ggml

A native CLI/library port of `src/models/esm2.rs`, built on
[ggml](https://github.com/ggml-org/ggml). Given a protein sequence, it
produces the same mean-pooled embedding as `get_embeddings` in `src/lib.rs`.

This is a standalone companion to the Rust/WASM web app — it does not affect
the existing build.

## 1. Convert a checkpoint to GGUF

Requires a HuggingFace ESM2 checkpoint directory containing `config.json` and
`model.safetensors` (e.g. `facebook/esm2_t6_8M_UR50D`), plus the `gguf` and
`safetensors` Python packages:

```sh
pip install gguf safetensors numpy
python3 cpp/convert_esm2.py chp/esm chp/esm/esm2-f32.gguf
```

## 2. Build

Requires ggml (e.g. `brew install ggml`):

```sh
make -C cpp
```

This produces `cpp/esm2`. Override `GGML_PREFIX` if ggml is installed
somewhere other than `/opt/homebrew`.

## 3. Run

```sh
cpp/esm2 chp/esm/esm2-f32.gguf [--normalize] <sequence> [sequence...]
```

Each sequence prints as one line of space-separated floats: the mean-pooled
hidden state over all tokens (including `<cls>`), optionally L2-normalized
with `--normalize`.

## Scope and limitations

- ESM2 only (no BERT), batch size 1, no padding/attention masks.
- The tokenizer (`cpp/tokenizer.h`) is a hardcoded re-implementation of the
  repo's `tokenizer.json`: prepend `<cls>`, then one token per residue
  character (case-insensitive), falling back to `<unk>` for unrecognized
  characters.
- Rotary position embeddings use ggml's built-in NEOX-style RoPE
  (`freq_base=10000`), which is bit-for-bit equivalent to the per-layer
  `inv_freq` stored in the checkpoint, so those tensors are not loaded.
- This port follows `esm2.rs`'s transformer/attention implementation, but
  additionally applies the `token_dropout` embedding rescale from
  HuggingFace's `EsmEmbeddings.forward` (zeroing `<mask>`-token embeddings and
  rescaling all token embeddings by `(1 - 0.12) / (1 - mask_ratio_observed)`),
  which `esm2.rs` itself omits. As a result, this port matches
  `transformers.EsmModel`'s `last_hidden_state` to ~1e-5, whereas `esm2.rs`
  differs from it by a small but non-negligible amount (cosine similarity
  ~0.9977 on a test sequence) due to that missing rescale.
