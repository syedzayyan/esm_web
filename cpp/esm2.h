#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Opaque handle for a loaded ESM2 (GGUF) model.
struct esm2_model;

// Load a model converted with convert_esm2.py. Returns nullptr on failure.
esm2_model * esm2_load(const std::string & path);

void esm2_free(esm2_model * model);

int esm2_n_embd(const esm2_model * model);

// Run the ESM2 encoder on a single sequence of token ids (no padding, batch
// size 1). Returns the per-token hidden states flattened as
// [n_tokens * n_embd], token-major (token t's vector is at
// result[t * n_embd .. (t + 1) * n_embd)).
std::vector<float> esm2_eval(esm2_model * model, const std::vector<int32_t> & tokens);
