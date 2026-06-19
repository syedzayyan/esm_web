// WASM entry point for the C/ggml ESM2 port.
// Compiled with Emscripten (see Makefile.wasm).
//
// JavaScript API (via Module.ccall / Module.cwrap):
//
//   esm2_wasm_load(ptr, size) -> int   load GGUF from heap buffer; 1=ok 0=fail
//   esm2_wasm_n_embd()        -> int   embedding dimension
//   esm2_wasm_embed(seq_ptr)  -> int   run inference; returns n_tokens (0=fail)
//   esm2_wasm_get_hidden()    -> ptr   float32 [n_tokens * n_embd], token-major
//   esm2_wasm_free()          -> void  release model

#include "esm2.h"
#include "tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <emscripten/emscripten.h>

static esm2_model         * g_model        = nullptr;
static std::vector<float>   g_hidden;
static int                  g_last_n_embd  = 0;
static int                  g_last_n_tok   = 0;

extern "C" {

// Write the GGUF bytes from the JS heap into Emscripten MEMFS, then load.
EMSCRIPTEN_KEEPALIVE
int esm2_wasm_load(const uint8_t * data, size_t size) {
    static const char * MEMFS_PATH = "/tmp/esm2.gguf";
    FILE * f = fopen(MEMFS_PATH, "wb");
    if (!f) { fprintf(stderr, "esm2_wasm_load: cannot write MEMFS\n"); return 0; }
    fwrite(data, 1, size, f);
    fclose(f);
    if (g_model) { esm2_free(g_model); g_model = nullptr; }
    g_model = esm2_load(MEMFS_PATH);
    if (!g_model) { fprintf(stderr, "esm2_wasm_load: esm2_load failed\n"); return 0; }
    g_last_n_embd = esm2_n_embd(g_model);
    return 1;
}

EMSCRIPTEN_KEEPALIVE
int esm2_wasm_n_embd() {
    return g_last_n_embd;
}

// Run inference on a null-terminated amino-acid string (1-letter codes).
// Returns the number of tokens (including BOS/EOS added by the tokenizer).
// Hidden states are available via esm2_wasm_get_hidden() until next call.
EMSCRIPTEN_KEEPALIVE
int esm2_wasm_embed(const char * seq) {
    if (!g_model || !seq) return 0;
    std::vector<int32_t> tokens = esm2_tokenizer::encode(std::string(seq));
    g_hidden = esm2_eval(g_model, tokens);
    g_last_n_tok  = (int)tokens.size();
    return g_last_n_tok;
}

// Returns a pointer into the internal buffer — valid until the next embed call.
EMSCRIPTEN_KEEPALIVE
const float * esm2_wasm_get_hidden() {
    return g_hidden.data();
}

EMSCRIPTEN_KEEPALIVE
void esm2_wasm_free() {
    if (g_model) { esm2_free(g_model); g_model = nullptr; }
    g_hidden.clear();
}

} // extern "C"
