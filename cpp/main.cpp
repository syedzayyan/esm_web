#include "esm2.h"
#include "tokenizer.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--normalize] <sequence> [sequence...]\n\n"
        "Computes a mean-pooled ESM2 embedding for each protein sequence and\n"
        "prints it as a line of space-separated floats.\n",
        argv0);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    bool normalize = false;
    std::vector<std::string> sequences;

    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--normalize") {
            normalize = true;
        } else {
            sequences.push_back(arg);
        }
    }

    if (sequences.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    esm2_model * model = esm2_load(model_path);
    if (!model) {
        fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }

    const int n_embd = esm2_n_embd(model);

    for (const std::string & seq : sequences) {
        std::vector<int32_t> tokens = esm2_tokenizer::encode(seq);
        std::vector<float> hidden = esm2_eval(model, tokens); // [n_tokens * n_embd]
        const int n_tokens = (int) tokens.size();

        // mean-pool over the sequence dimension (including <cls>), matching
        // lib.rs's `embeddings.sum(1) / n_tokens`
        std::vector<float> pooled(n_embd, 0.0f);
        for (int t = 0; t < n_tokens; t++) {
            for (int d = 0; d < n_embd; d++) {
                pooled[d] += hidden[(size_t) t * n_embd + d];
            }
        }
        for (int d = 0; d < n_embd; d++) {
            pooled[d] /= (float) n_tokens;
        }

        if (normalize) {
            float norm = 0.0f;
            for (int d = 0; d < n_embd; d++) norm += pooled[d] * pooled[d];
            norm = sqrtf(norm);
            if (norm > 0.0f) {
                for (int d = 0; d < n_embd; d++) pooled[d] /= norm;
            }
        }

        for (int d = 0; d < n_embd; d++) {
            printf("%s%.6f", d == 0 ? "" : " ", pooled[d]);
        }
        printf("\n");
    }

    esm2_free(model);
    return 0;
}
