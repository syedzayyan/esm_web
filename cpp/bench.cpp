// ESM2 C/ggml inference benchmark.
// Generates synthetic sequences of several lengths, runs warmup+timed passes,
// and reports mean/min/max latency and residues/second.
//
// Usage:
//   ./bench <model.gguf> [warmup=3] [runs=10]

#include "esm2.h"
#include "tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using hrc = std::chrono::high_resolution_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(hrc::now().time_since_epoch()).count();
}

// Synthetic sequence cycling over the 20 canonical amino acids.
static std::string make_seq(int n) {
    const char aa[] = "ACDEFGHIKLMNPQRSTVWY";
    std::string s(n, 'A');
    for (int i = 0; i < n; i++) s[i] = aa[i % 20];
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [warmup=3] [runs=10]\n", argv[0]);
        return 1;
    }
    const int n_warmup = argc > 2 ? atoi(argv[2]) : 3;
    const int n_runs   = argc > 3 ? atoi(argv[3]) : 10;

    double t0 = now_ms();
    esm2_model * model = esm2_load(argv[1]);
    if (!model) {
        fprintf(stderr, "failed to load model from '%s'\n", argv[1]);
        return 1;
    }
    printf("model load : %.1f ms\n\n", now_ms() - t0);

    const int lengths[] = {10, 50, 100, 200, 500};
    const int n_lengths = (int)(sizeof(lengths) / sizeof(lengths[0]));

    printf("%-6s  %8s  %8s  %8s  %12s  (warmup=%d, runs=%d)\n",
           "len", "mean_ms", "min_ms", "max_ms", "res/sec", n_warmup, n_runs);
    printf("%-6s  %8s  %8s  %8s  %12s\n",
           "------", "--------", "--------", "--------", "------------");

    for (int li = 0; li < n_lengths; li++) {
        int L = lengths[li];
        std::string seq = make_seq(L);
        std::vector<int32_t> tokens = esm2_tokenizer::encode(seq);

        for (int i = 0; i < n_warmup; i++) esm2_eval(model, tokens);

        std::vector<double> times(n_runs);
        for (int i = 0; i < n_runs; i++) {
            double t = now_ms();
            esm2_eval(model, tokens);
            times[i] = now_ms() - t;
        }

        double sum = 0.0, mn = times[0], mx = times[0];
        for (double t : times) { sum += t; if (t < mn) mn = t; if (t > mx) mx = t; }
        double mean = sum / n_runs;
        printf("%-6d  %8.1f  %8.1f  %8.1f  %12.0f\n",
               L, mean, mn, mx, (double)L / (mean / 1000.0));
    }

    esm2_free(model);
    return 0;
}
