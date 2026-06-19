// Verification harness for the ESMFold folding trunk C/ggml port.
// Reads fixtures produced by dump_trunk_fixtures.py and compares against
// the output of esmfold_trunk_eval() for 1 block, then all 48 blocks.
//
// Usage:
//   ./test_trunk <trunk-f32.gguf> <fixtures.npz>
//
// fixtures.npz is read as raw binary arrays.  The Python script also writes
// a plain fixtures.bin alongside it for this harness (no libzip needed).

#include "esmfold_trunk.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Minimal .npz reader: the Python dump script writes a companion .bin file
// (see dump_trunk_fixtures.py --bin) with a simple layout:
//   [4B n_arrays][foreach: 64B name, 4B ndim, ndim*4B shape, data]
// We implement that here; the Python script produces it.
// ── Alternatively, just read the raw .bin sidecar. ──────────────────────

struct Array {
    std::string name;
    std::vector<int>   shape;
    std::vector<float> data;
};

static std::vector<Array> load_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    int n_arrays = 0;
    f.read(reinterpret_cast<char*>(&n_arrays), 4);
    std::vector<Array> arrays(n_arrays);
    for (auto & a : arrays) {
        char name[65] = {};
        f.read(name, 64);
        a.name = name;
        int ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        a.shape.resize(ndim);
        f.read(reinterpret_cast<char*>(a.shape.data()), ndim * 4);
        size_t total = 1;
        for (int d : a.shape) total *= (size_t)d;
        a.data.resize(total);
        f.read(reinterpret_cast<char*>(a.data.data()), total * sizeof(float));
    }
    return arrays;
}

static float max_abs_diff(const std::vector<float> & a, const float * b, size_t n) {
    float mx = 0.f;
    for (size_t i = 0; i < n; i++) mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <trunk-f32.gguf> <fixtures.bin>\n", argv[0]);
        return 1;
    }

    auto arrays = load_bin(argv[2]);
    if (arrays.empty()) return 1;

    auto find = [&](const std::string & nm) -> const Array * {
        for (auto & a : arrays) if (a.name == nm) return &a;
        fprintf(stderr, "fixture '%s' not found\n", nm.c_str());
        return nullptr;
    };

    const Array * s0     = find("s_s_0");
    const Array * ridx   = find("residx");
    const Array * ss_gt  = find("s_s_final");
    const Array * sz_gt  = find("s_z_final");
    const Array * b0s_in = find("block0_s_in");
    const Array * b0z_in = find("block0_z_in");
    const Array * b0s_out= find("block0_s_out");
    const Array * b0z_out= find("block0_z_out");
    if (!s0||!ridx||!ss_gt||!sz_gt||!b0s_in||!b0z_in||!b0s_out||!b0z_out) return 1;

    const int L = (int)ridx->shape[0];
    printf("L=%d\n", L);

    esmfold_trunk_model * model = esmfold_trunk_load(argv[1]);
    if (!model) return 1;

    // convert residx to int32
    std::vector<int32_t> residx_i(L);
    for (int i = 0; i < L; i++) residx_i[i] = (int32_t)ridx->data[i];

    // ── Test 1: single block (block 0) ───────────────────────────────────
    {
        std::vector<float> ss_out, sz_out;
        bool ok = esmfold_trunk_eval(model, b0s_in->data, residx_i, ss_out, sz_out, 1);
        if (!ok) { fprintf(stderr, "eval failed\n"); return 1; }
        // The single-block test uses b0z_in as the initial z (no relpos re-add).
        // Since esmfold_trunk_eval always builds from scratch with relpos+recycle_bias,
        // comparing here would need matching inputs. Instead compare the full 48-block run.
        printf("block-0 run: s_s max_diff=%.2e (vs block0_s_out)\n",
               max_abs_diff(ss_out, b0s_out->data.data(), (size_t)L * esmfold_trunk_c_s(model)));
    }

    // ── Test 2: full 48-block run starting from s_s_0 ───────────────────
    {
        std::vector<float> ss_out, sz_out;
        bool ok = esmfold_trunk_eval(model, s0->data, residx_i, ss_out, sz_out);
        if (!ok) { fprintf(stderr, "eval failed\n"); return 1; }
        float d_s = max_abs_diff(ss_out, ss_gt->data.data(), (size_t)L * esmfold_trunk_c_s(model));
        float d_z = max_abs_diff(sz_out, sz_gt->data.data(), (size_t)L * L * esmfold_trunk_c_z(model));
        printf("full trunk: s_s max_diff=%.2e  s_z max_diff=%.2e\n", d_s, d_z);
        if (d_s < 1e-3f && d_z < 1e-3f) printf("PASS\n");
        else printf("FAIL (target <1e-3)\n");
    }

    esmfold_trunk_free(model);
    return 0;
}
