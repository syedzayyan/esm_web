#pragma once

#include "ggml.h"

// LayerNorm: normalize x, then affine transform with weight w and bias b.
static inline ggml_tensor * layer_norm(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

// Linear: x @ w^T + b  (w stored as [out, in]).
static inline ggml_tensor * linear(
        ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(ctx, w, x);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}
