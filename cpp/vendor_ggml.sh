#!/usr/bin/env bash
# Download the minimal ggml v0.9.7 sources needed for a WASM build.
# Run once before `make -f Makefile.wasm`.
#
# Note: ggml's tree was restructured well before this tag — there is no
# monolithic ggml-backend.c/gguf.c anymore, just ggml-backend.cpp/gguf.cpp
# plus a handful of split-out ggml-cpu/*.cpp files. The list below is the
# CPU-only subset (no CUDA/Metal/Vulkan/etc, no dynamic backend registry —
# we call ggml_backend_cpu_init() directly, so ggml-backend-reg.cpp/
# ggml-backend-dl.cpp and their dlopen/filesystem deps aren't needed).

set -euo pipefail

GGML_VERSION_TAG="v0.9.7"
GGML_REPO="https://github.com/ggml-org/ggml.git"
DEST="vendor/ggml"

if [ -d "$DEST/.git" ]; then
    echo "$DEST already present, skipping clone."
    exit 0
fi

echo "Fetching ggml ${GGML_VERSION_TAG} (sparse, source + include only)..."
mkdir -p "$DEST"
cd "$DEST"
git init -q
git remote add origin "$GGML_REPO"
git fetch --depth 1 origin "tag" "$GGML_VERSION_TAG"
git checkout -q FETCH_HEAD

git sparse-checkout init --no-cone
cat > .git/info/sparse-checkout <<'EOF'
/include/
/src/ggml.c
/src/ggml.cpp
/src/ggml-alloc.c
/src/ggml-backend.cpp
/src/ggml-backend-impl.h
/src/ggml-threading.cpp
/src/ggml-threading.h
/src/ggml-quants.c
/src/ggml-quants.h
/src/ggml-impl.h
/src/ggml-common.h
/src/gguf.cpp
/src/ggml-cpu/ggml-cpu.c
/src/ggml-cpu/ggml-cpu.cpp
/src/ggml-cpu/ggml-cpu-impl.h
/src/ggml-cpu/arch-fallback.h
/src/ggml-cpu/common.h
/src/ggml-cpu/simd-mappings.h
/src/ggml-cpu/quants.c
/src/ggml-cpu/quants.h
/src/ggml-cpu/traits.cpp
/src/ggml-cpu/traits.h
/src/ggml-cpu/binary-ops.cpp
/src/ggml-cpu/binary-ops.h
/src/ggml-cpu/unary-ops.cpp
/src/ggml-cpu/unary-ops.h
/src/ggml-cpu/vec.cpp
/src/ggml-cpu/vec.h
/src/ggml-cpu/ops.cpp
/src/ggml-cpu/ops.h
/src/ggml-cpu/repack.cpp
/src/ggml-cpu/repack.h
/src/ggml-cpu/hbm.cpp
/src/ggml-cpu/hbm.h
/src/ggml-cpu/amx/amx.h
/src/ggml-cpu/amx/amx.cpp
/src/ggml-cpu/amx/mmq.h
/src/ggml-cpu/amx/mmq.cpp
/src/ggml-cpu/amx/common.h
/src/ggml-cpu/arch/wasm/quants.c
EOF
git checkout

echo "Done. ggml sources in $DEST/"
echo "Now run: make -f cpp/Makefile.wasm"
