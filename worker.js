// ESM2 embedding worker — backed by the C/ggml WASM build.
//
// Built artifacts required (make -f cpp/Makefile.wasm):
//   ./esm2.js   — Emscripten glue
//   ./esm2.wasm — compiled module
//   ./esm2-small.gguf — model weights (copy from chp/esm/esm2-f32.gguf)
//
// Receives: { sentences: string[] }
// Sends:    { status: string }
//           { status: "complete", embeddings: number[][] }
//           { error: string }

import ESM2Module from "./esm2.js";

const GGUF_URL = "./esm2-small.gguf";

let Module    = null;
let loadFn, embedFn, getHiddenFn, nEmbdFn;
let modelReady = false;

async function ensureReady() {
    if (modelReady) return;

    // Instantiate Emscripten module
    if (!Module) {
        self.postMessage({ status: "Initialising WASM module…" });
        Module = await ESM2Module();
        loadFn      = Module.cwrap("esm2_wasm_load",      "number", ["number","number"]);
        embedFn     = Module.cwrap("esm2_wasm_embed",     "number", ["string"]);
        getHiddenFn = Module.cwrap("esm2_wasm_get_hidden","number", []);
        nEmbdFn     = Module.cwrap("esm2_wasm_n_embd",    "number", []);
    }

    // Fetch GGUF
    self.postMessage({ status: "Fetching model weights (28 MB)…", downloadProgress: { label: "Fetching model…", percent: 0 } });
    const resp = await fetch(GGUF_URL);
    if (!resp.ok) throw new Error(`GGUF fetch failed (${resp.status}): ${GGUF_URL}`);

    const buf  = await resp.arrayBuffer();
    self.postMessage({ status: "Loading model…", downloadProgress: { label: "Loading model…", percent: 90 } });

    const view = new Uint8Array(buf);
    const ptr  = Module._malloc(view.length);
    Module.HEAPU8.set(view, ptr);
    const ok = loadFn(ptr, view.length);
    Module._free(ptr);

    if (!ok) throw new Error("esm2_wasm_load returned failure");
    modelReady = true;
    self.postMessage({ status: `Model ready (${nEmbdFn()}-d embeddings)`, downloadProgress: { label: "Done", percent: 100 } });
}

self.onmessage = async (event) => {
    try {
        const { sentences } = event.data;

        await ensureReady();

        const nEmbd = nEmbdFn();
        const embeddings = [];

        for (let i = 0; i < sentences.length; i++) {
            // Strip to canonical amino-acid alphabet; U→C (selenocysteine)
            const seq = sentences[i]
                .toUpperCase()
                .replace(/U/g, "C")
                .replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, "");

            self.postMessage({
                status: `Embedding ${i + 1} / ${sentences.length}…`,
                downloadProgress: {
                    label: `Embedding sequences…`,
                    percent: Math.round(100 * i / sentences.length),
                },
            });

            if (!seq) { embeddings.push(new Array(nEmbd).fill(0)); continue; }

            const nTok = embedFn(seq);
            if (!nTok) { embeddings.push(new Array(nEmbd).fill(0)); continue; }

            // Copy hidden states out before the next call overwrites the buffer
            const ptr    = getHiddenFn();
            const hidden = new Float32Array(Module.HEAPF32.buffer, ptr, nTok * nEmbd).slice();

            // Mean-pool over all tokens (including BOS/EOS, matching training convention)
            const mean = new Float32Array(nEmbd);
            for (let t = 0; t < nTok; t++)
                for (let d = 0; d < nEmbd; d++) mean[d] += hidden[t * nEmbd + d];
            for (let d = 0; d < nEmbd; d++) mean[d] /= nTok;

            // L2 normalise — improves cosine-based t-SNE clustering
            let norm = 0;
            for (let d = 0; d < nEmbd; d++) norm += mean[d] * mean[d];
            norm = Math.sqrt(norm) || 1;
            const vec = Array.from(mean, v => v / norm);
            embeddings.push(vec);
        }

        self.postMessage({ status: "complete", embeddings });

    } catch (e) {
        console.error("[worker]", e);
        self.postMessage({ error: e.message ?? String(e) });
    }
};
