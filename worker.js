import init, { Model } from "./pkg/esm_rs.js";

// Global serde helpers (if not exported)
const serde_wasm_bindgen = {
  toJsValue: (obj) => {
    /* wasm-bindgen polyfill or import */
  },
  fromJsValue: async (jsval) => jsval, // Temp
};

let modelPromise = null;
let currentModelID = null;

const modelIDMap = {
  intfloat_e5_small_v2: "bert",
  intfloat_e5_base_v2: "bert",
  intfloat_multilingual_e5_small: "bert",
  sentence_transformers_all_MiniLM_L6_v2: "bert",
  sentence_transformers_all_MiniLM_L12_v2: "bert",
  facebook_esm2_t6_8M_UR50D: "esm2",
};

async function loadModel(weightsURL, tokenizerURL, configURL, modelID) {
  if (modelPromise && currentModelID === modelID) return modelPromise;

  currentModelID = modelID;
  modelPromise = (async () => {
    await init();

    const [wRes, tRes, cRes] = await Promise.all([
      fetch(weightsURL),
      fetch(tokenizerURL),
      fetch(configURL),
    ]);

    const [wBuf, tBuf] = await Promise.all([
      wRes.arrayBuffer(),
      tRes.arrayBuffer(),
    ]);

    const configText = await cRes.text();
    const cBuf = new TextEncoder().encode(configText);

    return new Model(new Uint8Array(wBuf), new Uint8Array(tBuf), cBuf);
  })().catch((e) => {
    // Reset cache on failure so next attempt retries
    modelPromise = null;
    currentModelID = null;
    throw e;
  });

  return modelPromise;
}

self.onmessage = async (event) => {
  try {
    const {
      weightsURL,
      tokenizerURL,
      configURL,
      modelID,
      sentences,
      normalize = true,
    } = event.data;

    self.postMessage({ status: `Loading ${modelID}...` });
    console.log("[worker] loading model...");
    const model = await loadModel(weightsURL, tokenizerURL, configURL, modelID);

    console.log("[worker] model loaded:", model);

    console.log("[worker] calling get_embeddings with sentences:", sentences);
    const result = model.get_embeddings({
      sentences,
      normalize_embeddings: normalize,
    });
    console.log("[worker] result:", result);

    self.postMessage({ status: "complete", modelID, embeddings: result.data });
  } catch (e) {
    console.error("[worker] error:", e);
    self.postMessage({ error: e.message ?? String(e) });
  }
};
