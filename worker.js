import init, { Model } from "./pkg/esm_rs.js";

let modelPromise = null;

async function loadModel(weightsURL, tokenizerURL, configURL) {
  if (!modelPromise) {
    modelPromise = (async () => {
      await init();

      const [wRes, tRes, cRes] = await Promise.all([
        fetch(weightsURL),
        fetch(tokenizerURL),
        fetch(configURL),
      ]);

      const [wBuf, tBuf, cBuf] = await Promise.all([
        wRes.arrayBuffer(),
        tRes.arrayBuffer(),
        cRes.arrayBuffer(),
      ]);

      const model = new Model(
        new Uint8Array(wBuf),
        new Uint8Array(tBuf),
        new Uint8Array(cBuf)
      );
      return model;
    })();
  }
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

    self.postMessage({ status: "loading", modelID });

    const model = await loadModel(weightsURL, tokenizerURL, configURL);

    const params = {
      sentences,
      normalize_embeddings: normalize,
    };

    const result = model.get_embeddings(params);
    self.postMessage({
      status: "complete",
      modelID,
      embeddings: result.data,
    });
  } catch (e) {
    self.postMessage({ error: e.message ?? String(e) });
  }
};
