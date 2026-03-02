# Embeddings -> Visualisations using tSNE

A lot of it is stolen code. [Candle RS's](https://github.com/huggingface/candle) repo has a lot of examples, from where the BERT model is implemented. The ESM2 model is inspired from BERT code and also the Pytorch code base. Implemeted models:

- BERT
- ESM2

The frontend is built with vanilla JavaScript and the UI is built using [Oat](https://oat.ink/). 

If you want to run the app locally, you need to install Rust and Cargo. Node.js is also required if you want to run the web app locally.

```bash
cargo build --release --target wasm32-unknown-unknown

wasm-bindgen target/wasm32-unknown-unknown/release/esm_rs.wasm \
  --out-dir pkg \
  --target web
  
npx serve # Start the server, if you have Node.js installed
```
