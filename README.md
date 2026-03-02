# Embeddings -> tSNE on the web

A lot of it is stolen code. Candle RS's repo has a lot of examples. HTMX, D3 and Bootstrap is used.


```Rust
cargo build --release --target wasm32-unknown-unknown

wasm-bindgen target/wasm32-unknown-unknown/release/esm_rs.wasm \
  --out-dir pkg \
  --target web
```