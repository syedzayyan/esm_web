use super::utils::{layer_norm, linear, Dropout, HiddenAct, HiddenActLayer, LayerNorm, Linear};
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ESM2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: HiddenAct,
    #[serde(default)]
    pub hidden_dropout_prob: f64,
    #[serde(default)]
    pub attention_probs_dropout_prob: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    #[serde(default)]
    pub emb_layer_norm_before: bool,
    #[serde(default)]
    pub use_cache: bool,
}

impl Default for ESM2Config {
    fn default() -> Self {
        Self {
            vocab_size: 33,
            hidden_size: 320,
            num_hidden_layers: 6,
            num_attention_heads: 20,
            intermediate_size: 1280,
            max_position_embeddings: 1026,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
            emb_layer_norm_before: false,
            use_cache: true,
        }
    }
}

// ── Embeddings ────────────────────────────────────────────────────────────────
// No LayerNorm in embeddings — just word_embeddings

struct ESM2Embeddings {
    word_embeddings: Embedding,
    dropout: Dropout,
}

impl ESM2Embeddings {
    fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        Ok(Self {
            word_embeddings,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let x = self.word_embeddings.forward(input_ids)?;
        self.dropout.forward(&x)
    }
}

// ── Rotary embeddings ─────────────────────────────────────────────────────────
// Load inv_freq from weights, compute cos/sin on the fly

fn compute_cos_sin(inv_freq: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
    let device = inv_freq.device();
    let t = Tensor::arange(0f32, seq_len as f32, device)?.to_dtype(inv_freq.dtype())?;
    // outer product: (seq_len, head_dim/2)
    let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    // repeat: (seq_len, head_dim)
    let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (batch, heads, seq, head_dim)
    // cos/sin: (seq, head_dim) -> unsqueeze to (1, 1, seq, head_dim)
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

// ── Self-attention ────────────────────────────────────────────────────────────

struct ESM2SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out_proj: Linear,
    inv_freq: Tensor,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl ESM2SelfAttention {
    fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        // weight keys: attention.self.query, .key, .value
        let query = linear(config.hidden_size, config.hidden_size, vb.pp("self.query"))?;
        let key = linear(config.hidden_size, config.hidden_size, vb.pp("self.key"))?;
        let value = linear(config.hidden_size, config.hidden_size, vb.pp("self.value"))?;
        let out_proj = linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        // Load rotary inv_freq from weights
        let inv_freq = vb
            .pp("self.rotary_embeddings")
            .get((attention_head_size / 2,), "inv_freq")?;
        Ok(Self {
            query,
            key,
            value,
            out_proj,
            inv_freq,
            dropout: Dropout::new(config.attention_probs_dropout_prob),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;
        x.reshape((
            batch,
            seq,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .transpose(1, 2)?
        .contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let q = self.transpose_for_scores(&self.query.forward(hidden_states)?)?;
        let k = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
        let v = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;

        let seq_len = q.dim(2)?;
        let (cos, sin) = compute_cos_sin(&self.inv_freq, seq_len)?;

        let q = apply_rotary(&q, &cos, &sin)?;
        let k = apply_rotary(&k, &cos, &sin)?;

        let scale = (self.attention_head_size as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let scores = scores.broadcast_add(attention_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let probs = self.dropout.forward(&probs)?;

        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .flatten_from(D::Minus2)?;

        self.out_proj.forward(&ctx)
    }
}

// ── Encoder layer ─────────────────────────────────────────────────────────────

struct ESM2Layer {
    attention: ESM2SelfAttention,
    attention_layer_norm: LayerNorm, // attention.LayerNorm
    intermediate: Linear,
    output: Linear,
    output_layer_norm: LayerNorm, // LayerNorm (top-level in layer)
    intermediate_act: HiddenActLayer,
    dropout: Dropout,
}

impl ESM2Layer {
    fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let attention = ESM2SelfAttention::load(vb.pp("attention"), config)?;
        let attention_layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attention.LayerNorm"),
        )?;
        let intermediate = linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate.dense"),
        )?;
        let output = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        let output_layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            attention,
            attention_layer_norm,
            intermediate,
            output,
            output_layer_norm,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, x: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Pre-norm: normalise input before attention, then residual
        let normed = self.attention_layer_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, attention_mask)?;
        let x = (x + attn_out)?;

        // Pre-norm: normalise before FFN, then residual
        let normed = self.output_layer_norm.forward(&x)?;
        let h = self
            .intermediate_act
            .forward(&self.intermediate.forward(&normed)?)?;
        let h = self.dropout.forward(&self.output.forward(&h)?)?;
        x + h
    }
}

// ── Top-level model ───────────────────────────────────────────────────────────

pub struct ESM2Model {
    embeddings: ESM2Embeddings,
    layers: Vec<ESM2Layer>,
    emb_layer_norm_after: LayerNorm, // encoder.emb_layer_norm_after
}

impl ESM2Model {
    pub fn load(vb: VarBuilder, config: ESM2Config) -> Result<Self> {
        let vb = vb.pp("esm");
        let embeddings = ESM2Embeddings::load(vb.pp("embeddings"), &config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| ESM2Layer::load(vb.pp(&format!("encoder.layer.{i}")), &config))
            .collect::<Result<Vec<_>>>()?;
        let emb_layer_norm_after = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("encoder.emb_layer_norm_after"),
        )?;
        Ok(Self {
            embeddings,
            layers,
            emb_layer_norm_after,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids)?;
        let attention_mask = get_extended_attention_mask(attention_mask, x.dtype())?;

        for layer in &self.layers {
            x = layer.forward(&x, &attention_mask)?;
        }
        self.emb_layer_norm_after.forward(&x)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?.to_dtype(dtype)?;
    (mask.ones_like()? - &mask)?.broadcast_mul(
        &Tensor::try_from(f32::MIN / 2.0)?
            .to_dtype(dtype)?
            .to_device(mask.device())?,
    )
}
