use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;


#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { inner, span })
}



impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}


#[derive(Clone, Debug)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

// impl LayerNorm {
//     pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
//         let inner = candle_nn::LayerNorm::new(weight, bias, eps);
//         let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
//         Self { inner, span }
//     }
// }

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

pub fn layer_norm<C: Into<candle_nn::LayerNormConfig>>(
    size: usize,
    c: C,
    vb: VarBuilder,
) -> Result<LayerNorm> {
    let inner = candle_nn::layer_norm(size, c, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "layer-norm");
    Ok(LayerNorm { inner, span })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    GeluApproximate,
    Relu,
}

#[derive(Clone)]
pub struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    pub fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu_erf(),
            HiddenAct::GeluApproximate => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Clone)]
pub struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    pub fn new(pr: f64) -> Self {
        Self { pr }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}