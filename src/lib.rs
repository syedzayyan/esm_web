use wasm_bindgen::prelude::*;
mod models;
pub use models::bert::{BertConfig, BertModel};
pub use models::esm2::{ESM2Config, ESM2Model};
pub use tokenizers::{PaddingParams, Tokenizer};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

#[wasm_bindgen]
pub enum ModelType {
    Bert,
    Esm2,
}

#[wasm_bindgen]
pub struct Model {
    model_type: ModelType,
    bert: Option<BertModel>,
    esm2: Option<ESM2Model>,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer_bytes: Vec<u8>,
        config_bytes: Vec<u8>,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");

        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;

        // Parse model_type from config JSON
        let config_str = String::from_utf8(config_bytes.to_vec())
            .map_err(|_| JsError::new("Invalid UTF-8 config"))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)?;

        let model_type = config_value["model_type"]
            .as_str()
            .ok_or_else(|| JsError::new("Missing 'model_type' in config"))?;

        let tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| JsError::new(&e.to_string()))?;

        let model = match model_type {
            "bert" => {
                let bert_config: BertConfig = serde_json::from_str(&config_str)?;
                let bert = BertModel::load(vb, &bert_config)?;
                Model {
                    model_type: ModelType::Bert,
                    bert: Some(bert),
                    esm2: None,
                    tokenizer,
                }
            }
            "esm" | "esm2" => {
                let esm2_config: ESM2Config = serde_json::from_str(&config_str)?;
                let esm2 = ESM2Model::load(vb, esm2_config)?;
                Model {
                    model_type: ModelType::Esm2,
                    bert: None,
                    esm2: Some(esm2),
                    tokenizer,
                }
            }
            _ => return Err(JsError::new(&format!("Unknown model_type: {}", model_type))),
        };

        Ok(model)
    }

    pub fn get_embeddings(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let params: Params =
            serde_wasm_bindgen::from_value(input).map_err(|e| JsError::new(&e.to_string()))?;
        let sentences = params.sentences;
        let normalize_embeddings = params.normalize_embeddings;

        let device = Device::Cpu;

        // Set padding strategy
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }

        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let ids = tokens.get_ids().to_vec();
                Tensor::new(ids.as_slice(), &device).map_err(|e| JsError::new(&e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let attention_mask: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let mask = tokens.get_attention_mask().to_vec();
                Tensor::new(mask.as_slice(), &device).map_err(|e| JsError::new(&e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)
            .map_err(|e| JsError::new(&format!("Stack token_ids error: {}", e)))?;
        let attention_mask = Tensor::stack(&attention_mask, 0)
            .map_err(|e| JsError::new(&format!("Stack attention_mask error: {}", e)))?;

        console_log!("running inference on batch {:?}", token_ids.shape());

        let embeddings = match &mut self.model_type {
            ModelType::Bert => {
                let bert = self.bert.as_mut().expect("BERT model not loaded");
                let token_type_ids = token_ids
                    .zeros_like()
                    .map_err(|e| JsError::new(&format!("Zeros error: {}", e)))?;
                bert.forward(&token_ids, &token_type_ids, Some(&attention_mask))
                    .map_err(|e| JsError::new(&format!("BERT forward error: {}", e)))
            }
            ModelType::Esm2 => {
                let esm2 = self.esm2.as_mut().expect("ESM2 model not loaded");
                esm2.forward(&token_ids, &attention_mask)
                    .map_err(|e| JsError::new(&format!("ESM2 forward error: {}", e)))
            }
        }?;

        console_log!("generated embeddings {:?}", embeddings.shape());

        // Avg-pool over sequence length (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings
            .dims3()
            .map_err(|e| JsError::new(&format!("Dims error: {}", e)))?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if normalize_embeddings {
            embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?
        } else {
            embeddings
        };

        let embeddings_data = embeddings
            .to_vec2()
            .map_err(|e| JsError::new(&format!("to_vec2 error: {}", e)))?;

        Ok(serde_wasm_bindgen::to_value(&Embeddings {
            data: embeddings_data,
        })
        .map_err(|e| JsError::new(&format!("Serialize error: {}", e)))?)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Embeddings {
    pub data: Vec<Vec<f32>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub sentences: Vec<String>,
    pub normalize_embeddings: bool,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}
