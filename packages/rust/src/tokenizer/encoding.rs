use wasm_bindgen::prelude::*;
use std::sync::Arc;

#[wasm_bindgen]
pub struct Encoding {
  inner: Arc<tokenizers::Encoding>,
}

impl From<tokenizers::Encoding> for Encoding {
  fn from(encoding: tokenizers::Encoding) -> Self {
    Encoding {
      inner: Arc::from(encoding),
    }
  }
}

impl From<Vec<tokenizers::Encoding>> for Encoding {
  fn from(encodings: Vec<tokenizers::Encoding>) -> Self {
    Encoding {
      inner: Arc::from(
        tokenizers::Encoding::merge(encodings, true)
      ),
    }
  }
}

#[wasm_bindgen]
impl Encoding {
  #[wasm_bindgen(getter)]
  pub fn input_ids_i64(&self) -> Box<[i64]> {
    self.inner.get_ids().iter().map(|n| *n as i64).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn attention_mask_i64(&self) -> Box<[i64]> {
    self.inner.get_attention_mask().iter().map(|n| *n as i64).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn attention_mask_f32(&self) -> Box<[f32]> {
    self.inner.get_attention_mask().iter().map(|n| *n as f32).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn token_type_ids_i64(&self) -> Box<[i64]> {
    self.inner.get_type_ids().iter().map(|n| *n as i64).collect()
  }
}
