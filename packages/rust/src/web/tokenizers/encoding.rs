use wasm_bindgen::prelude::*;
use std::sync::Arc;

#[wasm_bindgen]
pub struct Encoding {
  inner: Arc<tokenizers::Encoding>,
}

#[wasm_bindgen]
impl Encoding {
  #[wasm_bindgen(getter)]
  pub fn input_ids(&self) -> Box<[i64]> {
    self.inner.get_ids().iter().map(|n| *n as i64).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn attention_mask(&self) -> Box<[i64]> {
    self.inner.get_attention_mask().iter().map(|n| *n as i64).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn token_type_ids(&self) -> Box<[i64]> {
    self.inner.get_type_ids().iter().map(|n| *n as i64).collect()
  }
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
