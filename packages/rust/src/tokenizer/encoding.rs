use wasm_bindgen::prelude::*;
use std::sync::Arc;

/// Main type: u32 ///

#[wasm_bindgen]
pub struct EncodingU32 {
  inner: Arc<tokenizers::Encoding>,
}

#[wasm_bindgen]
impl EncodingU32 {
  #[wasm_bindgen(getter)]
  pub fn as_i64(&self) -> EncodingI64 {
    EncodingI64 {
      inner: self.inner.clone(),
    }
  }

  #[wasm_bindgen(getter)]
  pub fn as_f32(&self) -> EncodingF32 {
    EncodingF32 {
      inner: self.inner.clone(),
    }
  }

  #[wasm_bindgen(getter)]
  pub fn input_ids(&self) -> Box<[u32]> {
    Box::from(self.inner.get_ids())
  }

  #[wasm_bindgen(getter)]
  pub fn attention_mask(&self) -> Box<[u32]> {
    Box::from(self.inner.get_attention_mask())
  }

  #[wasm_bindgen(getter)]
  pub fn token_type_ids(&self) -> Box<[u32]> {
    Box::from(self.inner.get_type_ids())
  }
}

impl From<tokenizers::Encoding> for EncodingU32 {
  fn from(value: tokenizers::Encoding) -> Self {
    EncodingU32 {
      inner: Arc::from(value),
    }
  }
}

/// Other types ///

#[wasm_bindgen]
pub struct EncodingI64 {
  inner: Arc<tokenizers::Encoding>,
}

#[wasm_bindgen]
impl EncodingI64 {
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

#[wasm_bindgen]
pub struct EncodingF32 {
  inner: Arc<tokenizers::Encoding>,
}

#[wasm_bindgen]
impl EncodingF32 {
  #[wasm_bindgen(getter)]
  pub fn input_ids(&self) -> Box<[f32]> {
    self.inner.get_ids().iter().map(|n| *n as f32).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn attention_mask(&self) -> Box<[f32]> {
    self.inner.get_attention_mask().iter().map(|n| *n as f32).collect()
  }

  #[wasm_bindgen(getter)]
  pub fn token_type_ids(&self) -> Box<[f32]> {
    self.inner.get_type_ids().iter().map(|n| *n as f32).collect()
  }
}
