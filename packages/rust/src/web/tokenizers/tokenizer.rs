use wasm_bindgen::prelude::*;
use js_sys::{
  JSON,
  JsString,
  Object as JsObject,
};
use std::str::FromStr;
use crate::tokenizers::encoding::Encoding;

#[wasm_bindgen]
pub struct Tokenizer {
  inner: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl Tokenizer {
  #[wasm_bindgen(constructor)]
  pub fn new(
    options: &JsObject,
  ) -> Result<Tokenizer, JsValue> {
    Ok(
      Tokenizer::from(
        tokenizers::Tokenizer::from_str(
          &JSON::stringify(options)?
            .as_string()
            .unwrap_or("".into())
        ).map_err(|e| e.to_string())?,
      )
    )
  }

  pub fn encode(
    &self,
    input: &str,
    add_special_tokens: Option<bool>,
  ) -> Result<Encoding, JsValue> {
    Ok(
      Encoding::from(
        self.inner.encode(
          input,
          add_special_tokens.unwrap_or(true),
        ).map_err(|e| e.to_string())?
      )
    )
  }

  pub fn encode_batch(
    &self,
    inputs: Box<[JsString]>,
    add_special_tokens: Option<bool>,
  ) -> Result<Encoding, JsValue> {
    Ok(
      Encoding::from(
        self.inner.encode_batch(
          inputs
            .iter()
            .map(|s| s.as_string().unwrap_or("".into()))
            .collect::<Vec<_>>(),
          add_special_tokens.unwrap_or(true),
        ).map_err(|e| e.to_string())?
      )
    )
  }
}

impl From<tokenizers::Tokenizer> for Tokenizer {
  fn from(tokenizer: tokenizers::Tokenizer) -> Self {
    Tokenizer {
      inner: tokenizer,
    }
  }
}
