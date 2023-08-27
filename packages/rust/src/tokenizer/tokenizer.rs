use wasm_bindgen::prelude::*;
use js_sys::{
  JsString,
  Object as JsObject,
  JSON,
};
use std::str::FromStr;
use crate::encoding::Encoding;

#[wasm_bindgen]
pub struct Tokenizer {
  inner: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl Tokenizer {
  #[wasm_bindgen(constructor)]
  pub fn from_object(
    options: &JsObject,
  ) -> Result<Tokenizer, JsValue> {
    Ok(Tokenizer {
      inner:
        tokenizers::Tokenizer::from_str(
          JSON::stringify(options)?
            .as_string()
            .ok_or("Cannot stringify options")?
            .as_str()
        ).map_err(|e| e.to_string())?,
    })
  }

  pub fn encode_string(
    &self,
    input_string: &str,
    add_special_tokens: Option<bool>,
  ) -> Result<Encoding, JsValue> {
    Ok(Encoding::from(
      self.inner
        .encode(
          input_string,
          add_special_tokens.unwrap_or(true),
        ).map_err(|e| e.to_string())?
    ))
  }

  pub fn encode_strings(
    &self,
    input_strings: Box<[JsString]>,
    add_special_tokens: Option<bool>,
  ) -> Result<Encoding, JsValue> {
    Ok(Encoding::from(
      self.inner
        .encode_batch(
          input_strings
            .iter()
            .filter_map(|s| s.as_string())
            .collect::<Vec<_>>(),
          add_special_tokens.unwrap_or(true),
        ).map_err(|e| e.to_string())?
    ))
  }
}