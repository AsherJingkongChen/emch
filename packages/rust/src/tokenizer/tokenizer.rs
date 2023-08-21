use wasm_bindgen::prelude::*;
use std::str::FromStr;
use crate::encoding::EncodingU32;

#[wasm_bindgen]
pub struct Tokenizer {
  inner: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl Tokenizer {
  #[wasm_bindgen(constructor)]
  pub fn from_string(json_string: &str) -> Result<Tokenizer, String> {
    Ok(Tokenizer {
      inner: tokenizers::Tokenizer
        ::from_str(json_string)
        .map_err(|e| e.to_string())?
    })
  }

  pub fn encode_string(
    &self,
    input_string: &str,
    add_special_tokens: Option<bool>,
  ) -> Result<EncodingU32, String> {
    Ok(EncodingU32::from(
      self.inner
        .encode(
          input_string,
          add_special_tokens.unwrap_or(true),
        ).map_err(|e| e.to_string())?
    ))
  }
}