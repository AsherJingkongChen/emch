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
  /// builds with options from `tokenizer.json`
  #[wasm_bindgen(constructor)]
  pub fn from_js(
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
    parallelism: Option<bool>,
  ) -> Result<Encoding, JsValue> {
    let inputs =
      inputs.iter().map(|s| s.as_string().unwrap_or("".into()));
    let add_special_tokens =
      add_special_tokens.unwrap_or(true);
    let parallelism = parallelism.unwrap_or(false);
    Ok(
      Encoding::from(
        if parallelism {
          // crates tokenizers uses parallelism in encode_batch() by default
          self.inner
            .encode_batch(
              inputs.collect::<Vec<String>>(),
              add_special_tokens,
            ).map_err(|e| e.to_string())?
        } else {
          use tokenizers::{Result, utils::padding::pad_encodings};
          let mut encodings =
            inputs
              .map(|input| {
                self.inner.encode(input, add_special_tokens)
              }).collect::<Result<Vec<tokenizers::Encoding>>>()
              .map_err(|e| e.to_string())?;
          if let Some(params) = self.inner.get_padding() {
            pad_encodings(&mut encodings, params)
              .map_err(|e| e.to_string())?;
          }
          encodings
        }
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
