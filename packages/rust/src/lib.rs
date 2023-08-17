// use std::str::FromStr;

// use tokenizers::{
//   Tokenizer,
//   Error,
// };
// use js_sys::{
//   JsString,
// };
// use wasm_bindgen::prelude::*;

// fn main() -> Result<(), Error> {
//   let tokenizer = TokenizerForJS::from_string(include_str!("../../artifacts/sentence-transformers_all-MiniLM-L6-v2/onnx/tokenizer.json").into())?;
//   println!("Hello, {:?}!", tokenizer.tokenizer.get_vocab_size(true));
//   Ok(())
// }

// struct TokenizerForJS {
//   pub tokenizer: Tokenizer, // temp pub for dev
// }

// impl TokenizerForJS {
//   pub fn from_string(s: JsString) -> Result<Self, Error> {
//     Ok(Self {
//       tokenizer: Tokenizer::from_str(&s.as_string().unwrap())?
//     })
//   }
// }
use wasm_bindgen::prelude::*;
use std::str::FromStr;

// #[wasm_bindgen]
// pub fn get_vocab_size() -> Result<usize, String> {
//   let a = tokenizers::Tokenizer::from_str(
//     include_str!("../../../artifacts/sentence-transformers_all-MiniLM-L6-v2/onnx/tokenizer.json")
//   ).map_err(|e| e.to_string())?;

//   Ok(a.get_vocab_size(true))
// }

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

  #[wasm_bindgen]
  pub fn encode_string(
    &self,
    input_string: &str,
    add_special_tokens: Option<bool>,
  ) -> Result<Encoding, String> {
    Ok(Encoding {
      inner: self.inner
        .encode(input_string, add_special_tokens.unwrap_or(true))
        .map_err(|e| e.to_string())?
    })
  }
}

#[wasm_bindgen]
pub struct Encoding {
  inner: tokenizers::Encoding,
}

#[wasm_bindgen]
impl Encoding {
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
