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
use tokenizers::Tokenizer;
use std::str::FromStr;
use js_sys::JsString;

#[wasm_bindgen]
pub fn get_vocab_size() -> Result<usize, JsString> {
  let a = Tokenizer::from_str(
    include_str!("../../artifacts/sentence-transformers_all-MiniLM-L6-v2/onnx/tokenizer.json")
  ).map_err(|e| JsString::from(format!("{e}")))?;

  Ok(a.get_vocab_size(true))
}

#[wasm_bindgen]
pub fn addd(a: u32, b: u32) -> u32 {
  let result = a + b;
  result
}
