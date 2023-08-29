pub mod functional {
  pub mod metric;
  pub mod task;
}
pub mod tokenizers {
  pub mod encoding;
  pub mod tokenizer;
}

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn cu() -> String {
  format!("{:?}", "")
}
