#[path = "tokenizer/encoding.rs"]
pub mod encoding;

#[path = "tokenizer/tokenizer.rs"]
pub mod tokenizer;

use std::ops::{Mul, Div};
use wasm_bindgen::prelude::*;
use js_sys::Float32Array as JsFloat32Array;
use ndarray::{
  Array,
  Axis,
  Ix2,
  Ix3,
};

#[wasm_bindgen]
pub fn get_sentence_embeddings(
  last_hidden_state: &JsFloat32Array,
  attention_mask: &JsFloat32Array,
  batch_size: usize,
  sequence_size: usize,
  hidden_size: usize,
) -> Result<Box<[JsFloat32Array]>, JsValue> {
  let last_hidden_state = Array::<f32, Ix3>::from_shape_vec(
    (batch_size, sequence_size, hidden_size),
    last_hidden_state.to_vec(),
  ).map_err(|e| e.to_string())?;
  let attention_mask = Array::<f32, Ix3>::from_shape_vec(
    (batch_size, sequence_size, 1),
    attention_mask.to_vec(),
  ).map_err(|e| e.to_string())?;
  let embeddings_by_mean_pooling: Array<f32, Ix2> =
    last_hidden_state
      .mul(&attention_mask)
      .sum_axis(Axis(1))
      .div(
        attention_mask
          .sum_axis(Axis(1))
          .map(|n| n.max(f32::EPSILON))
      );
  Ok(
    embeddings_by_mean_pooling
      .axis_iter(Axis(0))
      .map(|arr| {
        JsFloat32Array::from(
          arr.as_slice().unwrap_or(&[])
        )
      }).collect::<Box<[JsFloat32Array]>>()
  )
}
