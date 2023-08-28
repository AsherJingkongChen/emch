#[path = "./tokenizers/encoding.rs"]
pub mod encoding;

#[path = "./tokenizers/tokenizer.rs"]
pub mod tokenizer;

use std::ops::{Mul, Div};
use wasm_bindgen::prelude::*;
use js_sys::{
  Float32Array as JsFloat32Array,
  BigInt64Array as JsBigInt64Array,
};
use ndarray::{
  Array,
  Axis,
  Ix2,
  Ix3, Ix1,
};

#[wasm_bindgen]
pub struct Emch;

#[wasm_bindgen]
impl Emch {
  #[wasm_bindgen]
  pub fn get_sentence_embeddings(
    last_hidden_state: &JsFloat32Array,
    attention_mask: &JsBigInt64Array,
    batch_size: usize,
    sequence_size: usize,
    hidden_size: usize,
  ) -> Result<Box<[JsFloat32Array]>, JsValue> {
    let last_hidden_state =
      Array::<f32, Ix3>::from_shape_vec(
        (batch_size, sequence_size, hidden_size),
        last_hidden_state.to_vec(),
      ).map_err(|e| e.to_string())?;
    let attention_mask =
      Array::<i64, Ix3>::from_shape_vec(
        (batch_size, sequence_size, 1),
        attention_mask.to_vec(),
      ).map_err(|e| e.to_string())?
      .map(|n| *n as f32);
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

  #[wasm_bindgen]
  pub fn cosine_similarity(
    embedding_0: &JsFloat32Array,
    embedding_1: &JsFloat32Array,
  ) -> f32 {
    let vecs = (
      Array::<f32, Ix1>::from(embedding_0.to_vec()),
      Array::<f32, Ix1>::from(embedding_1.to_vec()),
    );
    let norms = (
      f32::sqrt(vecs.0.dot(&vecs.0)).max(f32::EPSILON),
      f32::sqrt(vecs.1.dot(&vecs.1)).max(f32::EPSILON),
    );
    let normed_vecs = (
      vecs.0.div(norms.0),
      vecs.1.div(norms.1),
    );
    normed_vecs.0.dot(&normed_vecs.1)
  }
}