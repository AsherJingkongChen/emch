use std::ops::Div;
use wasm_bindgen::prelude::*;
use js_sys::Float32Array as JsFloat32Array;
use ndarray::{
  Array,
  Ix1,
};

#[wasm_bindgen]
pub struct Metric;

#[wasm_bindgen]
impl Metric {
  #[wasm_bindgen]
  pub fn get_cosine_similarity(
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