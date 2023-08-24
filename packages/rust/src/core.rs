#[path = "tokenizer/encoding.rs"]
pub mod encoding;

#[path = "tokenizer/tokenizer.rs"]
pub mod tokenizer;

use std::ops::{Mul, Div};

use wasm_bindgen::prelude::*;
use ndarray::{
  CowArray,
  Array,
  Axis,
  Ix1,
  Ix2,
  Dim,
  IxDynImpl,
};

#[wasm_bindgen]
pub fn get_sentence_embedding(
  last_hidden_state: &[f32], // (seq_length, hidden_size)
  attention_mask: &[f32], // (seq_length)
  seq_length: usize,
  hidden_size: usize,
) -> Result<Box<[f32]>, String> {
  let last_hidden_state: Array::<f32, Ix2> =
    Array::<f32, Ix1>
      ::from_iter(last_hidden_state.iter().map(|n| *n))
      .into_shape((seq_length, hidden_size))
      .map_err(|e| e.to_string())?;
  let attention_mask: Array::<f32, Ix2> =
    Array::<f32, Ix1>
      ::from_iter(attention_mask.iter().map(|n| *n))
      .insert_axis(Axis(1));
  let embedding_by_mean_pool = last_hidden_state
    .mul(&attention_mask)
    .sum_axis(Axis(0))
    .div(attention_mask.sum().max(f32::EPSILON));
  Ok(Box::from(
    embedding_by_mean_pool.as_slice().unwrap_or(&[])
  ))
}
//     model
//       .run(encoded_input)?[0]
//       .try_extract()?
//       .view()
//       .index_axis(Axis(0), 0)
//       .mul(&attention_mask)
//       .sum_axis(Axis(0))
//       .div(
//         attention_mask
//           .sum_axis(Axis(0))
//           .first()
//           .unwrap()
//           .max(f32::EPSILON)
//       ).into_dimensionality::<Ix1>()?;

//   // final attention_masks: (batch_size, seq_length, 1)
//   // **Inferencing**
//   // token_embeddings: (batch_size, seq_length, hidden_size)
//   // **Pooling**
//   // sentence_embeddings: (batch_size, hidden_size)
//       model
//         .run(encoded_inputs)?[0]
//         .try_extract()?
//         .view()
//         .mul(&attention_masks)
//         .sum_axis(Axis(1))
//         .div(
//           attention_masks
//             .sum_axis(Axis(1))
//             .map(|n| n.max(f32::EPSILON))
//         ).into_dimensionality::<Ix2>()?;