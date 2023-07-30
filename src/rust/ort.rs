pub mod pipelines;
use crate::pipelines::*;

use std::env;
use ort::{
  Environment,
  ExecutionProvider,
  GraphOptimizationLevel,
  SessionBuilder,
};
use tokenizers::{
  Tokenizer,
  PaddingStrategy,
};
// use kdam::tqdm;

const MODEL_PATH: &str = "assets/sbert/all-MiniLM-L6-v2/onnx/model.onnx";
const TOKENIZER_PATH: &str = "assets/sbert/all-MiniLM-L6-v2/onnx/tokenizer.json";
const BATCH_SIZE: i16 = 2;

fn main() -> Result<(), tokenizers::Error> {
  env::set_var("OMP_NUM_THREADS", stringify!(BATCH_SIZE));

  let environment =
    Environment::builder()
    .with_name(MODEL_PATH)
    .with_execution_providers(
      [ExecutionProvider::CPU(Default::default())]
    ).build()?
    .into_arc();

  let model =
    SessionBuilder::new(&environment)?
    .with_optimization_level(GraphOptimizationLevel::Level2)?
    .with_parallel_execution(true)?
    .with_inter_threads(1)?
    .with_intra_threads(BATCH_SIZE)?
    .with_model_from_file(MODEL_PATH)?;
  let mut tokenizer =
    Tokenizer::from_file(TOKENIZER_PATH)?;
  let mut padding_params =
    tokenizer
    .get_padding_mut()
    .ok_or("Expected padding parameters in the tokenizer")?;
  padding_params.strategy = PaddingStrategy::BatchLongest;

  let result = inference_semantic_search_on_scidata(
    &model, &tokenizer, BATCH_SIZE,
  )?;

  println!("{result}");
	Ok(())
}

// use rust_bert::pipelines::sentence_embeddings::{
//   SentenceEmbeddingsBuilder,
//   SentenceEmbeddingsModelType,
// };
// fn cosine_similarity_from_vec(
//   vec_1: Vec<f32>,
//   vec_2: Vec<f32>,
// ) -> f32 {
//   cosine_similarity(
//     &Array::<f32, Ix1>::from_vec(vec_1),
//     &Array::<f32, Ix1>::from_vec(vec_2)
//   )
// }

// let model =
//   SentenceEmbeddingsBuilder::remote(
//     SentenceEmbeddingsModelType::AllMiniLmL6V2
//   ).with_device(tch::Device::Cpu)
//   .create_model()?;
// let mut corpus_embeddings = Vec::<Vec<f32>>::new();
// let mut query_embeddings = Vec::<Vec<f32>>::new();
// for chunk_of_corpus in corpus.chunks(BATCH_SIZE as usize) {
//   corpus_embeddings.extend(
//     model
//     .encode(
//       &chunk_of_corpus
//       .iter()
//       .map(|corpus| corpus.sentence.as_str())
//       .collect::<Vec<_>>()
//     )?
//     .iter()
//     .map(Vec::to_owned)
//   );
// }
// for chunk_of_query in queries.chunks(BATCH_SIZE as usize) {
//   query_embeddings.extend(
//     model
//     .encode(
//       &chunk_of_query
//       .iter()
//       .map(|query| query.sentence.as_str())
//       .collect::<Vec<_>>()
//     )?
//     .iter()
//     .map(Vec::to_owned)
//   );
// }
