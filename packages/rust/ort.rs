pub mod pipelines;
use crate::pipelines::inference_semantic_search_on_scidata;

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
use std::env;
// use kdam::tqdm;

const MODEL_PATH: &str = "assets/sbert/all-MiniLM-L6-v2/onnx/model.onnx";
const TOKENIZER_PATH: &str = "assets/sbert/all-MiniLM-L6-v2/onnx/tokenizer.json";

fn main() -> Result<(), tokenizers::Error> {
  let batch_size: i16 =
    env::var("BATCH_SIZE")?.parse::<i16>()?;
  let environment =
    Environment::builder()
      .with_name(MODEL_PATH)
      .with_execution_providers([
        ExecutionProvider::CPU(Default::default())
      ]).build()?
      .into_arc();
  let model =
    SessionBuilder::new(&environment)?
      .with_optimization_level(GraphOptimizationLevel::Level2)?
      .with_model_from_file(MODEL_PATH)?;
  let mut tokenizer =
    Tokenizer::from_file(TOKENIZER_PATH)?;
  let mut padding_params =
    tokenizer
      .get_padding_mut()
      .unwrap();
  padding_params.strategy = PaddingStrategy::BatchLongest;
  let tokenizer = tokenizer;

  println!("{:#?}", tokenizer);

  // let result = inference_semantic_search_on_scidata(
  //   &model, &tokenizer, batch_size,
  // )?;

  // println!("{result}");
	Ok(())
}
