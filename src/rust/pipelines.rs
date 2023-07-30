use std::{
  ops::{Mul, Div},
  fs::File,
  iter::zip,
};
use ort::{
	Value,
  Session,
};
use tokenizers::{
  Tokenizer,
  EncodeInput,
};
use ndarray::{
  CowArray,
  Array,
  Axis,
  Ix1,
  Ix2,
  Dim,
  IxDynImpl,
};
use serde::Deserialize;

const JSON_DATA_PATH: &str = "tests/scifact/data/inference_semantic_search_on_scifact.json";

#[derive(Deserialize)]
struct Corpus {
  id: String,
  sentence: String,
}

#[derive(Deserialize)]
struct Query {
  // id: String,
  sentence: String,
  corpus_ids: Vec<String>,
}

#[derive(Deserialize)]
struct InferData {
  corpus: Vec<Corpus>,
  queries: Vec<Query>,
}

pub fn inference_semantic_search_on_scidata(
  model: &Session,
  tokenizer: &Tokenizer,
  batch_size: i16,
) -> Result<f32, tokenizers::Error> {
  let file_reader = File::open(JSON_DATA_PATH)?;
  let infer_data: InferData = serde_json::from_reader(file_reader)?;
  let corpus = infer_data.corpus;
  let queries = infer_data.queries;

  let (corpus_embeddings, query_embeddings) =
    if batch_size > 1 {
      (
        get_sentence_embedding_in_batch(
          &corpus
            .iter()
            .map(|corpus| corpus.sentence.to_owned())
            .collect::<Vec<_>>(),
          &model, &tokenizer, batch_size
        )?,
        get_sentence_embedding_in_batch(
          &queries
            .iter()
            .map(|query| query.sentence.to_owned())
            .collect::<Vec<_>>(),
          &model, &tokenizer, batch_size
        )?
      )
    } else {
      (
        // tqdm!(
          corpus
            .iter()
            .filter_map(|corpus| {
              get_sentence_embedding(
                corpus.sentence.as_str(),
                &model, &tokenizer
              ).ok()
            })
            .into_iter()
            .collect::<Vec<_>>(),
        // tqdm!(
          queries
            .iter()
            .filter_map(|query| {
              get_sentence_embedding(
                query.sentence.as_str(),
                &model, &tokenizer
              ).ok()
            })
            .into_iter()
            .collect::<Vec<_>>()
      )
    };

  let mut hits = 0;
  let total = queries.len();
  for (query_embedding, corpus_ids) in /* tqdm!( */zip(
    query_embeddings.iter(),
    queries.iter().map(|query| &query.corpus_ids),
  ) {
    let mut best_match = (&corpus[0].id, f32::MIN);
    for (corpus_embedding, corpus_id) in zip(
      corpus_embeddings.iter(),
      corpus.iter().map(|corpus| &corpus.id),
    ) {
      let similarity = cosine_similarity(
        query_embedding,
        corpus_embedding,
      );
      if best_match.1 < similarity {
        best_match = (corpus_id, similarity);
      }
    }
    if corpus_ids.contains(best_match.0) {
      hits += 1;
    }
  }

  Ok(hits as f32 / total as f32)
}

fn get_sentence_embedding_in_batch<'s, S>(
  inputs: &Vec<S>,
  model: &Session,
  tokenizer: &Tokenizer,
  batch_size: i16,
) -> Result<Vec<Array<f32, Ix1>>, tokenizers::Error>
where
  S: Into<EncodeInput<'s>> + Send + Clone,
{
  // inputs: (batch_size, seq_length)
  // intermediate encoded_inputs: (batch_size, 3, seq_length)
  // final encoded_inputs: (3, batch_size, seq_length) -> [(batch_size, seq_length); 3]
  // **Inferencing**
  // token_embeddings: (batch_size, seq_length, hidden_size)
  // **Pooling**
  // sentence_embeddings: (batch_size, hidden_size)
  let mut result = Vec::<Array<f32, Ix1>>::new();
  for chunk_of_inputs in /* tqdm!( */inputs.chunks(batch_size as usize) {
    let batch_size = chunk_of_inputs.len();
    let encoded_inputs = CowArray::from(
      Array::from_iter(
        tokenizer
          .encode_batch(chunk_of_inputs.to_vec(), true)?
          .iter()
          .map(|encoded_input| {
            encoded_input.get_ids().iter()
              .chain(encoded_input.get_attention_mask())
              .chain(encoded_input.get_type_ids())
          }).flatten()
          .map(|n: &u32| *n as i64)
      )
    );
    let seq_length = encoded_inputs.len() / batch_size / 3;
    let mut encoded_inputs =
      encoded_inputs
        .to_shape((batch_size, 3, seq_length))?;
    encoded_inputs.swap_axes(0, 1);
    let encoded_inputs =
      encoded_inputs
        .axis_iter(Axis(0))
        .map(|mat| CowArray::from(mat).into_dyn())
        .collect::<Vec<_>>();
    let attention_masks =
      encoded_inputs[1]
        .to_owned()
        .insert_axis(Axis(2))
        .map(|n| *n as f32);
    let encoded_inputs =
      encoded_inputs
        .iter()
        .filter_map(|array| {
          Value::from_array(model.allocator(), array).ok()
        }).collect::<Vec<_>>();
    let token_embeddings =
      model
        .run(encoded_inputs)?[0]
        .try_extract()?;
    let sentence_embeddings_mean_pooled =
      token_embeddings
        .view()
        .mul(&attention_masks)
        .sum_axis(Axis(1))
        .div(
          attention_masks
          .sum_axis(Axis(1))
          .map(|n| n.max(f32::EPSILON))
        ).into_dimensionality::<Ix2>()?;
    let sentence_embeddings_mean_pooled =
      sentence_embeddings_mean_pooled
        .axis_iter(Axis(0))
        .map(|array| array.to_owned());
    result.extend(sentence_embeddings_mean_pooled);
  }
  Ok(result)
}

fn get_sentence_embedding<'s, S>(
  input: S,
  model: &Session,
  tokenizer: &Tokenizer,
) -> Result<Array<f32, Ix1>, tokenizers::Error>
where
  S: Into<EncodeInput<'s>> + Send,
{
  // inputs: (seq_length)
  // encoded_inputs: (3, 1, seq_length) -> [(1, seq_length); 3]
  // **Inferencing**
  // token_embeddings: (1, seq_length, hidden_size)
  // **Pooling**
  // intermediate sentence_embedding: (1, hidden_size)
  // final sentence_embedding: (hidden_size)
  fn convert_to_input(from: &[u32]) -> CowArray<'_, i64, Dim<IxDynImpl>> {
    let cast_iter =
      from.iter().map(|n: &u32| *n as i64);
    let array = Array::from_iter(cast_iter);
    CowArray::from(array).insert_axis(Axis(0)).into_dyn()
  }
  let encoded_input = tokenizer.encode(input, true)?;
  let encoded_input = vec![
    convert_to_input(encoded_input.get_ids()),
    convert_to_input(encoded_input.get_attention_mask()),
    convert_to_input(encoded_input.get_type_ids()),
  ];
  let attention_mask =
    encoded_input[1]
    .to_owned()
    .insert_axis(Axis(2))
    .map(|n| *n as f32);
  let encoded_input =
    encoded_input
    .iter()
    .filter_map(|array| {
      Value::from_array(model.allocator(), array).ok()
    }).collect::<Vec<_>>();
  let token_embeddings =
    model
    .run(encoded_input)?[0]
    .try_extract()?;
  let sentence_embedding_mean_pooled =
    token_embeddings.view()
    .mul(&attention_mask)
    .sum_axis(Axis(1))
    .div(
      attention_mask
      .sum_axis(Axis(1))
      .map(|n| n.max(f32::EPSILON))
    ).remove_axis(Axis(0))
    .into_dimensionality::<Ix1>()?;
  Ok(sentence_embedding_mean_pooled)
}

fn cosine_similarity(
  array_1: &Array<f32, Ix1>,
  array_2: &Array<f32, Ix1>,
) -> f32 {
  array_1.div(
    f32::sqrt(array_1.dot(array_1))
    .max(f32::EPSILON)
  ).dot(&array_2.div(
    f32::sqrt(array_2.dot(array_2))
    .max(f32::EPSILON))
  )
}
