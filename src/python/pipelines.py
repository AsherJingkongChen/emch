from typing import Iterable
import torch
import json
from tqdm import tqdm
from .utils.get_data_for_inference_semantic_search_on_scifact import (
  JSON_DATA_PATH,
  InferData,
)

F32_EPSILON = torch.finfo(torch.float32).eps

def iter_to_chunks(iter: Iterable, chunk_size: int) -> Iterable:
  counter = 0
  chunk = []
  for i in iter:
    if counter >= chunk_size:
      yield chunk
      counter = 0
      chunk = []
    counter += 1
    chunk.append(i)
  return chunk

def get_sentence_embedding(
  input: str,
  model,
  tokenizer,
) -> torch.Tensor:
  encoded_inputs = tokenizer(
    input,
    padding = True,
    truncation = True,
    return_tensors = 'pt',
  )
  with torch.no_grad():
    outputs = model(**encoded_inputs)
  token_embeddings = outputs[0]
  attention_mask = encoded_inputs['attention_mask'].unsqueeze(-1).float()
  return ((token_embeddings * attention_mask).sum(1) /
    torch.clamp_min(attention_mask.sum(1), F32_EPSILON))

def get_sentence_embedding_in_batch(
  inputs: Iterable[str],
  model,
  tokenizer,
  batch_size: int,
) -> torch.Tensor:
  sentence_embedding_in_chunks: list[torch.Tensor] = []
  for chunk_of_inputs in tqdm(iter_to_chunks(inputs, batch_size)):
    sentence_embedding_in_chunks.append(
      get_sentence_embedding(chunk_of_inputs, model, tokenizer)
    )
  return torch.cat(sentence_embedding_in_chunks)

def inference_semantic_search_on_scidata(
  model,
  tokenizer,
  batch_size: int,
) -> float:
  infer_data: InferData = json.load(open(JSON_DATA_PATH, 'r'))
  corpus = infer_data['corpus']
  queries = infer_data['queries']
  if batch_size > 1:
    corpus_embeddings = get_sentence_embedding_in_batch([
      item['sentence'] for item in corpus
    ], model, tokenizer, batch_size)
    query_embeddings = get_sentence_embedding_in_batch([
      item['sentence'] for item in queries
    ], model, tokenizer, batch_size)
  else:
    corpus_embeddings = torch.cat([
      get_sentence_embedding(
        item['sentence'], model, tokenizer
      ) for item in tqdm(corpus)
    ])
    query_embeddings = torch.cat([
      get_sentence_embedding(
        item['sentence'], model, tokenizer
      ) for item in tqdm(queries)
    ])

  hits = 0; total = len(queries)
  for query_embedding, query in tqdm(zip(query_embeddings, queries)):
    hits += int(
      corpus[
        torch.cosine_similarity(
          query_embedding,
          corpus_embeddings,
          dim = 1,
          eps = F32_EPSILON,
        ).argmax()
      ]['id']
      in query['corpus_ids']
    )

  return (hits / total)
