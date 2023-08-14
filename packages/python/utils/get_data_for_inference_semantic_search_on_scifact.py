#! /usr/bin/env python3

import csv
import json
from pathlib import Path
from typing_extensions import TypedDict

ASSETS_PATH = Path('tests/scifact/assets')
CORPUS_PATH = ASSETS_PATH / 'corpus.jsonl'
QUERIES_PATH = ASSETS_PATH / 'queries.jsonl'
QREL_PATHS = [ASSETS_PATH / path for path in [
  'qrels/test.tsv', 'qrels/train.tsv'
]]

DATA_PATH = ASSETS_PATH / '../data'
JSON_DATA_PATH = DATA_PATH / 'inference_semantic_search_on_scifact.json'

class Corpus(TypedDict):
  id: str
  sentence: str

class Query(TypedDict):
  id: str
  sentence: str
  corpus_ids: list[str]

class InferData(TypedDict):
  corpus: list[Corpus]
  queries: list[Query]

def get_data_for_inference_semantic_search_on_scifact() -> InferData:
  query_dict: dict[str, Query] = {}
  corpus_dict: dict[str, Corpus] = {}

  with open(CORPUS_PATH, 'r') as file:
    for line in file:
      json_dict = json.loads(line)
      corpus_dict[json_dict['_id']] = Corpus(
        id = json_dict['_id'],
        sentence = json_dict['title'],
      )

  with open(QUERIES_PATH, 'r') as file:
    for line in file:
      json_dict = json.loads(line)
      query_dict[json_dict['_id']] = Query(
        id = json_dict['_id'],
        sentence = json_dict['text'],
        corpus_ids = [],
      )

  for qrel_path in QREL_PATHS:
    with open(qrel_path, 'r') as file:
      reader = csv.DictReader(
        file,
        delimiter = '\t',
        lineterminator = '\r\n'
      )
      for line in reader:
        (query_dict[line['query-id']]['corpus_ids']
          .append(line['corpus-id']))

  return InferData(
    queries = list(query_dict.values()),
    corpus = list(corpus_dict.values()),
  )

if __name__ == '__main__':
  DATA_PATH.mkdir(parents = True, exist_ok = True)
  json.dump(
    get_data_for_inference_semantic_search_on_scifact(),
    open(JSON_DATA_PATH, 'w')
  )
