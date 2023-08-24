#! /usr/bin/env python3

from transformers import AutoTokenizer, AutoModel, BertModel
from pathlib import Path
from optimum.bettertransformer import BetterTransformer
from .pipelines import inference_semantic_search_on_scidata
from os import getenv

BATCH_SIZE = int(getenv('BATCH_SIZE'))
MODEL_PATH = Path('artifacts/sentence-transformers_all-MiniLM-L6-v2/pt')

model = BetterTransformer.transform(
  AutoModel
    .from_pretrained(MODEL_PATH)
    .to('cpu')
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(inference_semantic_search_on_scidata(
  model, tokenizer, BATCH_SIZE
))
