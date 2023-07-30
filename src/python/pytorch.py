#! /usr/bin/env python3

from transformers import AutoTokenizer, AutoModel, BertModel
from pathlib import Path
from optimum.bettertransformer import BetterTransformer
from .pipelines import inference_semantic_search_on_scidata

MODEL_PATH = Path('assets/sbert/all-MiniLM-L6-v2/pt')
BATCH_SIZE = 8
model = BetterTransformer.transform(
  AutoModel
  .from_pretrained(MODEL_PATH)
  .to('cpu')
  .eval()
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(inference_semantic_search_on_scidata(model, tokenizer, BATCH_SIZE))
