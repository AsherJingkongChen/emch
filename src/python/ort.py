#! /usr/bin/env python3

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path
from .pipelines import inference_semantic_search_on_scidata
from os import getenv

BATCH_SIZE = int(getenv('BATCH_SIZE'))
MODEL_PATH = Path('artifacts/sentence-transformers_all-MiniLM-L6-v2/onnx')

model = (ORTModelForFeatureExtraction
  .from_pretrained(MODEL_PATH)
  .to('cpu'))
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(inference_semantic_search_on_scidata(
  model, tokenizer, BATCH_SIZE
))
