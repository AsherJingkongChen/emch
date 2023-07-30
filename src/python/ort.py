#! /usr/bin/env python3

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path
from .pipelines import inference_semantic_search_on_scidata

MODEL_PATH = Path('assets/sbert/all-MiniLM-L6-v2/onnx')
BATCH_SIZE = 2
model = (
  ORTModelForFeatureExtraction
  .from_pretrained(MODEL_PATH)
  .to('cpu')
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(inference_semantic_search_on_scidata(model, tokenizer, BATCH_SIZE))
