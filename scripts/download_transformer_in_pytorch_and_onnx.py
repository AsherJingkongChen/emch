#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer
from sys import argv
from pathlib import Path
from optimum.exporters import onnx

def download_transformer(
  model_id: str,
  save_pt_dir: str,
  save_onnx_dir: str,
):
  model = (AutoModel
    .from_pretrained(model_id, force_download = True)
    .eval())
  tokenizer = (AutoTokenizer
    .from_pretrained(model_id, force_download = True))

  # PyTorch
  model.save_pretrained(save_pt_dir)
  tokenizer.save_pretrained(save_pt_dir)

  # ONNX
  onnx.main_export(
    save_pt_dir,
    save_onnx_dir,
    framework = 'pt',
    local_files_only = True,
    task = 'feature-extraction',
    device = 'cpu',
    optimize = 'O3',
    batch_size = 2,
  )

# ./scripts/download_transformer_in_pytorch_and_onnx.py \
#   sentence-transformers/all-MiniLM-L6-v2 \
#   assets/sbert/all-MiniLM-L6-v2;
if __name__ == '__main__':
  if len(argv) > 1:
    model_id = argv[1]
  else:
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'

  if len(argv) > 2:
    models_path = Path(argv[2])
  else:
    models_path = Path('assets/sbert/all-MiniLM-L6-v2')

  download_transformer(
    model_id,
    models_path / 'pt',
    models_path / 'onnx',
  )