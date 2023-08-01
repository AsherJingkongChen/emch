#!/usr/bin/env python3

from sys import argv
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from optimum.exporters import onnx
from optimum.onnxruntime import (
  QuantizationConfig,
  ORTQuantizer,
  ORTQuantizableOperator,
)
from optimum.onnxruntime.quantization import (
  QuantFormat,
  QuantizationMode,
  QuantType,
)

def download_transformer(
  model_id: str,
  save_pt_dir: str,
  save_onnx_dir: str,
  save_onnx_q_dir: str,
):
  # PyTorch
  (AutoModel
    .from_pretrained(model_id, force_download = True)
    .to('cpu')
    .eval()
    .save_pretrained(save_pt_dir))
  (AutoTokenizer
    .from_pretrained(model_id, force_download = True)
    .save_pretrained(save_pt_dir))

  # ONNX
  onnx.main_export(
    save_pt_dir,
    output = save_onnx_dir,
    framework = 'pt',
    local_files_only = True,
    task = 'feature-extraction',
    device = 'cpu',
    optimize = 'O3',
    batch_size = 1,
  )

  # ONNX with quantization
  (ORTQuantizer
    .from_pretrained(save_onnx_dir)
    .quantize(
      save_dir = save_onnx_q_dir,
      quantization_config = QuantizationConfig(
        is_static = False,
        format = QuantFormat.QOperator,
        mode = QuantizationMode.IntegerOps,
        activations_dtype = QuantType.QUInt8,
        operators_to_quantize = list(
          ORTQuantizableOperator.__members__.keys()
        ),
      ),
    ))

# ./scripts/download_transformer_and_convert.py \
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
    models_path / 'onnx_q',
  )