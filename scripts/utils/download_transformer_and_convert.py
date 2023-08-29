#!/usr/bin/env python3

from transformers import AutoModel, AutoTokenizer
from optimum.exporters.onnx import main_export as onnx_main_export
from optimum.onnxruntime.configuration import (
  QuantizationConfig,
)
from optimum.onnxruntime.quantization import (
  ORTQuantizer,
  QuantFormat,
  QuantizationMode,
  QuantType,
  ORTQuantizableOperator,
)
from subprocess import run as run_subprocess
from sys import stdout, stderr, argv
from pathlib import Path
from shutil import copytree

def download_transformer(
  model_id: str,
  save_pt_dir: str,
  save_onnx_dir: str,
  save_onnx_q_dir: str,
  save_ort_q_dir: str,
):
  # PyTorch
  (AutoModel
    .from_pretrained(model_id)
    .to('cpu')
    .eval()
    .save_pretrained(save_pt_dir))
  (AutoTokenizer
    .from_pretrained(model_id)
    .save_pretrained(save_pt_dir))

  # ONNX
  onnx_main_export(
    save_pt_dir,
    output = save_onnx_dir,
    framework = 'pt',
    task = 'feature-extraction',
    device = 'cpu',
    optimize = 'O3',
  )

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

  # ORT with quantization
  copytree(
    src = save_onnx_q_dir,
    dst = save_ort_q_dir,
    dirs_exist_ok = True,
    ignore = lambda _, __: ('model_quantized.onnx',),
  )
  onnx_q_model_path = save_onnx_q_dir / 'model_quantized.onnx'
  run_subprocess(
    args = [
      'python3',
      '-m', 'onnxruntime.tools.convert_onnx_models_to_ort',
      '--output_dir', save_ort_q_dir,
      onnx_q_model_path,
    ],
    stdout = stdout,
    stderr = stderr,
  )

# ./scripts/download_transformer_and_convert.py \
#   sentence-transformers/all-MiniLM-L6-v2 \
#   artifacts/sentence-transformers_all-MiniLM-L6-v2;
if __name__ == '__main__':
  if len(argv) > 1:
    model_id = argv[1]
  else:
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  if len(argv) > 2:
    models_path = Path(argv[2])
  else:
    models_path = Path('artifacts/sentence-transformers_all-MiniLM-L6-v2')

  download_transformer(
    model_id,
    models_path / 'pt',
    models_path / 'onnx',
    models_path / 'onnx_q',
    models_path / 'ort_q',
  )