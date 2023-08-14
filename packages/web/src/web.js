import { InferenceSession, env, Tensor } from 'onnxruntime-web';

env.wasm.wasmPaths = 'wasm/';
// env.wasm.numThreads = 2;

const model_url = 'onnx_q/model_quantized.onnx';
const model_config = {
  graphOptimizationLevel: 'all'
};

const model = await InferenceSession.create(model_url, model_config);
console.log(model);
