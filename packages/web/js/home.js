import { InferenceSession, env, Tensor } from 'onnxruntime-web';
import initEmchWasm, { get_vocab_size } from 'emch-wasm';

env.wasm.wasmPaths = {
  'ort-wasm-simd.wasm': '../wasm/ort-wasm-simd.wasm',
};
const model_url = '../onnx/model_quantized.onnx';
const model_config = {
  graphOptimizationLevel: 'all'
};
// const model = await InferenceSession.create(model_url, model_config);
// console.log(model);

await initEmchWasm('../wasm/lib_bg.wasm');
console.log(get_vocab_size());
