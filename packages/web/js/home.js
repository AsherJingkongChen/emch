import { InferenceSession, env, Tensor } from 'onnxruntime-web';
import initEmchWasm, {
  Tokenizer,
} from 'emch-wasm';

/// load wasm modules ///
env.wasm.wasmPaths = {
  'ort-wasm-simd.wasm': '../wasm/ort-wasm-simd.wasm',
};
await initEmchWasm('../wasm/lib_bg.wasm');

/// tokenizer ///
const tokenizerJson =
  await fetch('../onnx/tokenizer.json').then((r) => r.json());

tokenizerJson.padding.strategy = {
  BatchLongest: null,
};
console.log({ tokenizerJson });

const tokenizer = new Tokenizer(JSON.stringify(tokenizerJson));
console.log({ tokenizer });

/// encoding ///
const encoding = tokenizer.encode_string('The IDs are the main input to a Language Model. They are the token indices, the numerical representations that a LM understands.');
let { input_ids, attention_mask, token_type_ids } = encoding;
console.log({ input_ids, attention_mask, token_type_ids });

input_ids = new Tensor(BigInt64Array.from(input_ids, BigInt), [1, input_ids.length]);
attention_mask = new Tensor(BigInt64Array.from(attention_mask, BigInt), [1, attention_mask.length]);
token_type_ids = new Tensor(BigInt64Array.from(token_type_ids, BigInt), [1, token_type_ids.length]);
console.log({ input_ids, attention_mask, token_type_ids });

encoding.free();

const model_url = '../onnx/model_quantized.onnx';
const model_config = {
  graphOptimizationLevel: 'all',
};
const model = await InferenceSession.create(
  model_url, model_config
);

const { inputNames, outputNames } = model;
console.log({ inputNames, outputNames });

const token_embeddings = await model.run({
  input_ids,
  attention_mask,
  token_type_ids,
});
console.log({ token_embeddings });