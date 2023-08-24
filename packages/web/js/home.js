import { env, InferenceSession, Tensor } from 'onnxruntime-web';
import initEmchWasm, {
  Tokenizer,
  get_sentence_embedding,
} from 'emch-wasm';

/// load wasm modules ///
env.wasm.wasmPaths = {
  'ort-wasm-simd.wasm': '../wasm/ort-wasm-simd.wasm',
};
await initEmchWasm('../wasm/core_bg.wasm');

/// tokenizer ///
const tokenizerJson = await fetch(
  '../onnx/tokenizer.json',
  {
    headers: {
      'Content-Type': 'application/json',
    },
  },
).then((r) => r.json());

tokenizerJson.padding.strategy = {
  BatchLongest: null,
};
console.log({ tokenizerJson });

const tokenizer = new Tokenizer(JSON.stringify(tokenizerJson));
console.log({ tokenizer });

/// encoding ///
const encoding = tokenizer.encode_string('The IDs are the main input to a Language Model. They are the token indices, the numerical representations that a LM understands.');
let { input_ids, attention_mask, token_type_ids } = encoding.as_i64;
const attention_mask_f32 = encoding.as_f32.attention_mask;

console.log({ input_ids, attention_mask, token_type_ids });

input_ids = new Tensor(input_ids, [1, input_ids.length]);
attention_mask = new Tensor(attention_mask, [1, attention_mask.length]);
token_type_ids = new Tensor(token_type_ids, [1, token_type_ids.length]);
console.log({ input_ids, attention_mask, token_type_ids });

const model_url = '../onnx/model_quantized.onnx';
const model_config = {
  graphOptimizationLevel: 'all',
};
const model = await InferenceSession.create(
  model_url, model_config
);

const { inputNames, outputNames } = model;
console.log({ inputNames, outputNames });

const { last_hidden_state } = await model.run({
  input_ids,
  attention_mask,
  token_type_ids,
});
console.log({ last_hidden_state });

const sentence_embedding = get_sentence_embedding(
  last_hidden_state.data,
  attention_mask_f32,
  last_hidden_state.dims[1],
  last_hidden_state.dims[2],
)
console.log({ sentence_embedding })
