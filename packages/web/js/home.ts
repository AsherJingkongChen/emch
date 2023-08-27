import { SentenceBertModel } from './SentenceBertModel';

/// tokenizer ///
const tokenizerOptions = await fetch(
  '../onnx/tokenizer.json',
  {
    headers: {
      'Content-Type': 'application/json',
    },
  },
).then((r) => r.json());

tokenizerOptions.padding.strategy = {
  BatchLongest: null,
};
console.log({ tokenizerOptions });

const sentences = [
  'The IDs are the main input to a Language Model.',
  'They are the token indices.',
  'The numerical representations that a LM understands.'
];
const model = await SentenceBertModel.create({
  modelURI: '../onnx/model_quantized.onnx',
  modelOptions: {
    graphOptimizationLevel: 'all',
  },
  tokenizerOptions,
  ortWasmDir: '../wasm/',
  emchWasmSource: '../wasm/emch_bg.wasm',
});
console.log({ model });

const embeddings = await model.getEmbeddings(sentences);
console.log({ embeddings });
