import { BertModel, Metric } from './lib';

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
  'That is a happy person',
  'That is a happy dog',
  'That is a very happy person',
  'Today is a sunny day',
];
const model = await BertModel.create({
  modelURI: '../onnx/model_quantized.onnx',
  modelOptions: {
    graphOptimizationLevel: 'all',
  },
  tokenizerOptions,
  ortWasmDir: '../wasm/',
  emchWasmSource: '../wasm/emch_rs_bg.wasm',
});
console.log({ model });

const sembeddings = await model.getSentenceEmbeddings(sentences);
console.log(
  sembeddings.map((e) => (
    e.map((n) => n ** 2).reduce((p, n) => p + n)
  ))
);
console.log({ sembeddings });
for (let i = 0; i < sembeddings.length; i++) {
  for (let j = i + 1; j < sembeddings.length; j++) {
    const cosine_sim = Metric.get_cosine_similarity(
      sembeddings[i],
      sembeddings[j],
    );
    console.log({
      sentences: [sentences[i], sentences[j]],
      cosine_sim,
    });
  }
}
