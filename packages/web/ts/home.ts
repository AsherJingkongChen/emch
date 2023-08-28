import { BertModel } from './BertModel';

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
  'const banner = "I\'m a banner";',
  'var banner_path;',
  'function ban(path);'
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
console.log({ sembeddings });

for (let i = 0; i < sembeddings.length; i++) {
  for (let j = i + 1; j < sembeddings.length; j++) {
    const cosine_sim = BertModel.cosineSimilarity(
      sembeddings[i],
      sembeddings[j],
    );
    console.log({
      sentences: [sentences[i], sentences[j]],
      cosine_sim,
    });
  }
}
