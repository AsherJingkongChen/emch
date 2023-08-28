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
  'const banner = "I\'m a banner";',
  'var banner_path;',
  'function ban(path);'
];
const model = await SentenceBertModel.create({
  modelURI: '../onnx/model_quantized.onnx',
  modelOptions: {
    graphOptimizationLevel: 'all',
  },
  tokenizerOptions,
  ortWasmDir: '../wasm/',
  emchWasmSource: '../wasm/emch_rs_bg.wasm',
});
console.log({ model });

const embeddings = await model.getEmbeddings(sentences);
console.log({ embeddings });

for (let i = 0; i < embeddings.length; i++) {
  for (let j = i + 1; j < embeddings.length; j++) {
    const cosine_sim = SentenceBertModel.cosineSimilarity(
      embeddings[i],
      embeddings[j],
    );
    console.log({
      sentences: [sentences[i], sentences[j]],
      cosine_sim,
    });
  }
}
