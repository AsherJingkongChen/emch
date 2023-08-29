import {
  env as OrtEnv,
  InferenceSession,
  Tensor,
} from 'onnxruntime-web';
import initEmch, {
  InitInput as EmchInitInput,
  Tokenizer,
  Pooling,
} from 'emch-rs';

export class BertModel {
  static async create({
    modelURI,
    modelOptions,
    tokenizerOptions,
    ortWasmDir,
    emchWasmSource,
  }: BertModel.CreateOptions
  ): Promise<BertModel> {
    OrtEnv.wasm.wasmPaths = ortWasmDir;
    await initEmch(emchWasmSource);

    const model = await InferenceSession.create(modelURI, modelOptions);
    const tokenizer = new Tokenizer(tokenizerOptions);

    console.assert(this.isModelFieldsValid(model));

    return new BertModel({
      model,
      tokenizer,
    });
  }

  free(): void {
    this.tokenizer.free();

    this.model = undefined as any;
    this.tokenizer = undefined as any;
    this.free = undefined as any;
  }

  async getSentenceEmbedding(
    inputSentence: string,
  ): Promise<Float32Array> {
    /// tokenization ///
    const encoding = this.tokenizer.encode(inputSentence, true);
    const { input_ids, attention_mask, token_type_ids } = encoding;
    const { length: sequence_size } = input_ids;
    const dimensions = [1, sequence_size];
    const model_input = {
      input_ids: new Tensor(input_ids, dimensions),
      attention_mask: new Tensor(attention_mask, dimensions),
      token_type_ids: new Tensor(token_type_ids, dimensions),
    };
    encoding.free();

    /// inferencing ///
    const {
      last_hidden_state: {
        data: last_hidden_state,
        dims: { 2: hidden_size },
      },
    } = await this.model.run(model_input);

    /// pooling ///
    const pooled_embedding =
      Pooling.get_mean_pooled_embedding(
        last_hidden_state as Float32Array,
        attention_mask,
        sequence_size,
        hidden_size,
      );

    return pooled_embedding;
  }

  async getSentenceEmbeddings(
    inputSentences: string[],
  ): Promise<Float32Array[]> {
    /// tokenization ///
    const encoding = this.tokenizer.encode_batch(inputSentences, true);
    const { input_ids, attention_mask, token_type_ids } = encoding;
    const { length: batch_size } = inputSentences;
    const { length: total_size } = input_ids;
    const sequence_size = total_size / batch_size;
    const dimensions = [batch_size, sequence_size];
    const model_input = {
      input_ids: new Tensor(input_ids, dimensions),
      attention_mask: new Tensor(attention_mask, dimensions),
      token_type_ids: new Tensor(token_type_ids, dimensions),
    };
    encoding.free();

    /// inferencing ///
    const {
      last_hidden_state: {
        data: last_hidden_state,
        dims: { 2: hidden_size },
      },
    } = await this.model.run(model_input);

    /// pooling ///
    const pooled_embeddings =
      Pooling.get_mean_pooled_embeddings(
        last_hidden_state as Float32Array,
        attention_mask,
        batch_size,
        sequence_size,
        hidden_size,
      );

    return pooled_embeddings;
  }

  /// private fields ///

  private model: InferenceSession;
  private tokenizer: Tokenizer;

  private constructor({
    model,
    tokenizer,
  }: {
    model: InferenceSession;
    tokenizer: Tokenizer;
  }) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  private static isModelFieldsValid(
    model: InferenceSession,
  ): boolean {
    const { inputNames, outputNames } = model;
    return (
      inputNames[0] === 'input_ids' &&
      inputNames[1] === 'attention_mask' &&
      inputNames[2] === 'token_type_ids' &&
      outputNames[0] === 'last_hidden_state'
    );
  }
}

export declare namespace BertModel {
  interface CreateOptions {
    modelURI: string;
    modelOptions?: InferenceSession.SessionOptions;
    tokenizerOptions: object;
    ortWasmDir: string;
    emchWasmSource: EmchInitInput;
  }
}

export default BertModel;

/// re-export ///
export { Metric } from 'emch-rs';
