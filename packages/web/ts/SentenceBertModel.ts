import {
  env as OrtEnv,
  InferenceSession,
  Tensor,
} from 'onnxruntime-web';
import emchInit, {
  InitInput as EmchInitInput,
  Tokenizer,
  get_sentence_embeddings,
} from 'emch-wasm';

export class SentenceBertModel {
  static async create({
    modelURI,
    modelOptions,
    tokenizerOptions,
    ortWasmDir,
    emchWasmSource,
  }: SentenceBertModel.CreateOptions
  ): Promise<SentenceBertModel> {
    OrtEnv.wasm.wasmPaths = ortWasmDir;
    await emchInit(emchWasmSource);

    const model = await InferenceSession.create(modelURI, modelOptions);
    const tokenizer = new Tokenizer(tokenizerOptions);

    console.assert(this.isModelFieldsValid(model));

    return new SentenceBertModel({
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

  async getEmbeddings(
    inputSentences: string[],
  ): Promise<Float32Array[]> {
    const encoding =
      this.tokenizer.encode_strings(inputSentences, true);
    const {
      input_ids_i64,
      attention_mask_i64,
      attention_mask_f32,
      token_type_ids_i64,
    } = encoding;
    encoding.free();
    const { length: batch_size } = inputSentences;
    const { length: total_size } = input_ids_i64;
    const sequence_size = total_size / batch_size;
    const dimensions = [batch_size, sequence_size];
    const model_input = {
      input_ids: new Tensor(input_ids_i64, dimensions),
      attention_mask: new Tensor(attention_mask_i64, dimensions),
      token_type_ids: new Tensor(token_type_ids_i64, dimensions),
    };
    const {
      last_hidden_state: {
        data: last_hidden_state,
        dims: { 2: hidden_size },
      },
    } = await this.model.run(model_input);
    return get_sentence_embeddings(
      last_hidden_state as Float32Array,
      attention_mask_f32,
      batch_size,
      sequence_size,
      hidden_size,
    );
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

export declare namespace SentenceBertModel {
  interface CreateOptions {
    modelURI: string;
    modelOptions?: InferenceSession.SessionOptions;
    tokenizerOptions: object;
    ortWasmDir: string;
    emchWasmSource: EmchInitInput;
  }
}

export default SentenceBertModel;
