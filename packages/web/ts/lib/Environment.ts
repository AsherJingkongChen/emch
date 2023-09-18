import {
  env as OrtEnv,
} from 'onnxruntime-web';
import initEmchWasm, {
  InitInput as WasmInitInput,
} from 'emch-rs';

class Environment {
  static create({
    emchWasmSource,
    ortWasmDir,
  }: Environment.CreateOptions = {}
  ): Environment {
    return new Environment(
      emchWasmSource ?? '/emch_rs_bg.wasm',
      ortWasmDir ?? '/',
    );
  }

  /**
   * Can only be applied once in a runtime
   */
  async apply(): Promise<Readonly<Environment>> {
    const thisRef = Object.freeze(this);

    if (Environment.isApplied) {
      return thisRef;
    }
    Environment.isApplied = true;

    await initEmchWasm(this.emchWasmSource);
    OrtEnv.wasm.wasmPaths = this.ortWasmDir;

    return thisRef;
  }

  emchWasmSource: WasmInitInput;
  ortWasmDir: string;

  static isApplied: boolean = false;

  /// private fields ///

  private constructor(
    emchWasmSource: WasmInitInput,
    ortWasmDir: string,
  ) {
    this.emchWasmSource = emchWasmSource;
    this.ortWasmDir = ortWasmDir;
  }
}

declare namespace Environment {
  interface CreateOptions {
    emchWasmSource?: WasmInitInput;
    ortWasmDir?: string;
  }
}

export default Environment;
export { Environment };
