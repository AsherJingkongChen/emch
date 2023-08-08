# Dev Notes

```
optimum-cli export onnx \
  --device cpu \
  --framework pt \
  --optimize O3 \
  --task feature-extraction \
  --model assets/sbert/all-MiniLM-L6-v2/pt/ \
  --batch_size 2 \
  assets/sbert/all-MiniLM-L6-v2/onnx/;
```

```
./scripts/download_transformer_and_convert.py \
  sentence-transformers/all-MiniLM-L6-v2 \
  assets/sbert/all-MiniLM-L6-v2;
```

```sh
export LIBTORCH=/path/to/libtorch;
export ORT_LIB_LOCATION=/path/to/onnxruntime;
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:$ORT_LIB_LOCATION/lib";
```

```
BSD time utility
%J
user:   %mU or %uU
kernel: %mS or %uS
total:  %mE or %uE
cpu:    %P
memory: %M KiB
```

```
cd cmake/external/emsdk/ && python2.7 ./emsdk.py install latest;
./build.sh --config MinSizeRel --minimal_build --build_wasm --skip_tests --enable_wasm_threads --enable_wasm_simd;
./build.sh --config MinSizeRel --minimal_build --build_shared_lib --skip_tests --osx_arch x86_64;
```

```
{
  "mangle": {
    "properties": {
      "reserved": ["_scriptDir", "startWorker"]
    }
  }
}
```

```diff
  if (filter) {
    const filterRegex = new RegExp(filter);
    return builds.filter(b => filterRegex.test(b.output.filename));
  }

+ builds.parallelism = 1;

  return builds;
};
```