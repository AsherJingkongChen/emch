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
./scripts/download_transformer_in_pytorch_and_onnx.py \
  sentence-transformers/all-MiniLM-L6-v2 \
  assets/sbert/all-MiniLM-L6-v2;
```
```sh
export LIBTORCH=/path/to/libtorch;
export ORT_LIB_LOCATION=/path/to/onnxruntime;
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH/lib:$ORT_LIB_LOCATION/lib";
```
```
scripts
src/rust
src/python
tests/scifact
assets/sbert/all-MiniLM-L6-v2
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
