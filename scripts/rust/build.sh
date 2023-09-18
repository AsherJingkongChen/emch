#! /usr/bin/env sh

cd ./packages/rust/;
wasm-pack build \
  --release \
  --target=web \
  --out-dir=./target/web/;
