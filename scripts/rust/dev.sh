#! /usr/bin/env sh

cd ./packages/rust/;
wasm-pack build \
  --dev \
  --target=web \
  --out-dir=./target/web/;
