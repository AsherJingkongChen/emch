#! /usr/bin/env sh

cd ./packages/rust/ || { exit 1; }
wasm-pack build \
  --dev \
  --target=web \
  --out-dir=./target/web/ || { exit 2; }

cd ../../ || { exit 3; }
./scripts/rust/link.sh || { exit 4; }
