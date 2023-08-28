#! /usr/bin/env sh

cd ./packages/rust/ || { exit 1; }
wasm-pack build \
  --release \
  --target=web \
  --out-dir=./target/web/ || { exit 2; }

# is it needed?
cd ../../ || { exit 3; }
./scripts/rust/link.sh || { exit 4; }
