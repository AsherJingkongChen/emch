#! /usr/bin/env sh

cd packages/rust/ || { exit 1; }
wasm-pack build --dev --target=web --out-dir=dist/ || { exit 2; }

cd ../web || { exit 3; }
[[ -L node_modules/emch-wasm/ ]] || {
  cd ../../ || { exit 4; }
  ./scripts/rust/prepare.sh || { exit 5; }
}
