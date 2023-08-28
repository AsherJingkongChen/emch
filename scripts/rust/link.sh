#! /usr/bin/env sh

cd ./packages/web/ || { exit 1; }
pnpm link \
  --save-dev \
  ../rust/target/web/ || { exit 2; }
