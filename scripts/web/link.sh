#! /usr/bin/env sh

cd ./packages/web/;
pnpm link \
  --save-dev \
  ../rust/target/web/;
pnpm link \
  --save-dev \
  ../glass-rs/build/web/;
