#! /usr/bin/env sh

pnpm exec tsc \
  --noEmit \
  --strict \
  --lib esnext,dom \
  --module esnext \
  --target esnext \
  --moduleResolution node \
  ./packages/web/ts/**/*.ts || { exit 1; }
