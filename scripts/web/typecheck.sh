#! /usr/bin/env zsh

pnpm exec tsc \
  --noEmit \
  --strict \
  --lib esnext,dom \
  --module esnext \
  --target esnext \
  --moduleResolution node \
  ./packages/web/js/**/*.ts || { exit 1; }
