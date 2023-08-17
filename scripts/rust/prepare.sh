#! /usr/bin/env sh

cd ./packages/web/ || { exit 1; }
pnpm link ../rust/dist/ || { exit 2; }
