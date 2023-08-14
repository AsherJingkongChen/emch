#! /usr/bin/env sh

cpx "public/**/*" dist/ --watch &\
esbuild packages/web.js --outdir=dist/ --format=esm --bundle --watch &\
http-server dist/ --port=8863 -o
