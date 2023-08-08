#! /usr/bin/env sh

cpx "public/**/*" dist/ --watch &\
esbuild src/web.js --outdir=dist/ --format=esm --bundle --watch &\
http-server dist/ --port=8863 -o
