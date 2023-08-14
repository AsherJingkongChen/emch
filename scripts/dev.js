#! /usr/bin/env node --experimental-modules

import cpx from 'cpx2';
import path from 'path';

const sources = {
  'artifacts': {
    'sentence-transformers_all-MiniLM-L6-v2/onnx_q': '*',
  },
  'packages': {
    'web': {
      'node_modules/onnxruntime-web/dist': [
        'ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm',
        'ort-wasm.wasm',
      ],
    },
  },
};

const targets = {
  'packages': {
    'web/dist': {
      'wasm': '/',
    },
  },
};

makePathtreeInplace(sources);
makePathtreeInplace(targets);

for (const source of sources['packages']['web']['node_modules/onnxruntime-web/dist']) {
  const target = targets['packages']['web/dist']['wasm'];
  cpx.watch(source, target)
    .on('watch-ready', () => {
      console.log({ source, target });
    });
}

/// functions ///

function makePathtreeInplace(recordRef, prefixes = []) {
  if (typeof recordRef !== 'object') {
    return path.join(...prefixes, recordRef);
  }
  if (recordRef instanceof Array) {
    for (const [prefix, nextRef] of Object.entries(recordRef)) {
      recordRef[prefix] = makePathtreeInplace(
        nextRef,
        prefixes
      );
    }
  } else {
    for (const [prefix, nextRef] of Object.entries(recordRef)) {
      recordRef[prefix] = makePathtreeInplace(
        nextRef,
        [...prefixes, prefix]
      );
    }
  }
  return recordRef;
}
