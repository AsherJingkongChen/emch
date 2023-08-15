#! /usr/bin/env node --experimental-modules

import cpx from 'cpx2';
import path from 'path';
import chalk from 'chalk';
import esbuild from 'esbuild';
import { getReasonPhrase } from 'http-status-codes';
import http from 'http';

/// Watch and copy ///
{
  const filesToWatch = [
    {
      source: './packages/web/html/**/*',
      target: './packages/web/dist/html/',
    }, {
      source: './packages/web/css/**/*',
      target: './packages/web/dist/css/',
    }, {
      source: './packages/web/favicons/**/*',
      target: './packages/web/dist/favicons/',
    }, {
      source: './artifacts/sentence-transformers_all-MiniLM-L6-v2/onnx_q/**/*',
      target: './packages/web/dist/onnx/',
    }, {
      source: './packages/web/node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
      target: './packages/web/dist/wasm/',
    }, {
      source: './packages/web/node_modules/emch-wasm/lib_bg.wasm',
      target: './packages/web/dist/wasm/',
    },
  ];
  for (const { source, target } of filesToWatch) {
    await watchFiles(source, target);
  }
}

/// Bundle JavaScript ///
{
  /// Development Service ///
  const devServerConfig = {
    url: {
      top: new URL('http://127.0.0.1:8863'),
      mid: new URL('http://127.0.0.1:9974'),
    },
  };
  const bundler = await esbuild.context({
    entryPoints: ['./packages/web/js/home.js'],
    outdir: './packages/web/dist/js/',
    format: 'esm',
    bundle: true,
  });
  await bundler.watch();
  await bundler.serve({
    host: devServerConfig.url.top.hostname,
    port: parseInt(devServerConfig.url.top.port),
    servedir: './packages/web/dist/',
    onRequest: devServerLogOnResponse,
  });

  // const devServerAddress =
  //   chalk.bold.underline(`http://${hostname}:${port}`);
  // console.log(`Development server is at ${devServerAddress} now`);
  http.createServer(
    { keepAliveTimeout: 60000 },
    interceptOnRequest.bind(null, devServerConfig.url.top),
  ).listen(
    devServerConfig.url.mid.port,
    devServerConfig.url.mid.hostname,
  );
}

function interceptOnRequest(targetBaseUrl, request, response) {
  const targetUrl = new URL(targetBaseUrl);
  const { url: pathname } = request;
  if (pathname === '/') {
    response.writeHead(301, { Location: '/html/' });
    response.end();
    return false;
  }
  targetUrl.pathname = pathname;

  const interRequest = http.request(
    targetUrl,
    request,
    interceptOnResponse.bind(null, response),
  );
  request.pipe(interRequest, { end: true });
}

function interceptOnResponse(response, interResponse) {
  const { statusCode, headers } = interResponse;
  response.writeHead(statusCode, headers);

  interResponse.pipe(response, { end: true });
}
// import { execSync, spawnSync } from 'child_process';
// import { chdir, cwd } from 'process';

// ***** IMPORTANT
// { // link wasm-pack bundle to npm
//   chdir('./packages/web/');
//   execSync('pnpm link ../rust/dist', { stdio: 'inherit' });
//   chdir('../../');
// }

function watchFiles(source, outputDir) {
  return new Promise((resolve, reject) => {
    cpx.watch(source, outputDir, { dereference: true })
      .on('watch-error', function logOnError(err) {
        console.error(err);
        reject(err);
      }).on('watch-ready', function logOnReady() {
        let { source, outputDir } = this;
        source = chalk.cyan(source);
        outputDir = chalk.green(outputDir);
        console.log(`Watching ${source} => ${outputDir}`);
        resolve();
      })
  });
}

function devServerLogOnResponse({
  method,
  path,
  status,
}) {
  const methodText = method;
  const pathText = chalk.gray(path);
  const clampedStatus = Math.trunc(status / 100) * 100;
  const statusPainter = ({
    100: chalk.cyanBright,
    200: chalk.green,
    300: chalk.yellow,
    400: chalk.red,
    500: chalk.magenta,
  }[clampedStatus]);
  const statusText = statusPainter(status);
  const statusReason = getReasonPhrase(status);
  const statusReasonText = statusPainter(statusReason);
  console.log(`\
${methodText} ${pathText} => \
${statusText} ${statusReasonText}`);
}
