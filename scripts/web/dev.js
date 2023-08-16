#! /usr/bin/env node --experimental-modules

import cpx from 'cpx2';
import path from 'path';
import chalk from 'chalk';
import esbuild from 'esbuild';
import { getReasonPhrase } from 'http-status-codes';
import http from 'http';
import os from 'os';

/// Watch and copy files ///
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

/// Bundle JavaScript and serve for development ///
{
  const devServerConfig = {
    url: {
      top: new URL('http://127.0.0.1:8863'),
      mid: new URL('http://127.0.0.1:9974'),
      ext: new URL(`http://${getAllExternalIPv4()[0]}:8080`),
    },
  };
  const bundler = await esbuild.context({
    entryPoints: ['./packages/web/js/home.js'],
    outdir: './packages/web/dist/js/',
    format: 'esm',
    bundle: true,
  });
  await bundler.watch();

  // Top service (internal)
  await bundler.serve({
    host: devServerConfig.url.top.hostname,
    port: parseInt(devServerConfig.url.top.port),
    servedir: './packages/web/dist/',
  });

  // Mid service (middleware)
  http.createServer(
    { keepAliveTimeout: 60000 },
    interceptOnRequest.bind(null, devServerConfig.url.top),
  ).listen(
    devServerConfig.url.mid.port,
    devServerConfig.url.mid.hostname,
  );

  // Ext service (external)
  http.createServer(
    { keepAliveTimeout: 60000 },
    interceptOnRequest.bind(null, devServerConfig.url.mid),
  ).listen(
    devServerConfig.url.ext.port,
    devServerConfig.url.ext.hostname,
  );

  const devServerAddressInternal =
    chalk.bold(devServerConfig.url.mid.href);
  const devServerAddressExternal =
    chalk.bold(devServerConfig.url.ext.href);
  console.log(`Development server is at ${devServerAddressInternal} now`);
  console.log(`Development server is at ${devServerAddressExternal} now`);
}

function interceptOnRequest(targetBaseUrl, request, response) {
  const targetUrl = new URL(targetBaseUrl);
  const { url: path } = request;
  if (path === '/') {
    response.writeHead(301, { Location: '/html/' });
    response.end();
    return;
  } else {
    targetUrl.pathname = path;
  }

  const interRequest = http.request(
    targetUrl,
    request,
    interceptOnResponse.bind(null, response),
  );
  request.pipe(interRequest, { end: true });
}

function interceptOnResponse(response, interResponse) {
  const { statusCode: status, headers } = interResponse;
  response.writeHead(status, headers);
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

function getAllExternalIPv4() {
  return Object
    .values(os.networkInterfaces())
    .flat()
    .filter(({ internal, family }) => (
      !internal && family === 'IPv4'
    )).map(({ address }) => address);
}
