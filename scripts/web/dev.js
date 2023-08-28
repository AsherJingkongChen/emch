#! /usr/bin/env node --experimental-modules

import cpx from 'cpx2';
import chalk from 'chalk';
import esbuild from 'esbuild';
import http from 'http';
import os from 'os';
import httpProxy from 'http-proxy';

/// Watch and copy files ///
{
  const filesToWatch = [
    {
      source: './packages/web/css/**/*',
      target: './packages/web/dist/css/',
    }, {
      source: './packages/web/favicons/**/*',
      target: './packages/web/dist/favicons/',
    }, {
      source: './packages/web/html/**/*',
      target: './packages/web/dist/html/',
    }, {
      source: './packages/web/onnx/**/*',
      target: './packages/web/dist/onnx/',
    }, {
      source: './packages/web/node_modules/onnxruntime-web/dist/*.wasm',
      target: './packages/web/dist/wasm/',
    }, {
      source: './packages/web/node_modules/emch-rs/*.wasm',
      target: './packages/web/dist/wasm/',
    },
  ];
  for (const { source, target } of filesToWatch) {
    await watchAndCopyFiles(source, target);
  }
}

/// Bundle JavaScript and serve for development ///
{
  // all IPv4 addresses are at private networks
  const devServerConfig = {
    url: {
      top: new URL('http://127.0.0.1:9901'),
      mid: new URL('http://127.0.0.1:8901'),
      ext: new URL(`http://${getExtIPv4s()[0] ?? '127.0.0.1'}:8080`),
    },
  };
  const bundler = await esbuild.context({
    entryPoints: ['./packages/web/ts/home.ts'],
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
  const proxyServer = httpProxy.createProxyServer({
    target: devServerConfig.url.top,
    keepAliveTimeout: 60000,
  }).listen(
    devServerConfig.url.mid.port
  ).on('proxyReq', (proxyRequest, request, response) => {
    const { host, path, method } = proxyRequest;
    if (path === '/') {
      response.writeHead(301, { Location: '/html/' });
      const { statusCode, statusMessage } = response;
      devServerLog({
        host,
        path,
        method,
        statusCode,
        statusMessage,
      });
      return;
    }
  }).on('proxyRes', (proxyResponse, request, response) => {
    const { url: path, headers: { host }, method } = request;
    const { statusCode, statusMessage } = proxyResponse;
    devServerLog({
      host,
      path,
      method,
      statusCode,
      statusMessage,
    });
  });

  // Ext service (external)
  http.createServer({
    keepAliveTimeout: 60000,
  }).listen(
    devServerConfig.url.ext.port,
    devServerConfig.url.ext.hostname,
  ).on('request', proxyServer.web.bind(proxyServer));

  const devServerAddressInternal =
    chalk.bold(devServerConfig.url.mid.href);
  const devServerAddressExternal =
    chalk.bold(devServerConfig.url.ext.href);
  console.log(`Development server is at ${devServerAddressInternal} now`);
  console.log(`Development server is at ${devServerAddressExternal} now`);
}

function watchAndCopyFiles(source, outputDir) {
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

function devServerLog({
  protocol,
  host,
  path,
  method,
  statusCode,
  statusMessage,
}) {
  protocol = protocol || 'http:';
  const methodText = method;
  const urlText = chalk.gray(`${protocol}//${host}${path}`);
  const clampedStatusCode = Math.trunc(statusCode / 100) * 100;
  const statusPainter = ({
    100: chalk.cyanBright,
    200: chalk.green,
    300: chalk.yellow,
    400: chalk.red,
    500: chalk.magenta,
  }[clampedStatusCode]);
  const statusCodeText = statusPainter(statusCode);
  const statusMessageText = statusPainter(statusMessage);
  console.log(`\
${methodText} ${urlText} => \
${statusCodeText} ${statusMessageText}`);
}

function getExtIPv4s() {
  return Object
    .values(os.networkInterfaces())
    .flat()
    .filter(({ internal, family }) => (
      !internal && family === 'IPv4'
    )).map(({ address }) => address);
}
