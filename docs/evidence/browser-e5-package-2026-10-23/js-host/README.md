# JavaScript host (ai-dotnet/samples/E5BrowserJs)

A smoke run of the React page that uses the e5 bundle from JavaScript. The bundle is a headless .NET WebAssembly app
with `[JSExport]`, running in a Web Worker. It was published from packages packed from this branch (local feed,
`0.3.1-dev.1`), served by the Vite dev server with the isolation headers, and driven in headless Chrome for Testing
153.0.8010.12 by `browser/conformance/smoke_js_sample.py` (`smoke-js-int8embed.json`, `smoke-js-int8embed.png`).

- **Load.** int8-embed from the app's own server was ready to search in 7.7 s. That covers the .NET
  runtime start in the worker, onnxruntime-web, tokenizer and graph download with SHA-256, the session, and the
  28-passage corpus.
- **Runtime.** The panel reads: profile `multilingual-e5-small/onnx-int8embed-wasm`, onnxruntime-web 1.30.0 (wasm
  bundle) in a Web Worker, 4 of 12 logical cores.
- **Main thread.** The page's frame counter advanced from 7 to 436 frames during the load (about
  59 frames per second), so loading never blocked the page.
- **Queries.** Each took 27, 21, 22 ms in the
  worker. Each ranked its passage first, with the same scores as the Blazor sample (`../smoke-sample-final.json`),
  because both use the same packages.
