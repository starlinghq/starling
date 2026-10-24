# JavaScript host (ai-dotnet/samples/E5BrowserJs)

A smoke run of the React page that uses the e5 bundle from JavaScript. The bundle is a headless .NET WebAssembly app
with `[JSExport]`, running in a Web Worker. It was published from packages packed from this branch (local feed,
`0.3.1-dev.1`), served by the Vite dev server with the isolation headers, and driven in headless Chrome for Testing
153.0.8010.12 by `browser/conformance/smoke_js_sample.py`. The run is in `smoke-js-int8embed.json` and
`smoke-js-int8embed.png`. The script exits non-zero on a load error, a page error or any failed request other than the
favicon, and this run exited 0.

- **Load.** int8-embed from the app's own server was ready to search in 12.0 s. That covers the .NET
  runtime start in the worker, onnxruntime-web, tokenizer and graph download with SHA-256, the session, and the
  28-passage corpus.
- **Runtime.** The panel reads: profile `multilingual-e5-small/onnx-int8embed-wasm`, onnxruntime-web 1.30.0 (wasm
  bundle) in a Web Worker, 4 of 12 logical cores.
- **Main thread.** The page's frame counter advanced from 15 to 695 frames during the load (about
  59 frames per second), so loading never blocked the page.
- **Queries.** Each took 49, 38, 92 ms in the
  worker. Each ranked its passage first, with the same scores as the Blazor sample (`../smoke-sample-final.json`),
  because both use the same packages.
- **Timings.** This run shared the machine with CI jobs (load average about 29). An earlier run of the same
  page at lower load was ready in 7.7 s, with queries at 21–27 ms.
