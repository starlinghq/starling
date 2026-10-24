# E5BrowserJs

`intfloat/multilingual-e5-small` used from a **JavaScript app**: a React page (Vite) calls a small promise API, and behind it a headless .NET WebAssembly bundle runs in a Web Worker. The bundle has no Blazor and no UI; it uses the same `Starlings.Models.E5.Browser` package as the Blazor sample [`E5BrowserSearch`](../E5BrowserSearch/README.md). Same tokenizer, same pooling, same numbers, as the evidence in `docs/evidence/browser-e5-package-2026-10-23/` shows.

```
React page ──postMessage──▶ Web Worker (public/e5/e5-worker.js)
 src/e5.js                    └─ dotnet.js ──▶ E5Wasm.dll ([JSExport] Load / Embed / EmbedBatch)
                                                └─ Starlings.Models.E5.Browser ──▶ onnxruntime-web (WASM threads or WebGPU)
```

The worker keeps tokenization, inference and pooling off the page's main thread. The page has a frame counter in its header, which keeps ticking (about 60 fps) while the model loads and while queries run.

## The JavaScript API

`web/src/e5.js` knows nothing about .NET; any JavaScript app can use it.

```js
import { createE5, cosine } from "./e5.js";

const e5 = await createE5({
  variant: "int8embed",                         // or "fp32" (loads from the Hugging Face Hub when modelUrl is empty)
  modelUrl: "models/model-int8-embed.onnx",            // resolved against the page
  provider: "wasm",                             // or "webgpu"
  onProgress: ({ stage, received, total }) => console.log(stage, received, total),
});
const query = await e5.embed("how do I keep basil fresh?", "query");      // Float32Array(384), L2-normalized
const docs = await e5.embedBatch(["...", "..."], "passage");              // Float32Array(384)[]
cosine(query, docs[0]);
e5.info;   // { profile, dimension, ortVersion, bundle, threads, cores, crossOriginIsolated }
```

- **Input types.** `"query"` and `"passage"` add e5's prefixes; `"none"` embeds the text as given. Unknown variants, providers or input types are rejected, never silently replaced.
- **Failure and teardown.** A failed load terminates its worker. `terminate()` rejects any call still in flight.
- **Returned vectors.** They are transferred from the worker, not copied.
- **One model per page.** The model loads once per `createE5` call. Keep one instance (for example in a React context).

## Run

```sh
cd web
npm install
npm run bundle          # dotnet publish ../E5Wasm, then copy _framework/ and _content/ into public/e5/
mkdir -p public/models  # optional: the self-hosted int8-embed graph (182 MB, quantize_e5_browser.py output)
cp <dir>/model-int8-embed.onnx public/models/
npm run dev             # http://localhost:5173/  (vite.config.js sends the isolation headers)
```

After `npm run bundle`, restart `npm run dev`. Vite's dev server caches the list of files in `public/`, and while it runs it answers replaced files with `index.html`. The worker then fails to import `dotnet.js` ("e5 worker error").

Pick **int8-embed** or **fp32 from the Hub**, then click **Load**. In the evidence run, int8-embed was ready in about 8 s with 4 inference threads, and each query took about 20 ms in the worker.

`E5Wasm/E5Wasm.csproj` references the packages by version (`StarlingsVersion`). Until that version is on nuget.org, build against a local feed packed from this repository, as for the other samples (`../README.md`).

## Ship it

- **Build.** `npm run build` writes `dist/`. The React code is bundled (about 50 KB gzipped); the .NET bundle stays plain static files under `dist/e5/`. Vite does not bundle `dotnet.js`, because it loads its fingerprinted parts at run time. Keep model files out of `dist/` (or out of `public/models/`) and host them where the app can fetch them.
- **Headers.** Serve with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` for several inference threads. Without them it runs one thread, 3–4x slower. The page must be https or localhost, because model SHA-256 verification uses SubtleCrypto.
- **Download.** The .NET runtime and assemblies are a few MB brotli, plus onnxruntime-web: 2.3 MB for the WASM bundle, about 4 MB with WebGPU. Remove `Starlings.Runtime.OnnxWeb.WebGpu` from `E5Wasm.csproj` (and the WebGPU option in `App.jsx`) to ship only WASM.
- **The bundle as a package.** To use it in several apps, publish `E5Wasm` once and ship `public/e5/` (worker plus `_framework/` and `_content/`) and `src/e5.js` as an npm package. `createE5({ base })` points at where the files are served. By default that's the app's base path plus `e5/`, so a Vite `base` such as `/app/` works. Browsers require the worker script to be on the page's origin, so serve the bundle from the app, not from a CDN on another origin.

## Maturity

These are the e5 browser profiles of `Starlings.Models.E5.Browser`, all Experimental (see that package's README). This sample shows the hosting pattern. It's checked by the smoke run in the evidence, not by its own gates.
