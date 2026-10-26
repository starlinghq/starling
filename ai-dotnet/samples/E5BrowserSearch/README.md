# E5BrowserSearch

Multilingual semantic search that runs entirely in the browser. It's a standalone Blazor WebAssembly app on `Starlings.Models.E5.Browser`: `intfloat/multilingual-e5-small` on onnxruntime-web, with the owned tokenizer and the same pooling code as the server-side `Starlings.Models.E5`.

The app does three things:
- **Loads the model once.** The graph comes from the Hugging Face Hub (fp32) or your own host (int8-embed). It is checked against its pinned SHA-256 and kept in the browser's Cache Storage, so later visits load from the cache.
- **Embeds a 28-passage corpus** in 16 languages with the `passage: ` prefix.
- **Ranks passages for a query** embedded with `query: `, by cosine, in any of those languages.

![The sample after a search](../../../docs/evidence/browser-e5-package-2026-10-23/sample-screenshot.png)

Nothing you type leaves the page. Inference is Experimental: WebAssembly has no support badge (see the package README).

## Demo (three commands)

```sh
dotnet publish -c Release -o out                   # add -p:RunAOTCompilation=true with the wasm-tools workload
python model-pipeline/reference/encoder/quantize_e5_browser.py --model <e5 snapshot>/onnx/model.onnx --out models
python3 serve.py out/wwwroot --models models       # http://127.0.0.1:8080/
```

Pick **int8-embed, self-hosted** (URL `models/model-int8-embed.onnx`) and **WebAssembly**, then click **Load**. It's ready in about 8 s, and the runtime panel should say "4 of N logical cores". Then click the example queries; most are in another language than the passage they find.

- **What `serve.py` does.** It's standard-library Python. It serves the publish output with the cross-origin isolation headers (several inference threads), its precompressed `.br` files, and the graph directory under `/models/`.
- **The graph.** The int8-embed graph (182 MB, cosine 0.9998 to fp32) has no public URL. `quantize_e5_browser.py` builds it deterministically. The loader rejects any file whose SHA-256 differs from the pin (`374a3cc0…fb19`). Don't commit it.
- **No preparation.** Pick **fp32 reference, from the Hugging Face Hub**. It downloads 470 MB from the Hub on the first visit (about 20 s on a fast line), then loads from the browser cache.
- **WebGPU.** Pick **WebGPU** in a browser that exposes it. Each call costs about 18 ms whatever the batch size, against about 25 ms for one short text on 4 WASM threads.

`dotnet run` also works, but its development server sends no isolation headers, so the panel shows one thread (3–4x slower). Everything else is the same.

## Publish and host

```sh
dotnet workload install wasm-tools        # once, for AOT
dotnet publish -c Release -p:RunAOTCompilation=true -o out
```

Serve `out/wwwroot` from any static host, with these response headers so onnxruntime-web can use several threads:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

- **Netlify and Cloudflare Pages:** `wwwroot/_headers` already sets them.
- **Azure Static Web Apps:** `wwwroot/staticwebapp.config.json` already sets them.
- **nginx:** `add_header Cross-Origin-Opener-Policy same-origin always; add_header Cross-Origin-Embedder-Policy require-corp always;`
- **GitHub Pages:** it cannot set headers, so the app runs one thread there.

Hosting rules:
- **Secure context.** The page must be https or localhost, because SHA-256 verification uses the browser's SubtleCrypto.
- **Self-hosted graph.** Host it with the app, or on a server that sends CORS headers.
- **Notices.** onnxruntime's license and third-party notices are published next to its binaries (`_content/Starlings.Runtime.OnnxWeb/ort/`).

## What to expect

These are measured in headless Chrome for Testing 153 on an i7-5820K (6 cores), not a qualified benchmark. Raw results: `docs/evidence/browser-e5-package-2026-10-23/`.

| | fp32 | int8-embed |
| --- | --- | --- |
| Download (first visit) | 470 MB, from the Hub | 182 MB, from your host |
| Load to searchable, first visit (download, verify, session, 28 passages) | 20 s | 11 s (from localhost) |
| Session creation | about 1.4 s | about 0.9 s |
| Renderer memory | about 2.3 GB | about 1.4 GB |
| One 34-token passage, 4 WASM threads | about 25 ms | about 25 ms |
| Any call on WebGPU (RTX 4060 Ti) | about 18 ms | about 18 ms |

The .NET runtime and assemblies were 2.4 MB (interpreter) to 4.5 MB (AOT) brotli in the browser spike; onnxruntime-web's WASM bundle adds about 2.3 MB.

## Packages

| Package | Why |
| --- | --- |
| `Starlings.Models.E5.Browser` | the embedder, `IEmbeddingGenerator`, pinned assets; brings `Starlings.Runtime.OnnxWeb` (onnxruntime-web, WASM) |
| `Starlings.Runtime.OnnxWeb.WebGpu` | optional: the WebGPU backend option. Remove it and the WebGPU choice in `App.razor` to ship only WASM |

`StarlingsVersion` in the project file is the release this sample targets. Until that release is on nuget.org, build against a local feed packed from this repository: pack the packages with a version that is not on nuget.org, then pass `-p:StarlingsVersion=<that version>` with a `nuget.config` that maps `Starlings.*` to the feed.
