# Browser spike evidence (2026-10-11)

Raw results behind [docs/m0/browser-wasm-spike.md](../../m0/browser-wasm-spike.md), copied by `browser/spike/harness/collect_evidence.py` (embeddings and user agents removed). Exploratory, one workstation (i7-5820K, RTX 4060 Ti as GPU 1), not a qualified benchmark.

| File | Content |
| --- | --- |
| `tokenizer.json` | `TokenizerSuite` output natively (CoreCLR) and in headless Chrome for Testing 153 (interpreter and AOT publish): per profile, field checks, mismatches, load and encode timings; startup download per path |
| `variants.json` | e5 graph variants: SHA-256, sizes, node types, the exact quantization calls (`model-pipeline/reference/encoder/quantize_e5_browser.py`) |
| `native-ort.json` | Python ONNX Runtime 1.30.0 CPU on each variant: drift against the protected fp32 PyTorch embeddings, batch-of-8 vs single, warm latency samples |
| `browser-drift.json` | each browser run (variant x backend): drift against the reference and against the native run of the same graph, load, latency medians, memory |
| `browser-e5-runs.json` | raw latency samples, bytes downloaded per run, `nvidia-smi` before, during and after (WebGPU), ORT placement warnings |
| `probes.json` | load-only probes of EmbeddingGemma 2 text fp32, GLiNER2 fp32, Qwen3.5-0.8B int4 and fp32 graphs in onnxruntime-web (WASM) |
| `gpu-pinning.json` | the WebGPU adapter Chrome reports with and without `VK_LOADER_DEVICE_ID_FILTER=0x2805` |
| `compressed-sizes.txt` | raw, gzip and brotli sizes of the graphs, tokenizer files and onnxruntime-web binaries |
| `load-average.txt` | load average at the start of each step of `run_all.sh` |
