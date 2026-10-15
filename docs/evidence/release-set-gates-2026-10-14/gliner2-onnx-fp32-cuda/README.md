# gliner2-base-v1 `onnx-fp32-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The sets are the same as for the CPU profile ([`../gliner2-onnx-fp32-cpu/`](../gliner2-onnx-fp32-cpu/README.md)). The bounds and budget come from `protected/gliner2-base-v1/onnx-fp32-cuda/`. Every stage ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 tokenizer release (232 rows) | **passed** | processor outputs exact 232/232 (device-independent; the CPU run's candidate, evaluated against this manifest) | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1, G2 extraction release (30 rows) | **G1 passed, G2 passed**, first divergence none | processor exact 30/30; span logits relative 1.24e-5 (1e-4); count logits 3.1e-5 (5e-4); scores 3.07e-6 (2e-5); entities 30/30 with 0 exceptions; confidence 1.88e-6 (2e-5); C# vs Python ORT span logits 0 (1e-6); hidden state 1.34e-5 (1e-4) | [`g1-g2-extraction-release/report.md`](g1-g2-extraction-release/report.md) |
| G3 WNUT-17 (300 sentences) | **passed** | exact micro F1 **0.3626** (minimum 0.352), partial **0.4134** (minimum 0.403), equal to CPU and the reference; producer on CUDA device 0 (#29) | [`g3-ner/report.md`](g3-ner/report.md) |

Commands: as for CPU, with the CUDA build and `--device cuda:0`, and with `stage_gliner2.py export --provider cuda --profile gliner2-base-v1/onnx-fp32-cuda` in an environment from `gliner2/requirements-cuda.txt`. Wall clock: .NET stages 7.8 s, Python stage 2 32.3 s, G3 producer 9.8 s.
