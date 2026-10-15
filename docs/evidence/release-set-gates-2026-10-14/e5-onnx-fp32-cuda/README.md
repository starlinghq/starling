# multilingual-e5-small `onnx-fp32-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` (main after #29, plus this branch's evidence commits) on the RTX 3090, GPU 0 only. The sets are the same as for the CPU profile ([`../e5-onnx-fp32-cpu/`](../e5-onnx-fp32-cpu/README.md)). The bounds come from `protected/multilingual-e5-small/onnx-fp32-cuda/tolerances.yaml` and the budget from its `task-quality.yaml`. The evaluator verified the protected evidence against Git. Every stage ran on `CUDAExecutionProvider` with `use_tf32: 0`, and the evaluator's `provider_options` check confirms it.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 tokenizer holdout (232 rows) | **passed** | ids and attention mask exact 232/232 (the owned tokenizer is device-independent; its candidate is the CPU run's, evaluated against this profile's manifest) | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G1, G2 staged on the text-encoder release set | **G1 passed, G2 passed**, first divergence none | .NET embedding from ids and from text 1.2e-7 (1e-5); C# vs Python ORT CUDA 8.5e-8 (1e-6); Python ORT CUDA vs PyTorch hidden state 2.25e-6 (1e-4), embedding 1.3e-7 | [`g1-g2-staged/report.md`](g1-g2-staged/report.md) |
| G3 MIRACL | **passed** | nDCG@10 **0.7978** (minimum 0.792), MRR@10 **0.7717** (minimum 0.765), equal to the CPU run and the PyTorch reference. The producer printed `device: CUDA (cuda device 0, CUDA_VISIBLE_DEVICES=0)` (`--device` from #29). | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

Commands: as for CPU, with the CUDA build (`ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda`) and `--device cuda:0`; `stage_e5_export.py --provider cuda --profile multilingual-e5-small/onnx-fp32-cuda` in an environment from `encoder/requirements-cuda.txt`; `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, with CUDA 13 and cuDNN 9 from that environment's `nvidia-*` wheels on `LD_LIBRARY_PATH`. Wall clock: .NET stages 8.4 s, Python stage 2 37.6 s, G3 producer 12.6 s.
