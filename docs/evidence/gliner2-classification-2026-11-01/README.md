# GLiNER2 classification head on CPU, 2026-11-01

Issue #183, step 1. This is a parity run of GLiNER2's classification head on the existing profile `gliner2-base-v1/onnx-fp32-cpu`: the .NET path compared with the pinned Python reference. Result: parity holds. The capability `classification` moves from `unsupported` to `experimental` on the CPU profile and stays `unsupported` on the CUDA profile. It is not `supported`, because a task-quality set (G3) for classification belongs to the maintainer and none exists.

This is one machine and not qualification.

## Route

GLiNER2 2.0.0 classifies with the same encoder pass it uses for entities. The schema `( [P] task ( [L] label1 [L] label2 ... ) )` goes before `[SEP_TEXT]` and the words. The hidden state at each `[L]` marker goes through `classifier`, which is `Linear(768, 1536)`, ReLU, then `Linear(1536, 1)`. Single-label tasks take a softmax over the labels. Multi-label tasks take a sigmoid per label against `cls_threshold`, and fall back to the best label when none reaches it (`inference/runtime.py`, `_extract_classification_result`).

The locked graph (`onnx/gliner2-entities-fp32.onnx`, SHA-256 `0ba2db95…`) does not contain `classifier`. It does expose `last_hidden_state` as a third output, because route A was exported with the hidden state for stage diagnosis. The .NET path therefore needs no new graph and no new hosted asset:

1. `Gliner2Preprocessor.Build` builds the classification schema with `[L]` in place of `[E]` and the task name as the prompt.
2. `Gliner2EntityExtractor` runs the locked graph and requests only `last_hidden_state`. It gathers the rows at the `[L]` markers.
3. `Gliner2ClassificationHead` reads the four `classifier.*` tensors (4.7 MB) from the checkpoint's `model.safetensors`. That file is already pinned in the manifest's `origin.files`, SHA-256 `845fc4bd…`. The head checks those tensors' bytes against `HeadSha256` (`9a7a68ec…`) and applies the MLP.
4. `Gliner2TextClassifier` applies GLiNER2's decision rule.

`Gliner2ModelAssets.ClassificationLock` is the entity lock plus `model.safetensors` from `fastino/gliner2-base-v1` at `f9634218`. The Hub repository is public, so no Starling hosting step is needed. The download is 834 MB to obtain a 4.7 MB head, which is a cost the maintainer may want to remove later by hosting the head on its own.

## Reference

`model-pipeline/reference/gliner2/run_gliner2_classification.py` ran GLiNER2 2.0.0 (source `55656fbf`) with torch 2.14.1 CPU, transformers 4.57.6, one thread, fp32. It used the public API, `classify_text(..., include_confidence=True)`, and hooked the encoder (input ids) and `classifier` (logits). The 26 cases are under `model-pipeline/proposed/gliner2-base-v1/classification/` (see `PROPOSED.md`). They are repository-written texts:

- 19 single-label cases: sentiment, topic, support intent, language, urgency and spam
- 7 multi-label cases, including one where no label reaches the threshold and one with `cls_threshold` 0.8

The same script ran the locked graph with ONNX Runtime 1.30.0 CPU and applied the head in NumPy. This is the route the .NET package takes, so it bounds what a faithful port can reach.

## Results

Hardware: Intel i7-5820K (12 threads), Linux x64, CPU provider only. Windows x64 CPU builds but was not run. No GPU was used. .NET SDK 10.0.100, ONNX Runtime 1.30.0.

There is no protected bound for classification. The table judges with the CPU profile's protected bounds for the nearest quantities. The classifier logit has the same form as the span logit (an MLP and a dot product over the same hidden states), so it is judged with `span_logits_relative`. The score is judged with `scores_max_abs`. Whether these bounds should be the classification bounds is the maintainer's call.

| Comparison | Inputs (ids, marker positions) | Decisions | Logits, max relative | Scores, max abs |
| --- | --- | --- | --- | --- |
| Python: ORT graph + NumPy head vs PyTorch | same ids | 26 / 26 argmax | 2.75e-5 | 8.46e-6 |
| C# `Gliner2TextClassifier` vs PyTorch | exact, 26 / 26 | exact, 26 / 26 selected label sets | 2.74e-5 (bound 1e-4) | 8.46e-6 (bound 2e-5) |

Per-case output is in [classification-parity-cpu.log](classification-parity-cpu.log), taken from the test's TRX report. The whole `Starlings.Models.Gliner2.Tests` project passed with the model present (112 of 112, none skipped). [gliner2-tests-cpu.log](gliner2-tests-cpu.log) holds that run's summary, copied from the console. An earlier run of the same build, started while the machine's memory level was `tight`, reported 102 passed, 0 failed and one test-host error, and was not investigated further. The rerun above is the clean result. That includes the protected entity parity, which confirms the shared preprocessor change kept entity inputs exact.

## Parity is not quality

Parity here means the .NET path gives the same answers as the pinned GLiNER2, not that those answers are right. On the 26 cases, at least 2 of the 19 single-label answers are wrong for GLiNER2 itself:

- `spam`: an obvious prize-scam text ("Congratulations!!! You have WON a free cruise ...") is `not spam` at 1.0.
- `sentiment-neu-1`: store opening hours are `positive` at 0.796, not `neutral`.

The cases were written to exercise the route, not to measure accuracy, so classification quality is unmeasured until a task-quality (G3) set exists.

## What changed

- `Gliner2TextClassifier`, `Gliner2ClassificationHead`, `Gliner2ClassificationResult` and `Gliner2LabelScore` (new). Also `Gliner2ModelAssets.ClassificationLock` with the `ClassificationWeights*` constants, and `Gliner2EntityExtractor.LastHiddenStateName`.
- `Gliner2Capabilities.Statuses`:
  - `classification` is `experimental` (it was `unsupported`)
  - new `classification.cuda`, `unsupported`: the classifier rejects a CUDA session
  - new `classification.multiple_tasks`, `classification.label_descriptions` and `classification.batch`, all `unsupported`
- Manifests: `onnx-fp32-cpu` has `classification: experimental`. `onnx-fp32-cuda` keeps `unsupported`, with the reason. The catalog and `ProfileCapabilities.g.cs` were regenerated.
- Nothing under `model-pipeline/protected/` or in any `tolerances.*` file changed, and no maturity or gate changed.

## Not done

- **A download through `ClassificationLock`.** It was not run end to end. The pinned size and SHA-256 of `model.safetensors` were checked against a local copy, and the tests read that copy. A Hub file with no explicit source takes the same path as `tokenizer.json`.
- **Task quality (G3).** A classification task-quality set and its bound are maintainer-owned.
- **CUDA.** There is no CUDA run. A 4060 Ti run could support only an Experimental CUDA status.
- **Several tasks per call, label descriptions or examples, temperature, batching and long-text chunking.** All are `unsupported`.
- **Issue #183, step 2.** The NLI route for mDeBERTa (`IZeroShotClassifier`) stays open.
