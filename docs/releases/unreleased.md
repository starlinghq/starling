# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Starlings.Models.Gliner2: text classification, Experimental on CPU (#183, step 1)

The new `Gliner2TextClassifier` runs GLiNER2's classification head over an extractor's session. It takes a task name and at least two labels, and returns a logit and a score per label and GLiNER2's answer (single-label: softmax argmax; multi-label: sigmoid against a threshold, with the best label as fallback). It reads the locked graph's `last_hidden_state` at the `[L]` markers, so no new graph is needed. The head comes from the checkpoint's `model.safetensors`: `Gliner2ClassificationHead.FromSafetensors` reads only the 4.7 MB head and checks it against `HeadSha256`. `Gliner2ModelAssets.ClassificationLock` adds that file (834 MB, from the Hub repository at the pinned commit) to the entity lock.

The capability `classification` is now `experimental` on `gliner2-base-v1/onnx-fp32-cpu`. On 26 proposed cases run by the pinned GLiNER2, input ids and decisions were exact, logits were within 2.74e-5 relative and scores within 8.46e-6 (`docs/evidence/gliner2-classification-2026-11-01/`). It stays `unsupported` on CUDA: the classifier rejects CUDA sessions. Several tasks per call, label descriptions, temperature and batching are `unsupported`. There is no task-quality set yet, and parity is not quality: at least 2 of the 19 single-label answers in those cases are wrong for GLiNER2 itself. Entity extraction is unchanged: the protected entity parity passes, with exact inputs.
