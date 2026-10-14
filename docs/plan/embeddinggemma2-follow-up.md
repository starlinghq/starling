# EmbeddingGemma 2 follow-up plan

## Starting point

EmbeddingGemma 2 text, image and video embeddings are merged into `main` (video: PR #50, `c4726c7`). The full requested contract is implemented at maturity Experimental. Evidence for video is in `docs/evidence/embeddinggemma2-video/`:

| Check | CPU | CUDA (`use_tf32=0`) | Bound |
| --- | --- | --- | --- |
| Frame sampling, ids, decoded frames, resizing, positions | exact | exact (same code) | exact |
| Embedding max abs | 1.7e-6 | 3.3e-6 | 1e-5 |
| Min cosine | 0.99999999992 | 0.99999999968 | 0.9999995 |
| Batched vs single | 0 | 8.9e-8 | 1e-5 |
| FFmpeg adapter vs torchcodec, 22 videos, 5,367 frames | exact | | exact |
| Protected evaluator, G1 and G2 | pass | not run | required |

What is not finished:

- Nothing is published. The EmbeddingGemma 2 packages are in the 0.1.0-preview.5 candidate (`docs/releases/ai-0.1.0-preview.5.md`, PR #52, W5 done): 14 publishable packages, the FFmpeg adapter built and drilled but held until the D2 legal review. Publishing itself still waits for the `release` environment and the NuGet prefix reservation.
- The registry manifests for all six EmbeddingGemma 2 profiles keep G1 and G2 at "not run".
- CUDA profiles (image and video) have test-project evidence only. The fidelity tool references the CPU natives, so the protected evaluator has never run on CUDA.
- The CUDA video error rose from the spike's 4.5e-7 to 3.3e-6 after the default moved from 4 frames per vision call to 2.
- The FFmpeg adapter supports FFmpeg 8 only. The workstation has FFmpeg 9.0.1, which the adapter rejects with a typed error.
- Input between 2,049 and 8,192 tokens is untested (admission record, feature contract).
- The browser (WebAssembly) target has eight open decisions (`docs/m0/browser-wasm-spike.md` section 10).

## 1. Maintainer decisions

All seven were approved by Mark on 2026-10-12 as recommended, and recorded in `docs/kickoff.md`. Mark added one rule: every adapter is a voluntary addition. Users choose to reference it, so they decide how deep to go. That covers SkiaSharp imaging, each FFmpeg-major video adapter, the browser's WebGPU support, and any future adapter. No core package may reference an adapter, directly or transitively.

What remains with Mark: the legal review itself (D2), the P-030 sets (D5), and committing the protected tolerance files in section 1.1 (the browser files now, the FFmpeg 9 gate with W4). The safety controls on agent sessions block agents from writing under `model-pipeline/protected/`.

The browser spike made no recommendation on hosting (its decision 4). The recorded default is a GitHub release asset, as for the GLiNER2 and EmbeddingGemma 2 graphs. The spike's quantization script stays available for users who build the graph themselves.

| ID | Decision | Recommendation | Unblocks |
| --- | --- | --- | --- |
| D1 (approved) | Browser decisions 1 to 8 from the spike report: target, e5 variant, registry profiles for quantized variants, hosting, onnxruntime-web as a redistributed dependency, CI, AOT, spike code. | As in the spike: e5 only, WASM qualified and WebGPU experimental, int8-embed default, separate registry profiles with maintainer-set tolerances, AOT for consumers, keep `browser/spike/` as the package seed. | W6 |
| D2 (approved) | Legal review of the FFmpeg adapter before any publish. Scope for the reviewer: dynamic loading of the user's LGPL FFmpeg, no FFmpeg binaries shipped, GPL and nonfree builds rejected unless opted in, codec patent responsibility placed on the deployer, the adapter's own FFmpeg bindings generated from FFmpeg's headers (`ai-dotnet/THIRD-PARTY-NOTICES.md`; FFmpeg.AutoGen was replaced on 2026-10-13). | Review before the adapter package is published, not before other packages. The adapter can be held back from a release on its own. | W5 (adapter package only) |
| D3 (approved) | FFmpeg 9 support, and how: a second adapter package with bindings generated from the FFmpeg 9 headers, or one adapter that binds both majors. | Separate package per FFmpeg major. Each keeps a single decoder profile and its own exactness gate. Bindings differ per major, so one package would carry two binding sets. | W4 |
| D4 (approved) | Confirm the gate policy: G1 and G2 move into the manifests only after the protected release set (P-030) exists. Or accept development-fixture evidence for Experimental profiles now. | Keep the existing policy, starting with `video-fp32-cpu` once P-030 lands. | W7 |
| D5 (approved) | Curate P-030 release sets for EmbeddingGemma 2 (text, image, video). Plan 17 budgets 10 to 20 maintainer hours for the encoder and tokenizer sets; multimodal sets add to that. | Start with a small retrieval-style text set and a handful of licensed images and videos with known sources. Agents do not author these sets. | W7 |
| D6 (approved) | Cut a release candidate that includes the EmbeddingGemma 2 packages (next would be 0.1.0-preview.5), with or without the FFmpeg adapter. | Yes, with the adapter included only if D2 is done. Publishing still waits for the release environment and the NuGet prefix (`docs/kickoff.md`). | W5 |
| D7 (approved) | CUDA vision frame-call default, after W1 reports. | Decide on W1's numbers. Keep 2 unless 4 fits the shared GPU with headroom. | none |

### 1.1 Protected tolerance files for Mark to commit

These carry the approved values. Agents cannot write them. The browser files can be committed now. The FFmpeg 9 gate waits for W4, for the reason given with it.

**FFmpeg 9 decoder gate (commit with W4, not before).** Append to `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/tolerances.yaml` and `video-fp32-cuda/tolerances.yaml`. The FFmpeg 8 keys stay unchanged.

```yaml
decoder_ffmpeg9:
  # Set by Mark on 2026-10-12: the separate FFmpeg 9 adapter (user-supplied LGPL FFmpeg 9.x, x86-64) must reproduce
  # torchcodec's decoded RGB exactly on FFmpeg 9. The spike compared torchcodec on FFmpeg 8 and 9: 22/22 equal.
  decoded_rgb_vs_torchcodec: exact
  supported_ffmpeg_major: 9
```

Why it waits: the video evaluator's decoder check (`_decoder` in `model-pipeline/src/model_pipeline/evaluate_video.py`) reads the `decoder_ffmpeg.*` keys by name and nothing else. `EXPECTED_KEYS` only decides which keys are reported as unknown. Adding `decoder_ffmpeg9.*` there without a check would make the key known and never evaluated, which breaks the evaluator's own rule that a bound nobody checks is not silently satisfied. Leaving it out of `EXPECTED_KEYS` instead would report the key as not evaluated and stop the existing `video-fp32-cpu` G1 from passing until W4 lands. So the W4 PR carries the check, the key text above (proposed, for Mark to commit) and the design for one evaluator run holding one FFmpeg major (today the stage environment records a single `ffmpeg_major`; gating two majors needs a decoder stage result per adapter or an equivalent).

**Browser profiles.** Values from `docs/m0/browser-wasm-spike.md` section 3.4. The reference is the PyTorch fp32 fixture set of `multilingual-e5-small/onnx-fp32-cpu`.

Two things to expect once these files exist. First, `browser_vs_native_same_graph.*` and `task_metric_before_preview.*` are not in the text evaluator's `KNOWN_KEYS` (`model-pipeline/src/model_pipeline/evaluate.py`), so the evaluator reports them as not evaluated and G2 for these profiles cannot pass until W6 implements the checks. That is the intended behaviour, not a regression. Second, the files omit `float.last_hidden_state_max_abs` on purpose: the browser path has no hidden-state export stage. The export check reads that key by name, so W6 must make it tolerate the key's absence or add a browser-specific export stage. Nothing validates these files until W6 adds the matching registry manifests.

`model-pipeline/protected/multilingual-e5-small/onnx-fp32-wasm/tolerances.yaml`, and the same file with `webgpu` in the profile name and path:

```yaml
# Binding tolerances, set by Mark on 2026-10-12, plan 10.1. Values from docs/m0/browser-wasm-spike.md section 3.4.
# Observed worst (Chrome for Testing 153, onnxruntime-web 1.30.0): embedding 2.1e-7, padded batch vs single 1.5e-7.
profile: multilingual-e5-small/onnx-fp32-wasm
reference: PyTorch fp32 embeddings from model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/
discrete:
  token_ids: exact
  attention_mask: exact
  offsets: exact
float:
  embedding_max_abs: 1.0e-5
  padded_batch_vs_single_embedding_max_abs: 1.0e-5
  nan_or_inf: fail
  report_only: [mean_abs, p99_abs, min_cosine]
```

`model-pipeline/protected/multilingual-e5-small/onnx-int8embed-wasm/tolerances.yaml`, and the same file with `webgpu`:

```yaml
# Binding tolerances, set by Mark on 2026-10-12, plan 10.1. Values from docs/m0/browser-wasm-spike.md section 3.4.
# Observed worst: embedding 3.5e-3, min cosine 0.99980, padded batch vs single 1.6e-7, browser vs native ORT 1.8e-7.
# Before Preview: retrieval top-10 overlap with fp32 at least 0.95 on a P-030 held-out query set.
profile: multilingual-e5-small/onnx-int8embed-wasm
reference: PyTorch fp32 embeddings from model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/
discrete:
  token_ids: exact
  attention_mask: exact
  offsets: exact
float:
  embedding_max_abs: 1.0e-2
  min_cosine: 0.9995
  padded_batch_vs_single_embedding_max_abs: 1.0e-5
  nan_or_inf: fail
  report_only: [mean_abs, p99_abs]
browser_vs_native_same_graph:
  embedding_max_abs: 1.0e-5
task_metric_before_preview:
  retrieval_top10_overlap_vs_fp32: 0.95
```

## 2. Work items

Each item is one draft PR. Bounded authority from `CLAUDE.md` applies throughout: no edits to `model-pipeline/protected/` or `tolerances.*`, new edge cases go in a separate change with their reference source, weights stay out of Git, and two failed repair attempts stop the item with a reproducer.

### Ready now

**W1. Measure the CUDA frame-call size.** Run `CudaVideoParityTests` on GPU 1 with frame calls of 2, 3 and 4. Record embedding max abs, min cosine, peak GPU memory and wall clock for each, plus how much memory the other workload leaves. Acceptance: a short evidence note under `docs/evidence/embeddinggemma2-video/` with the table and the memory headroom. No default changes in this PR; the default is D7.

**W2. Run the protected evaluator on CUDA.** Give the fidelity tool a CUDA mode, or a separate CUDA build of it, so `model-pipeline evaluate --require-gates G1,G2` runs for `image-fp32-cuda` and `video-fp32-cuda`. Use the same reference tensors and the CPU stage-1 checks unchanged. Acceptance: unedited `report.md` and `report.json` for both CUDA profiles, protected evidence verified against Git, and the CPU path unchanged (CPU reports reproduce). If CUDA natives cannot coexist with the CPU natives in one tool, document why and build two. Status 2026-10-13 (R7): two builds, because both native packages ship `libonnxruntime` under one file name. `ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda` compiles the tool's sources against `Microsoft.ML.OnnxRuntime.Gpu.Linux` with its own lock file, outside `AI.slnx` like the CUDA test projects; the EmbeddingGemma 2 dumps take `--device cpu|cuda[:N]` (`ai-dotnet/README.md`, "CUDA build"). Built and run on CPU only: its CPU results equal the CPU build's exactly. The GPU run and the two CUDA reports are the maintainer's (GPU allowance).

**W3. Propose long-input cases.** In a separate PR, propose fixtures between 2,049 and 8,192 tokens for text, interleaved text and image, and video near the 32-frame limit, generated by the pinned reference runner. Acceptance: a `PROPOSED.md` with sources and reference environment, marked "proposed". The contract row moves from "untested" only after the maintainer accepts the fixtures and a parity run passes. Status 2026-10-13 (R7): proposed in `model-pipeline/proposed/embeddinggemma-2/long-input/` (17 rows: text, text with images, text with video at the 32-frame limit, four rejections just over 8,192), generated by `run_embeddinggemma2_long.py`; the C# package (`Starlings.Fidelity.Tool embeddinggemma2-long`, CPU) gives equal ids, the expected rejections, embedding max abs 1.0e-7 on text rows and 6.8e-7 on media rows. `input_2050_to_8192_tokens` stays `untested`.

### After a decision

**W4. FFmpeg 9 adapter (D3).** A new package, for example `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`, with bindings generated from the FFmpeg 9 headers (`tools/generate-ffmpeg-bindings.py`), pinned by FFmpeg commit. It reuses the scan, sampling, display-matrix and conversion logic through shared source, not a copy. Acceptance: the same 22 fixture videos decode exactly like torchcodec on FFmpeg 9; the spike already measured 22 of 22 equal. Loader errors, license classification and the no-bundled-FFmpeg CI check cover the new package. No core package references either adapter; CI rejects any third-party FFmpeg bindings package in every project. Notices and README updated in the same change. The same PR adds the FFmpeg 9 decoder check to `evaluate_video.py`, with a design that lets one evaluation hold decoder results for both FFmpeg majors, and proposes the `decoder_ffmpeg9` tolerance text from section 1.1 for Mark to commit alongside it. Adding the key to `EXPECTED_KEYS` without the check is not acceptable (section 1.1 explains why). Status 2026-10-13 (R7): done except the maintainer's commit of the tolerance text. `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` shares the adapter code with the FFmpeg 8 package (`ai-dotnet/src/Shared/EmbeddingGemma2.Video.FFmpeg/`) and has bindings generated from the FFmpeg 9.0.1 headers (commit `bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa`); on the machine's FFmpeg 9.0.1 it decodes every frame of the 22 videos exactly like torchcodec on FFmpeg 9.0.1, which equals the protected digests. The evaluator checks `decoder_ffmpeg9` when the protected tolerances carry it, from FFmpeg 9 frames written next to the FFmpeg 8 ones in one run. The tolerance text is proposed in `model-pipeline/proposed/embeddinggemma-2/decoder-ffmpeg9/`. The package is held from publishing like the FFmpeg 8 one (D2). `video_decode_ffmpeg9` is experimental in both video manifests.

**W5. Release candidate with EmbeddingGemma 2 (D6, D2 for the adapter).** Bump the ai-dotnet set, run `tools/release-drill-ai.sh` with the models and the FFmpeg 8 directory, run the tokenizer drill, and write `docs/releases/<version>.md` listing each EmbeddingGemma 2 package, its maturity, platform statuses, and the Gemma Prohibited Use Policy notice. Acceptance: drills pass from a clean clone; the record states plainly that nothing is published. Done 2026-10-12 (PR #52): `docs/releases/ai-0.1.0-preview.5.md`, both drills passed, FFmpeg adapter held per D2. Not run in that drill: G1 and G2 with the models (wait on P-030), anything on CUDA, and the EmbeddingGemma 2 ML.NET package beyond its unit tests.

**W6. Browser package (D1).** Scope follows D1. With the recommended answers: a browser package for e5 on onnxruntime-web, the int8-embed profile proposed with fixtures for maintainer acceptance, CI with the wasm-tools workload and headless Chromium on WASM only. Acceptance also covers the evaluator: implement the `browser_vs_native_same_graph` and `task_metric_before_preview` checks so the browser profiles' G2 can pass (until then they report as not evaluated, see section 1.1), make the export check tolerate the missing hidden-state bound or add a browser export stage, and add the four registry manifests. EmbeddingGemma 2 in the browser is out of scope until e5 is qualified there.

**W7. Gate flip (D4, D5).** Once P-030 exists for EmbeddingGemma 2, run the protected evaluator against it and update the manifests' G1 and G2 entries, starting with `video-fp32-cpu`, then the other CPU profiles, then CUDA once W2 has landed. Acceptance: each manifest change links its report.

## 3. Order

1. W1 and W2 now, in parallel. Both are local GPU runs inside the monthly GPU allowance.
2. W3 alongside them; it only needs the CPU reference environment.
3. D1 to D7 as the maintainer has time. D2 and D6 matter most for shipping; D1 is the largest scope choice.
4. W5 after D6 (done, PR #52). W4 after D3. W6 after D1. W7 after D4 and D5.

## 4. Out of scope

- Audio embeddings: not requested (admission record).
- Changes to existing tolerances or fixtures.
- Publishing to nuget.org, tags, CI permissions and secrets: maintainer actions.
