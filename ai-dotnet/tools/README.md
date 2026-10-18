# ai-dotnet tools

Command-line tools for acquisition, fidelity stage results, operational qualification and local reproducers. None is a package. They apply no thresholds and write nothing under `model-pipeline/protected/`; the evaluators in `model-pipeline` read the bounds. Where [`../README.md`](../README.md) already documents a tool, this page links there instead of repeating it.

| Tool | Purpose | In `AI.slnx` |
| --- | --- | --- |
| [`fetch-e5-model.sh`](#fetch-e5-modelsh) | Download and verify the e5 runtime files. | n/a (script) |
| [`Starlings.Acquisition.Tool`](#starlingsacquisitiontool) | Download, verify, print or locate a model asset lock. | yes |
| [`Starlings.Fidelity.Tool`](#starlingsfidelitytool) | Stage results for e5, GLiNER2 and EmbeddingGemma 2; G1 tokenizer and G3 task-quality candidate outputs. | yes |
| [`Starlings.Fidelity.Tool.Cuda`](#starlingsfidelitytoolcuda) | The same sources against the CUDA native runtime. | no (local) |
| [`Starlings.Qwen35.Fidelity`](#starlingsqwen35fidelity) | Qwen3.5-0.8B parity, semantics, long-context and stress runs; G3 generation outputs. | yes |
| [`Starlings.Bert.Fidelity.Tool`](#starlingsbertfidelitytool) | The .NET stage of the `bert-onnx-cpu-v1` onboarding route. | yes |
| [`Starlings.Generation.Fidelity`](#starlingsgenerationfidelity) | The .NET stage of the `qwen-genai-cpu-v1` onboarding route (QwenChat). | yes |
| [`Starlings.Kev.Fidelity.Tool`](#starlingskevfidelitytool) | The .NET stage of the `kev-onnx-cpu-v1` onboarding route. | yes |
| [`Starlings.Soak.Tool`](#starlingssoaktool) | Gate G4 soak for every model profile. | yes |
| [`Starlings.Gliner2.CudaStress`](#starlingsgliner2cudastress) | Local reproducer for the GLiNER2 CUDA wrong output. | no (local) |
| [`Shared/ToolMain.cs`](#sharedtoolmaincs) | Common entry-point wrapper. | n/a (source) |

## fetch-e5-model.sh

```sh
ai-dotnet/tools/fetch-e5-model.sh [cache-dir]
```

A wrapper over `Starlings.Acquisition.Tool -- download multilingual-e5-small --cache <dir>`. The cache defaults to `$STARLING_MODELS`, else `<repo>/.models`, where the tests look. Uses `$DOTNET_ROOT/dotnet` when set, else `dotnet` on `PATH`. Exits 1 on a digest mismatch, a refused or missing file, or a network failure. See [Model acquisition (P-009)](../README.md#model-acquisition-p-009) and [Running the model tests](../README.md#running-the-model-tests).

## Starlings.Acquisition.Tool

```sh
dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- <download|verify|lock|path> <lock> [--cache <dir>] [--size-only]
```

- `download`: download missing or invalid files of the lock into the cache and verify them. The only command that uses the network.
- `verify`: offline check that the cache holds every file with the right size and SHA-256.
- `lock`: print the lock as JSON.
- `path`: print the snapshot directory.

`<lock>` is a known name (`multilingual-e5-small`, `gliner2-base-v1`, `embeddinggemma-2`, `embeddinggemma-2-image`) or the path of a JSON lock file (`ModelAssetLock.ToJson`). Other locks (Qwen3.5, Kev) are not known by name. `--cache` defaults to `$STARLING_MODELS`, else the user cache directory. `--size-only` checks cached files by size only. The token for gated repositories is read from `$HF_TOKEN` and never printed. Exit codes: 0 success, 1 an acquisition error, 2 usage error, 130 cancelled (partial downloads are kept and resumed). See [Model acquisition (P-009)](../README.md#model-acquisition-p-009).

## Starlings.Fidelity.Tool

```sh
dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- <subcommand> --out <dir|file> [options]
```

Runs the C# packages on fixture or dataset rows and writes inputs for `model-pipeline evaluate`. Subcommands:

| Subcommand | Writes | Documented in |
| --- | --- | --- |
| `e5` | `tokenizer.results.json`, `dotnet.results.json` (`--hidden` adds last hidden states) | [Fidelity stage results (P-007)](../README.md#fidelity-stage-results-p-007) |
| `gliner2` | `tokenizer.results.json`, `dotnet.results.json` (task `entity-extraction`) | [GLiNER2 entity extraction (P-015)](../README.md#gliner2-entity-extraction-p-015) |
| `embeddinggemma2`, `embeddinggemma2-images` (`--images <dir>`), `embeddinggemma2-videos` (`--videos <dir> [--ffmpeg <dir>]`) | EmbeddingGemma 2 stage results | [Fidelity stage results (P-007)](../README.md#fidelity-stage-results-p-007), "CUDA build" |
| `embeddinggemma2-long` | `long.results.json` for the proposed long-input rows (`--rows <long-embeddings.jsonl> --images <dir> --videos <dir>`), compared by `model-pipeline/reference/embeddinggemma2/compare_embeddinggemma2_long_csharp.py` | header of `EmbeddingGemma2LongDump.cs` |
| `retrieval --model e5\|embeddinggemma2\|embeddinggemma2-image\|embeddinggemma2-video --dataset <dir>` (media: `--media <dir>`) | query and document embeddings for `model-pipeline evaluate --g3` | header of `TaskQualityDump.cs` |
| `ner --dataset <gold.jsonl>` | GLiNER2 predictions for `model-pipeline evaluate --g3` | header of `TaskQualityDump.cs` |
| `tokenizer --model e5\|embeddinggemma2\|qwen35\|gliner2 --tokenizer <tokenizer.json> --fixtures <jsonl>` | `tokenizer.candidate.jsonl`, `environment.json` for `model-pipeline evaluate --tokenizer-fixtures` | header of `TokenizerDump.cs` |

Common options (where the subcommand accepts them): `--snapshot <dir>` (e5 default `$STARLING_E5_SNAPSHOT`, else the acquisition cache), `--threads <n>`, `--device cpu|cuda|cuda:N` (`cuda` needs the CUDA build below). The e5 dump also takes `--fixtures <dir>` and `--profile <id>`. A row that the package rejects with a typed per-input error is written as `rejected` and the run exits 1; see "Errors and exit codes" in [Fidelity stage results (P-007)](../README.md#fidelity-stage-results-p-007).

## Starlings.Fidelity.Tool.Cuda

The same sources as `Starlings.Fidelity.Tool`, built against `Microsoft.ML.OnnxRuntime.Gpu.Linux` with its own `packages.lock.json`; the assembly keeps the name `Starlings.Fidelity.Tool`. Not in `AI.slnx`, so CI never restores the GPU package. It also runs `--device cpu`. Needs the CUDA 13 runtime and cuDNN 9 on the loader path and an explicit `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>`. Usage: "CUDA build" in [Fidelity stage results (P-007)](../README.md#fidelity-stage-results-p-007).

## Starlings.Qwen35.Fidelity

```sh
dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release [-p:UseCuda=true] -- \
  --bundle <dir> --variant fp32x|fp16x --device cpu|cuda --prompts <jsonl> --reference <dir> --out <dir> \
  [--modes parity,semantics] [--threads N] [--prefill-chunk N] [--serialize true|false]
```

Modes: `parity` (template, ids and greedy generation against the reference; `results.jsonl`, `step0/<id>.f32`), `semantics` (stop handling, caching, cancellation, concurrency, sampling; `semantics.json`), `fixture-semantics` (the G2 subset of `semantics` on any generation fixture), `longcontext` (`longcontext.json`), `stress` and `rawstress`. `semantics`, `stress` and `rawstress` need the fixed prompts of `model-pipeline/reference/generation/prompts-v1.jsonl`; without them the tool exits 2 before loading the model. `-p:UseCuda=true` swaps in the GenAI CUDA natives (`packages.cuda.lock.json`).

`-- generation --bundle <dir> --prompts <jsonl> --out <dir> [--variant] [--device] [--threads]` writes `generations.jsonl` for `model-pipeline evaluate --g3 generation`. Exit codes: 0 done, 1 rows rejected (`generation`), 2 usage or input error, 70 an exception escaped. Score parity runs with `model-pipeline/reference/generation/compare.py`. See the fidelity runner in [Generation backend and pilot LLM (P-016)](../README.md#generation-backend-and-pilot-llm-p-016).

## Starlings.Bert.Fidelity.Tool

```sh
Starlings.Bert.Fidelity.Tool --profile <candidate.json> --snapshot <dir> --reference <reference.json> --out <dir> [--threads 2]
```

The .NET stage of the `bert-onnx-cpu-v1` onboarding route (`model-pipeline/src/model_pipeline/onboarding_bert.py`), which publishes the tool and runs it. It loads the candidate profile with `BertModelProfile.Load`, checks that the reference belongs to that candidate (profile id and candidate SHA-256), and runs every reference case through `Starlings.Models.Bert` from the reference ids and from text. It writes `dotnet.json` (format `starlings.bert-candidate-results`) and, for embedding profiles, the hidden states in `dotnet-hidden.f32`. CPU only. Exit 2 on a usage error. Recipes: `model-pipeline/recipes/{minilm,bge-small,ms-marco,nomic-embed}-cpu.yaml`; see [`model-pipeline/ONBOARDING.md`](../../model-pipeline/ONBOARDING.md).

## Starlings.Generation.Fidelity

```sh
Starlings.Generation.Fidelity --candidate <candidate.json> --bundle <dir> --corpus <jsonl> --out <dir> [--threads 2] [--templates-only true]
```

The .NET stage of the `qwen-genai-cpu-v1` onboarding route (`model-pipeline/src/model_pipeline/onboarding_generation.py`) for `Starlings.Models.QwenChat` (Qwen3-0.6B, Qwen2.5-0.5B-Instruct). It checks the candidate's profile id and revision against `QwenChatProfile`, refuses an output directory inside the bundle and corpus rows with `enable_thinking`, and runs each corpus row through `QwenChatModel`. It writes `results.jsonl`, or `templates.jsonl` with `--templates-only true` (template rendering and ids, no model load). Arguments are `--name value` pairs. Recipes: `model-pipeline/recipes/{qwen3,qwen25}-cpu.yaml`.

## Starlings.Kev.Fidelity.Tool

```sh
Starlings.Kev.Fidelity.Tool --profile <candidate.json> --snapshot <bundle dir> --reference <reference.json> --out <dir> [--threads 2] [--device cpu|cuda[:N]]
```

The .NET stage of the `kev-onnx-cpu-v1` onboarding route (`model-pipeline/src/model_pipeline/onboarding_kev.py`). It reads the reference stage's `reference.json`, runs every case through `Starlings.Models.Kev` from the reference ids (cached and uncached) and from the record text, and writes `dotnet.json` for the decision evaluator (`model_pipeline/evaluate_decision.py`). `--device cuda[:N]` runs the same bundle as the `onnx-fp32-cuda` profile (`use_tf32=0`) and needs the `-p:UseCuda=true` build, which swaps in `Microsoft.ML.OnnxRuntime.Gpu.Linux` with its own lock file (`packages.cuda.lock.json`). Exit 2 on a usage error. See [Kev decision models](../README.md#kev-decision-models).

## Starlings.Soak.Tool

See [Operational qualification (gate G4)](../README.md#operational-qualification-gate-g4).

## Starlings.Gliner2.CudaStress

```sh
CUDA_VISIBLE_DEVICES=<gpu> STARLING_GLINER2_SNAPSHOT=<dir> ai-dotnet/tools/Starlings.Gliner2.CudaStress/run.sh [options]
```

A local-only stress reproducer for the GLiNER2 CUDA wrong output of the 2026-10-15 G4 soak ([evidence](../../docs/evidence/gliner2-cuda-wrong-output/README.md)). Not in `AI.slnx`; references `Microsoft.ML.OnnxRuntime.Gpu.Linux`. `run.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, builds the project and passes the snapshot (`tokenizer.json` and `onnx/gliner2-entities-fp32.onnx`) as the first argument. It needs the CUDA 13 runtime and cuDNN 9 on `LD_LIBRARY_PATH` or in the ldconfig cache.

- Default mode: the soak's pattern. Options `--concurrency`, `--cycles`, `--requests` (per cycle), `--deadline-every`, `--deadline-min`, `--deadline-max` (ms), `--keep-loaded`, `--seed`. On a mismatch it prints the raw count and span logits and the request history.
- `--pairs n`: one thread runs `n` batches of `--long-count` texts (`--batch` per graph call) cancelled after a deadline; `--waiters` threads run short requests meanwhile.
- `--probe n`: `n` cancelled batches, then times `cudaDeviceSynchronize` after each call.
- `--noise n`: `n` other extractors run uncancelled batches on the same GPU.
- `--cpu`: the CPU provider instead of CUDA.

It does not run through `Shared/ToolMain.cs`.

## Shared/ToolMain.cs

The entry-point wrapper that the tools above (except `Starlings.Gliner2.CudaStress`) and the benchmark harnesses compile in. An escaping exception is written to stderr with the prefix `error:` and the process exits 70 instead of aborting. See "Errors and exit codes" in [Fidelity stage results (P-007)](../README.md#fidelity-stage-results-p-007).
