# Qwen3.5-0.8B int4 on CPU: Experimental profile `ortgenai-int4-cpu` (issue #163, step 1)

Run on 2026-11-01. This is the evidence for the Experimental profile `qwen3.5-0.8b/ortgenai-int4-cpu` (`model-pipeline/registry/qwen3.5-0.8b/ortgenai-int4-cpu/manifest.yaml`, `Qwen35Variant.Int4`). It also shows the evidence shape proposed for quantized generator profiles. Nothing here sets a tolerance or a gate. The fixture is proposed (`model-pipeline/proposed/qwen3.5-0.8b/ortgenai-int4-cpu/`). The maintainer sets the int4 bounds and task-quality floor in step 2.

**CPU only. No GPU was used.** The host has an RTX 4060 Ti, but no run here touched it, so nothing here says anything about CUDA. `Qwen35ChatModel.Load` rejects the int4 variant on CUDA, and the manifest lists CUDA as unsupported.

## 1. Environment

- Host: Intel Core i7-5820K (6 cores, 12 threads, AVX2, no AVX-512), 94 GB RAM, Linux x86-64. The host was shared with CI runners and other jobs (load average 10 to 46 during the runs), so no timing here is a benchmark.
- Python 3.12.14 with `model-pipeline/reference/generation/requirements.txt`: onnxruntime-genai 0.17.1 (pip wheel, producer string `0.17.1+c98b9fc`), onnxruntime 1.30.0, torch 2.14.1+cu130 (builder only), transformers 5.18.0, tokenizers 0.23.2, numpy 2.5.3.
- .NET 10.0.14, Microsoft.ML.OnnxRuntimeGenAI 0.17.1 (NuGet, CPU), `ai-dotnet/tools/Starlings.Qwen35.Fidelity` built from this branch.
- Checkpoint: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, downloaded with `fetch_snapshot.py`. Every hashed file equals `docs/m0/hub-pins.md`. The build checks the four files it reads.

## 2. Build

```sh
python model-pipeline/reference/generation/build_ortgenai_bundle.py --snapshot <snapshot> --out <bundle> --variant int4
```

The new `int4` variant of the bundle script is the float32 config overlay (sha256 `295154cd…`), then `-p int4 -e cpu --extra_options exclude_mtp=true`. The builder's CPU defaults give round-to-nearest quantization with block 32 and accuracy level 4. The graph has 151 `MatMulNBits` nodes and a `GatherBlockQuantized` embedding.

| File | SHA-256 | Bytes | Against M0 (`docs/m0/generation-backend-check.md` section 2) |
| --- | --- | ---: | --- |
| `ortgenai-int4/model.onnx.data` | `709e08797497f609547f7aef1387cfb35bd0deb3f4039acf001f0680b23f9745` | 541,523,968 | byte-identical |
| `ortgenai-int4/model.onnx` | `a41f341f106a1bb5b50eb67ca824cd9c324705141d88b027de0888f140e39cf7` | 364,388 | differs from M0's `766e49f3…` (size not recorded in M0) |
| `ortgenai-int4/genai_config.json` | `e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba` | 1,935 | not recorded in M0; equal to the fp32x build's file |
| `transformers-5.18.0/tokenizer.json`, `chat_template.jinja` | as in every Qwen3.5 lock | | equal |

I built twice into separate work directories and got the same digests both times, so the build is reproducible in this environment. The weights are byte-identical to M0's. Only the graph proto differs from M0's recorded digest. I did not investigate why. M0's builder environment may have differed (the producer string of the M0 file was not recorded). The lock (`Qwen35ModelAssets`), the manifest and the bundle script pin the digests above.

**Not hosted.** Hosting is out of scope for this step (no uploads), so the profile uses a local path only. `Qwen35ModelAssets.DownloadAsync(Qwen35Variant.Int4)` throws `NotSupportedException`, and the lock's graph files have no source. Users build the files and open them with `Qwen35ModelAssets.OpenDirectory(dir, Qwen35Variant.Int4)`. The manifest records `redistribution: blocked`, because no public int4 bundle exists. The profile is still useful without hosting: the build is one command from the pinned checkpoint, and it reproduces the pinned digests.

## 3. Reference: the same graph through the upstream Python binding

The transformers fp32 reference cannot be the reference of an int4 graph (section 5). The reference here is the same int4 graph run by the upstream Python binding, onnxruntime-genai 0.17.1 from pip: `model-pipeline/reference/generation/run_genai_reference.py`.

- `run` feeds the prompt ids of the accepted fixture `model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl` (prompts-v1, 22 prompts), so the prompt side stays the accepted one. Settings equal the C# runtime's: CPU provider, `pad_token_id` overlaid to 248319, greedy, `max_length` = prompt + `max_new_tokens`, the prompt in 1,024-token `append_tokens` calls (the C# CPU default), and `intra_op_num_threads` 6. It checks that each step's token is the argmax of that step's logits. It also checks that the step-0 vector read after `generate_next_token` equals the one read after `append_tokens`.
- `annotate` loads only the transformers tokenizer. It re-renders every prompt with the snapshot's chat template, checks the rendering and ids against the fixture (22 of 22 equal), and decodes `output_text`.
- Why there are two processes: the first two `run` attempts loaded transformers in the same process, and both were killed with SIGKILL at the 1,666-token prompt (`long-license`), the first with a one-call prefill and the second with 1,024-token chunks. Two runs without transformers in the process completed, each beside another job. No out-of-memory record was readable, so the cause is unconfirmed. Per the two-repair rule, I did not retry that configuration. The runner now keeps the tokenizer out of the generation process.

Determinism of the reference (`results.jsonl` fields compared per prompt against run B, and the step-0 `.f32` files compared byte for byte):

| Runs | Threads | Prompts | Result |
| --- | ---: | ---: | --- |
| B, C, E | 6, 6, 6 | 22 each | identical to B: output ids, finish reasons, per-step log-probabilities and margins, top-5 at step 0, and all 22 step-0 vectors. Only `elapsed_s` differs, and `output_text`, which only the run that `annotate` processed carries. |
| D | 4 | 11 of 22 | identical to B on the 11 prompts it ran (same fields, 11 step-0 vectors). The run stopped after `ru-de-mix` without an error in its log and was not repeated, so the 4-thread check covers half of the set and none of the long prompts. |

## 4. .NET against the Python int4 reference

`Starlings.Qwen35.Fidelity --variant int4 --device cpu --threads 6 --modes parity,semantics --reference <run B>`, default prefill. `compare.py` then compared its output with run B:

| Check | Result |
| --- | --- |
| Greedy sequences exact | **22/22** (token match 1.000, finish reasons 22/22) |
| Step-0 log-probabilities, full vocabulary | **max abs difference 0.0** (bit-identical; argmax 22/22) |
| Owned template rendering and owned-tokenizer prompt ids | 21/21 equal (the `tools` prompt is fed by ids; the fidelity tool's text path does not render tools) |
| Streaming | updates reassemble to the result on 22/22 |

Semantics mode on int4 (`semantics.json`, prompts of prompts-v1, with the Python int4 run as the reference):

| Check | Result |
| --- | --- |
| Max-token boundary (`en-short`, limits 1, 6, 7, 8) | finish reasons and ids as expected on all four |
| Stop sequence, extra stop token id | cut text equal to the reference before the stop; the extra stop id ends at its first occurrence |
| Cancellation | `OperationCanceledException` between steps, before the prefill for a cancelled token, reuse afterwards equal to the reference |
| Concurrency | two threads, four prompts: prefixes equal to the reference; interleaved streams equal |
| Seeded sampling | same seed identical, another seed differs, top-k 1 equals greedy |
| Chunked prefill (`long-license` at 128 and 100, `multi-turn` at 7) | tokens equal to the whole prefill; step-0 difference 0 |
| **Cached vs uncached decoding** (`en-list`, 12 steps) | tokens equal on 12/12, but **log-probabilities differ by up to 0.456** (0 at step 1, 0.26 to 0.46 at steps 2 to 12) |

The cached-versus-uncached gap is the main numerical finding. On fp32x it is 3.4e-5 against a protected bound of 2e-4. On int4 it is about 2,000 times the fp32x bound. The .NET cached path equals the Python binding's cached path bit for bit (the table above), so the gap comes from inside the native int4 graph, not from the .NET loop. My guess, not tested: with accuracy level 4, `MatMulNBits` quantizes activations to int8 per block, so a one-token step and a many-token prefill compute through different paths. Any int4 bound for this check has to come from int4 observations, not from the fp32x tolerance file.

## 5. The int4 graph against the transformers fp32 reference (context, not a target)

Python run B compared with the protected fp32 fixture: 1/22 exact sequences (`tools`), token match 0.051, step-0 argmax 13/22, and a maximum step-0 difference over the reference's top 5 of 6.97 (`en-short`). M0 recorded 1/22, 0.052, 14/22 and 6.97 for its int4x CPU run. That run used GenAI's own `pad_token_id` 248044 and its own prefill. Here the overlay is 248319, with 1,024-token chunks. The two agree except for one step-0 argmax and one prefix token. The fixture's per-step margins show that int4 greedy output is a different sequence from fp32 within a few tokens on almost every prompt. That is why this profile claims parity with the Python binding, and no fidelity to the checkpoint.

## 6. Candidate values for the maintainer (not set here)

These are observations, offered as starting points for step 2. Nothing below is written to a tolerances file.

| Check | int4 observation | Candidate |
| --- | --- | --- |
| Greedy ids against the Python int4 reference | 22/22 exact | exact, as fp32x |
| Step-0 log-probabilities against the Python int4 reference | 0.0 (same native code) | an fp32x-sized bound (for example 5e-3 / 2e-3) leaves room for a different ONNX Runtime native build (NuGet against pip) without hiding a defect |
| Template rendering, prompt ids, streaming, usage, finish reason | exact | exact, as fp32x |
| Cached vs uncached log-probabilities | 0.456 on `en-list` (12 steps) | an int4-specific bound, measured on more prompts first (for example on the release prompts), or token-level equality only |
| Chunked vs one-call prefill | 0 | as fp32x (1e-4) |
| Task quality | not measured | GSM8K (the 50-problem `gsm8k-test-v1` set), measured on this profile before a floor is set |

## 7. What remains (issue #163)

1. **Step 2, maintainer:** a `protected-change` that accepts or rejects the proposed fixture, sets `tolerances.yaml` for `ortgenai-int4-cpu`, and sets the task-quality floor after an int4 GSM8K measurement.
2. Hosting the int4 files (and `DownloadAsync` for the variant), if wanted. Until then the profile is local-path only.
3. Step 3: CUDA int4 (needs paid GPU time, #210). Step 4: int4 for the next generator (#181).
4. Not run on int4: Native AOT, the `IChatClient` adapter, tool calling, prompts above 1,666 tokens, the release prompt sets (G1 to G3), soak (G4), performance against Python (G5). The manifest marks the corresponding capabilities untested.

## 8. Reproduce

```sh
python model-pipeline/reference/generation/fetch_snapshot.py --repo Qwen/Qwen3.5-0.8B \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --out <snapshot>
python model-pipeline/reference/generation/build_ortgenai_bundle.py --snapshot <snapshot> --out <bundle> --variant int4
python model-pipeline/reference/generation/run_genai_reference.py run --graph <bundle>/ortgenai-int4 \
  --fixture model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl \
  --prompts model-pipeline/reference/generation/prompts-v1.jsonl --out <ref> --threads 6
python model-pipeline/reference/generation/run_genai_reference.py annotate --results <ref> --snapshot <snapshot> \
  --fixture model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl \
  --prompts model-pipeline/reference/generation/prompts-v1.jsonl
python model-pipeline/reference/generation/make_generation_fixture.py --reference <ref> \
  --prompts model-pipeline/reference/generation/prompts-v1.jsonl --out <fixture dir>
dotnet run -c Release --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -- --bundle <bundle> --variant int4 \
  --device cpu --threads 6 --prompts model-pipeline/reference/generation/prompts-v1.jsonl --reference <ref> \
  --out <out> --modes parity,semantics
python model-pipeline/reference/generation/compare.py --reference <ref> --candidate <out>
STARLING_QWEN35_SNAPSHOT=<bundle> dotnet test --project ai-dotnet/tests/Starlings.Models.Qwen35.Tests  # Int4CpuModelTests
```
