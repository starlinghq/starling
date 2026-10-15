# Draft upstream report: microsoft/onnxruntime, NuGet and pip builds of 1.30.0 on Linux x64 (not filed)

**Status: draft, not filed, and not ready to file.** The maintainer decides whether to file it. It is not ready because the one direct test so far did not reproduce a slower NuGet build. That test put the pip `libonnxruntime.so` into the C# process, and the result is included below. Before filing, run the Python-only reproducer below on a quiet machine. File only if it shows a difference between the two builds that is larger than the run-to-run spread. If it does not, close this draft with that result.

Context: [README.md, Qwen CPU time to first token](README.md#qwen-cpu-time-to-first-token).

---

## Title

CPU prefill latency differs between the NuGet and pip builds of ONNX Runtime 1.30.0 (linux-x64), same model, same GenAI library

## Summary

ONNX Runtime 1.30.0 ships two different `libonnxruntime.so` binaries for linux-x64:

- `runtimes/linux-x64/native/libonnxruntime.so` in the NuGet package `Microsoft.ML.OnnxRuntime` 1.30.0;
- `onnxruntime/capi/libonnxruntime.so.1.30.0` in the pip wheel `onnxruntime` 1.30.0 (CPython 3.13, manylinux x86_64).

Both report version 1.30.0. Both were driven by the same `libonnxruntime-genai.so` (onnxruntime-genai 0.17.1; the NuGet and pip copies are byte-identical), on the same model bundle, threads and prompts.

On CPU, one prefill call of Qwen3.5-0.8B (fp32) took longer under the NuGet build in a matched benchmark. The median time to first token was 1.119 times Python's over five short prompts. The largest gap was on a 22-token prompt (1.156); prompts of 20, 40 and 61 tokens were 1.02 to 1.04. Overhead outside the native prefill call was measured as 2.2 to 2.4 ms against 2.1 ms, so it does not account for the gap.

Is it the build? We do not know yet; see "Status of the evidence". We are asking whether the two builds are expected to differ in CPU kernels or build flags (compiler, MLAS dispatch, OpenMP or thread-pool settings, `-O` level, LTO). We are also asking whether a CPU latency difference between them would be considered a bug.

## Binaries

| Build | File | SHA-256 |
| --- | --- | --- |
| NuGet `Microsoft.ML.OnnxRuntime` 1.30.0 | `runtimes/linux-x64/native/libonnxruntime.so` | `245a6f8c38127551057a1cd1ffd59f0a186a227ade4f3492dea2494eb565542e` |
| pip `onnxruntime` 1.30.0 | `onnxruntime/capi/libonnxruntime.so.1.30.0` | `c902c70b3003c0e99fada202f37478c515ae9bba7944c2b2abd0017bae0c82ed` |
| `onnxruntime-genai` 0.17.1 (NuGet `Microsoft.ML.OnnxRuntimeGenAI` and pip, identical) | `libonnxruntime-genai.so` | `0cdf8076ecf75eebe2bd1507c7904651e3d27d6f0b853ed361772ab25c1f2d40` |

## Environment

- CPU: Intel Core i7-5820K (Haswell-E, 6 cores, 12 threads, AVX2 and FMA, no AVX-512); 94 GB RAM.
- OS: Linux x86_64, kernel 7.2.5 (Arch-based distribution); CPU governor recorded per run.
- Session: CPU execution provider, `intra_op_num_threads=6` set through a GenAI config overlay; other session options at the GenAI defaults.
- Model: Qwen3.5-0.8B, the onnxruntime-genai fp32 CPU bundle (`ortgenai-fp32x`), greedy, the prompt appended in one call.
- Python 3.13.15, `onnxruntime==1.30.0`, `onnxruntime-genai==0.17.1`. .NET 10.0.12 with the NuGet packages of the same versions.

## Measurements (matched benchmark, 2026-10-15)

Median time to first token, C# (NuGet build) over Python (pip build), 3 warm runs per side, 2 samples per prompt per run:

| Prompt tokens | NuGet (ms) | pip (ms) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 143.67 | 137.83 | 1.042 |
| 22 | 167.72 | 145.03 | 1.156 |
| 29 | 184.16 | 166.74 | 1.104 |
| 40 | 223.97 | 220.17 | 1.017 |
| 61 | 318.79 | 313.21 | 1.018 |
| 286 | | | 0.982 |
| 1,666 | | | 1.011 |

## Status of the evidence

A follow-up loaded the pip `libonnxruntime.so` into the C# process in place of the NuGet one, to test the build alone. It ran C# with the NuGet build, C# with the pip build, and Python, interleaved, for two rounds of a one-token probe on the same five prompts. The pip build did **not** make C# faster:

- Pooled medians: 207.5 ms (NuGet build), 229.2 ms (pip build), 205.3 ms (Python).
- At 22 tokens: 171.2, 204.4 and 151.0 ms.
- Per-length minima were nearly equal in all three (22 tokens: 138.8, 138.1 and 137.9 ms).

The swapped library's digest was not recorded, and its build directory was not kept, so the swap is as reported, not independently verified. That machine was loaded (load average 6 to 10). A later run of the same C# binary had a pooled median of 296.7 ms, so the spread between runs was larger than the effect. Neither result settles the question; a quiet-machine run of the reproducer below is needed.

## Minimal reproducer (Python only; the builds swapped by file)

Two virtual environments that differ only in `onnxruntime/capi/libonnxruntime.so.1.30.0`. onnxruntime-genai's Python package loads that file by path (`_dll_directory.add_onnxruntime_dependency` sets `ORT_LIB_PATH` to it). Copying the NuGet binary over it therefore swaps the build and changes nothing else.

```sh
uv venv pip-build --python 3.13 && uv pip install --python pip-build/bin/python onnxruntime==1.30.0 onnxruntime-genai==0.17.1
cp -a pip-build nuget-build   # or create it the same way
nuget=$(mktemp -d) && curl -sL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/1.30.0 -o "$nuget/ort.zip" \
  && unzip -q "$nuget/ort.zip" runtimes/linux-x64/native/libonnxruntime.so -d "$nuget"
capi=$(nuget-build/bin/python -c 'import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))')/capi
cp "$nuget/runtimes/linux-x64/native/libonnxruntime.so" "$capi/libonnxruntime.so.1.30.0"
sha256sum pip-build/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0 "$capi/libonnxruntime.so.1.30.0"
# interleave the two builds, several rounds, on a quiet machine
for round in 1 2 3 4 5; do
  pip-build/bin/python prefill.py <model-dir> && nuget-build/bin/python prefill.py <model-dir>
done
```

`prefill.py`:

```python
"""Median time of one prefill plus the first token, per prompt length, with the libonnxruntime.so in use."""
import hashlib, json, statistics, sys, time
import onnxruntime_genai as og

model_dir = sys.argv[1]
config = og.Config(model_dir)
config.clear_providers()  # CPU
# The benchmark's overlays, in its order: 6 intra-op threads, then the pad token id of the Qwen3.5 bundle.
config.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": 6}}}}))
config.overlay(json.dumps({"model": {"pad_token_id": 248319}}))
model = og.Model(config)
lib = next(l.split()[-1] for l in open("/proc/self/maps") if "libonnxruntime.so" in l)
print("libonnxruntime:", lib, hashlib.sha256(open(lib, "rb").read()).hexdigest()[:12])

for n in (20, 22, 29, 40, 61, 286):
    prompt = [9707 + (i % 50) for i in range(n)]  # fixed token ids; any valid ids of this length
    times = []
    for i in range(32):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=n + 1, do_sample=False)
        generator = og.Generator(model, params)
        t = time.perf_counter()
        generator.append_tokens(prompt)  # the whole prompt in one call (prefill)
        generator.generate_next_token()
        _ = generator.get_next_tokens()
        if i >= 2:  # two warm-up calls per length
            times.append((time.perf_counter() - t) * 1000)
    print(f"{n:5d} tokens: median {statistics.median(times):8.2f} ms  min {min(times):8.2f} ms")
```

Report the per-length medians and minima of both builds over all rounds, with the load average. A build difference shows as one build slower in every round, by more than the spread between rounds of the same build.
