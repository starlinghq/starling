# Draft: upstream report for microsoft/onnxruntime-genai (not filed)

Status: **draft for the maintainer (M9)**. Not filed. The text below the line is the proposed issue body. Evidence: section 5 of `README.md` in this directory (`rawstress-cuda-fp32x.json`, `rawstress-cuda-fp16x.json`, `rawstress-cpu-fp32x.json`, `stress-*.json`).

Before filing, the maintainer should:

1. Run the reproducer below once more, as written (a single file, no Starling code), on a GPU that no other process is using, and paste the actual output. Our runs used the same API calls inside our fidelity runner (`--modes rawstress`) with the prompt ids below, and compared against a transformers reference; the standalone file compares concurrent runs with sequential runs of the same API instead, so no reference is needed. The standalone file compiles (`dotnet build repro.cs`, .NET SDK 10.0.100, against the cached 0.17.1 package). It has not been run yet, because this draft was written without GPU access.
2. Decide whether to publish the model build. The model is a local build (recipe below), and the hosting repository is private. Either link a public copy, or keep the build recipe as the only way to get the model.
3. Search open issues for "concurrent" / "multiple generators" / "thread" on 0.17.x first, and link a matching issue instead of filing a duplicate.

---

### Title

Wrong greedy tokens when several `Generator`s on one `Model` run concurrently on the CUDA EP (Qwen3.5 hybrid model, 0.17.1)

### Describe the bug

Several `Generator` instances created from one `Model` and run on different threads at the same time (one generator per thread, never shared) produce different greedy tokens on the CUDA execution provider. The same requests run one at a time give the correct tokens every time, and the same concurrent runs on the CPU execution provider are always correct.

The differences are not near-ties. Some requests diverge at the first generated token (step 0, right after `AppendTokens`), and others diverge at steps where the top-1 minus top-2 logit margin of the correct token is up to 6.2. So these are wrong results, not floating-point noise.

Results: 8 rounds of 6 requests (32 greedy tokens each), compared with the expected tokens.

| Configuration | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| CUDA, fp32 graph (`-p fp32 -e cpu` build, `use_tf32=0`), plain C# API | 0/48 wrong | 10/48 wrong | |
| CUDA, fp16 graph (`-p fp16 -e cuda` build), plain C# API | 0/48 | 11/48 | |
| CUDA, fp32 graph, our decode loop over the same API | 0/48 | 14/48 (15/48 in a second run) | 19/48 |
| CUDA, either graph, all native calls of all generators under one lock | 0/48 | 0/48 | 0/48 |
| CPU, fp32 graph, plain C# API | 0/48 | 0/48 | |

Holding one process-wide lock around every native call (generator creation, `AppendTokens`, `GenerateNextToken`, `GetNextTokens`, `GetLogits`, `Dispose`) removes the problem. So the trigger is native calls from different generators overlapping in time. We did not investigate whether the cause is in GenAI (for example, state shared between generators of one model, such as buffers or the CUDA stream) or in the ONNX Runtime CUDA EP.

### To reproduce

1. Build the model with the onnxruntime-genai 0.17.1 builder from `Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17`. First make a copy whose `config.json` has `torch_dtype`/`dtype` set to `float32`, with the weights unchanged, and then run:

   ```sh
   python -m onnxruntime_genai.models.builder -m <float32-config copy> -o fp32x -p fp32 -e cpu -c <cache> --extra_options exclude_mtp=true
   ```

   The output's SHA-256 values: `genai_config.json` `e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba`, `model.onnx` `4a052c4a1444c9437e00ffd48baecb0838d3d9ca1703c3153dbd46e945e84703`, `model.onnx.data` `129c58a320b6de7192713cc9c7e4ba744dcf7fabb0f57c6535021b45b3dc5143`. The fp16 variant, `-p fp16 -e cuda`, shows the same behavior.

2. Save the following as `repro.cs` and run it with `CUDA_VISIBLE_DEVICES=<gpu> dotnet run repro.cs -- <model dir> [threads]` (.NET 10 SDK, file-based app):

   ```csharp
   #:package Microsoft.ML.OnnxRuntimeGenAI.Cuda@0.17.1
   using System.Collections.Concurrent;
   using Microsoft.ML.OnnxRuntimeGenAI;

   // Prompt ids: Qwen3.5 chat-template renderings of six short user messages (tokenized with the model's tokenizer).
   int[][] prompts =
   [
       [248045, 846, 198, 139054, 96177, 98933, 113552, 109757, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
       [248045, 846, 198, 7734, 264, 12654, 709, 421, 4523, 279, 307, 7324, 76938, 1324, 5199, 7638, 13, 8020, 1970, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
       [248045, 846, 198, 32, 5257, 33060, 220, 16, 17, 15, 12754, 303, 220, 16, 13, 20, 3979, 13, 3437, 369, 1141, 5289, 4478, 303, 12754, 7294, 30, 6712, 279, 21261, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
       [248045, 846, 198, 161607, 104647, 153178, 96706, 45220, 247206, 172808, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
       [248045, 846, 198, 826, 2250, 14542, 10104, 364, 4237, 2708, 10597, 9417, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
       [248045, 846, 198, 213193, 151773, 155775, 149651, 149826, 159124, 201047, 149136, 148693, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
   ];
   const int Tokens = 32, Rounds = 8;
   int threads = args.Length > 1 ? int.Parse(args[1]) : 2;

   using var config = new Config(args[0]);
   config.ClearProviders();
   config.AppendProvider("cuda");
   config.SetProviderOption("cuda", "use_tf32", "0");
   config.Overlay("{\"model\":{\"pad_token_id\":248319}}"); // as in our runs; the default pad id masks <|endoftext|> in prompts
   using var model = new Model(config);

   List<int> Run(int[] prompt)
   {
       using var p = new GeneratorParams(model);
       p.SetSearchOption("max_length", prompt.Length + Tokens);
       p.SetSearchOption("do_sample", false);
       using var g = new Generator(model, p);
       g.AppendTokens(prompt);
       var o = new List<int>();
       while (!g.IsDone() && o.Count < Tokens)
       {
           g.GenerateNextToken();
           o.Add(g.GetNextTokens()[0]);
       }
       return o;
   }

   var expected = prompts.Select(Run).ToArray(); // sequential: the correct tokens
   int wrong = 0;
   for (int round = 0; round < Rounds; round++)
   {
       var outs = new ConcurrentDictionary<int, List<int>>();
       Parallel.For(0, prompts.Length, new ParallelOptions { MaxDegreeOfParallelism = threads }, i => outs[i] = Run(prompts[i]));
       for (int i = 0; i < prompts.Length; i++)
       {
           int n = 0;
           while (n < expected[i].Count && n < outs[i].Count && expected[i][n] == outs[i][n]) n++;
           if (n < Math.Max(expected[i].Count, outs[i].Count))
           {
               wrong++;
               Console.WriteLine($"round {round} prompt {i}: first difference at step {n}");
           }
       }
   }
   Console.WriteLine($"{threads} threads: {wrong} of {Rounds * prompts.Length} requests differ from the sequential result");
   ```

   Our runs of these calls on these prompt ids gave 0 wrong requests with 1 thread, and 10 of 48 (fp32 graph) and 11 of 48 (fp16 graph) with 2 threads. Those runs compared against a transformers reference, which equals the sequential result here.

### Expected behavior

Each generator's greedy output does not depend on what other generators of the same model are doing at the same time. That is the case on the CPU EP. If concurrent generators of one model are not supported on CUDA, the documentation should say so, and ideally the API should reject or serialize them.

### Versions

- `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1 (C# API) with the ONNX Runtime 1.30.0 it ships with. The builder was onnxruntime-genai 0.17.1 (`rel-0.17.1` head `83de55a1f3886cddbb32a5517e26adb3f9d65159`).
- .NET SDK 10.0.100, runtime 10.0.12, Linux x86-64.
- NVIDIA RTX 4060 Ti 16 GB, driver 610.57.04, CUDA 13.3. The GPU was shared with another process, which held about 10.7 GB and was often at 100% utilization.
- Model type `qwen3_5_text`: hybrid linear-attention (Gated DeltaNet) and full-attention layers, with `past_conv` / `past_recurrent` state in addition to the KV cache, and `past_present_share_buffer: true`.

### Workaround

Serialize all native calls of all generators of one model behind one lock on CUDA. Requests can still interleave step by step, but CUDA throughput then does not scale with concurrent requests.
