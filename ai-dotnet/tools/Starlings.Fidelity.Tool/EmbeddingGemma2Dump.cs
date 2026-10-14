// EmbeddingGemma 2 stage results for `model-pipeline evaluate` (task "embedding", stage results format version 1).
// Runs the C# path on every row of the protected embeddings.jsonl (46 texts x no prompt, query, document; prompt rows
// are cases "<id>@<prompt>", as the evaluator keys them) and writes:
//
//   tokenizer.results.json   stage "tokenizer": EmbeddingGemma2Tokenizer ids and masks of prompt_text + text.
//   dotnet.results.json      stage "dotnet": embeddings from the fixture ids (EmbeddingGemma2Encoder) and from text
//                            (EmbeddingGemma2TextEmbedder with the row's prompt); the protected padded batch, every
//                            row in padded batches of 8 (ids), and one padded text batch per prompt.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- embeddinggemma2 --out <dir>
//       [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>] [--device cpu|cuda[:N]]
//
// --device cuda needs the CUDA build (ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda) and defaults the profile to
// embeddinggemma-2/onnx-fp32-cuda (same fixtures). Applies no bounds; writes nothing under protected/.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

internal static class EmbeddingGemma2Dump
{
    public static int Run(string[] args)
    {
        string? outDir = null;
        string? snapshot = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        string? fixturesDir = null;
        string? profile = null;
        int? threads = null;
        var device = ToolDevice.Cpu;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--out" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
                case "--snapshot" when i + 1 < args.Length:
                    snapshot = args[++i];
                    break;
                case "--fixtures" when i + 1 < args.Length:
                    fixturesDir = args[++i];
                    break;
                case "--profile" when i + 1 < args.Length:
                    profile = args[++i];
                    break;
                case "--threads" when i + 1 < args.Length && int.TryParse(args[i + 1], out int n) && n > 0:
                    threads = n;
                    i++;
                    break;
                case "--device" when i + 1 < args.Length && ToolDevice.TryParse(args[i + 1], out var parsed):
                    device = parsed;
                    i++;
                    break;
                default:
                    Console.Error.WriteLine($"error: unknown option {args[i]}");
                    return 2;
            }
        }

        if (outDir is null)
        {
            Console.Error.WriteLine("error: --out is required");
            return 2;
        }

        if (device.Unavailable() is { } unavailable)
        {
            Console.Error.WriteLine($"error: {unavailable}");
            return 2;
        }

        profile ??= device.Profile("embeddinggemma-2/onnx-fp32-cpu");

        string? repoRoot = ToolSupport.FindRepositoryRoot();
        fixturesDir ??= repoRoot is null ? null : Path.Combine(repoRoot, "model-pipeline", "protected", "embeddinggemma-2", "onnx-fp32-cpu");
        if (fixturesDir is null || !File.Exists(Path.Combine(fixturesDir, "embeddings.jsonl")))
        {
            Console.Error.WriteLine($"error: protected fixtures not found ({fixturesDir ?? "no repository root"}); pass --fixtures");
            return 2;
        }

        EmbeddingGemma2ModelFiles files;
        try
        {
            files = string.IsNullOrEmpty(snapshot) ? EmbeddingGemma2ModelAssets.OpenLocal() : EmbeddingGemma2ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
        }
        catch (ModelAcquisitionException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        Directory.CreateDirectory(outDir);
        string? commit = ToolSupport.GitHead(repoRoot);
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath);
        using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
        {
            Session = device.Apply(new OnnxSessionOptions { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256, IntraOpNumThreads = threads }),
        });
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var d = encoder.Session.Diagnostics;
        var tokenizerAsset = new Asset(EmbeddingGemma2ModelAssets.TokenizerFile, EmbeddingGemma2ModelAssets.TokenizerSha256);
        var modelAsset = new Asset(EmbeddingGemma2ModelAssets.ModelFile, d.ModelSha256 ?? ToolSupport.Sha256(files.ModelPath));
        var rows = ReadRows(Path.Combine(fixturesDir, "embeddings.jsonl"));
        var environment = StageEnvironment(d, device);

        // Stage 1: the owned tokenizer on prompt_text + text.
        var tokenizerStage = new StageWriter(profile, "tokenizer", "Starlings.Tokenizers via EmbeddingGemma2Tokenizer (profile gemma-bpe-embeddinggemma2, no truncation), on prompt text + text", commit, environment, [tokenizerAsset]);
        foreach (var r in rows)
        {
            var input = tokenizer.Tokenize(r.Prompt.Apply(r.Text));
            tokenizerStage.Case(r.Key).Add("input_ids", input.InputIds.ToArray()).Add("attention_mask", input.AttentionMask.ToArray());
        }

        tokenizerStage.Write(Path.Combine(outDir, "tokenizer.results.json"));

        // Stage 3: from ids and from text.
        var dotnet = new StageWriter(profile, "dotnet", $"C# EmbeddingGemma2Encoder + EmbeddingGemma2TextEmbedder, ONNX Runtime {d.OrtVersion} {device.Label} (managed binding {d.ManagedBindingVersion})", commit, environment, [modelAsset, tokenizerAsset]);
        foreach (var r in rows)
        {
            dotnet.Case(r.Key)
                .Add("input_ids", r.InputIds)
                .Add("embedding", encoder.Embed(new EmbeddingGemma2Input(r.InputIds)))
                .Add("text_embedding", embedder.Embed(r.Text, r.Prompt));
        }

        var batchRows = ReadPaddedBatch(Path.Combine(fixturesDir, "tokenizer-batch.json"));
        dotnet.Batch("fixture-padded", "embedding", batchRows.Select(b => b.Id).ToList(), encoder.EmbedBatch(batchRows.Select(b => new EmbeddingGemma2Input(b.Ids, b.Mask)).ToList()));
        var batched = new List<float[]>();
        for (int start = 0; start < rows.Count; start += 8)
        {
            batched.AddRange(encoder.EmbedBatch(rows.Skip(start).Take(8).Select(r => new EmbeddingGemma2Input(r.InputIds)).ToList()));
        }

        dotnet.Batch("all-cases-batch8", "embedding", rows.Select(r => r.Key).ToList(), batched);
        foreach (var group in rows.GroupBy(r => r.Prompt.Name))
        {
            var list = group.ToList();
            dotnet.Batch($"text-batch-{group.Key}", "text_embedding", list.Select(r => r.Key).ToList(), embedder.EmbedBatch(list.Select(r => r.Text).ToList(), list[0].Prompt));
        }

        dotnet.Write(Path.Combine(outDir, "dotnet.results.json"));
        Console.WriteLine($"wrote {Path.Combine(outDir, "tokenizer.results.json")} and dotnet.results.json: {rows.Count} cases");
        return 0;
    }

    private sealed record Row(string Key, string Text, EmbeddingGemma2Prompt Prompt, long[] InputIds);

    private static List<Row> ReadRows(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l =>
            {
                using var doc = JsonDocument.Parse(l);
                var r = doc.RootElement;
                string id = r.GetProperty("id").GetString()!;
                string prompt = r.GetProperty("prompt").GetString()!;
                return new Row(
                    prompt == "none" ? id : $"{id}@{prompt}",
                    r.GetProperty("text").GetString()!,
                    EmbeddingGemma2Prompt.FromName(prompt),
                    r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray());
            })
            .ToList();

    private static List<(string Id, long[] Ids, long[] Mask)> ReadPaddedBatch(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("rows").EnumerateArray()
            .Select(r => (
                r.GetProperty("id").GetString()!,
                r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                r.GetProperty("attention_mask").EnumerateArray().Select(e => e.GetInt64()).ToArray()))
            .ToList();
    }

    private static Dictionary<string, object?> StageEnvironment(OnnxSessionDiagnostics d, ToolDevice device)
    {
        var environment = StageEnvironment(d, device.Name);
        device.Describe(environment);
        return environment;
    }

    private static Dictionary<string, object?> StageEnvironment(OnnxSessionDiagnostics d, string provider) => new()
    {
        ["machine"] = Environment.MachineName,
        ["provider"] = provider,
        ["provider_options"] = d.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
        ["dotnet"] = RuntimeInformation.FrameworkDescription,
        ["os"] = RuntimeInformation.OSDescription,
        ["process_architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
        ["cpu_count"] = Environment.ProcessorCount,
        ["cpu"] = ToolSupport.CpuModel(),
        ["onnxruntime"] = d.OrtVersion,
        ["onnxruntime_managed"] = d.ManagedBindingVersion,
        ["session_providers"] = d.SessionProviders,
        ["intra_op_threads"] = d.IntraOpNumThreads,
        ["graph_optimization_level"] = d.GraphOptimizationLevel,
    };
}
