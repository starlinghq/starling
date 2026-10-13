// EmbeddingGemma 2 image stage results for `model-pipeline evaluate` (task "image-embedding", stage results format
// version 2). Runs the C# image path on every in-contract row of the protected image-embeddings.jsonl (rows with a
// 16-bit image are rejected by contract and skipped) and writes:
//
//   tokenizer.results.json   stage "tokenizer": rendered ids (EmbeddingGemma2MultimodalEmbedder.Tokenize with the owned
//                            tokenizer) and, per image, the SHA-256 of the SkiaSharp-decoded RGB, the resized bytes, the
//                            pixel values and the positions (EmbeddingGemma2ImageProcessor).
//   dotnet.results.json      stage "dotnet": the embedding of every row alone; soft tokens per image and token
//                            embeddings per image row (sidecars soft_tokens.f32, token_embeddings.f32); the mixed rows in
//                            vision batches of 8 and padded to 2,520 patches.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- embeddinggemma2-images --out <dir>
//       --images <dir> [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]
//
// CPU only (the tool references the CPU native runtime). Applies no bounds; writes nothing under protected/.
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Runtime.Onnx;

internal static class EmbeddingGemma2ImageDump
{
    public static int Run(string[] args)
    {
        string? outDir = null;
        string? snapshot = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        string? imagesDir = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_IMAGES");
        string? fixturesDir = null;
        string profile = "embeddinggemma-2/image-fp32-cpu";
        int? threads = null;
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
                case "--images" when i + 1 < args.Length:
                    imagesDir = args[++i];
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
                default:
                    Console.Error.WriteLine($"error: unknown option {args[i]}");
                    return 2;
            }
        }

        if (outDir is null || imagesDir is null)
        {
            Console.Error.WriteLine("error: --out and --images (the make_eg2_images.py output) are required");
            return 2;
        }

        string? repoRoot = ToolSupport.FindRepositoryRoot();
        fixturesDir ??= repoRoot is null ? null : Path.Combine(repoRoot, "model-pipeline", "protected", "embeddinggemma-2", "image-fp32-cpu");
        if (fixturesDir is null || !File.Exists(Path.Combine(fixturesDir, "image-embeddings.jsonl")))
        {
            Console.Error.WriteLine($"error: protected image fixtures not found ({fixturesDir ?? "no repository root"}); pass --fixtures");
            return 2;
        }

        EmbeddingGemma2ImageModelFiles files;
        try
        {
            files = string.IsNullOrEmpty(snapshot) ? EmbeddingGemma2ModelAssets.OpenLocalImageAssets() : EmbeddingGemma2ModelAssets.OpenImageDirectory(snapshot);
        }
        catch (ModelAcquisitionException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        Directory.CreateDirectory(outDir);
        string? commit = ToolSupport.GitHead(repoRoot);
        var session = new OnnxSessionOptions { IntraOpNumThreads = threads };
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = session });
        var manifest = ReadManifest(Path.Combine(fixturesDir, "images.json"));
        var rows = ReadRows(Path.Combine(fixturesDir, "image-embeddings.jsonl"))
            .Where(r => r.Images.All(i => manifest[i].Mode != "I;16"))
            .ToList();
        var decoded = new Dictionary<string, EmbeddingGemma2Image>(StringComparer.Ordinal);
        EmbeddingGemma2Image Decode(string id) =>
            decoded.TryGetValue(id, out var image) ? image : decoded[id] = EmbeddingGemma2ImageDecoder.DecodeFile(Path.Combine(imagesDir, manifest[id].File));

        var vd = embedder.VisionSession.Diagnostics;
        var environment = new Dictionary<string, object?>
        {
            ["machine"] = Environment.MachineName,
            ["provider"] = "cpu",
            ["provider_options"] = vd.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
            ["dotnet"] = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            ["cpu_count"] = Environment.ProcessorCount,
            ["cpu"] = ToolSupport.CpuModel(),
            ["onnxruntime"] = vd.OrtVersion,
            ["onnxruntime_managed"] = vd.ManagedBindingVersion,
            ["session_providers"] = vd.SessionProviders,
            ["intra_op_threads"] = vd.IntraOpNumThreads,
            ["image_decoder"] = "Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp (SkiaSharp 4.153.1)",
        };
        var tokenizerAsset = new Asset(EmbeddingGemma2ModelAssets.TokenizerFile, EmbeddingGemma2ModelAssets.TokenizerSha256);
        Asset[] graphAssets =
        [
            new(EmbeddingGemma2ModelAssets.VisionModelFile, vd.ModelSha256 ?? ToolSupport.Sha256(files.VisionModelPath)),
            new(EmbeddingGemma2ModelAssets.MultimodalModelFile, embedder.MultimodalSession.Diagnostics.ModelSha256 ?? ToolSupport.Sha256(files.MultimodalModelPath)),
            new(EmbeddingGemma2ModelAssets.ModelFile, ToolSupport.Sha256(files.Text.ModelPath)),
        ];

        // Stage 1: rendering and preprocessing.
        var stage1 = new StageWriter(profile, "tokenizer", "EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer; EmbeddingGemma2ImageProcessor on SkiaSharp-decoded RGB", commit, environment, [tokenizerAsset], task: "image-embedding");
        var inputs = new Dictionary<string, EmbeddingGemma2MultimodalInput>(StringComparer.Ordinal);
        foreach (var r in rows)
        {
            var input = BuildInput(r, Decode);
            inputs[r.Id] = input;
            var c = stage1.Case(r.Id).Add("input_ids", embedder.Tokenize(input, r.Prompt).InputIds.ToArray());
            for (int k = 0; k < r.Images.Count; k++)
            {
                var image = Decode(r.Images[k]);
                var p = EmbeddingGemma2ImageProcessor.Preprocess(image);
                c.Add($"image{k}_rgb_sha256", Sha(image.Pixels.Span))
                    .Add($"image{k}_resized_sha256", Sha(p.ResizedPixels.Span))
                    .Add($"image{k}_pixel_values_sha256", Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(p.PixelValues.Span)))
                    .Add($"image{k}_positions_sha256", Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(p.Positions.Span)));
            }
        }

        stage1.Write(Path.Combine(outDir, "tokenizer.results.json"));

        // Stage 3: every row alone, with its soft tokens and token embeddings.
        var stage3 = new StageWriter(profile, "dotnet", $"C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime {vd.OrtVersion} CPU (managed binding {vd.ManagedBindingVersion})", commit, environment, [.. graphAssets, tokenizerAsset], task: "image-embedding");
        var collector = new Collector();
        var singles = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var r in rows)
        {
            collector.Row = r.Id;
            var destination = new float[embedder.Dimension];
            embedder.EmbedCore([inputs[r.Id]], r.Prompt, destination, default, collector, CancellationToken.None);
            singles[r.Id] = destination;
            stage3.Case(r.Id).Add("embedding", destination);
            Console.WriteLine($"{r.Id}: embedded");
        }

        stage3.Tensor("soft_tokens", outDir, "soft_tokens.f32", collector.Soft.Select(s => (s.Id, (new[] { s.Values.Length / 512, 512 }, s.Values))));
        stage3.Tensor("token_embeddings", outDir, "token_embeddings.f32", collector.Tokens.Select(t => (t.Id, (new[] { t.Values.Length / 768, 768 }, t.Values))));

        string[] mixed = ["q-cat", "photo-chelsea", "aspect-wide-2000x40", "size-tiny-8x6", "mm-two-images", "q-rocket", "mm-caption-placeholder", "aspect-tall-30x1500"];
        var mixedRows = rows.Where(r => mixed.Contains(r.Id) && r.Prompt == EmbeddingGemma2Prompt.None).ToList();
        foreach (var (name, options) in new[]
        {
            ("vision-batch-8", new EmbeddingGemma2MultimodalOptions { Session = session, MaxVisionBatchSize = 8, MaxVisionBatchPatches = 8 * 2_520, MaxBatchSize = 4, VerifyModelSha256 = false }),
            ("padded-2520-vision-batch-3", new EmbeddingGemma2MultimodalOptions { Session = session, PadImagesToMaxPatches = true, MaxVisionBatchSize = 3, MaxVisionBatchPatches = 3 * 2_520, VerifyModelSha256 = false }),
        })
        {
            using var batchEmbedder = EmbeddingGemma2MultimodalEmbedder.Create(files, options);
            stage3.Batch(name, "embedding", mixedRows.Select(r => r.Id).ToList(), batchEmbedder.EmbedBatch(mixedRows.Select(r => inputs[r.Id]).ToList()));
        }

        stage3.Write(Path.Combine(outDir, "dotnet.results.json"));
        Console.WriteLine($"wrote {Path.Combine(outDir, "tokenizer.results.json")} and dotnet.results.json: {rows.Count} rows");
        return 0;
    }

    private sealed record ImageEntry(string File, string Mode);

    private sealed record Row(string Id, EmbeddingGemma2Prompt Prompt, string? Text, List<string> Images);

    private sealed class Collector : IEmbeddingGemma2Diagnostics
    {
        public string Row { get; set; } = string.Empty;

        public List<(string Id, float[] Values)> Soft { get; } = [];

        public List<(string Id, float[] Values)> Tokens { get; } = [];

        public void OnSoftTokens(int input, int image, float[] softTokens) => Soft.Add(($"{Row}/image{image}", softTokens));

        public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) => Tokens.Add((Row, tokenEmbeddings.ToArray()));
    }

    /// <summary>Upstream's rendering: placeholders in the text mark the image positions; without one, images follow the text.</summary>
    private static EmbeddingGemma2MultimodalInput BuildInput(Row r, Func<string, EmbeddingGemma2Image> decode)
    {
        var images = r.Images.Select(decode).ToList();
        if (r.Text is null)
        {
            return images.Count == 0 ? EmbeddingGemma2MultimodalInput.FromText(string.Empty) : EmbeddingGemma2MultimodalInput.Create(images.Cast<object>());
        }

        if (images.Count == 0)
        {
            return EmbeddingGemma2MultimodalInput.FromText(r.Text);
        }

        var pieces = r.Text.Split("<|image|>");
        var parts = new List<object>();
        if (pieces.Length == 1)
        {
            parts.Add(r.Text);
            parts.AddRange(images);
        }
        else
        {
            for (int i = 0; i < pieces.Length; i++)
            {
                parts.Add(pieces[i]);
                if (i < images.Count)
                {
                    parts.Add(images[i]);
                }
            }
        }

        return EmbeddingGemma2MultimodalInput.Create(parts);
    }

    private static Dictionary<string, ImageEntry> ReadManifest(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("images").EnumerateArray()
            .ToDictionary(i => i.GetProperty("id").GetString()!, i => new ImageEntry(i.GetProperty("file").GetString()!, i.GetProperty("mode").GetString()!), StringComparer.Ordinal);
    }

    private static List<Row> ReadRows(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l =>
            {
                using var doc = JsonDocument.Parse(l);
                var r = doc.RootElement;
                return new Row(
                    r.GetProperty("id").GetString()!,
                    EmbeddingGemma2Prompt.FromName(r.GetProperty("prompt").GetString()!),
                    r.GetProperty("text").ValueKind == JsonValueKind.Null ? null : r.GetProperty("text").GetString(),
                    r.GetProperty("images").EnumerateArray().Select(i => i.GetProperty("image").GetString()!).ToList());
            })
            .ToList();

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes));
}
