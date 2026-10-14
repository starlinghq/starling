// EmbeddingGemma 2 long inputs (2,049 to 8,192 tokens; W3 proposal): the C# package on the PROPOSED long-input rows
// (model-pipeline/proposed/embeddinggemma-2/long-input/long-embeddings.jsonl, made by
// model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_long.py). These rows are not protected fixtures and no
// evaluator profile reads them, so this writes its own outputs file, `long.results.json`, compared with the reference by
// model-pipeline/reference/embeddinggemma2/compare_embeddinggemma2_long_csharp.py. Per row: the rendered ids
// (EmbeddingGemma2MultimodalEmbedder.Tokenize with the owned tokenizer), and either the embedding or the typed rejection
// (EmbeddingGemma2InputTooLongException, for rows above 8,192 tokens). Two padded batches of two rows each (text 2,600 +
// 8,192 tokens; text with six images + text with a 32-frame video) for batch-vs-single. Text-only rows run on the
// released text graph, media rows on the vision and multimodal text graphs. Applies no bounds.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- embeddinggemma2-long --out <dir>
//       --rows <long-embeddings.jsonl> --images <dir> --videos <dir> [--ffmpeg <FFmpeg 8 lib dir>] [--snapshot <dir>]
//       [--threads <n>] [--device cpu|cuda[:N]]
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
using Starlings.Runtime.Onnx;

internal static class EmbeddingGemma2LongDump
{
    private const string ImagePlaceholder = "<|image|>";
    private const string VideoPlaceholder = "<|video|>";

    public static int Run(string[] args)
    {
        string? outDir = null;
        string? rowsPath = null;
        string? snapshot = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        string? imagesDir = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_IMAGES");
        string? videosDir = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_VIDEOS");
        string? ffmpegDir = null;
        int? threads = null;
        var device = ToolDevice.Cpu;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--out" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
                case "--rows" when i + 1 < args.Length:
                    rowsPath = args[++i];
                    break;
                case "--snapshot" when i + 1 < args.Length:
                    snapshot = args[++i];
                    break;
                case "--images" when i + 1 < args.Length:
                    imagesDir = args[++i];
                    break;
                case "--videos" when i + 1 < args.Length:
                    videosDir = args[++i];
                    break;
                case "--ffmpeg" when i + 1 < args.Length:
                    ffmpegDir = args[++i];
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

        if (outDir is null || rowsPath is null || imagesDir is null || videosDir is null)
        {
            Console.Error.WriteLine("error: --out, --rows (long-embeddings.jsonl), --images and --videos are required");
            return 2;
        }

        if (device.Unavailable() is { } unavailable)
        {
            Console.Error.WriteLine($"error: {unavailable}");
            return 2;
        }

        EmbeddingGemma2ImageModelFiles files;
        EmbeddingGemma2VideoDecoder decoder;
        try
        {
            files = string.IsNullOrEmpty(snapshot) ? EmbeddingGemma2ModelAssets.OpenLocalImageAssets() : EmbeddingGemma2ModelAssets.OpenImageDirectory(snapshot);
            decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = ffmpegDir });
        }
        catch (Exception ex) when (ex is ModelAcquisitionException or EmbeddingGemma2FFmpegException)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        Directory.CreateDirectory(outDir);
        string? repoRoot = ToolSupport.FindRepositoryRoot();
        var session = device.Apply(new OnnxSessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = threads is null ? null : 1 });
        var options = device.IsCuda
            ? new EmbeddingGemma2MultimodalOptions { Session = session }
            : new EmbeddingGemma2MultimodalOptions { Session = session, MaxVideoFramesPerVisionCall = 8, MaxVisionBatchPatches = 8 * 1_260 };
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, options);
        var rows = File.ReadAllLines(rowsPath).Where(l => l.Length > 0).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
        var images = new Dictionary<string, EmbeddingGemma2Image>(StringComparer.Ordinal);
        var videos = new Dictionary<string, EmbeddingGemma2Video>(StringComparer.Ordinal);

        EmbeddingGemma2MultimodalInput BuildInput(JsonObject row)
        {
            string text = row["text"]!.GetValue<string>();
            var media = row["media"]!.AsArray();
            if (media.Count == 0)
            {
                return EmbeddingGemma2MultimodalInput.FromText(text);
            }

            // Media in placeholder order, as upstream fills the placeholders.
            var imageQueue = new Queue<object>();
            var videoQueue = new Queue<object>();
            foreach (var m in media)
            {
                string file = m!["file"]!.GetValue<string>();
                if (m["kind"]!.GetValue<string>() == "image")
                {
                    imageQueue.Enqueue(images.TryGetValue(file, out var image) ? image : images[file] = EmbeddingGemma2ImageDecoder.DecodeFile(Path.Combine(imagesDir, file)));
                }
                else
                {
                    videoQueue.Enqueue(videos.TryGetValue(file, out var video) ? video : videos[file] = decoder.DecodeFile(Path.Combine(videosDir, file)).Video);
                }
            }

            var parts = new List<object>();
            int start = 0;
            while (true)
            {
                int iv = text.IndexOf(VideoPlaceholder, start, StringComparison.Ordinal);
                int ii = text.IndexOf(ImagePlaceholder, start, StringComparison.Ordinal);
                int next = iv < 0 ? ii : ii < 0 ? iv : Math.Min(iv, ii);
                if (next < 0)
                {
                    parts.Add(text[start..]);
                    break;
                }

                parts.Add(text[start..next]);
                parts.Add(next == iv ? videoQueue.Dequeue() : imageQueue.Dequeue());
                start = next + (next == iv ? VideoPlaceholder.Length : ImagePlaceholder.Length);
            }

            if (imageQueue.Count + videoQueue.Count != 0)
            {
                throw new InvalidOperationException($"{row["id"]}: media without placeholders");
            }

            return EmbeddingGemma2MultimodalInput.Create(parts);
        }

        var vd = embedder.VisionSession.Diagnostics;
        var environment = new Dictionary<string, object?>
        {
            ["machine"] = Environment.MachineName,
            ["provider"] = device.Name,
            ["provider_options"] = vd.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
            ["dotnet"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["cpu"] = ToolSupport.CpuModel(),
            ["onnxruntime"] = vd.OrtVersion,
            ["intra_op_threads"] = vd.IntraOpNumThreads,
            ["vision_graph_sha256"] = vd.ModelSha256,
            ["multimodal_graph_sha256"] = embedder.MultimodalSession.Diagnostics.ModelSha256,
            ["ffmpeg_version"] = decoder.Libraries.Version,
            ["max_video_frames_per_vision_call"] = options.MaxVideoFramesPerVisionCall,
        };
        device.Describe(environment);

        var perRow = new JsonObject();
        var inputs = new Dictionary<string, EmbeddingGemma2MultimodalInput>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            string id = row["id"]!.GetValue<string>();
            var prompt = EmbeddingGemma2Prompt.FromName(row["prompt"]!.GetValue<string>());
            var input = inputs[id] = BuildInput(row);
            var rec = new JsonObject { ["expect"] = row["expect"]!.GetValue<string>() };
            try
            {
                rec["input_ids"] = new JsonArray(embedder.Tokenize(input, prompt).InputIds.ToArray().Select(x => (JsonNode?)x).ToArray());
            }
            catch (ArgumentException ex)
            {
                rec["tokenize_exception"] = $"{ex.GetType().Name}: {ex.Message}";
            }

            var watch = Stopwatch.StartNew();
            try
            {
                float[] embedding = embedder.Embed(input, prompt);
                rec["outcome"] = "embedded";
                rec["embedding"] = new JsonArray(embedding.Select(x => (JsonNode?)x).ToArray());
            }
            catch (EmbeddingGemma2InputTooLongException ex)
            {
                rec["outcome"] = "rejected";
                rec["exception"] = new JsonObject
                {
                    ["type"] = ex.GetType().Name,
                    ["token_count"] = ex.TokenCount,
                    ["maximum_tokens"] = ex.MaximumTokens,
                    ["message"] = ex.Message,
                };
            }

            rec["seconds"] = Math.Round(watch.Elapsed.TotalSeconds, 2);
            perRow[id] = rec;
            Console.WriteLine($"{id}: {rec["outcome"]} ({rec["input_ids"]?.AsArray().Count} tokens, {rec["seconds"]} s)");
        }

        // Padded batches of two rows (no prompt) against the single rows above.
        var batches = new JsonObject();
        foreach (var (name, pair) in new[]
        {
            ("text-2600-and-8192", new[] { "txt-alice-2600-none", "txt-alice-8192-none" }),
            ("six-images-and-32-frame-video", new[] { "mm-alice-6-images-6000-none", "mm-alice-video32-7000-none" }),
        })
        {
            if (!pair.All(inputs.ContainsKey))
            {
                continue;
            }

            using var batchEmbedder = EmbeddingGemma2MultimodalEmbedder.Create(files, options with { MaxBatchSize = 2, MaxBatchTokens = 2 * 8_192, VerifyModelSha256 = false });
            var embeddings = batchEmbedder.EmbedBatch(pair.Select(p => inputs[p]).ToList(), EmbeddingGemma2Prompt.None);
            batches[name] = new JsonObject
            {
                ["rows"] = new JsonArray(pair.Select(p => (JsonNode?)p).ToArray()),
                ["embeddings"] = new JsonArray(embeddings.Select(e => (JsonNode?)new JsonArray(e.Select(x => (JsonNode?)x).ToArray())).ToArray()),
            };
            Console.WriteLine($"batch {name}: {pair.Length} rows");
        }

        var result = new JsonObject
        {
            ["producer"] = "ai-dotnet/tools/Starlings.Fidelity.Tool embeddinggemma2-long",
            ["repository_commit"] = ToolSupport.GitHead(repoRoot),
            ["candidate"] = $"C# EmbeddingGemma2MultimodalEmbedder (text-only rows on the released text graph), ONNX Runtime {vd.OrtVersion} {device.Label}, FFmpeg adapter {decoder.Libraries.Version}, SkiaSharp image decoder",
            ["rows_file_sha256"] = ToolSupport.Sha256(rowsPath),
            ["environment"] = JsonSerializer.SerializeToNode(environment),
            ["rows"] = perRow,
            ["batches"] = batches,
        };
        string path = Path.Combine(outDir, "long.results.json");
        File.WriteAllText(path, result.ToJsonString() + "\n");
        Console.WriteLine($"wrote {path}: {rows.Count} rows, {batches.Count} batches");
        return 0;
    }
}
