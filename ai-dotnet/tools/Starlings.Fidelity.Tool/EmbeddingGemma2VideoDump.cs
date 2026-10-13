// EmbeddingGemma 2 video stage results for `model-pipeline evaluate` (task "video-embedding", stage results format
// version 2). Decodes the fixture videos with the FFmpeg adapter (the user's own FFmpeg 8.x), runs the C# video path on
// every row of the protected video-embeddings.jsonl (the metadata-less array row with AllowVideosWithoutTiming, as
// upstream treats it) and writes:
//
//   tokenizer.results.json   stage "tokenizer": rendered ids (EmbeddingGemma2MultimodalEmbedder.Tokenize with the owned
//                            tokenizer) and, per video, the sampled frame indices, the SHA-256 of each decoded frame and
//                            of the sampled frames, the resized frames, the pixel values and the positions
//                            (EmbeddingGemma2ImageProcessor at 140 soft tokens); the still image's decoded RGB.
//   dotnet.results.json      stage "dotnet": the embedding of every row alone; soft tokens per video and token
//                            embeddings per media row (sidecars soft_tokens.f32, token_embeddings.f32); mixed rows in
//                            batches (frame calls of 2 and text batch 4; padded to 1,260 patches per frame).
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- embeddinggemma2-videos --out <dir>
//       --videos <dir> [--ffmpeg <FFmpeg 8 lib dir>] [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]
//
// CPU only (the tool references the CPU native runtime). Applies no bounds; writes nothing under protected/.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
using Starlings.Runtime.Onnx;

internal static class EmbeddingGemma2VideoDump
{
    private const string NoMetadataRow = "vid-array-no-metadata";

    public static int Run(string[] args)
    {
        string? outDir = null;
        string? snapshot = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        string? videosDir = Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_VIDEOS");
        string? ffmpegDir = null;
        string? fixturesDir = null;
        string profile = "embeddinggemma-2/video-fp32-cpu";
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
                case "--videos" when i + 1 < args.Length:
                    videosDir = args[++i];
                    break;
                case "--ffmpeg" when i + 1 < args.Length:
                    ffmpegDir = args[++i];
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

        if (outDir is null || videosDir is null)
        {
            Console.Error.WriteLine("error: --out and --videos (the make_eg2_videos.py output) are required");
            return 2;
        }

        string? repoRoot = ToolSupport.FindRepositoryRoot();
        fixturesDir ??= repoRoot is null ? null : Path.Combine(repoRoot, "model-pipeline", "protected", "embeddinggemma-2", "video-fp32-cpu");
        if (fixturesDir is null || !File.Exists(Path.Combine(fixturesDir, "video-embeddings.jsonl")))
        {
            Console.Error.WriteLine($"error: protected video fixtures not found ({fixturesDir ?? "no repository root"}); pass --fixtures");
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
        string? commit = ToolSupport.GitHead(repoRoot);
        var session = new OnnxSessionOptions { IntraOpNumThreads = threads };
        var options = new EmbeddingGemma2MultimodalOptions { Session = session, AllowVideosWithoutTiming = true, MaxVideoFramesPerVisionCall = 8, MaxVisionBatchPatches = 8 * 1_260 };
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, options);
        var videoFiles = ReadVideoFiles(Path.Combine(fixturesDir, "videos.json"));
        var rows = ReadRows(Path.Combine(fixturesDir, "video-embeddings.jsonl"));

        var sampled = new Dictionary<string, EmbeddingGemma2DecodedVideo>(StringComparer.Ordinal);
        var all = new Dictionary<string, EmbeddingGemma2DecodedVideo>(StringComparer.Ordinal);
        string PathOf(string id) => Path.Combine(videosDir, videoFiles[id]);
        EmbeddingGemma2Video VideoFor(string id, string form) => form switch
        {
            "array+metadata" => (all.TryGetValue(id, out var a) ? a : all[id] = decoder.DecodeAllFrames(PathOf(id))).Video,
            "array" => EmbeddingGemma2Video.FromFramesWithoutTiming((all.TryGetValue(id, out var b) ? b : all[id] = decoder.DecodeAllFrames(PathOf(id))).Video.Frames),
            _ => (sampled.TryGetValue(id, out var s) ? s : sampled[id] = decoder.DecodeFile(PathOf(id))).Video,
        };

        var libs = decoder.Libraries;
        var vd = embedder.VisionSession.Diagnostics;
        var environment = new Dictionary<string, object?>
        {
            ["machine"] = Environment.MachineName,
            ["provider"] = "cpu",
            ["provider_options"] = vd.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
            ["dotnet"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["cpu_count"] = Environment.ProcessorCount,
            ["cpu"] = ToolSupport.CpuModel(),
            ["onnxruntime"] = vd.OrtVersion,
            ["onnxruntime_managed"] = vd.ManagedBindingVersion,
            ["session_providers"] = vd.SessionProviders,
            ["intra_op_threads"] = vd.IntraOpNumThreads,
            ["video_decoder"] = "Starlings.Models.EmbeddingGemma2.Video.FFmpeg (FFmpeg.AutoGen 8.1.0)",
            ["ffmpeg_version"] = libs.Version,
            ["ffmpeg_major"] = EmbeddingGemma2FFmpeg.SupportedFFmpegMajor, // the loader accepts only the FFmpeg 8 library majors
            ["ffmpeg_libraries"] = libs.LibraryVersions,
            ["ffmpeg_license"] = libs.Licenses["avcodec"],
            ["max_video_frames_per_vision_call"] = options.MaxVideoFramesPerVisionCall,
        };
        var tokenizerAsset = new Asset(EmbeddingGemma2ModelAssets.TokenizerFile, EmbeddingGemma2ModelAssets.TokenizerSha256);
        Asset[] graphAssets =
        [
            new(EmbeddingGemma2ModelAssets.VisionModelFile, vd.ModelSha256 ?? ToolSupport.Sha256(files.VisionModelPath)),
            new(EmbeddingGemma2ModelAssets.MultimodalModelFile, embedder.MultimodalSession.Diagnostics.ModelSha256 ?? ToolSupport.Sha256(files.MultimodalModelPath)),
            new(EmbeddingGemma2ModelAssets.ModelFile, ToolSupport.Sha256(files.Text.ModelPath)),
        ];

        // Stage 1: decoding, sampling, rendering and preprocessing.
        var stage1 = new StageWriter(profile, "tokenizer", $"FFmpeg adapter on FFmpeg {libs.Version}; EmbeddingGemma2VideoSampling; EmbeddingGemma2ImageProcessor at 140 soft tokens; EmbeddingGemma2MultimodalEmbedder rendering with EmbeddingGemma2Tokenizer", commit, environment, [tokenizerAsset], task: "video-embedding");
        var inputs = new Dictionary<string, EmbeddingGemma2MultimodalInput>(StringComparer.Ordinal);
        foreach (var r in rows)
        {
            var videos = r.Videos.Select(v => VideoFor(v.Video, v.Form)).ToList();
            var still = r.Still is { } st ? decoder.DecodeFramesAt(PathOf(st.Video), [st.Frame])[0] : null;
            var input = BuildInput(r.Text, videos, still);
            inputs[r.Id] = input;
            var c = stage1.Case(r.Id).Add("input_ids", embedder.Tokenize(input, r.Prompt).InputIds.ToArray());
            for (int k = 0; k < videos.Count; k++)
            {
                int[] indices = videos[k].GetSampledFrameIndices(allowWithoutTiming: true);
                var frames = indices.Select(i => videos[k].Frames[i]).ToList();
                int[] source = videos[k].SourceFrameIndices is { } si ? [.. si] : indices;
                var patches = frames.Select(f => EmbeddingGemma2ImageProcessor.Preprocess(f, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens)).ToList();
                c.Add($"video{k}_frame_indices", source.Select(i => (long)i).ToArray())
                    .Add($"video{k}_frame_sha256", frames.Select(f => Sha(f.Pixels.Span)).ToArray())
                    .Add($"video{k}_frames_sha256", Sha(frames.Select(f => MemoryMarshal.AsBytes(f.Pixels.Span).ToArray())))
                    .Add($"video{k}_resized_sha256", Sha(patches.Select(p => p.ResizedPixels.ToArray())))
                    .Add($"video{k}_pixel_values_sha256", Sha(patches.Select(p => MemoryMarshal.AsBytes(p.PixelValues.Span).ToArray())))
                    .Add($"video{k}_positions_sha256", Sha(patches.Select(p => MemoryMarshal.AsBytes(p.Positions.Span).ToArray())));
            }

            if (still is not null)
            {
                c.Add("image0_rgb_sha256", Sha(still.Pixels.Span));
            }
        }

        stage1.Write(Path.Combine(outDir, "tokenizer.results.json"));
        Console.WriteLine($"stage 1: {rows.Count} rows ({sampled.Count + all.Count} decoded videos)");

        // Stage 3: every row alone, with its soft tokens and token embeddings.
        var stage3 = new StageWriter(profile, "dotnet", $"C# EmbeddingGemma2MultimodalEmbedder, ONNX Runtime {vd.OrtVersion} CPU (managed binding {vd.ManagedBindingVersion}), frames from the FFmpeg adapter", commit, environment, [.. graphAssets, tokenizerAsset], task: "video-embedding");
        var collector = new Collector();
        foreach (var r in rows)
        {
            collector.Row = r.Id;
            var destination = new float[embedder.Dimension];
            var watch = System.Diagnostics.Stopwatch.StartNew();
            embedder.EmbedCore([inputs[r.Id]], r.Prompt, destination, default, collector, CancellationToken.None);
            stage3.Case(r.Id).Add("embedding", destination);
            Console.WriteLine($"{r.Id}: embedded in {watch.Elapsed.TotalSeconds:F1} s");
        }

        stage3.Tensor("soft_tokens", outDir, "soft_tokens.f32", collector.Soft.Select(s => (s.Id, (new[] { s.Values.Length / 512, 512 }, s.Values))));
        stage3.Tensor("token_embeddings", outDir, "token_embeddings.f32", collector.Tokens.Select(t => (t.Id, (new[] { t.Values.Length / 768, 768 }, t.Values))));

        string[] mixed = ["q-bars", "h264-1s-320x240-30fps", "h264-60fps-480x270-3s", "mm-caption-then-video", "h264-1fps-64x48-8s", "vp9-odd-321x179-30fps-3s", "q-cat", "mm-two-videos", "q-shapes"];
        var mixedRows = rows.Where(r => mixed.Contains(r.Id) && r.Prompt == EmbeddingGemma2Prompt.None).ToList();
        foreach (var (name, batchOptions) in new[]
        {
            ("frame-calls-2-text-batch-4", new EmbeddingGemma2MultimodalOptions { Session = session, MaxBatchSize = 4, VerifyModelSha256 = false }),
            ("padded-1260-frame-calls-3", new EmbeddingGemma2MultimodalOptions { Session = session, PadImagesToMaxPatches = true, MaxVideoFramesPerVisionCall = 3, MaxVisionBatchPatches = 3 * 2_520, VerifyModelSha256 = false }),
        })
        {
            using var batchEmbedder = EmbeddingGemma2MultimodalEmbedder.Create(files, batchOptions);
            stage3.Batch(name, "embedding", mixedRows.Select(r => r.Id).ToList(), batchEmbedder.EmbedBatch(mixedRows.Select(r => inputs[r.Id]).ToList()));
            Console.WriteLine($"batch {name}: {mixedRows.Count} rows");
        }

        stage3.Write(Path.Combine(outDir, "dotnet.results.json"));
        Console.WriteLine($"wrote {Path.Combine(outDir, "tokenizer.results.json")} and dotnet.results.json: {rows.Count} rows");
        return 0;
    }

    private sealed record RowVideo(string Video, string Form);

    private sealed record Row(string Id, EmbeddingGemma2Prompt Prompt, string? Text, (string Video, int Frame)? Still, List<RowVideo> Videos);

    private sealed class Collector : IEmbeddingGemma2Diagnostics
    {
        public string Row { get; set; } = string.Empty;

        public List<(string Id, float[] Values)> Soft { get; } = [];

        public List<(string Id, float[] Values)> Tokens { get; } = [];

        public void OnSoftTokens(int input, int image, float[] softTokens)
        {
        }

        public void OnVideoSoftTokens(int input, int video, float[] softTokens) => Soft.Add(($"{Row}/video{video}", softTokens));

        public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) => Tokens.Add((Row, tokenEmbeddings.ToArray()));
    }

    /// <summary>Upstream's layout: placeholders in the text mark media positions; without any, the image and then the videos follow the text.</summary>
    private static EmbeddingGemma2MultimodalInput BuildInput(string? text, List<EmbeddingGemma2Video> videos, EmbeddingGemma2Image? still)
    {
        var media = new List<object>();
        if (text is null)
        {
            if (still is not null)
            {
                media.Add(still);
            }

            media.AddRange(videos);
            return media.Count == 0 ? EmbeddingGemma2MultimodalInput.FromText(string.Empty) : EmbeddingGemma2MultimodalInput.Create(media);
        }

        if (videos.Count == 0 && still is null)
        {
            return EmbeddingGemma2MultimodalInput.FromText(text);
        }

        var parts = new List<object>();
        int v = 0, start = 0;
        bool placed = false;
        while (true)
        {
            int iv = text.IndexOf("<|video|>", start, StringComparison.Ordinal);
            int ii = text.IndexOf("<|image|>", start, StringComparison.Ordinal);
            int next = iv < 0 ? ii : ii < 0 ? iv : Math.Min(iv, ii);
            if (next < 0)
            {
                parts.Add(text[start..]);
                break;
            }

            parts.Add(text[start..next]);
            parts.Add(next == iv ? videos[v++] : still!);
            placed = true;
            start = next + "<|video|>".Length;
        }

        if (!placed)
        {
            if (still is not null)
            {
                parts.Add(still);
            }

            parts.AddRange(videos);
        }

        return EmbeddingGemma2MultimodalInput.Create(parts);
    }

    private static Dictionary<string, string> ReadVideoFiles(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("videos").EnumerateArray()
            .ToDictionary(v => v.GetProperty("id").GetString()!, v => v.GetProperty("file").GetString()!, StringComparer.Ordinal);
    }

    private static List<Row> ReadRows(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l =>
            {
                using var doc = JsonDocument.Parse(l);
                var r = doc.RootElement;
                var image = r.GetProperty("image");
                return new Row(
                    r.GetProperty("id").GetString()!,
                    EmbeddingGemma2Prompt.FromName(r.GetProperty("prompt").GetString()!),
                    r.GetProperty("text").ValueKind == JsonValueKind.Null ? null : r.GetProperty("text").GetString(),
                    image.ValueKind == JsonValueKind.Null ? null : (image.GetProperty("video").GetString()!, image.GetProperty("frame_index").GetInt32()),
                    r.GetProperty("videos").EnumerateArray().Select(v => new RowVideo(v.GetProperty("video").GetString()!, v.GetProperty("input_form").GetString()!)).ToList());
            })
            .ToList();

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static string Sha(IEnumerable<byte[]> parts)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var part in parts)
        {
            hash.AppendData(part);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
}
