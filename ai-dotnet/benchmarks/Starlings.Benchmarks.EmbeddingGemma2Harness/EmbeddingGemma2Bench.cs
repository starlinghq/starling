using System.Diagnostics;
using System.Text.Json.Nodes;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// EmbeddingGemma 2 through the public API (CPU profiles; the CUDA build: the <c>-cuda</c> profiles, judged against the
/// same reference rows with their own binding tolerances). Text path (<c>onnx-fp32-cpu</c>): <c>tokenize</c>
/// (<see cref="EmbeddingGemma2Tokenizer.Tokenize"/> with the row's prompt, per text) and <c>encode</c>
/// (<see cref="EmbeddingGemma2Encoder.EmbedBatch(IReadOnlyList{EmbeddingGemma2Input}, CancellationToken)"/>: padding,
/// the ORT run, pooling and normalization). Image path (<c>image-fp32-cpu</c>): <c>decode</c> (SkiaSharp adapter, from
/// the file bytes in memory) and <c>embed</c> (<see cref="EmbeddingGemma2MultimodalEmbedder"/>: image preprocessing,
/// the vision graph and the multimodal text graph; the public API does not split them, so the
/// <c>preprocess-only</c> workload times the image processor alone). Video path (<c>video-fp32-cpu</c>), from decoded
/// frames (<see cref="EmbeddingGemma2VideoFrames"/>: the reference's sampled frames, no decoder in the process):
/// <c>embed</c> (the same embedder: per-frame preprocessing at 140 soft tokens, the vision graph in calls of at most
/// <see cref="EmbeddingGemma2MultimodalOptions.MaxVideoFramesPerVisionCall"/> frames, the multimodal text graph);
/// <c>video-preprocess-only</c> times the per-frame preprocessing alone.
/// </summary>
internal sealed class EmbeddingGemma2Driver : IBenchDriver, ISweepDriver
{
    public const string TextReference = "model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/embeddings.jsonl";

    public const string ImageReference = "model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/image-embeddings.jsonl";

    private static readonly string[] s_allPaths = ["text", "image", "video"];

    public string Model => "embeddinggemma-2";

    public string Profile(HarnessContext context) =>
        string.Join('+', new[] { ("text", "onnx-fp32"), ("image", "image-fp32"), ("video", "video-fp32") }.Where(p => Paths(context).Contains(p.Item1)).Select(p => context.ProfileOf(p.Item2)));

    public string Reference => $"{TextReference}; {ImageReference}; {EmbeddingGemma2VideoFrames.Reference}";

    public string HarnessName => "Starlings.Benchmarks.EmbeddingGemma2Harness";

    public string WorkloadFile => "embeddinggemma2-v1.json";

    public string Usage => $"""
          --paths text,image,video  paths to load and run (default text,image; image needs the image set, video the
                                    decoded-frame fixtures)
          --cold-path text|image|video  the cold-start request (default text)
          --preliminary all|workloads  preliminary sweep over every fixture row (default) or only the rows the workloads use
          model: ${EmbeddingGemma2TestAssets.SnapshotEnvironmentVariable} or $STARLING_MODELS; image set: ${EmbeddingGemma2ImageTestAssets.ImagesEnvironmentVariable}
          (make_eg2_images.py output) or <models>/embeddinggemma-2-images; decoded video frames: ${EmbeddingGemma2VideoFrames.TensorsEnvironmentVariable}
          (run_embeddinggemma2_video.py --tensors output) or <models>/embeddinggemma-2-video-tensors.
        """;

    public IBenchRun Load(HarnessContext context) => new EmbeddingGemma2Run(context, Paths(context));

    /// <summary>--max-batch-tokens maps to the existing <see cref="EmbeddingGemma2EncoderOptions.MaxBatchTokens"/>.</summary>
    public IReadOnlyCollection<string> WiredSweepHooks => ["max-batch-tokens"];

    /// <summary>The text path only (image and video are outside the batching investigation).</summary>
    public ISweepTarget LoadSweep(HarnessContext context, SweepOptions options) => new EmbeddingGemma2SweepTarget(context, options);

    public JsonObject Artifacts(HarnessContext context)
    {
        var paths = Paths(context);
        var text = TextFiles();
        var fixtures = new List<string>
        {
            Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "embeddings.jsonl"),
            Path.Combine(BenchPaths.RepositoryRoot, EmbeddingGemma2Tolerances.Load(context.ProfileOf("onnx-fp32")).Source),
        };
        if (paths.Contains("image"))
        {
            fixtures.Add(Path.Combine(EmbeddingGemma2ImageTestAssets.FixtureDirectory, "image-embeddings.jsonl"));
            fixtures.Add(Path.Combine(EmbeddingGemma2ImageTestAssets.FixtureDirectory, "images.json"));
            fixtures.Add(Path.Combine(BenchPaths.RepositoryRoot, EmbeddingGemma2Tolerances.Load(context.ProfileOf("image-fp32")).Source));
        }

        if (paths.Contains("video"))
        {
            fixtures.Add(Path.Combine(EmbeddingGemma2VideoFrames.FixtureDirectory, "video-embeddings.jsonl"));
            fixtures.Add(Path.Combine(EmbeddingGemma2VideoFrames.FixtureDirectory, "tensors.json"));
            fixtures.Add(Path.Combine(BenchPaths.RepositoryRoot, EmbeddingGemma2Tolerances.Load(context.ProfileOf("video-fp32")).Source));
        }

        var artifacts = BenchEnvironment.Artifacts(
            text.ModelPath,
            null,
            text.TokenizerPath,
            context.WorkloadPath,
            BenchEnvironment.FileDigests([.. fixtures]),
            HarnessHost.Packages(
                ("Starlings.Models.EmbeddingGemma2", BenchEnvironment.InformationalVersion(typeof(EmbeddingGemma2Encoder).Assembly)),
                ("Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp", BenchEnvironment.InformationalVersion(typeof(EmbeddingGemma2ImageDecoder).Assembly)),
                ("SkiaSharp", BenchEnvironment.InformationalVersion(typeof(SkiaSharp.SKBitmap).Assembly))));
        artifacts["model_id"] = $"{EmbeddingGemma2ModelAssets.Repository}@{EmbeddingGemma2ModelAssets.Revision} ({EmbeddingGemma2ModelAssets.ModelFile}, release {EmbeddingGemma2ModelAssets.ReleaseTag})";
        artifacts["expected_model_sha256"] = EmbeddingGemma2ModelAssets.ModelSha256;
        artifacts["tokenizer_version"] = BenchEnvironment.TokenizerVersion;
        if (paths.Contains("image") || paths.Contains("video"))
        {
            var image = ImageFiles();
            artifacts["image_model_id"] = $"{EmbeddingGemma2ModelAssets.ImageReleaseTag} ({EmbeddingGemma2ModelAssets.VisionModelFile}, {EmbeddingGemma2ModelAssets.MultimodalModelFile})";
            artifacts["vision_model_sha256"] = BenchEnvironment.Sha256File(image.VisionModelPath);
            artifacts["multimodal_model_sha256"] = BenchEnvironment.Sha256File(image.MultimodalModelPath);
        }

        return artifacts;
    }

    internal static EmbeddingGemma2ModelFiles TextFiles()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        return files ?? throw new BenchAssetsMissingException(reason);
    }

    internal static EmbeddingGemma2ImageModelFiles ImageFiles()
    {
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        return files ?? throw new BenchAssetsMissingException(reason);
    }

    private static HashSet<string> Paths(HarnessContext context)
    {
        var paths = (context.Args.GetList("paths") ?? ["text", "image"]).ToHashSet(StringComparer.Ordinal);
        if (paths.Count == 0 || paths.Except(s_allPaths).Any())
        {
            throw new ArgumentException("--paths takes text, image and/or video.");
        }

        return paths;
    }
}

internal sealed class EmbeddingGemma2Run : IBenchRun
{
    private static readonly string[] s_textStages = ["tokenize", "encode"];
    private static readonly string[] s_imageStages = ["decode", "embed"];
    private static readonly string[] s_preprocessStages = ["decode", "preprocess"];
    private static readonly string[] s_videoStages = ["embed"];
    private static readonly string[] s_videoPreprocessStages = ["preprocess"];
    private readonly HarnessContext _context;
    private readonly HashSet<string> _paths;
    private readonly WorkloadSpec _spec;
    private readonly EmbeddingGemma2Tokenizer? _tokenizer;
    private readonly EmbeddingGemma2Encoder? _encoder;
    private readonly EmbeddingGemma2MultimodalEmbedder? _multimodal;
    private readonly Dictionary<string, EmbeddingGemma2Row> _textRows = new(StringComparer.Ordinal);
    private readonly EmbeddingGemma2Tolerances _textTolerances;
    private readonly EmbeddingGemma2Tolerances _imageTolerances;
    private readonly Dictionary<string, ImageRow> _imageRows = new(StringComparer.Ordinal);
    private readonly Dictionary<string, byte[]> _imageBytes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _selfReference = new(StringComparer.Ordinal);
    private readonly EmbeddingGemma2Tolerances _videoTolerances;
    private readonly Dictionary<string, VideoBenchRow> _videoRows = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (EmbeddingGemma2Video[] Videos, EmbeddingGemma2MultimodalInput Input)> _videoInputs = new(StringComparer.Ordinal);

    public EmbeddingGemma2Run(HarnessContext context, HashSet<string> paths)
    {
        _context = context;
        _paths = paths;
        var load = new JsonObject();
        long t = Stopwatch.GetTimestamp();
        _spec = WorkloadSpec.Load(context.WorkloadPath);
        foreach (var row in EmbeddingGemma2Fixtures.LoadRows())
        {
            _textRows[row.Key] = row;
        }

        _textTolerances = EmbeddingGemma2Tolerances.Load(context.ProfileOf("onnx-fp32"));
        _imageTolerances = EmbeddingGemma2Tolerances.Load(context.ProfileOf("image-fp32"));
        _videoTolerances = EmbeddingGemma2Tolerances.Load(context.ProfileOf("video-fp32"));
        if (paths.Contains("video"))
        {
            foreach (var (id, row) in EmbeddingGemma2VideoFrames.LoadRows())
            {
                _videoRows[id] = row;
            }
        }
        if (paths.Contains("image"))
        {
            foreach (var row in EmbeddingGemma2ImageTestAssets.Rows)
            {
                _imageRows[row.Id] = row;
            }
        }

        load["fixture_read"] = MonotonicClock.ElapsedMs(t);
        var session = context.SessionOptions();
        bool verify = context.HashModelAtLoad;
        if (paths.Contains("text"))
        {
            var files = EmbeddingGemma2Driver.TextFiles();
            t = Stopwatch.GetTimestamp();
            _tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath);
            load["tokenizer_load"] = MonotonicClock.ElapsedMs(t);
            t = Stopwatch.GetTimestamp();
            _encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
            {
                Session = session with { ExpectedModelSha256 = verify ? EmbeddingGemma2ModelAssets.ModelSha256 : null },
            });
            load["session_create"] = MonotonicClock.ElapsedMs(t);
        }

        if (paths.Contains("image") || paths.Contains("video"))
        {
            var files = EmbeddingGemma2Driver.ImageFiles();
            t = Stopwatch.GetTimestamp();
            _multimodal = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = session, VerifyModelSha256 = verify });
            load["multimodal_create"] = MonotonicClock.ElapsedMs(t);
        }

        LoadStages = load;
    }

    public JsonObject LoadStages { get; }

    public JsonObject Settings
    {
        get
        {
            var settings = _context.SessionSettings((_encoder?.Session ?? _multimodal?.VisionSession)?.Diagnostics.ProviderOptions);
            settings["pad_id"] = EmbeddingGemma2Encoder.PadTokenId;
            settings["dimension"] = EmbeddingGemma2Encoder.NativeDimension;
            settings["paths"] = new JsonArray(_paths.Order(StringComparer.Ordinal).Select(p => (JsonNode)p).ToArray());
            if (_encoder is not null)
            {
                settings["max_batch_size"] = _encoder.Options.MaxBatchSize;
                settings["max_batch_tokens"] = _encoder.Options.MaxBatchTokens;
            }

            if (_multimodal is not null)
            {
                settings["max_image_soft_tokens"] = _multimodal.Options.MaxImageSoftTokens;
                settings["max_vision_batch_size"] = _multimodal.Options.MaxVisionBatchSize;
            }

            return settings;
        }
    }

    public IReadOnlyList<string>? SessionProviders => (_encoder?.Session ?? _multimodal?.VisionSession)?.Diagnostics.SessionProviders;

    public JsonObject PreliminaryQuality()
    {
        var result = new JsonObject
        {
            ["reference"] = string.Join("; ", new[]
            {
                _encoder is null ? null : EmbeddingGemma2Driver.TextReference,
                _paths.Contains("image") ? EmbeddingGemma2Driver.ImageReference : null,
                _paths.Contains("video") ? EmbeddingGemma2VideoFrames.Reference : null,
            }.OfType<string>()),
        };
        bool pass = true;
        int cases = 0;
        var stages = new double[4];
        bool all = _context.Args.Get("preliminary", "all") switch
        {
            "all" => true,
            "workloads" => false,
            _ => throw new ArgumentException("--preliminary all|workloads."),
        };
        var used = _spec.Workloads.Where(w => w.Raw.TryGetProperty("requests", out _))
            .SelectMany(w => w.Raw.GetProperty("requests").EnumerateArray().SelectMany(WorkloadSpec.Strings))
            .ToHashSet(StringComparer.Ordinal);
        result["scope"] = all ? "every fixture row" : "the rows the workloads use";
        if (_encoder is not null)
        {
            // Every row alone (batches of one), then in padded batches of 8 (the long rows alone).
            var rows = _textRows.Values.Where(r => all || used.Contains(r.Key)).ToArray();
            var singles = JudgeText(rows.Select((r, i) => (i, (object)EmbedText([r], stages))).ToList(), rows.Select(r => new[] { r }).ToArray());
            var batches = rows.Where(r => r.InputIds.Length < 512).Chunk(8).ToArray();
            var batched = JudgeText(batches.Select((b, i) => (i, (object)EmbedText(b, stages))).ToList(), batches);
            result["text_single"] = singles;
            result["text_batch8"] = batched;
            result["text_tolerance_file"] = _textTolerances.Source;
            pass &= singles["pass"]!.GetValue<bool>() && batched["pass"]!.GetValue<bool>();
            cases += rows.Length;
        }

        if (_paths.Contains("image"))
        {
            var rows = _imageRows.Values.Where(r => r.InContract && r.Images.All(i => i.Image != "jpeg-exif-orientation") && (all || used.Contains(r.Id))).ToArray();
            var requests = rows.Select(r => new[] { r }).ToArray();
            var available = requests.Where(r => ImagesPresent(r)).ToArray();
            var image = JudgeImage(available.Select((r, i) => (i, (object)EmbedImages(r, stages))).ToList(), available);
            image["rows_without_image_files"] = requests.Length - available.Length;
            result["image_single"] = image;
            result["image_tolerance_file"] = _imageTolerances.Source;
            pass &= image["pass"]!.GetValue<bool>();
            cases += available.Length;
        }

        if (_paths.Contains("video"))
        {
            // Every row of text and videos (a row with a still image is not a video workload row), each alone.
            var rows = _videoRows.Values.Where(r => !r.HasImage && r.Videos.Count > 0 && (all || used.Contains(r.Id))).Select(r => new[] { r }).ToArray();
            var video = JudgeVideo(rows.Select((r, i) => (i, (object)EmbedVideos(r, stages))).ToList(), rows);
            video["rows_not_run"] = _videoRows.Count - rows.Length;
            result["video_single"] = video;
            result["video_tolerance_file"] = _videoTolerances.Source;
            result["video_frames"] = "reference sampled frames (run_embeddinggemma2_video.py --tensors), checked against tensors.json";
            pass &= video["pass"]!.GetValue<bool>();
            cases += rows.Length;
        }

        result["cases"] = cases;
        result["pass"] = pass;
        return result;
    }

    public IEnumerable<BenchSeries> WarmSeries()
    {
        foreach (var w in _spec.Workloads.Where(w => _paths.Contains(w.Path)))
        {
            foreach (int c in w.Concurrency)
            {
                yield return Series(w.Name, w.Path, w.Input, w.Batch, c, w.Samples, w.WarmupCalls ?? _spec.WarmupCalls, w.Raw);
            }
        }
    }

    public BenchSeries ColdSeries()
    {
        string path = _context.Args.Get("cold-path", "text");
        if (!_paths.Contains(path))
        {
            throw new ArgumentException($"--cold-path {path} needs --paths to include it.");
        }

        var request = WorkloadSpec.Strings(_spec.Cold.GetProperty(path));
        return path switch
        {
            "text" => TextSeries("cold-start", "short", request.Length, 1, 1, 0, [request.Select(TextRow).ToArray()]),
            "image" => ImageSeries("cold-start", "photo", request.Length, 1, 1, 0, [request.Select(ImageRowOf).ToArray()], preprocessOnly: false),
            _ => VideoSeries("cold-start", "video", request.Length, 1, 1, 0, [request.Select(VideoRowOf).ToArray()], preprocessOnly: false),
        };
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        _multimodal?.Dispose();
    }

    private BenchSeries Series(string name, string path, string input, int batch, int concurrency, int samples, int warmup, System.Text.Json.JsonElement raw)
    {
        if (path == "video")
        {
            var requests = raw.GetProperty("requests").EnumerateArray().Select(r => WorkloadSpec.Strings(r).Select(VideoRowOf).ToArray()).ToArray();
            bool preprocessOnly = raw.TryGetProperty("preprocess_only", out var vp) && vp.GetBoolean();
            return VideoSeries(name, input, batch, concurrency, samples, warmup, requests, preprocessOnly);
        }

        if (path == "image")
        {
            var requests = raw.GetProperty("requests").EnumerateArray().Select(r => WorkloadSpec.Strings(r).Select(ImageRowOf).ToArray()).ToArray();
            bool preprocessOnly = raw.TryGetProperty("preprocess_only", out var p) && p.GetBoolean();
            return ImageSeries(name, input, batch, concurrency, samples, warmup, requests, preprocessOnly);
        }

        if (raw.TryGetProperty("synthetic", out var synthetic))
        {
            var source = TextRow(synthetic.GetProperty("row").GetString()!);
            int words = synthetic.GetProperty("words").GetInt32();
            string text = string.Join(' ', source.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Take(words));
            var row = new EmbeddingGemma2Row($"{source.Id}:first-{words}-words", source.Prompt, source.PromptText, text, [], []);
            if (!_selfReference.ContainsKey(row.Key))
            {
                _selfReference[row.Key] = EmbedText([row], new double[4])[0];
            }

            return TextSeries(name, input, batch, concurrency, samples, warmup, [[row]]);
        }

        var textRequests = raw.GetProperty("requests").EnumerateArray().Select(r => WorkloadSpec.Strings(r).Select(TextRow).ToArray()).ToArray();
        return TextSeries(name, input, batch, concurrency, samples, warmup, textRequests);
    }

    private BenchSeries TextSeries(string name, string input, int batch, int concurrency, int samples, int warmup, EmbeddingGemma2Row[][] requests) => new()
    {
        Workload = name,
        Path = "text",
        Input = input,
        Batch = batch,
        Concurrency = concurrency,
        Samples = samples,
        WarmupCalls = Math.Max(warmup, requests.Length),
        Requests = requests.Length,
        Stages = s_textStages,
        Call = (r, stages, _) => EmbedText(requests[r], stages),
        Judge = outputs => JudgeText(outputs, requests),
    };

    private BenchSeries ImageSeries(string name, string input, int batch, int concurrency, int samples, int warmup, ImageRow[][] requests, bool preprocessOnly)
    {
        foreach (var request in requests)
        {
            if (!ImagesPresent(request))
            {
                throw new BenchAssetsMissingException(EmbeddingGemma2ImageTestAssets.ImagesSkipReason ?? "image files missing");
            }

            if (request.Select(r => r.Prompt).Distinct().Count() != 1)
            {
                throw new ArgumentException($"{name}: a batch is one call with one prompt; its rows have different prompts.");
            }
        }

        return new BenchSeries
        {
            Workload = name,
            Path = "image",
            Input = input,
            Batch = batch,
            Concurrency = concurrency,
            Samples = samples,
            WarmupCalls = Math.Max(warmup, requests.Length),
            Requests = requests.Length,
            Stages = preprocessOnly ? s_preprocessStages : s_imageStages,
            Call = preprocessOnly ? (r, stages, _) => Preprocess(requests[r], stages) : (r, stages, _) => EmbedImages(requests[r], stages),
            Judge = preprocessOnly ? outputs => JudgePreprocess(outputs, requests) : outputs => JudgeImage(outputs, requests),
        };
    }

    private BenchSeries VideoSeries(string name, string input, int batch, int concurrency, int samples, int warmup, VideoBenchRow[][] requests, bool preprocessOnly)
    {
        foreach (var request in requests)
        {
            if (request.Select(r => r.Prompt).Distinct().Count() != 1)
            {
                throw new ArgumentException($"{name}: a batch is one call with one prompt; its rows have different prompts.");
            }

            foreach (var row in request)
            {
                _ = VideoInput(row);
            }
        }

        return new BenchSeries
        {
            Workload = name,
            Path = "video",
            Input = input,
            Batch = batch,
            Concurrency = concurrency,
            Samples = samples,
            WarmupCalls = Math.Max(warmup, requests.Length),
            Requests = requests.Length,
            Stages = preprocessOnly ? s_videoPreprocessStages : s_videoStages,
            Call = preprocessOnly ? (r, stages, _) => PreprocessVideos(requests[r], stages) : (r, stages, _) => EmbedVideos(requests[r], stages),
            Judge = preprocessOnly ? outputs => JudgeVideoPreprocess(outputs, requests) : outputs => JudgeVideo(outputs, requests),
        };
    }

    private VideoBenchRow VideoRowOf(string id) =>
        _videoRows.TryGetValue(id, out var row) ? row : throw new ArgumentException($"Workload names unknown video row '{id}'.");

    /// <summary>The row's decoded frames and input, built once (outside every timed call).</summary>
    private (EmbeddingGemma2Video[] Videos, EmbeddingGemma2MultimodalInput Input) VideoInput(VideoBenchRow row)
    {
        if (!_videoInputs.TryGetValue(row.Id, out var prepared))
        {
            var videos = EmbeddingGemma2VideoFrames.LoadVideos(row);
            prepared = (videos, EmbeddingGemma2VideoFrames.BuildInput(row, videos));
            _videoInputs[row.Id] = prepared;
        }

        return prepared;
    }

    private float[][] EmbedVideos(VideoBenchRow[] request, double[] stages)
    {
        var inputs = request.Select(r => VideoInput(r).Input).ToArray();
        long t = Stopwatch.GetTimestamp();
        var output = _multimodal!.EmbedBatch(inputs, request[0].PackagePrompt);
        stages[0] = MonotonicClock.ElapsedMs(t);
        return output;
    }

    /// <summary>
    /// Per-frame preprocessing at <see cref="EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens"/> as the embedder
    /// runs it: the batch processor on up to <see cref="FramesPreprocessedAhead"/> frames at a time (frames in
    /// parallel); one pixel-values digest per video.
    /// </summary>
    private string[] PreprocessVideos(VideoBenchRow[] request, double[] stages)
    {
        var videos = request.SelectMany(r => VideoInput(r).Videos).ToArray();
        long t = Stopwatch.GetTimestamp();
        var patches = videos.Select(v => v.Frames.Chunk(FramesPreprocessedAhead)
            .SelectMany(frames => EmbeddingGemma2ImageProcessor.Preprocess(frames, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens)).ToArray()).ToArray();
        stages[0] = MonotonicClock.ElapsedMs(t);
        return patches.Select(p => VideoPixelSha(p)).ToArray();
    }

    /// <summary>The embedder's internal <c>VideoFramesPreprocessedAhead</c>: frames preprocessed together ahead of their vision calls.</summary>
    private const int FramesPreprocessedAhead = 8;

    private static string VideoPixelSha(EmbeddingGemma2ImagePatches[] frames)
    {
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        foreach (var f in frames)
        {
            hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(f.PixelValues.Span));
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private JsonObject JudgeVideo(IReadOnlyList<(int Request, object Output)> outputs, VideoBenchRow[][] requests)
    {
        var q = Quality.Empty;
        int rows = 0;
        foreach (var (r, o) in outputs)
        {
            var embeddings = (float[][])o;
            q = q.Merge(Quality.Of(embeddings, requests[r].Select(row => row.Embedding).ToArray()));
            rows += embeddings.Length;
        }

        double tolerance = _videoTolerances.EmbeddingMaxAbs;
        var json = q.ToJson();
        json["rows"] = rows;
        json["tolerance"] = tolerance;
        json["pass"] = q.MaxAbs <= tolerance && q.Finite;
        return json;
    }

    private static JsonObject JudgeVideoPreprocess(IReadOnlyList<(int Request, object Output)> outputs, VideoBenchRow[][] requests)
    {
        int videos = 0, equal = 0;
        foreach (var (r, o) in outputs)
        {
            var digests = (string[])o;
            var expected = requests[r].SelectMany(row => row.Videos).Select(v => v.PixelValuesSha256).ToArray();
            for (int i = 0; i < digests.Length; i++)
            {
                videos++;
                equal += digests[i] == expected[i] ? 1 : 0;
            }
        }

        return new JsonObject
        {
            ["check"] = "per-video pixel_values SHA-256 (every sampled frame, 140 soft tokens) equal to the fixture digest (exact)",
            ["videos"] = videos,
            ["equal"] = equal,
            ["pass"] = equal == videos,
        };
    }

    private EmbeddingGemma2Row TextRow(string key) =>
        _textRows.TryGetValue(key, out var row) ? row : throw new ArgumentException($"Workload names unknown text row '{key}'.");

    private ImageRow ImageRowOf(string id) =>
        _imageRows.TryGetValue(id, out var row) ? row : throw new ArgumentException($"Workload names unknown image row '{id}'.");

    private float[][] EmbedText(EmbeddingGemma2Row[] request, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        var inputs = new EmbeddingGemma2Input[request.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = _tokenizer!.Tokenize(request[i].PackagePrompt.Apply(request[i].Text));
        }

        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var output = _encoder!.EmbedBatch(inputs);
        stages[1] = MonotonicClock.ElapsedMs(t);
        return output;
    }

    private float[][] EmbedImages(ImageRow[] request, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        var decoded = new Dictionary<string, EmbeddingGemma2Image>(StringComparer.Ordinal);
        foreach (var image in request.SelectMany(r => r.Images))
        {
            if (!decoded.ContainsKey(image.Image))
            {
                decoded[image.Image] = EmbeddingGemma2ImageDecoder.Decode(_imageBytes[image.Image]);
            }
        }

        var inputs = request.Select(r => EmbeddingGemma2ImageTestAssets.BuildInput(r, id => decoded[id])).ToArray();
        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var output = _multimodal!.EmbedBatch(inputs, request[0].PackagePrompt);
        stages[1] = MonotonicClock.ElapsedMs(t);
        return output;
    }

    private string[] Preprocess(ImageRow[] request, double[] stages)
    {
        long t = Stopwatch.GetTimestamp();
        var images = request.Select(r => EmbeddingGemma2ImageDecoder.Decode(_imageBytes[r.Images[0].Image])).ToArray();
        stages[0] = MonotonicClock.ElapsedMs(t);
        t = Stopwatch.GetTimestamp();
        var patches = images.Select(i => EmbeddingGemma2ImageProcessor.Preprocess(i, _multimodal!.Options.MaxImageSoftTokens)).ToArray();
        stages[1] = MonotonicClock.ElapsedMs(t);
        return patches.Select(p => EmbeddingGemma2ImageTestAssets.Sha(p.PixelValues.Span)).ToArray();
    }

    /// <summary>True when every image of the request is in the image set with its pinned SHA-256 (read into memory once).</summary>
    private bool ImagesPresent(ImageRow[] request)
    {
        foreach (var image in request.SelectMany(r => r.Images))
        {
            if (_imageBytes.ContainsKey(image.Image))
            {
                continue;
            }

            var entry = EmbeddingGemma2ImageTestAssets.Manifest.Single(m => m.Id == image.Image);
            string path = Path.Combine(EmbeddingGemma2ImageTestAssets.ImagesDirectory, entry.File);
            if (!File.Exists(path))
            {
                return false;
            }

            var bytes = File.ReadAllBytes(path);
            if (Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes)) != entry.Sha256)
            {
                throw new InvalidDataException($"{path}: not the fixture image (SHA-256 differs from images.json).");
            }

            _imageBytes[image.Image] = bytes;
        }

        return true;
    }

    /// <summary>Fixture rows against their reference embedding; synthetic rows against the same text embedded alone at load.</summary>
    private JsonObject JudgeText(IReadOnlyList<(int Request, object Output)> outputs, EmbeddingGemma2Row[][] requests)
    {
        var fixture = Quality.Empty;
        var self = Quality.Empty;
        int fixtureRows = 0, selfRows = 0;
        foreach (var (r, o) in outputs)
        {
            var embeddings = (float[][])o;
            for (int i = 0; i < embeddings.Length; i++)
            {
                var row = requests[r][i];
                if (_selfReference.TryGetValue(row.Key, out var reference))
                {
                    self = self.Merge(Quality.Of([embeddings[i]], [reference]));
                    selfRows++;
                }
                else
                {
                    fixture = fixture.Merge(Quality.Of([embeddings[i]], [row.Embedding]));
                    fixtureRows++;
                }
            }
        }

        double tolerance = _textTolerances.EmbeddingMaxAbs;
        double selfTolerance = _textTolerances.PaddedBatchVsSingleMaxAbs;
        var json = fixture.ToJson();
        json["rows"] = fixtureRows;
        json["tolerance"] = tolerance;
        bool pass = fixture.MaxAbs <= tolerance && fixture.Finite;
        if (selfRows > 0)
        {
            json["synthetic_rows"] = selfRows;
            json["synthetic_vs_single_at_load"] = self.ToJson();
            json["synthetic_tolerance"] = selfTolerance;
            pass &= self.MaxAbs <= selfTolerance && self.Finite;
            json["finite"] = fixture.Finite && self.Finite;
        }

        json["pass"] = pass;
        return json;
    }

    private JsonObject JudgeImage(IReadOnlyList<(int Request, object Output)> outputs, ImageRow[][] requests)
    {
        var q = Quality.Empty;
        int rows = 0;
        foreach (var (r, o) in outputs)
        {
            var embeddings = (float[][])o;
            q = q.Merge(Quality.Of(embeddings, requests[r].Select(row => row.Embedding).ToArray()));
            rows += embeddings.Length;
        }

        double tolerance = _imageTolerances.EmbeddingMaxAbs;
        var json = q.ToJson();
        json["rows"] = rows;
        json["tolerance"] = tolerance;
        json["pass"] = q.MaxAbs <= tolerance && q.Finite;
        return json;
    }

    private static JsonObject JudgePreprocess(IReadOnlyList<(int Request, object Output)> outputs, ImageRow[][] requests)
    {
        int images = 0, equal = 0;
        foreach (var (r, o) in outputs)
        {
            var digests = (string[])o;
            for (int i = 0; i < digests.Length; i++)
            {
                images++;
                equal += digests[i] == requests[r][i].Images[0].PixelValuesSha256 ? 1 : 0;
            }
        }

        return new JsonObject
        {
            ["check"] = "pixel_values SHA-256 equal to the fixture digest (exact)",
            ["images"] = images,
            ["equal"] = equal,
            ["pass"] = equal == images,
        };
    }
}
