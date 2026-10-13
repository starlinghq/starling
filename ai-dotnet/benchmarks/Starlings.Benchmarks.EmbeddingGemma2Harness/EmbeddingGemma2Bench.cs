using System.Diagnostics;
using System.Text.Json.Nodes;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>
/// EmbeddingGemma 2 through the public API. Text path (<c>onnx-fp32-cpu</c>): <c>tokenize</c>
/// (<see cref="EmbeddingGemma2Tokenizer.Tokenize"/> with the row's prompt, per text) and <c>encode</c>
/// (<see cref="EmbeddingGemma2Encoder.EmbedBatch(IReadOnlyList{EmbeddingGemma2Input}, CancellationToken)"/>: padding,
/// the ORT run, pooling and normalization). Image path (<c>image-fp32-cpu</c>): <c>decode</c> (SkiaSharp adapter, from
/// the file bytes in memory) and <c>embed</c> (<see cref="EmbeddingGemma2MultimodalEmbedder"/>: image preprocessing,
/// the vision graph and the multimodal text graph; the public API does not split them, so the
/// <c>preprocess-only</c> workload times the image processor alone).
/// </summary>
internal sealed class EmbeddingGemma2Driver : IBenchDriver
{
    public const string TextReference = "model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/embeddings.jsonl";

    public const string ImageReference = "model-pipeline/protected/embeddinggemma-2/image-fp32-cpu/image-embeddings.jsonl";

    public string Model => "embeddinggemma-2";

    public string Profile => "onnx-fp32-cpu+image-fp32-cpu";

    public string Reference => $"{TextReference}; {ImageReference}";

    public string HarnessName => "Starlings.Benchmarks.EmbeddingGemma2Harness";

    public string WorkloadFile => "embeddinggemma2-v1.json";

    public string Usage => $"""
          --paths text,image        paths to load and run (default text,image; image needs the image set)
          --cold-path text|image    the cold-start request (default text)
          --preliminary all|workloads  preliminary sweep over every fixture row (default) or only the rows the workloads use
          model: ${EmbeddingGemma2TestAssets.SnapshotEnvironmentVariable} or $STARLING_MODELS; image set: ${EmbeddingGemma2ImageTestAssets.ImagesEnvironmentVariable}
          (make_eg2_images.py output) or <models>/embeddinggemma-2-images. Video is not benchmarked (needs FFmpeg).
        """;

    public IBenchRun Load(HarnessContext context) => new EmbeddingGemma2Run(context, Paths(context));

    public JsonObject Artifacts(HarnessContext context)
    {
        var paths = Paths(context);
        var text = TextFiles();
        var fixtures = new List<string>
        {
            Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "embeddings.jsonl"),
            Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "tolerances.yaml"),
        };
        if (paths.Contains("image"))
        {
            fixtures.Add(Path.Combine(EmbeddingGemma2ImageTestAssets.FixtureDirectory, "image-embeddings.jsonl"));
            fixtures.Add(Path.Combine(EmbeddingGemma2ImageTestAssets.FixtureDirectory, "images.json"));
            fixtures.Add(Path.Combine(EmbeddingGemma2ImageTestAssets.FixtureDirectory, "tolerances.yaml"));
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
        if (paths.Contains("image"))
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
        if (paths.Except(["text", "image"]).Any())
        {
            throw new ArgumentException("--paths takes text and/or image (video is not benchmarked: its fixtures need FFmpeg).");
        }

        return paths;
    }
}

internal sealed class EmbeddingGemma2Run : IBenchRun
{
    private static readonly string[] s_textStages = ["tokenize", "encode"];
    private static readonly string[] s_imageStages = ["decode", "embed"];
    private static readonly string[] s_preprocessStages = ["decode", "preprocess"];
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

        _textTolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cpu");
        _imageTolerances = EmbeddingGemma2Tolerances.Load("image-fp32-cpu");
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

        if (paths.Contains("image"))
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
            var settings = _context.SessionSettings();
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
            ["reference"] = string.Join("; ", new[] { _encoder is null ? null : EmbeddingGemma2Driver.TextReference, _multimodal is null ? null : EmbeddingGemma2Driver.ImageReference }.OfType<string>()),
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

        if (_multimodal is not null)
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
        return path == "text"
            ? TextSeries("cold-start", "short", request.Length, 1, 1, 0, [request.Select(TextRow).ToArray()])
            : ImageSeries("cold-start", "photo", request.Length, 1, 1, 0, [request.Select(ImageRowOf).ToArray()], preprocessOnly: false);
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        _multimodal?.Dispose();
    }

    private BenchSeries Series(string name, string path, string input, int batch, int concurrency, int samples, int warmup, System.Text.Json.JsonElement raw)
    {
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
