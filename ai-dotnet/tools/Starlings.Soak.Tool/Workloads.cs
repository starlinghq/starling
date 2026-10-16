using System.Text;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.Gliner2;
using Starlings.Models.Qwen35;
using Starlings.Runtime.Onnx;

/// <summary>
/// The result of one request, reduced to what the soak compares across requests and cycles: a discrete part that must
/// be equal (token ids, entity spans) and a continuous part compared within <see cref="SoakResult.ValueTolerance"/>.
/// This is a state-leakage check (the same input gives the same output after cancellations, concurrency and reloads),
/// not a fidelity check; fidelity is gates G1 and G2.
/// </summary>
internal sealed record SoakResult(string Discrete, float[]? Values)
{
    public const double ValueTolerance = 1e-4;

    public string? Difference(SoakResult reference)
    {
        if (!string.Equals(Discrete, reference.Discrete, StringComparison.Ordinal))
        {
            return $"discrete output differs: '{Truncate(Discrete)}' vs reference '{Truncate(reference.Discrete)}'";
        }

        if (Values is null || reference.Values is null)
        {
            return Values is null == reference.Values is null ? null : "value presence differs";
        }

        if (Values.Length != reference.Values.Length)
        {
            return $"length {Values.Length} vs reference {reference.Values.Length}";
        }

        double max = 0;
        for (int i = 0; i < Values.Length; i++)
        {
            double d = Math.Abs(Values[i] - reference.Values[i]);
            if (!(d <= max))
            {
                max = double.IsNaN(d) ? double.PositiveInfinity : d;
            }
        }

        return max <= ValueTolerance ? null : $"max abs difference {max:E3} from the reference exceeds {ValueTolerance:E0}";
    }

    private static string Truncate(string s) => s.Length <= 80 ? s : s[..80] + "...";
}

/// <summary>One loaded model instance: requests run on it concurrently; <see cref="IDisposable.Dispose"/> unloads it.</summary>
internal interface ISoakInstance : IDisposable
{
    IReadOnlyList<string> SessionProviders { get; }

    /// <summary>The native memory settings in effect for the ONNX Runtime sessions (CPU arena, heap trim on release); null for other runtimes.</summary>
    string? SessionMemory => null;

    SoakResult Run(int input, CancellationToken cancellationToken);

    /// <summary>A long request used for the timed-out (deadline) requests; its result is discarded.</summary>
    void RunLong(CancellationToken cancellationToken);
}

/// <summary>A model profile the soak can load repeatedly.</summary>
internal abstract class SoakWorkload
{
    public abstract string Name { get; }

    public abstract int DefaultRequestsPerCycle { get; }

    public static int InputCount => SoakTexts.Short.Length;

    public abstract string Profile(SoakOptions options);

    /// <summary>Finds and verifies (SHA-256) the files offline. Returns null when available, else the reason.</summary>
    public abstract string? Locate(SoakOptions options);

    public abstract JsonObject Artifacts();

    public abstract ISoakInstance Load(SoakOptions options);

    protected static ModelAcquisitionOptions Acquisition => new() { Verification = CacheVerification.Sha256 };

    internal static string Memory(OnnxSessionDiagnostics d) =>
        $"cpu_memory_arena={(d.CpuMemoryArena ? "on" : "off")}, trim_native_heap_on_release={(d.TrimNativeHeapOnRelease ? "on" : "off")}, dedicated_run_threads={(d.DedicatedRunThreads ? "on" : "off")}";

    protected static OnnxSessionOptions Session(SoakOptions options, bool computeSha) => new()
    {
        Provider = options.Cuda ? OnnxExecutionProvider.Cuda : OnnxExecutionProvider.Cpu,
        IntraOpNumThreads = options.Threads,
        ComputeModelSha256 = computeSha,
        CpuMemoryArena = options.CpuArena,
        DedicatedRunThreads = options.RunThreads,
    };

    protected static JsonObject Lock(ModelAssetLock assetLock) => new()
    {
        ["name"] = assetLock.Name,
        ["repository"] = assetLock.Repository,
        ["revision"] = assetLock.Revision,
        ["files"] = new JsonArray(assetLock.Files.Select(f => (JsonNode)new JsonObject { ["path"] = f.Path, ["size"] = f.Size, ["sha256"] = f.Sha256 }).ToArray()),
        ["verified"] = "sha256 (offline, at startup)",
    };

    protected static string? Snapshot(string variable) => Environment.GetEnvironmentVariable(variable) is { Length: > 0 } s ? s : null;

    public static IReadOnlyList<SoakWorkload> All { get; } =
    [
        new E5Workload(), new Gliner2Workload(), new EmbeddingGemma2TextWorkload(), new EmbeddingGemma2ImageWorkload(), new EmbeddingGemma2VideoWorkload(), new Qwen35Workload(), new Qwen35Workload(Qwen35Variant.Fp16x),
    ];
}

/// <summary>Fixed request inputs (development text, written for the soak; not task data).</summary>
internal static class SoakTexts
{
    public static readonly string[] Short =
    [
        "Ada Lovelace met Charles Babbage in London in June 1833.",
        "The committee in Geneva approved the budget on 4 March 2025 after a long debate.",
        "Marie Curie worked at the University of Paris and won two Nobel Prizes.",
        "Der Zug nach München fährt um 14 Uhr vom Hauptbahnhof Berlin ab.",
        "El museo del Prado en Madrid abre a las nueve de la mañana.",
        "東京タワーは1958年に完成しました。",
        "Satya Nadella spoke at the Microsoft Build conference in Seattle.",
        "A short one.",
        "Researchers at CERN in Switzerland announced new results from the Large Hadron Collider on Tuesday.",
        "Jean Dupont, directeur de la société Lumière, a visité Lyon et Marseille en avril.",
        "The river Thames flows through Oxford, Reading and London before reaching the North Sea.",
        "Квантовые компьютеры обещают ускорить некоторые вычисления.",
    ];

    /// <summary>About 230 words: below the GLiNER2 512-token limit, long enough that a batch takes visible time.</summary>
    public static readonly string Long = string.Join(' ', Enumerable.Repeat(
        "In October the board of Northwind Traders met in Copenhagen with representatives of the city council, " +
        "the harbour authority and two universities to discuss a new logistics centre near the old shipyard. " +
        "Anna Larsen presented the plan, and Peter Holm asked about the cost and the timetable.", 4));

    /// <summary>About 1,500 tokens for EmbeddingGemma 2.</summary>
    public static readonly string VeryLong = string.Join(' ', Enumerable.Repeat(Long, 7));

    public static string Discrete(ReadOnlySpan<float> values)
    {
        // Embeddings have no discrete part; the sign pattern of the first 16 values guards against gross corruption.
        var sb = new StringBuilder(16);
        foreach (float v in values[..Math.Min(16, values.Length)])
        {
            sb.Append(v >= 0 ? '+' : '-');
        }

        return sb.ToString();
    }
}

internal sealed class E5Workload : SoakWorkload
{
    private E5ModelFiles? _files;

    public override string Name => "e5";

    public override int DefaultRequestsPerCycle => 200;

    public override string Profile(SoakOptions options) => "multilingual-e5-small/" + (options.Cuda ? "onnx-fp32-cuda" : "onnx-fp32-cpu");

    public override string? Locate(SoakOptions options)
    {
        try
        {
            _files = Snapshot("STARLING_E5_SNAPSHOT") is { } dir ? E5ModelAssets.OpenDirectory(dir) : E5ModelAssets.OpenLocal(Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(E5ModelAssets.Lock);

    public override ISoakInstance Load(SoakOptions options)
    {
        var tokenizer = E5Tokenizer.FromFile(_files!.TokenizerPath, E5ModelAssets.TokenizerSha256);
        var encoder = E5Encoder.Create(_files.ModelPath, new E5EncoderOptions { Session = Session(options, computeSha: false) });
        return new Instance(new E5TextEmbedder(tokenizer, encoder), encoder);
    }

    private sealed class Instance(E5TextEmbedder embedder, E5Encoder encoder) : ISoakInstance
    {
        private static readonly string[] LongBatch = Enumerable.Repeat("passage: " + SoakTexts.Long, 16).ToArray();

        public IReadOnlyList<string> SessionProviders => encoder.Session.Diagnostics.SessionProviders;

        public string? SessionMemory => SoakWorkload.Memory(encoder.Session.Diagnostics);

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            var v = embedder.Embed("query: " + SoakTexts.Short[input], cancellationToken);
            return new SoakResult(SoakTexts.Discrete(v), v);
        }

        public void RunLong(CancellationToken cancellationToken) => embedder.EmbedBatch(LongBatch, cancellationToken);

        public void Dispose() => encoder.Dispose();
    }
}

internal sealed class Gliner2Workload : SoakWorkload
{
    private static readonly string[] Labels = ["person", "organization", "location", "date"];
    private Gliner2ModelFiles? _files;

    public override string Name => "gliner2";

    public override int DefaultRequestsPerCycle => 100;

    public override string Profile(SoakOptions options) => "gliner2-base-v1/" + (options.Cuda ? "onnx-fp32-cuda" : "onnx-fp32-cpu");

    public override string? Locate(SoakOptions options)
    {
        try
        {
            _files = Snapshot("STARLING_GLINER2_SNAPSHOT") is { } dir ? Gliner2ModelAssets.OpenDirectory(dir) : Gliner2ModelAssets.OpenLocal(Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(Gliner2ModelAssets.Lock);

    public override ISoakInstance Load(SoakOptions options)
    {
        // The startup check verified the graph's SHA-256; skip re-hashing it on every cycle.
        var extractor = Gliner2EntityExtractor.Create(_files!.ModelPath, _files.TokenizerPath, new Gliner2ExtractorOptions { Session = Session(options, computeSha: false) });
        return new Instance(extractor);
    }

    private sealed class Instance(Gliner2EntityExtractor extractor) : ISoakInstance
    {
        private static readonly string[] LongBatch = Enumerable.Repeat(SoakTexts.Long, 16).ToArray();

        public IReadOnlyList<string> SessionProviders => extractor.Session.Diagnostics.SessionProviders;

        public string? SessionMemory => SoakWorkload.Memory(extractor.Session.Diagnostics);

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            var result = extractor.Extract(SoakTexts.Short[input], Labels, cancellationToken: cancellationToken);
            string spans = string.Join(";", result.Entities.Select(e => $"{e.Label}:{e.Start}-{e.End}"));
            return new SoakResult(spans, result.Entities.Select(e => e.Confidence).ToArray());
        }

        public void RunLong(CancellationToken cancellationToken) => extractor.ExtractBatch(LongBatch, Labels, cancellationToken: cancellationToken);

        public void Dispose() => extractor.Dispose();
    }
}

internal sealed class EmbeddingGemma2TextWorkload : SoakWorkload
{
    private EmbeddingGemma2ModelFiles? _files;

    public override string Name => "embeddinggemma2";

    public override int DefaultRequestsPerCycle => 60;

    public override string Profile(SoakOptions options) => "embeddinggemma-2/" + (options.Cuda ? "onnx-fp32-cuda" : "onnx-fp32-cpu");

    public override string? Locate(SoakOptions options)
    {
        try
        {
            _files = Snapshot("STARLING_EMBEDDINGGEMMA2_SNAPSHOT") is { } dir ? EmbeddingGemma2ModelAssets.OpenDirectory(dir) : EmbeddingGemma2ModelAssets.OpenLocal(Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(EmbeddingGemma2ModelAssets.Lock);

    public override ISoakInstance Load(SoakOptions options)
    {
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(_files!.TokenizerPath);
        var encoder = EmbeddingGemma2Encoder.Create(_files.ModelPath, new EmbeddingGemma2EncoderOptions { Session = Session(options, computeSha: false) });
        return new Instance(new EmbeddingGemma2TextEmbedder(tokenizer, encoder), encoder);
    }

    private sealed class Instance(EmbeddingGemma2TextEmbedder embedder, EmbeddingGemma2Encoder encoder) : ISoakInstance
    {
        private static readonly string[] LongBatch = Enumerable.Repeat(SoakTexts.VeryLong, 4).ToArray();

        public IReadOnlyList<string> SessionProviders => encoder.Session.Diagnostics.SessionProviders;

        public string? SessionMemory => SoakWorkload.Memory(encoder.Session.Diagnostics);

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            var v = embedder.Embed(SoakTexts.Short[input], cancellationToken);
            return new SoakResult(SoakTexts.Discrete(v), v);
        }

        public void RunLong(CancellationToken cancellationToken) => embedder.EmbedBatch(LongBatch, cancellationToken);

        public void Dispose() => encoder.Dispose();
    }
}

internal sealed class EmbeddingGemma2ImageWorkload : SoakWorkload
{
    private EmbeddingGemma2ImageModelFiles? _files;

    public override string Name => "embeddinggemma2-image";

    // A CPU image request takes seconds (vision tower at 280 soft tokens); keep cycles short enough for load/unload.
    public override int DefaultRequestsPerCycle => 5;

    public override string Profile(SoakOptions options) => "embeddinggemma-2/" + (options.Cuda ? "image-fp32-cuda" : "image-fp32-cpu");

    public override string? Locate(SoakOptions options)
    {
        try
        {
            _files = Snapshot("STARLING_EMBEDDINGGEMMA2_SNAPSHOT") is { } dir
                ? EmbeddingGemma2ModelAssets.OpenImageDirectory(dir)
                : EmbeddingGemma2ModelAssets.OpenLocalImageAssets(Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(EmbeddingGemma2ModelAssets.ImageLock);

    public override ISoakInstance Load(SoakOptions options) =>
        new Instance(EmbeddingGemma2MultimodalEmbedder.Create(_files!, new EmbeddingGemma2MultimodalOptions
        {
            Session = Session(options, computeSha: false),
            VerifyModelSha256 = false,
        }));

    /// <summary>Deterministic synthetic RGB images (gradients and stripes) of a few sizes; no decoder needed.</summary>
    private static EmbeddingGemma2Image Image(int input)
    {
        int w = 96 + (input * 37 % 160);
        int h = 80 + (input * 53 % 144);
        var rgb = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int o = ((y * w) + x) * 3;
                rgb[o] = (byte)((x * 255 / w) ^ (input * 17));
                rgb[o + 1] = (byte)(y * 255 / h);
                rgb[o + 2] = (byte)(((x / (4 + input)) + (y / 8)) % 2 == 0 ? 200 : 40);
            }
        }

        return EmbeddingGemma2Image.FromRgb24(w, h, rgb);
    }

    private sealed class Instance(EmbeddingGemma2MultimodalEmbedder embedder) : ISoakInstance
    {
        private static readonly EmbeddingGemma2Image[] Images = Enumerable.Range(0, SoakTexts.Short.Length).Select(Image).ToArray();
        private static readonly EmbeddingGemma2Image[] LongBatch = Enumerable.Range(0, 6).Select(i => Image(i + 3)).ToArray();

        public IReadOnlyList<string> SessionProviders => embedder.VisionSession.Diagnostics.SessionProviders;

        public string? SessionMemory => $"vision: {SoakWorkload.Memory(embedder.VisionSession.Diagnostics)}; multimodal text: {SoakWorkload.Memory(embedder.MultimodalSession.Diagnostics)}";

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            var v = embedder.Embed(Images[input], cancellationToken);
            return new SoakResult(SoakTexts.Discrete(v), v);
        }

        public void RunLong(CancellationToken cancellationToken) => embedder.EmbedBatch(LongBatch, cancellationToken);

        public void Dispose() => embedder.Dispose();
    }
}

internal sealed class EmbeddingGemma2VideoWorkload : SoakWorkload
{
    private EmbeddingGemma2ImageModelFiles? _files;

    public override string Name => "embeddinggemma2-video";

    // A CPU video request embeds 2 or 3 sampled frames at 140 soft tokens each (seconds per request on CPU); keep
    // cycles short enough for load/unload.
    public override int DefaultRequestsPerCycle => 4;

    public override string Profile(SoakOptions options) => "embeddinggemma-2/" + (options.Cuda ? "video-fp32-cuda" : "video-fp32-cpu");

    public override string? Locate(SoakOptions options)
    {
        try
        {
            // Videos run on the image path's graphs (vision and multimodal text); no video decoder is needed.
            _files = Snapshot("STARLING_EMBEDDINGGEMMA2_SNAPSHOT") is { } dir
                ? EmbeddingGemma2ModelAssets.OpenImageDirectory(dir)
                : EmbeddingGemma2ModelAssets.OpenLocalImageAssets(Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(EmbeddingGemma2ModelAssets.ImageLock);

    public override ISoakInstance Load(SoakOptions options) =>
        new Instance(EmbeddingGemma2MultimodalEmbedder.Create(_files!, new EmbeddingGemma2MultimodalOptions
        {
            Session = Session(options, computeSha: false),
            VerifyModelSha256 = false,
        }));

    /// <summary>
    /// One decoded frame of a deterministic synthetic clip (a gradient with a square moving across it and stripes that
    /// depend on the clip), so consecutive frames differ.
    /// </summary>
    internal static EmbeddingGemma2Image Frame(int input, int index, int width, int height)
    {
        var rgb = new byte[width * height * 3];
        int side = Math.Max(4, height / 4);
        int x0 = index * 7 % Math.Max(1, width - side);
        int y0 = ((index * 5) + (input * 3)) % Math.Max(1, height - side);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = ((y * width) + x) * 3;
                bool square = x >= x0 && x < x0 + side && y >= y0 && y < y0 + side;
                rgb[o] = square ? (byte)250 : (byte)((x * 255 / width) ^ (input * 17));
                rgb[o + 1] = square ? (byte)250 : (byte)(y * 255 / height);
                rgb[o + 2] = (byte)(((x / (4 + input)) + (y / 8) + index) % 2 == 0 ? 200 : 40);
            }
        }

        return EmbeddingGemma2Image.FromRgb24(width, height, rgb);
    }

    /// <summary>
    /// Every decoded frame of a synthetic clip with its timing, as a decoder returns it: <paramref name="seconds"/> s at
    /// 4 frames per second, so the package samples one frame per second (upstream's sampling).
    /// </summary>
    internal static EmbeddingGemma2Video Clip(int input, int seconds)
    {
        const int fps = 4;
        int width = 96 + (input * 37 % 160);
        int height = 80 + (input * 53 % 144);
        var frames = Enumerable.Range(0, fps * seconds).Select(i => Frame(input, i, width, height)).ToArray();
        return EmbeddingGemma2Video.FromDecodedFrames(frames, fps, seconds);
    }

    private sealed class Instance(EmbeddingGemma2MultimodalEmbedder embedder) : ISoakInstance
    {
        // 2 or 3 sampled frames per request; the long request is a batch of two 6-frame clips.
        private static readonly EmbeddingGemma2Video[] Videos = Enumerable.Range(0, SoakTexts.Short.Length).Select(i => Clip(i, 2 + (i % 2))).ToArray();
        private static readonly EmbeddingGemma2MultimodalInput[] LongBatch = [EmbeddingGemma2MultimodalInput.FromVideo(Clip(3, 6)), EmbeddingGemma2MultimodalInput.FromVideo(Clip(4, 6))];

        public IReadOnlyList<string> SessionProviders => embedder.VisionSession.Diagnostics.SessionProviders;

        public string? SessionMemory => $"vision: {SoakWorkload.Memory(embedder.VisionSession.Diagnostics)}; multimodal text: {SoakWorkload.Memory(embedder.MultimodalSession.Diagnostics)}";

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            var v = embedder.Embed(Videos[input], cancellationToken);
            return new SoakResult(SoakTexts.Discrete(v), v);
        }

        public void RunLong(CancellationToken cancellationToken) => embedder.EmbedBatch(LongBatch, EmbeddingGemma2Prompt.None, cancellationToken);

        public void Dispose() => embedder.Dispose();
    }
}

/// <summary>Qwen3.5 chat: <c>qwen35</c> is the fp32x graph (CPU or CUDA), <c>qwen35-fp16x</c> the fp16x graph (CUDA only).</summary>
internal sealed class Qwen35Workload(Qwen35Variant variant = Qwen35Variant.Fp32x) : SoakWorkload
{
    private Qwen35ModelFiles? _files;

    public override string Name => variant == Qwen35Variant.Fp32x ? "qwen35" : "qwen35-fp16x";

    public override int DefaultRequestsPerCycle => 12;

    public override string Profile(SoakOptions options) => variant == Qwen35Variant.Fp32x
        ? "qwen3.5-0.8b/" + (options.Cuda ? "ortgenai-fp32x-cuda" : "ortgenai-fp32x-cpu")
        : "qwen3.5-0.8b/ortgenai-fp16x-cuda";

    public override string? Locate(SoakOptions options)
    {
        if (variant == Qwen35Variant.Fp16x && !options.Cuda)
        {
            return "the fp16x graph is a CUDA profile (ortgenai-fp16x-cuda); run with --cuda on a -p:UseCuda=true build";
        }

        try
        {
            _files = Snapshot("STARLING_QWEN35_SNAPSHOT") is { } dir
                ? Qwen35ModelAssets.OpenDirectory(dir, variant)
                : Qwen35ModelAssets.OpenLocal(variant, Acquisition);
            return null;
        }
        catch (ModelAcquisitionException ex)
        {
            return ex.Message;
        }
    }

    public override JsonObject Artifacts() => Lock(Qwen35ModelAssets.GetLock(variant));

    public override ISoakInstance Load(SoakOptions options) =>
        new Instance(Qwen35ChatModel.Load(_files!, new Qwen35ModelOptions
        {
            Provider = options.Cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu,
            IntraOpThreads = options.Threads,
        }));

    private sealed class Instance(Qwen35ChatModel model) : ISoakInstance
    {
        public IReadOnlyList<string> SessionProviders => [model.Info.Provider.ToString()];

        public SoakResult Run(int input, CancellationToken cancellationToken)
        {
            // Greedy decoding: the same prompt gives the same ids on one provider.
            var r = model.Generate([Qwen35Message.User("Answer in one sentence: " + SoakTexts.Short[input])], new Qwen35ChatOptions { MaxNewTokens = 12 }, cancellationToken);
            return new SoakResult(string.Join(',', r.TokenIds), null);
        }

        public void RunLong(CancellationToken cancellationToken)
        {
            foreach (var update in model.GenerateStreaming(
                [Qwen35Message.User("Write a long story about this: " + SoakTexts.Long)], new Qwen35ChatOptions { MaxNewTokens = 256 }, cancellationToken))
            {
                _ = update;
            }
        }

        public void Dispose() => model.Dispose();
    }
}
