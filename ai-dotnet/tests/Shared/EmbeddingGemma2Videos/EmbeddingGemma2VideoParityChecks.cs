using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// One <see cref="EmbeddingGemma2MultimodalEmbedder"/> over the real graphs per test class (SHA-256 enforced), for the
/// video profiles. When a graph, the video set or FFmpeg is absent the fixture holds a skip reason.
/// </summary>
public abstract class EmbeddingGemma2VideoModelFixture : IDisposable
{
    private readonly EmbeddingGemma2MultimodalEmbedder? _embedder;
    private readonly Lock _gate = new();
    private Dictionary<string, float[]>? _singles;

    protected EmbeddingGemma2VideoModelFixture(OnnxSessionOptions sessionOptions, EmbeddingGemma2MultimodalOptions? options = null, Func<string?>? gate = null)
    {
        if (gate?.Invoke() is { } gateReason)
        {
            SkipReason = gateReason;
            return;
        }

        if ((EmbeddingGemma2VideoTestAssets.VideosSkipReason ?? EmbeddingGemma2VideoTestAssets.FFmpegSkipReason) is { } assetsReason)
        {
            SkipReason = assetsReason;
            return;
        }

        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        Files = files;
        SessionOptions = sessionOptions with { CapturePlacement = true };
        Options = (options ?? new EmbeddingGemma2MultimodalOptions()) with { Session = SessionOptions };
        _embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, Options);
    }

    public string? SkipReason { get; }

    public EmbeddingGemma2ImageModelFiles? Files { get; }

    public OnnxSessionOptions SessionOptions { get; } = new();

    public EmbeddingGemma2MultimodalOptions Options { get; } = new();

    /// <summary>Video soft tokens and token embeddings of the single-row runs.</summary>
    internal VideoTensorCollector Collector { get; } = new();

    /// <summary>Wall time per row of the single-row runs.</summary>
    internal Dictionary<string, TimeSpan> Timings { get; } = new(StringComparer.Ordinal);

    public EmbeddingGemma2MultimodalEmbedder Embedder
    {
        get
        {
            Assert.SkipWhen(_embedder is null, SkipReason ?? "model unavailable");
            return _embedder!;
        }
    }

    /// <summary>Every default-contract fixture row embedded alone (computed once per fixture): row id to embedding.</summary>
    internal Dictionary<string, float[]> Singles()
    {
        lock (_gate)
        {
            if (_singles is not null)
            {
                return _singles;
            }

            var embedder = Embedder;
            var result = new Dictionary<string, float[]>(StringComparer.Ordinal);
            foreach (var row in EmbeddingGemma2VideoTestAssets.Rows.Where(r => r.InDefaultContract))
            {
                var input = EmbeddingGemma2VideoTestAssets.BuildInput(row);
                var destination = new float[embedder.Dimension];
                var watch = System.Diagnostics.Stopwatch.StartNew();
                Collector.Row = row.Id;
                embedder.EmbedCore([input], row.PackagePrompt, destination, default, Collector, TestContext.Current.CancellationToken);
                Timings[row.Id] = watch.Elapsed;
                result[row.Id] = destination;
            }

            _singles = result;
            return result;
        }
    }

    public void Dispose()
    {
        _embedder?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Video soft tokens and token embeddings by row, for the row-relative checks.</summary>
internal sealed class VideoTensorCollector : IEmbeddingGemma2Diagnostics
{
    public string Row { get; set; } = string.Empty;

    public Dictionary<(string Row, int Video), float[]> Soft { get; } = [];

    public Dictionary<string, float[]> Tokens { get; } = new(StringComparer.Ordinal);

    public void OnSoftTokens(int input, int image, float[] softTokens)
    {
    }

    public void OnVideoSoftTokens(int input, int video, float[] softTokens) => Soft[(Row, video)] = softTokens;

    public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) => Tokens[Row] = tokenEmbeddings.ToArray();
}

/// <summary>Video parity checks against the accepted fixtures, shared by the CPU and CUDA test projects.</summary>
internal static class EmbeddingGemma2VideoParityChecks
{
    /// <summary>Scene of each text query (the reference's retrieval sanity: each ranks a video of its scene first).</summary>
    public static readonly IReadOnlyDictionary<string, string> QueryScene = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["q-bars"] = "bars",
        ["q-shapes"] = "shapes",
        ["q-text"] = "text",
    };

    /// <summary>Every default-contract row alone vs the reference embedding; soft tokens and token embeddings row-relative when the tensors are available.</summary>
    public static void RowsAgainstReference(EmbeddingGemma2VideoModelFixture fixture, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var singles = fixture.Singles();
        var rows = EmbeddingGemma2VideoTestAssets.Rows.Where(r => r.InDefaultContract).ToList();
        Assert.Equal(35, rows.Count);
        var all = new EmbeddingGemma2ErrorStats();
        var videoRows = new EmbeddingGemma2ErrorStats();
        var textRows = new EmbeddingGemma2ErrorStats();
        foreach (var row in rows)
        {
            all.Add(row.Id, singles[row.Id], row.Embedding);
            (row.Videos.Count > 0 ? videoRows : textRows).Add(row.Id, singles[row.Id], row.Embedding);
        }

        output.WriteLine($"{rows.Count} rows (default contract; {EmbeddingGemma2VideoTestAssets.NoMetadataRowId} needs the opt-in) vs reference: {all}");
        output.WriteLine($"  video and interleaved rows: {videoRows}");
        output.WriteLine($"  text-only rows (released text graph): {textRows}");
        output.WriteLine($"  time per row: {string.Join(", ", fixture.Timings.OrderByDescending(t => t.Value).Take(5).Select(t => $"{t.Key} {t.Value.TotalSeconds:F1} s"))}; total {fixture.Timings.Values.Sum(t => t.TotalSeconds):F0} s");
        output.WriteLine($"bounds: embedding_max_abs {tolerances.EmbeddingMaxAbs:E1}, min_cosine {tolerances.MinCosine} ({tolerances.Source})");
        Assert.True(tolerances.NanOrInfFails);
        Assert.False(all.NonFinite, "NaN or Inf in embeddings");
        Assert.True(all.Max <= tolerances.EmbeddingMaxAbs, $"embedding max abs {all.Max:E3} > {tolerances.EmbeddingMaxAbs:E1} ({all.WorstKey})");
        Assert.True(all.MinCosine >= tolerances.MinCosine, $"min cosine {all.MinCosine:F12} < {tolerances.MinCosine}");

        var (tensors, reason) = EmbeddingGemma2VideoTestAssets.FindTensors();
        if (tensors is null)
        {
            output.WriteLine("soft tokens / token embeddings: not compared: " + reason);
            return;
        }

        double worstSoft = 0, worstTokens = 0;
        string worstSoftKey = string.Empty, worstTokensKey = string.Empty;
        int softCount = 0, tokenCount = 0;
        foreach (var row in rows.Where(r => r.Videos.Count > 0))
        {
            for (int k = 0; k < row.Videos.Count; k++)
            {
                var reference = EmbeddingGemma2VideoTestAssets.ReadReferenceTensor(tensors, row.Id, $"video{k}_soft_tokens");
                var actual = fixture.Collector.Soft[(row.Id, k)];
                Assert.False(actual.Any(v => !float.IsFinite(v)), $"{row.Id}/video{k}: NaN or Inf in soft tokens");
                double rel = EmbeddingGemma2ImageParityChecks.RowRelative(actual, reference);
                softCount++;
                if (rel > worstSoft)
                {
                    (worstSoft, worstSoftKey) = (rel, $"{row.Id}/video{k}");
                }
            }

            var refTokens = EmbeddingGemma2VideoTestAssets.ReadReferenceTensor(tensors, row.Id, "token_embeddings");
            double relTokens = EmbeddingGemma2ImageParityChecks.RowRelative(fixture.Collector.Tokens[row.Id], refTokens);
            tokenCount++;
            if (relTokens > worstTokens)
            {
                (worstTokens, worstTokensKey) = (relTokens, row.Id);
            }
        }

        output.WriteLine($"soft tokens row-relative worst {worstSoft:E2} ({worstSoftKey}) over {softCount} videos; bound {tolerances.SoftTokensRowRelative:E1}");
        output.WriteLine($"token embeddings row-relative worst {worstTokens:E2} ({worstTokensKey}) over {tokenCount} rows; bound {tolerances.TokenEmbeddingsRowRelative:E1}");
        Assert.True(worstSoft <= tolerances.SoftTokensRowRelative, $"soft tokens row-relative {worstSoft:E3} ({worstSoftKey})");
        Assert.True(worstTokens <= tolerances.TokenEmbeddingsRowRelative, $"token embeddings row-relative {worstTokens:E3} ({worstTokensKey})");
    }

    /// <summary>Mixed batches (text, short and long videos, interleaved, prompted) vs single rows.</summary>
    public static void BatchesAgainstSingles(EmbeddingGemma2VideoModelFixture fixture, EmbeddingGemma2MultimodalOptions batchOptions, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output, string label)
    {
        var singles = fixture.Singles();
        var files = fixture.Files!;
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, batchOptions with { Session = fixture.SessionOptions, VerifyModelSha256 = false });
        string[] ids = ["q-bars", "h264-1s-320x240-30fps", "h264-60fps-480x270-3s", "mm-caption-then-video", "h264-1fps-64x48-8s", "vp9-odd-321x179-30fps-3s", "q-cat", "mm-two-videos", "vid-query-prompt", "q-shapes"];
        var rows = ids.Select(id => EmbeddingGemma2VideoTestAssets.Rows.Single(r => r.Id == id)).ToList();
        var stats = new EmbeddingGemma2ErrorStats();
        foreach (var group in rows.GroupBy(r => r.Prompt))
        {
            var list = group.ToList();
            var inputs = list.Select(EmbeddingGemma2VideoTestAssets.BuildInput).ToList();
            var batch = embedder.EmbedBatch(inputs, list[0].PackagePrompt, TestContext.Current.CancellationToken);
            for (int i = 0; i < list.Count; i++)
            {
                stats.Add(list[i].Id, batch[i], singles[list[i].Id]);
            }
        }

        output.WriteLine($"{label}: {rows.Count} rows in batches vs single rows: {stats}; bound {tolerances.PaddedBatchVsSingleMaxAbs:E1}");
        Assert.False(stats.NonFinite);
        Assert.True(stats.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"{label}: batch vs single {stats.Max:E3} ({stats.WorstKey})");
    }

    /// <summary>Upstream's metadata-less behaviour on opt-in reproduces its row; and differs from the file (reported).</summary>
    public static void FramesWithoutTimingOnOptIn(EmbeddingGemma2VideoModelFixture fixture, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var singles = fixture.Singles();
        var row = EmbeddingGemma2VideoTestAssets.Rows.Single(r => r.Id == EmbeddingGemma2VideoTestAssets.NoMetadataRowId);
        var input = EmbeddingGemma2VideoTestAssets.BuildInput(row);
        Assert.Throws<EmbeddingGemma2VideoTimingRequiredException>(() => fixture.Embedder.Embed(input, TestContext.Current.CancellationToken));
        using var optIn = EmbeddingGemma2MultimodalEmbedder.Create(fixture.Files!, fixture.Options with { AllowVideosWithoutTiming = true, VerifyModelSha256 = false });
        var embedding = optIn.Embed(input, TestContext.Current.CancellationToken);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add(row.Id, embedding, row.Embedding);
        var vsFile = new EmbeddingGemma2ErrorStats();
        vsFile.Add("vs file", embedding, singles["h264-bars-640x360-30fps-5s"]);
        output.WriteLine($"{row.Id} with AllowVideosWithoutTiming vs reference: {stats}; vs the same file sampled at 1 fps: cosine {vsFile.MinCosine:F4} (upstream quirk, documented)");
        Assert.True(stats.Max <= tolerances.EmbeddingMaxAbs, $"{row.Id}: {stats.Max:E3}");
        Assert.True(stats.MinCosine >= tolerances.MinCosine);
        Assert.True(vsFile.MinCosine < 0.99);
    }

    /// <summary>Each scene query (query prompt, released text graph) ranks a video of its scene first among the 22.</summary>
    public static void RetrievalSanity(EmbeddingGemma2VideoModelFixture fixture, ITestOutputHelper output)
    {
        var singles = fixture.Singles();
        var videos = EmbeddingGemma2VideoTestAssets.Manifest;
        foreach (var (query, scene) in QueryScene)
        {
            var q = singles[query];
            var ranked = videos.Select(v => (v, Dot(q, singles[v.Id]))).OrderByDescending(x => x.Item2).ToList();
            output.WriteLine($"{query}: top {ranked[0].v.Id} ({ranked[0].v.Scene}) {ranked[0].Item2:F4}, runner-up {ranked[1].v.Id} {ranked[1].Item2:F4}");
            Assert.Equal(scene, ranked[0].v.Scene);
        }
    }

    /// <summary>Text-only inputs through the multimodal embedder equal the text embedder (released graph) bit for bit.</summary>
    public static void TextOnlyEqualsTheTextEmbedder(EmbeddingGemma2VideoModelFixture fixture)
    {
        _ = fixture.Embedder;
        using var encoder = EmbeddingGemma2Encoder.Create(fixture.Files!.Text.ModelPath, new EmbeddingGemma2EncoderOptions { Session = fixture.SessionOptions });
        var text = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2TestAssets.Tokenizer, encoder);
        var singles = fixture.Singles();
        foreach (var row in EmbeddingGemma2VideoTestAssets.Rows.Where(r => r.Kind == "text"))
        {
            Assert.Equal(text.Embed(row.Text!, row.PackagePrompt, TestContext.Current.CancellationToken), singles[row.Id]);
        }
    }

    /// <summary>A Matryoshka size of a video embedding equals the truncated and renormalized 768-dimensional one.</summary>
    public static void Matryoshka(EmbeddingGemma2VideoModelFixture fixture, EmbeddingGemma2Tolerances tolerances)
    {
        _ = fixture.Embedder;
        using var small = EmbeddingGemma2MultimodalEmbedder.Create(fixture.Files!, fixture.Options with { Dimension = 128, VerifyModelSha256 = false });
        const string Id = "h264-60fps-480x270-3s";
        var e128 = small.Embed(EmbeddingGemma2VideoTestAssets.Decode(Id).Video, TestContext.Current.CancellationToken);
        var expected = EmbeddingGemma2Encoder.Truncate(fixture.Singles()[Id], 128);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add(Id + "@128", e128, expected);
        Assert.Equal(128, e128.Length);
        Assert.True(stats.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"Matryoshka 128 {stats.Max:E3}");
    }

    private static double Dot(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++)
        {
            s += (double)a[i] * b[i];
        }

        return s;
    }
}
