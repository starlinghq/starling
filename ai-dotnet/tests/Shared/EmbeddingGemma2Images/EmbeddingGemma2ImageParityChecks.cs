using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// One <see cref="EmbeddingGemma2MultimodalEmbedder"/> over the real image graphs per test class (SHA-256 enforced). When
/// a graph or the image set is absent the fixture holds a skip reason.
/// </summary>
public abstract class EmbeddingGemma2ImageModelFixture : IDisposable
{
    private readonly EmbeddingGemma2MultimodalEmbedder? _embedder;
    private readonly Lock _gate = new();
    private Dictionary<string, float[]>? _singles;

    protected EmbeddingGemma2ImageModelFixture(OnnxSessionOptions sessionOptions, Func<string?>? gate = null)
    {
        string? gateReason = gate?.Invoke();
        if (gateReason is not null)
        {
            SkipReason = gateReason;
            return;
        }

        if (EmbeddingGemma2ImageTestAssets.ImagesSkipReason is { } imagesReason)
        {
            SkipReason = imagesReason;
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
        _embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = SessionOptions });
    }

    public string? SkipReason { get; }

    public EmbeddingGemma2ImageModelFiles? Files { get; }

    public OnnxSessionOptions SessionOptions { get; } = new();

    /// <summary>Soft tokens and token embeddings of the single-row image runs.</summary>
    internal TensorCollector Collector { get; } = new();

    public EmbeddingGemma2MultimodalEmbedder Embedder
    {
        get
        {
            Assert.SkipWhen(_embedder is null, SkipReason ?? "model unavailable");
            return _embedder!;
        }
    }

    /// <summary>
    /// Every in-contract fixture row embedded alone (computed once per fixture): row id to embedding. Soft tokens and token
    /// embeddings of the image rows are kept in <see cref="Collector"/>.
    /// </summary>
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
            foreach (var row in EmbeddingGemma2ImageTestAssets.Rows.Where(r => r.InContract))
            {
                var input = EmbeddingGemma2ImageTestAssets.BuildInput(row, EmbeddingGemma2ImageTestAssets.Decode);
                var destination = new float[embedder.Dimension];
                embedder.EmbedCore([input], row.PackagePrompt, destination, default, new RowDiagnostics(row.Id, Collector), TestContext.Current.CancellationToken);
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

    /// <summary>Maps the batch-local input index 0 to the row id.</summary>
    private sealed class RowDiagnostics(string rowId, IEmbeddingGemma2Diagnostics inner) : IEmbeddingGemma2Diagnostics
    {
        public void OnSoftTokens(int input, int image, float[] softTokens) =>
            (inner as IRowSink)?.SoftTokens(rowId, image, softTokens);

        public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings) =>
            (inner as IRowSink)?.TokenEmbeddings(rowId, tokenEmbeddings.ToArray());
    }
}

/// <summary>Collects intermediate tensors by fixture row id.</summary>
internal interface IRowSink
{
    void SoftTokens(string rowId, int image, float[] softTokens);

    void TokenEmbeddings(string rowId, float[] tokenEmbeddings);
}

/// <summary>Soft tokens and token embeddings by row, for the row-relative checks.</summary>
internal sealed class TensorCollector : IEmbeddingGemma2Diagnostics, IRowSink
{
    public Dictionary<(string Row, int Image), float[]> Soft { get; } = [];

    public Dictionary<string, float[]> Tokens { get; } = new(StringComparer.Ordinal);

    public void OnSoftTokens(int input, int image, float[] softTokens)
    {
    }

    public void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings)
    {
    }

    public void SoftTokens(string rowId, int image, float[] softTokens) => Soft[(rowId, image)] = softTokens;

    public void TokenEmbeddings(string rowId, float[] tokenEmbeddings) => Tokens[rowId] = tokenEmbeddings;
}

/// <summary>Image parity checks against the accepted fixtures, shared by the CPU and CUDA test projects.</summary>
internal static class EmbeddingGemma2ImageParityChecks
{
    /// <summary>Each query's intended image (the reference's retrieval sanity table).</summary>
    public static readonly IReadOnlyDictionary<string, string> IntendedImage = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["q-cat"] = "photo-chelsea",
        ["q-coffee"] = "photo-coffee",
        ["q-rocket"] = "photo-rocket",
        ["q-astronaut"] = "photo-astronaut",
        ["q-galaxies"] = "photo-hubble-deep-field",
        ["q-cameraman"] = "photo-camera",
        ["q-shapes"] = "synth-shapes",
        ["q-text"] = "synth-text",
    };

    /// <summary>Every in-contract row alone vs the reference embedding; soft tokens and token embeddings row-relative when the tensors are available.</summary>
    public static void RowsAgainstReference(EmbeddingGemma2ImageModelFixture fixture, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var singles = fixture.Singles();
        var rows = EmbeddingGemma2ImageTestAssets.Rows.Where(r => r.InContract).ToList();
        Assert.Equal(42, rows.Count);
        var all = new EmbeddingGemma2ErrorStats();
        var imageRows = new EmbeddingGemma2ErrorStats();
        var textRows = new EmbeddingGemma2ErrorStats();
        foreach (var row in rows)
        {
            all.Add(row.Id, singles[row.Id], row.Embedding);
            (row.Images.Count > 0 ? imageRows : textRows).Add(row.Id, singles[row.Id], row.Embedding);
        }

        output.WriteLine($"{rows.Count} rows (in contract; {EmbeddingGemma2ImageTestAssets.SixteenBitImageId} is rejected by contract) vs reference: {all}");
        output.WriteLine($"  image and interleaved rows: {imageRows}");
        output.WriteLine($"  text-only rows (released text graph): {textRows}");
        output.WriteLine($"bounds: embedding_max_abs {tolerances.EmbeddingMaxAbs:E1}, min_cosine {tolerances.MinCosine} ({tolerances.Source})");
        Assert.True(tolerances.NanOrInfFails);
        Assert.False(all.NonFinite, "NaN or Inf in embeddings");
        Assert.True(all.Max <= tolerances.EmbeddingMaxAbs, $"embedding max abs {all.Max:E3} > {tolerances.EmbeddingMaxAbs:E1} ({all.WorstKey})");
        Assert.True(all.MinCosine >= tolerances.MinCosine, $"min cosine {all.MinCosine:F12} < {tolerances.MinCosine}");

        var (tensors, reason) = EmbeddingGemma2ImageTestAssets.FindTensors();
        if (tensors is null)
        {
            output.WriteLine("soft tokens / token embeddings: not compared: " + reason);
            return;
        }

        double worstSoft = 0, worstTokens = 0;
        string worstSoftKey = string.Empty, worstTokensKey = string.Empty;
        foreach (var row in rows.Where(r => r.Images.Count > 0))
        {
            for (int k = 0; k < row.Images.Count; k++)
            {
                var reference = EmbeddingGemma2ImageTestAssets.ReadReferenceTensor(tensors, row.Id, $"image{k}_soft_tokens");
                double rel = RowRelative(fixture.Collector.Soft[(row.Id, k)], reference);
                if (rel > worstSoft)
                {
                    (worstSoft, worstSoftKey) = (rel, $"{row.Id}/image{k}");
                }
            }

            var refTokens = EmbeddingGemma2ImageTestAssets.ReadReferenceTensor(tensors, row.Id, "token_embeddings");
            double relTokens = RowRelative(fixture.Collector.Tokens[row.Id], refTokens);
            if (relTokens > worstTokens)
            {
                (worstTokens, worstTokensKey) = (relTokens, row.Id);
            }
        }

        output.WriteLine($"soft tokens row-relative worst {worstSoft:E2} ({worstSoftKey}); bound {tolerances.SoftTokensRowRelative:E1}");
        output.WriteLine($"token embeddings row-relative worst {worstTokens:E2} ({worstTokensKey}); bound {tolerances.TokenEmbeddingsRowRelative:E1}");
        Assert.True(worstSoft <= tolerances.SoftTokensRowRelative, $"soft tokens row-relative {worstSoft:E3} ({worstSoftKey})");
        Assert.True(worstTokens <= tolerances.TokenEmbeddingsRowRelative, $"token embeddings row-relative {worstTokens:E3} ({worstTokensKey})");
    }

    /// <summary>Mixed batches (text-only, single images of different sizes, interleaved, two images) vs single rows.</summary>
    public static void BatchesAgainstSingles(EmbeddingGemma2ImageModelFixture fixture, EmbeddingGemma2MultimodalOptions batchOptions, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output, string label)
    {
        var singles = fixture.Singles();
        _ = fixture.Embedder; // skips when the model is absent
        var files = fixture.Files!;
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, batchOptions with { Session = fixture.SessionOptions, VerifyModelSha256 = false });
        string[] ids = ["q-cat", "photo-chelsea", "aspect-wide-2000x40", "size-tiny-8x6", "mm-two-images", "q-rocket", "mm-caption-placeholder", "aspect-tall-30x1500", "img-query-prompt"];
        var rows = ids.Select(id => EmbeddingGemma2ImageTestAssets.Rows.Single(r => r.Id == id)).ToList();

        // One prompt per call: group by prompt, batch within the group.
        var stats = new EmbeddingGemma2ErrorStats();
        foreach (var group in rows.GroupBy(r => r.Prompt))
        {
            var list = group.ToList();
            var inputs = list.Select(r => EmbeddingGemma2ImageTestAssets.BuildInput(r, EmbeddingGemma2ImageTestAssets.Decode)).ToList();
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

    /// <summary>Each query (query prompt, released text graph) ranks its intended image first among the eight candidates.</summary>
    public static void RetrievalSanity(EmbeddingGemma2ImageModelFixture fixture, ITestOutputHelper output)
    {
        var singles = fixture.Singles();
        var candidates = IntendedImage.Values.ToList();
        foreach (var (query, intended) in IntendedImage)
        {
            var q = singles[query];
            var ranked = candidates.Select(c => (c, Dot(q, singles[c]))).OrderByDescending(x => x.Item2).ToList();
            output.WriteLine($"{query}: top {ranked[0].c} {ranked[0].Item2:F4}, runner-up {ranked[1].c} {ranked[1].Item2:F4}");
            Assert.Equal(intended, ranked[0].c);
        }
    }

    /// <summary>Text-only inputs through the multimodal embedder equal the text embedder (released graph) bit for bit.</summary>
    public static void TextOnlyEqualsTheTextEmbedder(EmbeddingGemma2ImageModelFixture fixture)
    {
        _ = fixture.Embedder; // skips when the model is absent
        var files = fixture.Files!;
        using var encoder = EmbeddingGemma2Encoder.Create(files.Text.ModelPath, new EmbeddingGemma2EncoderOptions { Session = fixture.SessionOptions });
        var text = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2TestAssets.Tokenizer, encoder);
        var singles = fixture.Singles();
        foreach (var row in EmbeddingGemma2ImageTestAssets.Rows.Where(r => r.Kind == "text"))
        {
            Assert.Equal(text.Embed(row.Text!, row.PackagePrompt, TestContext.Current.CancellationToken), singles[row.Id]);
        }
    }

    /// <summary>A Matryoshka size of an image embedding equals the truncated and renormalized 768-dimensional one.</summary>
    public static void Matryoshka(EmbeddingGemma2ImageModelFixture fixture, EmbeddingGemma2Tolerances tolerances)
    {
        _ = fixture.Embedder; // skips when the model is absent
        var files = fixture.Files!;
        using var small = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = fixture.SessionOptions, Dimension = 256, VerifyModelSha256 = false });
        var image = EmbeddingGemma2ImageTestAssets.Decode("synth-shapes");
        var e256 = small.Embed(image, TestContext.Current.CancellationToken);
        var expected = EmbeddingGemma2Encoder.Truncate(fixture.Singles()["synth-shapes"], 256);
        Assert.Equal(256, e256.Length);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add("synth-shapes@256", e256, expected);
        Assert.True(stats.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"Matryoshka 256 {stats.Max:E3}");
    }

    public static double RowRelative(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxDiff = 0, maxRef = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs((double)actual[i] - expected[i]));
            maxRef = Math.Max(maxRef, Math.Abs((double)expected[i]));
        }

        return maxDiff / maxRef;
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
