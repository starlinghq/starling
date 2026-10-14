using System.Globalization;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// One encoder over the real text-path graph per test class, with the graph's SHA-256 enforced. When the graph is
/// absent the fixture holds a skip reason and the tests skip.
/// </summary>
public abstract class EmbeddingGemma2ModelFixture : IDisposable
{
    private readonly EmbeddingGemma2Encoder? _encoder;

    protected EmbeddingGemma2ModelFixture(OnnxSessionOptions sessionOptions, Func<string?>? gate = null)
    {
        Rows = EmbeddingGemma2Fixtures.LoadRows();
        string? gateReason = gate?.Invoke();
        if (gateReason is not null)
        {
            SkipReason = gateReason;
            return;
        }

        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        if (files is null)
        {
            SkipReason = reason;
            return;
        }

        Files = files;
        _encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
        {
            Session = sessionOptions with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256, CapturePlacement = true },
        });
    }

    /// <summary>Why the model tests skip, or null when the model is loaded.</summary>
    public string? SkipReason { get; }

    /// <summary>The verified files, or null when absent.</summary>
    public EmbeddingGemma2ModelFiles? Files { get; }

    /// <summary>The 138 protected rows (always available: the fixtures are in Git).</summary>
    public IReadOnlyList<EmbeddingGemma2Row> Rows { get; }

    /// <summary>The encoder (768 dimensions); skips the calling test when the model is unavailable.</summary>
    public EmbeddingGemma2Encoder Encoder
    {
        get
        {
            Assert.SkipWhen(_encoder is null, SkipReason ?? "model unavailable");
            return _encoder!;
        }
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>Absolute-error and cosine summary over many vectors.</summary>
internal sealed class EmbeddingGemma2ErrorStats
{
    private readonly List<double> _abs = [];

    public double Max { get; private set; }

    public double MinCosine { get; private set; } = double.PositiveInfinity;

    public bool NonFinite { get; private set; }

    public string WorstKey { get; private set; } = string.Empty;

    public double Mean => _abs.Count == 0 ? 0 : _abs.Average();

    public double P99
    {
        get
        {
            if (_abs.Count == 0)
            {
                return 0;
            }

            var sorted = _abs.Order().ToArray();
            double pos = (sorted.Length - 1) * 0.99;
            int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
            return sorted[lo] + ((sorted[hi] - sorted[lo]) * (pos - lo));
        }
    }

    public void Add(string key, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double dot = 0, na = 0, ne = 0, rowMax = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (!float.IsFinite(actual[i]))
            {
                NonFinite = true;
            }

            double d = Math.Abs((double)actual[i] - expected[i]);
            _abs.Add(d);
            rowMax = Math.Max(rowMax, d);
            dot += (double)actual[i] * expected[i];
            na += (double)actual[i] * actual[i];
            ne += (double)expected[i] * expected[i];
        }

        if (WorstKey.Length == 0 || rowMax > Max)
        {
            Max = rowMax;
            WorstKey = key;
        }

        MinCosine = Math.Min(MinCosine, dot / (Math.Sqrt(na) * Math.Sqrt(ne)));
    }

    public override string ToString() => string.Create(
        CultureInfo.InvariantCulture,
        $"max abs {Max:E2}, mean {Mean:E2}, p99 {P99:E2}, min cosine {MinCosine:F15} (worst {WorstKey})");
}

/// <summary>
/// Parity checks against the protected EmbeddingGemma 2 embeddings, shared by the CPU and CUDA test projects. Inputs
/// are the fixture ids; nothing is tokenized. Bounds come from the profile's protected tolerances.yaml.
/// </summary>
internal static class EmbeddingGemma2ParityChecks
{
    /// <summary>Lengths around the 512-token sliding window, as the admission spike's parity run (511 to 2,049 tokens).</summary>
    public static readonly int[] SlidingWindowLengths = [511, 512, 513, 514, 515, 600, 1024, 1025, 1026, 1027, 1100, 1537, 2049];

    public static Dictionary<string, float[]> SingleRows(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var stats = new EmbeddingGemma2ErrorStats();
        var singles = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            var embedding = encoder.Embed(new EmbeddingGemma2Input(row.InputIds), TestContext.Current.CancellationToken);
            singles[row.Key] = embedding;
            stats.Add(row.Key, embedding, row.Embedding);
        }

        output.WriteLine($"single rows from fixture ids vs protected ({rows.Count} rows, prompts none/query/document): {stats}");
        output.WriteLine($"bounds: embedding_max_abs {tolerances.EmbeddingMaxAbs:E1}, min_cosine {tolerances.MinCosine} from {tolerances.Source}");
        Assert.Equal(138, rows.Count);
        Assert.Equal(138, singles.Count);
        Assert.False(stats.NonFinite, "NaN or Inf in embeddings");
        Assert.True(stats.Max <= tolerances.EmbeddingMaxAbs, $"embedding max abs {stats.Max:E3} > {tolerances.EmbeddingMaxAbs:E1} ({stats.WorstKey})");
        Assert.True(stats.MinCosine >= tolerances.MinCosine, $"min cosine {stats.MinCosine:F12} < {tolerances.MinCosine}");
        return singles;
    }

    public static void PaddedBatchFixture(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, EmbeddingGemma2Tolerances tolerances, IReadOnlyDictionary<string, float[]> singles, ITestOutputHelper output)
    {
        var none = rows.Where(r => r.Prompt == "none").ToDictionary(r => r.Id, StringComparer.Ordinal);
        var batch = EmbeddingGemma2Fixtures.LoadPaddedBatch();
        var ct = TestContext.Current.CancellationToken;

        // As a tokenizer would hand them over: already right-padded with explicit masks.
        var padded = encoder.EmbedBatch(batch.Select(r => new EmbeddingGemma2Input(r.Ids, r.Mask)).ToList(), ct);

        // Unpadded rows, padded by the encoder.
        var unpadded = encoder.EmbedBatch(batch.Select(r => new EmbeddingGemma2Input(none[r.Id].InputIds)).ToList(), ct);
        var vsProtected = new EmbeddingGemma2ErrorStats();
        var vsSingle = new EmbeddingGemma2ErrorStats();
        var encoderPadded = new EmbeddingGemma2ErrorStats();
        for (int i = 0; i < batch.Count; i++)
        {
            var (id, ids, _) = batch[i];
            int length = none[id].InputIds.Length;
            Assert.Equal(none[id].InputIds, ids[..length]);
            Assert.All(ids[length..], x => Assert.Equal(EmbeddingGemma2Encoder.PadTokenId, x));
            vsProtected.Add(id, padded[i], none[id].Embedding);
            vsSingle.Add(id, padded[i], singles[id]);
            encoderPadded.Add(id, unpadded[i], singles[id]);
        }

        output.WriteLine($"{batch.Count}-row protected padded batch vs protected: {vsProtected}");
        output.WriteLine($"{batch.Count}-row protected padded batch vs own single rows: {vsSingle}");
        output.WriteLine($"{batch.Count} rows padded by the encoder vs own single rows: {encoderPadded}");
        Assert.False(vsProtected.NonFinite || encoderPadded.NonFinite);
        Assert.True(vsProtected.Max <= tolerances.EmbeddingMaxAbs, $"batch vs protected {vsProtected.Max:E3}");
        Assert.True(vsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"batch vs single {vsSingle.Max:E3}");
        Assert.True(encoderPadded.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"encoder-padded batch vs single {encoderPadded.Max:E3}");
    }

    public static void AllRowsInBatches(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, int batchSize, EmbeddingGemma2Tolerances tolerances, IReadOnlyDictionary<string, float[]> singles, ITestOutputHelper output)
    {
        var vsProtected = new EmbeddingGemma2ErrorStats();
        var vsSingle = new EmbeddingGemma2ErrorStats();
        for (int start = 0; start < rows.Count; start += batchSize)
        {
            var chunk = rows.Skip(start).Take(batchSize).ToList();
            var embeddings = encoder.EmbedBatch(chunk.Select(r => new EmbeddingGemma2Input(r.InputIds)).ToList(), TestContext.Current.CancellationToken);
            for (int i = 0; i < chunk.Count; i++)
            {
                vsProtected.Add(chunk[i].Key, embeddings[i], chunk[i].Embedding);
                vsSingle.Add(chunk[i].Key, embeddings[i], singles[chunk[i].Key]);
            }
        }

        output.WriteLine($"{rows.Count} rows in padded batches of {batchSize} vs protected: {vsProtected}");
        output.WriteLine($"{rows.Count} rows in padded batches of {batchSize} vs own single rows: {vsSingle}");
        Assert.False(vsProtected.NonFinite);
        Assert.True(vsProtected.Max <= tolerances.EmbeddingMaxAbs, $"batches vs protected {vsProtected.Max:E3}");
        Assert.True(vsProtected.MinCosine >= tolerances.MinCosine, $"batches min cosine {vsProtected.MinCosine:F12}");
        Assert.True(vsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"batches vs single {vsSingle.Max:E3}");
    }

    /// <summary>
    /// A sequence of exactly <paramref name="length"/> tokens: <c>&lt;bos&gt;</c>, the interior of the longest fixture
    /// row repeated as needed, <c>&lt;eos&gt;</c>.
    /// </summary>
    public static long[] SyntheticSequence(IReadOnlyList<EmbeddingGemma2Row> rows, int length)
    {
        var longest = rows.MaxBy(r => r.InputIds.Length)!.InputIds;
        var interior = longest[1..^1];
        var seq = new long[length];
        seq[0] = EmbeddingGemma2Encoder.BosTokenId;
        for (int i = 1; i < length - 1; i++)
        {
            seq[i] = interior[(i - 1) % interior.Length];
        }

        seq[^1] = EmbeddingGemma2Encoder.EosTokenId;
        return seq;
    }

    /// <summary>
    /// Batch independence across the sliding window: each length padded in one batch with the 12-token
    /// <c>ascii-sentence</c> row, against both rows' own single runs (padded-batch bound), finite throughout.
    /// </summary>
    public static void SlidingWindowBatchIndependence(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var ct = TestContext.Current.CancellationToken;
        var shortIds = rows.First(r => r.Id == "ascii-sentence" && r.Prompt == "none").InputIds;
        var shortSingle = encoder.Embed(new EmbeddingGemma2Input(shortIds), ct);
        var longStats = new EmbeddingGemma2ErrorStats();
        var shortStats = new EmbeddingGemma2ErrorStats();
        foreach (int n in SlidingWindowLengths)
        {
            var seq = SyntheticSequence(rows, n);
            var single = encoder.Embed(new EmbeddingGemma2Input(seq), ct);
            var batch = encoder.EmbedBatch([new EmbeddingGemma2Input(seq), new EmbeddingGemma2Input(shortIds)], ct);
            var rowLong = new EmbeddingGemma2ErrorStats();
            rowLong.Add($"{n}", batch[0], single);
            longStats.Add($"{n}", batch[0], single);
            shortStats.Add($"short-with-{n}", batch[1], shortSingle);
            output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{n,5} tokens: batch vs single {rowLong.Max:E2}; norm {Norm(single):F7}"));
            Assert.InRange(Norm(single), 0.9999, 1.0001);
        }

        output.WriteLine($"long rows, batch vs single: {longStats}");
        output.WriteLine($"12-token row padded next to them vs single: {shortStats}");
        Assert.False(longStats.NonFinite || shortStats.NonFinite);
        Assert.True(longStats.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"long batch vs single {longStats.Max:E3}");
        Assert.True(shortStats.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"short batch vs single {shortStats.Max:E3}");
    }

    /// <summary>
    /// Matryoshka sizes against the protected 768-dimensional reference sliced and L2-normalized again (what
    /// sentence-transformers <c>truncate_dim</c> computes; the fixtures hold no truncated rows), with the embedding
    /// bound; and against <see cref="EmbeddingGemma2Encoder.Truncate"/> of the encoder's own 768 output.
    /// </summary>
    public static void Matryoshka(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, EmbeddingGemma2Tolerances tolerances, IReadOnlyDictionary<string, float[]> singles, ITestOutputHelper output)
    {
        var ct = TestContext.Current.CancellationToken;
        foreach (int dimension in new[] { 512, 256, 128 })
        {
            using var truncated = new EmbeddingGemma2Encoder(encoder.Session, encoder.Options with { Dimension = dimension });
            var vsReference = new EmbeddingGemma2ErrorStats();
            var vsOwn = new EmbeddingGemma2ErrorStats();
            var subset = rows.Where((_, i) => i % 3 == 0 || rows[i].InputIds.Length > 1000).ToList();
            var embeddings = truncated.EmbedBatch(subset.Select(r => new EmbeddingGemma2Input(r.InputIds)).ToList(), ct);
            for (int i = 0; i < subset.Count; i++)
            {
                Assert.Equal(dimension, embeddings[i].Length);
                vsReference.Add(subset[i].Key, embeddings[i], EmbeddingGemma2Encoder.Truncate(subset[i].Embedding, dimension));
                vsOwn.Add(subset[i].Key, embeddings[i], EmbeddingGemma2Encoder.Truncate(singles[subset[i].Key], dimension));
                Assert.InRange(Norm(embeddings[i]), 0.9999, 1.0001);
            }

            output.WriteLine($"Matryoshka {dimension} ({subset.Count} rows) vs sliced+renormalized reference: {vsReference}");
            output.WriteLine($"Matryoshka {dimension} vs Truncate(own 768 output): {vsOwn}");
            Assert.True(vsReference.Max <= tolerances.EmbeddingMaxAbs, $"Matryoshka {dimension} vs reference {vsReference.Max:E3}");
            Assert.True(vsReference.MinCosine >= tolerances.MinCosine, $"Matryoshka {dimension} min cosine {vsReference.MinCosine:F12}");
            Assert.True(vsOwn.Max <= 1e-6, $"Matryoshka {dimension} vs Truncate(own) {vsOwn.Max:E3}");
        }
    }

    /// <summary>
    /// From text: every protected row through <see cref="EmbeddingGemma2TextEmbedder"/> with the owned tokenizer and the
    /// row's prompt, singly and in padded batches per prompt; ids exact, embeddings within the binding bounds.
    /// </summary>
    public static void TextRows(EmbeddingGemma2Encoder encoder, EmbeddingGemma2Tokenizer tokenizer, IReadOnlyList<EmbeddingGemma2Row> rows, EmbeddingGemma2Tolerances tolerances, ITestOutputHelper output)
    {
        var ct = TestContext.Current.CancellationToken;
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var single = new EmbeddingGemma2ErrorStats();
        var batched = new EmbeddingGemma2ErrorStats();
        var vsSingle = new EmbeddingGemma2ErrorStats();
        foreach (var group in rows.GroupBy(r => r.Prompt))
        {
            var list = group.ToList();
            var prompt = list[0].PackagePrompt;
            var counts = new int[list.Count];
            var batch = embedder.EmbedBatch(list.Select(r => r.Text).ToList(), prompt, counts, ct);
            for (int i = 0; i < list.Count; i++)
            {
                Assert.Equal(list[i].InputIds, tokenizer.Tokenize(prompt.Apply(list[i].Text)).InputIds.ToArray());
                Assert.Equal(list[i].InputIds.Length, counts[i]);
                var one = embedder.Embed(list[i].Text, prompt, ct);
                single.Add(list[i].Key, one, list[i].Embedding);
                batched.Add(list[i].Key, batch[i], list[i].Embedding);
                vsSingle.Add(list[i].Key, batch[i], one);
            }
        }

        output.WriteLine($"from text, {rows.Count} rows (prompts {string.Join("/", rows.Select(r => r.Prompt).Distinct())}), ids exact; single vs reference: {single}");
        output.WriteLine($"from text, padded batches vs reference: {batched}");
        output.WriteLine($"from text, padded batches vs single: {vsSingle}");
        Assert.True(single.Max <= tolerances.EmbeddingMaxAbs, $"text single {single.Max:E3}");
        Assert.True(single.MinCosine >= tolerances.MinCosine, $"text min cosine {single.MinCosine:F12}");
        Assert.True(batched.Max <= tolerances.EmbeddingMaxAbs, $"text batch {batched.Max:E3}");
        Assert.True(vsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"text batch vs single {vsSingle.Max:E3}");
        Assert.False(single.NonFinite || batched.NonFinite);
    }

    public static void InputsOver8192TokensAreRejected(EmbeddingGemma2Encoder encoder, IReadOnlyList<EmbeddingGemma2Row> rows, ITestOutputHelper output)
    {
        var tooLong = SyntheticSequence(rows, 8193);
        var ok = rows[3].InputIds;
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() =>
            encoder.EmbedBatch([new EmbeddingGemma2Input(ok), new EmbeddingGemma2Input(tooLong)], TestContext.Current.CancellationToken));
        Assert.Equal(1, ex.InputIndex);
        Assert.Equal(8193, ex.TokenCount);
        Assert.Equal(8192, ex.MaximumTokens);
        output.WriteLine($"8,193 tokens rejected before inference: {ex.Message}");
    }

    public static double Norm(ReadOnlySpan<float> v)
    {
        double s = 0;
        foreach (float x in v)
        {
            s += (double)x * x;
        }

        return Math.Sqrt(s);
    }
}
