// ML.NET clean-consumer check for the Starlings release-candidate drill (G6; production-readiness R9).
// tools/release-drill-ai.sh copies this file into a temporary directory outside the repository and builds it against
// the packed Starlings.Models.E5.MLNet and Starlings.Models.EmbeddingGemma2.MLNet from a local folder feed. It runs with
// no network on files the drill already placed in the cache.
//
//   dotnet MlNetConsumer.dll e5 <cache>     the e5 transform (Starlings.Models.E5.MLNet)
//   dotnet MlNetConsumer.dll eg2 <cache>    the EmbeddingGemma 2 transform (Starlings.Models.EmbeddingGemma2.MLNet)
//
// For each: the pipeline's output column type (Vector<Single, n>), the row-wise transform equal to the typed API one
// text per call (exactly), the look-ahead batched transform equal to the typed batch (exactly), the token-count column
// equal to the typed API's counts, the expected passage ranked first for every query, a Matryoshka 256 column
// (EmbeddingGemma 2), Save rejected (persistence unsupported), and a video column rejected with a schema error
// (EmbeddingGemma 2; ML.NET video is unsupported). Exit code 0 only if every check passes.
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.MLNet;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.MLNet;
using Starlings.Runtime.Onnx;

// An exception ends with "error: " and the full exception on stderr and exit code 70 (EX_SOFTWARE), not an
// unhandled-exception abort and core dump (as ai-dotnet/tools/Shared/ToolMain.cs does for the in-repository tools).
try
{
    return ToolBody();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex}");
    return 70;
}

int ToolBody()
{
    if (args.Length != 2 || args[0] is not ("e5" or "eg2"))
    {
        Console.Error.WriteLine("usage: MlNetConsumer e5|eg2 <cache>");
        return 2;
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = args[1], UseEnvironmentToken = false };
    int failures = 0;
    void Check(bool ok, string what)
    {
        Console.WriteLine($"{(ok ? "ok:  " : "FAIL:")} {what}");
        if (!ok)
        {
            failures++;
        }
    }

    foreach (Type t in new[] { typeof(MLContext), args[0] == "e5" ? typeof(E5EmbeddingTransformer) : typeof(EmbeddingGemma2EmbeddingTransformer) })
    {
        Assembly a = t.Assembly;
        Console.WriteLine($"assembly: {a.GetName().Name} {a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion?.Split('+')[0]}");
    }

    // Query i should rank passage i first.
    string[] queries = ["how much protein should a female eat", "Warum rostet Eisen?", "日本の首都はどこですか"];
    string[] passages =
    [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Eisen rostet, wenn es mit Sauerstoff und Wasser zu Eisenoxid reagiert.",
        "東京は日本の首都であり、最大の都市です。",
    ];

    var ml = new MLContext(seed: 1);
    IDataView passageView = ml.Data.LoadFromEnumerable([.. passages.Select(p => new TextRow { Text = p })]);
    IDataView queryView = ml.Data.LoadFromEnumerable([.. queries.Select(q => new TextRow { Text = q })]);

    if (args[0] == "e5")
    {
        E5ModelFiles files = E5ModelAssets.OpenLocal(acquisition);
        using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions { Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 } });
        var embedder = new E5TextEmbedder(E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256), encoder);
        string[] prefixed = [.. passages.Select(p => E5TextEmbedder.PassagePrefix + p)];
        float[][] single = [.. prefixed.Select(p => embedder.Embed(p))];
        int[] counts = new int[prefixed.Length];
        float[][] batch = embedder.EmbedBatch(prefixed, counts);
        float[][] typedQueries = [.. queries.Select(q => embedder.Embed(E5TextEmbedder.QueryPrefix + q))];

        using var rowWise = ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text",
            new E5EmbeddingOptions { Prefix = E5TextPrefix.Passage, TokenCountColumnName = "Tokens" }).Fit(passageView);
        using var batched = ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text",
            new E5EmbeddingOptions { Prefix = E5TextPrefix.Passage, BatchSize = passages.Length }).Fit(passageView);
        using var fromFiles = ml.Transforms.ApplyE5Embedding(files, "Embedding", "Text", new E5EmbeddingOptions { Prefix = E5TextPrefix.Query }).Fit(queryView);
        RunChecks("e5", rowWise, batched, fromFiles, single, batch, counts, typedQueries, 384);
    }
    else
    {
        EmbeddingGemma2ModelFiles files = EmbeddingGemma2ModelAssets.OpenLocal(acquisition);
        using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions { Session = new OnnxSessionOptions { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 } });
        var embedder = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath), encoder);
        float[][] single = [.. passages.Select(p => embedder.Embed(p, EmbeddingGemma2Prompt.Document))];
        int[] counts = new int[passages.Length];
        float[][] batch = embedder.EmbedBatch(passages, EmbeddingGemma2Prompt.Document, counts);
        float[][] typedQueries = [.. embedder.EmbedBatch(queries, EmbeddingGemma2Prompt.Query)];

        using var rowWise = ml.Transforms.ApplyEmbeddingGemma2Embedding(embedder, "Embedding", "Text",
            new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, TokenCountColumnName = "Tokens" }).Fit(passageView);
        using var batched = ml.Transforms.ApplyEmbeddingGemma2Embedding(embedder, "Embedding", "Text",
            new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, BatchSize = passages.Length }).Fit(passageView);
        using var fromFiles = ml.Transforms.ApplyEmbeddingGemma2Embedding(files, "Embedding", "Text",
            new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Query }).Fit(queryView);
        RunChecks("EmbeddingGemma 2", rowWise, batched, fromFiles, single, batch, counts, typedQueries, 768);

        // Matryoshka: a 256-dimension column is the typed API's 256-dimension output.
        using var encoder256 = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions { Dimension = 256 });
        var embedder256 = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath), encoder256);
        using var small = ml.Transforms.ApplyEmbeddingGemma2Embedding(files, "Embedding", "Text",
            new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, Encoder = new EmbeddingGemma2EncoderOptions { Dimension = 256 } }).Fit(passageView);
        IDataView smallOut = small.Transform(passageView);
        Check(smallOut.Schema["Embedding"].Type is VectorDataViewType { Size: 256 }, $"EmbeddingGemma 2 Matryoshka column type {smallOut.Schema["Embedding"].Type}");
        float[][] smallRows = [.. smallOut.GetColumn<float[]>("Embedding")];
        Compare("EmbeddingGemma 2 Matryoshka 256 column vs typed API at 256", smallRows, [.. passages.Select(p => embedder256.Embed(p, EmbeddingGemma2Prompt.Document))]);

        // Video (and image) columns are unsupported in ML.NET (capability mlnet: unsupported for the image and video
        // profiles): a column of encoded video bytes is rejected with a schema error, never embedded as something else.
        IDataView videoView = ml.Data.LoadFromEnumerable([new VideoRow { Video = [0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70] }]);
        string? videoError = null;
        try
        {
            ml.Transforms.ApplyEmbeddingGemma2Embedding(embedder, "Embedding", "Video").Fit(videoView);
        }
        catch (ArgumentOutOfRangeException ex)
        {
            videoError = ex.Message;
        }

        Check(videoError is not null && videoError.Contains("Schema mismatch for input column 'Video'", StringComparison.Ordinal),
            $"EmbeddingGemma 2: a video column (Vector<Byte>) is rejected by the ML.NET transform (mlnet unsupported for video): {videoError}");
    }

    Console.WriteLine(failures == 0 ? "RESULT: PASS" : $"RESULT: FAIL ({failures})");
    return failures == 0 ? 0 : 1;

    void RunChecks(string model, ITransformer rowWise, ITransformer batched, ITransformer fromFiles,
        float[][] single, float[][] batch, int[] counts, float[][] typedQueries, int dimension)
    {
        IDataView rowOut = rowWise.Transform(passageView);
        Check(rowOut.Schema["Embedding"].Type is VectorDataViewType { Size: var n } && n == dimension,
            $"{model}: output column type {rowOut.Schema["Embedding"].Type} (expected Vector<Single, {dimension}>)");
        float[][] rows = [.. rowOut.GetColumn<float[]>("Embedding")];
        int[] tokens = [.. rowOut.GetColumn<int>("Tokens")];
        Compare($"{model}: row-wise transform vs typed API one text per call", rows, single);
        Compare($"{model}: look-ahead batched transform vs typed batch", [.. batched.Transform(passageView).GetColumn<float[]>("Embedding")], batch);
        Check(tokens.SequenceEqual(counts), $"{model}: token-count column {string.Join(",", tokens)} equals the typed API's {string.Join(",", counts)}");
        float[][] queryRows = [.. fromFiles.Transform(queryView).GetColumn<float[]>("Embedding")];
        Compare($"{model}: transform from verified files (own session) vs typed queries", queryRows, typedQueries);
        for (int q = 0; q < queryRows.Length; q++)
        {
            int top = Enumerable.Range(0, rows.Length).MaxBy(p => queryRows[q].Zip(rows[p], (x, y) => x * y).Sum());
            Check(top == q, $"{model}: query '{queries[q]}' ranks passage {top} first (expected {q})");
        }

        bool saveRejected = false;
        try
        {
            ml.Model.Save(rowWise, passageView.Schema, new MemoryStream());
        }
        catch (Exception ex) when (ex is NotSupportedException || ex.InnerException is NotSupportedException)
        {
            saveRejected = true;
        }

        Check(saveRejected, $"{model}: ML.NET Save of the transformer is rejected (persistence unsupported)");
    }

    void Compare(string what, float[][] actual, float[][] expected)
    {
        double max = 0;
        bool finite = actual.Length == expected.Length;
        for (int r = 0; r < Math.Min(actual.Length, expected.Length); r++)
        {
            finite &= actual[r].Length == expected[r].Length;
            for (int i = 0; i < Math.Min(actual[r].Length, expected[r].Length); i++)
            {
                finite &= float.IsFinite(actual[r][i]);
                max = Math.Max(max, Math.Abs((double)actual[r][i] - expected[r][i]));
            }
        }

        Check(finite && max == 0, $"{what}: {actual.Length} rows, max abs {max:G3} (exact)");
    }
}

/// <summary>Input row of the pipelines.</summary>
internal sealed class TextRow
{
    public string Text { get; set; } = string.Empty;
}

internal sealed class VideoRow
{
    [VectorType]
    public byte[] Video { get; set; } = [];
}
