// Semantic search with intfloat/multilingual-e5-small from the Starlings packages.
//
//   dotnet run -- download [--cache <dir>]           explicit acquisition (network): downloads and verifies the model
//   dotnet run -- search   [--cache <dir>] [query]   offline: loads the verified files, embeds, ranks by cosine
//
// Without --cache the files go to $STARLING_MODELS or the user cache directory. Inference never downloads anything.
using System.Reflection;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Starlings.Runtime.Onnx;
using Microsoft.Extensions.AI;

// A failure ends with a one-line message and exit code 1, not an unhandled-exception abort.
try
{
    return await ToolBody();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> ToolBody()
{
    if (args.Length == 0 || args[0] is not ("download" or "search"))
    {
        Console.Error.WriteLine("usage: E5Search download [--cache <dir>] | search [--cache <dir>] [query]");
        return 2;
    }

    string? cache = null;
    var rest = new List<string>();
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--cache" && i + 1 < args.Length)
        {
            cache = args[++i];
        }
        else
        {
            rest.Add(args[i]);
        }
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };

    if (args[0] == "download")
    {
        // 1. Explicit acquisition: files of the pinned lock (repository, commit, size, SHA-256). Cached files are verified,
        //    missing ones downloaded and verified. This is the only step that uses the network.
        Console.WriteLine($"lock:  {E5ModelAssets.Lock}");
        Console.WriteLine($"cache: {acquisition.ResolveCacheDirectory()}");
        var progress = new FileProgress();
        E5ModelFiles downloaded = await E5ModelAssets.DownloadAsync(acquisition, progress);
        foreach (ModelAssetFile file in E5ModelAssets.Lock.Files)
        {
            string how = progress.Downloaded.Contains(file.Path) ? "downloaded" : "already cached";
            Console.WriteLine($"{file.Path}: {how}, {file.Size} bytes, sha256 {file.Sha256} verified");
        }

        Console.WriteLine($"snapshot: {downloaded.Snapshot.Directory}");
        return 0;
    }

    // 2. Offline load: verifies the cached files (size and SHA-256) and never downloads. Missing files fail with
    //    ModelAssetMissingException, whose message gives the download call.
    E5ModelFiles files = E5ModelAssets.OpenLocal(acquisition);
    using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
    {
        Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
    });
    var embedder = new E5TextEmbedder(E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256), encoder);

    Console.WriteLine($"packages: {Version<E5Encoder>()}, {Version<Starlings.Tokenizers.Tokenizer>()}, {Version<E5EmbeddingGenerator>()}");
    Console.WriteLine($"onnxruntime: native {OnnxRuntimeInfo.NativeVersion}, managed {OnnxRuntimeInfo.ManagedVersion}, session providers {string.Join(",", encoder.Session.Diagnostics.SessionProviders)}");

    (string Id, string Text)[] passages =
    [
        ("protein", "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day."),
        ("summit", "The summit of Mount Everest is 8,849 metres above sea level, the highest point on Earth."),
        ("rust", "Eisen rostet, wenn es Sauerstoff und Wasser ausgesetzt ist; dabei entsteht Eisenoxid."),
        ("tokyo", "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。"),
        ("bread", "Para hacer pan se necesita harina, agua, sal y levadura; la masa reposa antes de hornearse."),
    ];
    string[] queries = rest.Count > 0
        ? [.. rest]
        : ["how much protein should a female eat", "Warum rostet Eisen?", "日本の首都はどこですか", "highest mountain in the world"];

    // 3. Typed API. e5 needs "passage: " on documents and "query: " on queries; nothing adds them implicitly.
    float[][] passageVectors = embedder.EmbedBatch([.. passages.Select(p => E5TextEmbedder.PassagePrefix + p.Text)]);
    float[][] queryVectors = embedder.EmbedBatch([.. queries.Select(q => E5TextEmbedder.QueryPrefix + q)]);
    for (int q = 0; q < queries.Length; q++)
    {
        // Vectors are L2-normalized, so the dot product is the cosine similarity.
        var ranked = passages.Select((p, i) => (p.Id, Score: Dot(queryVectors[q], passageVectors[i]))).OrderByDescending(r => r.Score).ToArray();
        Console.WriteLine($"query: {queries[q]}");
        Console.WriteLine($"  top: {ranked[0].Id} ({ranked[0].Score:F4}); next: {ranked[1].Id} ({ranked[1].Score:F4})");
    }

    // 4. The same model through Microsoft.Extensions.AI. The generator wraps the embedder (and owns nothing); the input
    //    type adds the prefix, here per generator for queries and per request for passages.
    using IEmbeddingGenerator<string, Embedding<float>> generator =
        new E5EmbeddingGenerator(embedder, new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });
    GeneratedEmbeddings<Embedding<float>> generatedQueries = await generator.GenerateAsync(queries);
    GeneratedEmbeddings<Embedding<float>> generatedPassages = await generator.GenerateAsync(
        passages.Select(p => p.Text),
        new EmbeddingGenerationOptions { AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = "passage" } });

    double maxDiff = 0;
    for (int q = 0; q < queries.Length; q++)
    {
        maxDiff = Math.Max(maxDiff, MaxAbsDiff(generatedQueries[q].Vector.Span, queryVectors[q]));
    }

    for (int p = 0; p < passages.Length; p++)
    {
        maxDiff = Math.Max(maxDiff, MaxAbsDiff(generatedPassages[p].Vector.Span, passageVectors[p]));
    }

    EmbeddingGeneratorMetadata? metadata = generator.GetService<EmbeddingGeneratorMetadata>();
    Console.WriteLine($"extensions.ai: {metadata?.ProviderName} {metadata?.DefaultModelId} dims {metadata?.DefaultModelDimensions}; " +
        $"{generatedQueries.Count + generatedPassages.Count} embeddings, {generatedQueries.Usage?.InputTokenCount + generatedPassages.Usage?.InputTokenCount} tokens, " +
        $"max abs difference from the typed API {maxDiff:G3}");
    return 0;

    static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    static double MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        }

        return max;
    }

    static string Version<T>()
    {
        AssemblyName name = typeof(T).Assembly.GetName();
        string? informational = typeof(T).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        return $"{name.Name} {informational?.Split('+')[0]}";
    }
}

/// <summary>Records which files were actually transferred. Called synchronously by the store, unlike <see cref="Progress{T}"/>.</summary>
internal sealed class FileProgress : IProgress<ModelAssetProgress>
{
    public HashSet<string> Downloaded { get; } = [];

    public void Report(ModelAssetProgress value)
    {
        if (value.Stage == ModelAssetStage.Downloading)
        {
            Downloaded.Add(value.Path);
        }
    }
}
