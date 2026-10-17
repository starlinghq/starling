using System.Globalization;
using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;

try
{
    return await RunAsync();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> RunAsync()
{
    if (args.Length == 0 || args[0] is not ("download" or "search"))
        return Usage();

    string? cache = null;
    string? question = null;
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--cache" && i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            cache = args[++i];
        else if (!args[i].StartsWith("--", StringComparison.Ordinal) && args[0] == "search" && question is null)
            question = args[i];
        else
            return Usage();
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };
    if (args[0] == "download")
    {
        var files = await E5ModelAssets.DownloadAsync(acquisition);
        Console.WriteLine($"Verified model: {files.Snapshot.Directory}");
        return 0;
    }

    // No automatic prefix: the vector property supplies "passage: ", and SearchAsync supplies "query: ".
    using IEmbeddingGenerator<string, Embedding<float>> generator =
        E5EmbeddingGenerator.Create(E5ModelAssets.OpenLocal(acquisition));
    using var collection = new InMemoryCollection<string, Passage>("passages", new()
    {
        EmbeddingGenerator = generator,
    });
    await collection.EnsureCollectionExistsAsync();
    Passage[] passages =
    [
        new() { Id = "rhine", Text = "Der Rhein fließt durch Köln." },
        new() { Id = "tokyo", Text = "東京は日本の首都です。" },
        new() { Id = "bread", Text = "Para hacer pan se necesita harina, agua, sal y levadura." },
        new() { Id = "everest", Text = "Mount Everest is the highest mountain above sea level." },
    ];
    // The collection embeds the string vector property on upsert; no manual vector generation is needed.
    await collection.UpsertAsync(passages);
    Console.WriteLine($"Indexed {passages.Length} multilingual passages in memory.");
    question ??= "Which river flows through Cologne?";
    Console.WriteLine($"Query: {question}");
    await foreach (var hit in collection.SearchAsync("query: " + question, top: 3))
    {
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{hit.Score:F4}  {hit.Record.Id}: {hit.Record.Text}"));
    }
    return 0;
}

static int Usage()
{
    Console.Error.WriteLine("usage: E5VectorStore download [--cache <dir>] | search [--cache <dir>] [\"question\"]");
    return 2;
}

internal sealed class Passage
{
    [VectorStoreKey]
    public required string Id { get; set; }

    [VectorStoreData]
    public required string Text { get; set; }

    // A string vector property tells VectorData to use the collection's embedding generator.
    [VectorStoreVector(384, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public string Vector => "passage: " + Text;
}
