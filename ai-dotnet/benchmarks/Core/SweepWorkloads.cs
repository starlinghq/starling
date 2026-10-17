using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Benchmarks;

/// <summary>
/// One sweep item: an id, the raw text and its kind (<c>query</c> or <c>passage</c> for retrieval, <c>text</c> for
/// extraction). The target applies its own prefix or prompt from the kind.
/// </summary>
internal sealed record SweepItem(string Id, string Text, string Kind);

/// <summary>
/// The batching-sweep workload file (<c>ai-dotnet/benchmarks/workloads/batching-sweep-v1.json</c>) and the item pools
/// it describes. Built deterministically from text in the repository; see the file's description.
/// </summary>
internal sealed class SweepWorkloadFile
{
    public static readonly string[] Categories = ["short", "medium", "long"];

    private SweepWorkloadFile(string path, JsonElement root)
    {
        Path = path;
        Root = root;
        Seed = root.GetProperty("seed").GetInt32();
        Pool = root.GetProperty("pool").GetInt32();
    }

    public string Path { get; }

    public JsonElement Root { get; }

    public int Seed { get; }

    public int Pool { get; }

    public static string DefaultPath => BenchPaths.Workload("batching-sweep-v1.json");

    public static SweepWorkloadFile Load(string? path)
    {
        path = System.IO.Path.GetFullPath(path ?? DefaultPath);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        return new SweepWorkloadFile(path, doc.RootElement.Clone());
    }

    /// <summary>The source files this model's pools read, repository-relative, for the artifact digests.</summary>
    public string[] SourceFiles(string model)
    {
        var m = Model(model);
        var source = Root.GetProperty("sources").GetProperty(m.GetProperty("source").GetString()!);
        return source.EnumerateObject().Select(p => System.IO.Path.Combine(BenchPaths.RepositoryRoot, p.Value.GetString()!)).ToArray();
    }

    private JsonElement Model(string model) =>
        Root.GetProperty("models").TryGetProperty(model, out var m) ? m : throw new ArgumentException($"{Path}: no entry for model '{model}'.");

    /// <summary>
    /// The pools of every category and of <c>mixed</c>, each with at most <paramref name="pool"/> items (the file's
    /// pool when null). <paramref name="tokens"/> counts an item's model tokens as the target runs it (prefix, prompt or
    /// schema included); items above <paramref name="maxTokens"/> are dropped.
    /// </summary>
    public SweepPools Build(string model, Func<SweepItem, int> tokens, int maxTokens, int? pool)
    {
        var m = Model(model);
        int size = pool ?? Pool;
        int longTokens = Math.Min(m.GetProperty("long_tokens").GetInt32(), maxTokens);
        double minFraction = m.GetProperty("long_min_fraction").GetDouble();
        var source = m.GetProperty("source").GetString()!;
        var cache = new Dictionary<string, int>(StringComparer.Ordinal);
        int Count(SweepItem item)
        {
            if (!cache.TryGetValue(item.Id, out int n))
            {
                n = tokens(item);
                cache[item.Id] = n;
            }

            return n;
        }

        var rng = new Random(Seed);
        var categories = source switch
        {
            "retrieval" => Retrieval(Root.GetProperty("sources").GetProperty("retrieval"), Count, maxTokens, longTokens, minFraction, size, rng),
            "ner" => Ner(Root.GetProperty("sources").GetProperty("ner"), m, Count, maxTokens, longTokens, minFraction, size, rng),
            _ => throw new ArgumentException($"{Path}: unknown source '{source}'."),
        };

        var pools = new Dictionary<string, SweepItem[]>(categories, StringComparer.Ordinal);

        // Mixed: each item's category from the configured probabilities, then the next unused item of that category.
        var mix = Root.GetProperty("mixed");
        double pShort = mix.GetProperty("short").GetDouble(), pMedium = mix.GetProperty("medium").GetDouble();
        rng = new Random(Seed + 100);
        var next = Categories.ToDictionary(k => k, _ => 0, StringComparer.Ordinal);
        var mixed = new List<SweepItem>(size);
        for (int i = 0; i < size; i++)
        {
            double r = rng.NextDouble();
            string cat = r < pShort ? "short" : r < pShort + pMedium ? "medium" : "long";
            var list = categories[cat];
            if (list.Length == 0)
            {
                continue;
            }

            mixed.Add(list[next[cat]++ % list.Length]);
        }

        pools["mixed"] = [.. mixed];
        var counts = pools.Values.SelectMany(p => p).DistinctBy(i => i.Id).ToDictionary(i => i.Id, Count, StringComparer.Ordinal);
        return new SweepPools(pools, counts);
    }

    private static Dictionary<string, SweepItem[]> Retrieval(JsonElement src, Func<SweepItem, int> count, int maxTokens, int longTokens, double minFraction, int size, Random rng)
    {
        var queries = ReadJsonl(src.GetProperty("queries").GetString()!).Select(r => new SweepItem("q:" + r.Id, r.Text, "query")).ToArray();
        var corpus = ReadJsonl(src.GetProperty("corpus").GetString()!);
        var passages = corpus.Select(r => new SweepItem("p:" + r.Id, r.Text, "passage")).ToArray();
        return new Dictionary<string, SweepItem[]>(StringComparer.Ordinal)
        {
            ["short"] = Take(queries, i => count(i) <= maxTokens, size, rng),
            ["medium"] = Take(passages, i => count(i) <= maxTokens, size, rng),
            ["long"] = Joined(corpus.Select(r => (Group: r.Id.Split(':')[0], r.Id, r.Text)).ToList(), "passage", count, longTokens, minFraction, size, rng),
        };
    }

    private static Dictionary<string, SweepItem[]> Ner(JsonElement src, JsonElement model, Func<SweepItem, int> count, int maxTokens, int longTokens, double minFraction, int size, Random rng)
    {
        var rows = ReadJsonl(src.GetProperty("texts").GetString()!);
        var sentences = rows.Select(r => new SweepItem("s:" + r.Id, r.Text, "text")).ToArray();
        var range = model.GetProperty("medium_sentences").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var medium = new List<SweepItem>();
        for (int i = 0; i + range[0] <= rows.Count;)
        {
            int k = Math.Min(rng.Next(range[0], range[1] + 1), rows.Count - i);
            var part = rows.Skip(i).Take(k).ToList();
            medium.Add(new SweepItem($"m:{part[0].Id}+{part.Count}", string.Join(' ', part.Select(p => p.Text)), "text"));
            i += k;
        }

        return new Dictionary<string, SweepItem[]>(StringComparer.Ordinal)
        {
            ["short"] = Take(sentences, i => count(i) <= maxTokens, size, rng),
            ["medium"] = Take([.. medium], i => count(i) <= maxTokens, size, rng),
            ["long"] = Joined(rows.Select(r => (Group: "wnut17", r.Id, r.Text)).ToList(), "text", count, longTokens, minFraction, size, rng),
        };
    }

    /// <summary>The first <paramref name="size"/> items in a seeded shuffle that pass <paramref name="keep"/> (only those are tokenized).</summary>
    private static SweepItem[] Take(SweepItem[] items, Func<SweepItem, bool> keep, int size, Random rng)
    {
        var shuffled = items.ToArray();
        rng.Shuffle(shuffled);
        return shuffled.Where(keep).Take(size).ToArray();
    }

    /// <summary>
    /// Long items: from start rows in a seeded shuffle, the start row and the rows after it of the same group joined
    /// with a space while the joined text stays within <paramref name="target"/> tokens; kept when it reaches
    /// <paramref name="minFraction"/> of the target. Stops after <paramref name="size"/> items.
    /// </summary>
    private static SweepItem[] Joined(List<(string Group, string Id, string Text)> rows, string kind, Func<SweepItem, int> count, int target, double minFraction, int size, Random rng)
    {
        var starts = Enumerable.Range(0, rows.Count).ToArray();
        rng.Shuffle(starts);
        var result = new List<SweepItem>();
        foreach (int i in starts)
        {
            if (result.Count >= size)
            {
                break;
            }

            var current = new SweepItem($"l:{rows[i].Id}+1", rows[i].Text, kind);
            if (count(current) > target)
            {
                continue;
            }

            int parts = 1;
            for (int j = i + 1; j < rows.Count && rows[j].Group == rows[i].Group; j++)
            {
                var candidate = new SweepItem($"l:{rows[i].Id}+{parts + 1}", current.Text + " " + rows[j].Text, kind);
                if (count(candidate) > target)
                {
                    break;
                }

                current = candidate;
                parts++;
            }

            if (count(current) >= minFraction * target)
            {
                result.Add(current);
            }
        }

        return [.. result];
    }

    private static List<(string Id, string Text)> ReadJsonl(string relative)
    {
        string path = System.IO.Path.Combine(BenchPaths.RepositoryRoot, relative);
        var rows = new List<(string, string)>();
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            rows.Add((doc.RootElement.GetProperty("id").GetString()!, doc.RootElement.GetProperty("text").GetString()!));
        }

        return rows;
    }
}

/// <summary>The item pools of one model (category or <c>mixed</c> to items) and their token counts.</summary>
internal sealed class SweepPools(Dictionary<string, SweepItem[]> pools, Dictionary<string, int> tokens)
{
    public IReadOnlyDictionary<string, SweepItem[]> Pools => pools;

    /// <summary>Model tokens of a pool item (counted once at build; safe to read from every worker).</summary>
    public int Tokens(SweepItem item) => tokens[item.Id];

    /// <summary>Per pool: items, and token count p10/p50/p90/max/mean (recorded in the run record).</summary>
    public JsonObject Describe()
    {
        var json = new JsonObject();
        foreach (var (name, items) in pools)
        {
            var t = items.Select(Tokens).Order().ToArray();
            json[name] = t.Length == 0
                ? new JsonObject { ["items"] = 0 }
                : new JsonObject
                {
                    ["items"] = t.Length,
                    ["tokens_p10"] = t[t.Length / 10],
                    ["tokens_p50"] = t[t.Length / 2],
                    ["tokens_p90"] = t[Math.Min(t.Length - 1, t.Length * 9 / 10)],
                    ["tokens_max"] = t[^1],
                    ["tokens_mean"] = Math.Round(t.Average(), 1),
                    ["first_ids"] = new JsonArray(items.Take(3).Select(i => (JsonNode)i.Id).ToArray()),
                };
        }

        return json;
    }

    public override string ToString() => string.Join(", ", pools.Select(p => string.Create(CultureInfo.InvariantCulture, $"{p.Key} {p.Value.Length}")));
}
