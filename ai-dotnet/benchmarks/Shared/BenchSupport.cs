using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>One protected fixture case: ids and reference embedding from <c>embeddings.jsonl</c>, text from <c>tokenizer.jsonl</c>.</summary>
internal sealed record BenchCase(string Id, long[] Ids, string Text, float[] Embedding);

/// <summary>One workload of the shared workload file (<c>ai-dotnet/benchmarks/workloads/e5-v1.json</c>).</summary>
internal sealed record BenchWorkload(string Name, string Input, int Batch, int[] Concurrency, IReadOnlyDictionary<string, int> Samples, BenchCase[][] Requests);

/// <summary>The workload file, resolved against the protected fixtures.</summary>
internal sealed record BenchWorkloadSpec(string Path, int WarmupCalls, BenchCase[] ColdRequest, IReadOnlyList<BenchWorkload> Workloads)
{
    public static string DefaultPath => System.IO.Path.Combine(E5TestAssets.RepositoryRoot, "ai-dotnet", "benchmarks", "workloads", "e5-v1.json");

    public static BenchWorkloadSpec Load(string? path = null)
    {
        path = System.IO.Path.GetFullPath(path ?? DefaultPath);
        var cases = BenchFixtures.Cases;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        BenchCase[] Resolve(JsonElement ids) => ids.EnumerateArray().Select(e => cases[e.GetString()!]).ToArray();
        var workloads = root.GetProperty("workloads").EnumerateArray().Select(w => new BenchWorkload(
            w.GetProperty("name").GetString()!,
            w.GetProperty("input").GetString()!,
            w.GetProperty("batch").GetInt32(),
            w.GetProperty("concurrency").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            w.GetProperty("samples").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal),
            w.GetProperty("requests").EnumerateArray().Select(Resolve).ToArray())).ToList();
        return new BenchWorkloadSpec(
            path,
            root.GetProperty("warmup_calls").GetInt32(),
            Resolve(root.GetProperty("cold").GetProperty("request")),
            workloads);
    }
}

/// <summary>The protected e5 fixtures as benchmark cases (read only).</summary>
internal static class BenchFixtures
{
    private static readonly Lazy<IReadOnlyDictionary<string, BenchCase>> s_cases = new(() =>
        E5Fixtures.Cases.ToDictionary(
            c => c.Id,
            c => new BenchCase(c.Id, c.InputIds, E5Fixtures.Texts[c.Id], c.Embedding),
            StringComparer.Ordinal));

    /// <summary>Case id to case, in <c>embeddings.jsonl</c> order when enumerated through <see cref="Ordered"/>.</summary>
    public static IReadOnlyDictionary<string, BenchCase> Cases => s_cases.Value;

    public static IReadOnlyList<BenchCase> Ordered => E5Fixtures.Cases.Select(c => Cases[c.Id]).ToList();

    public static string FixtureDirectory(OnnxExecutionProvider provider) =>
        provider == OnnxExecutionProvider.Cuda ? E5TestAssets.CudaFixtureDirectory : E5TestAssets.CpuFixtureDirectory;

    /// <summary>Model and tokenizer paths: <c>$STARLING_E5_SNAPSHOT</c>, else the acquisition cache. Never downloads.</summary>
    public static E5ModelFiles Files(string? snapshot)
    {
        if (!string.IsNullOrEmpty(snapshot))
        {
            return E5ModelAssets.OpenDirectory(snapshot, CacheVerification.Size);
        }

        var (files, reason) = E5TestAssets.FindFiles();
        return files ?? throw new InvalidOperationException(reason);
    }

    /// <summary>The reference embeddings of a request, in order.</summary>
    public static float[][] Embeddings(IReadOnlyList<BenchCase> request) => request.Select(c => c.Embedding).ToArray();

    /// <summary>The artifact block of an e5 run record.</summary>
    public static JsonObject Artifacts(E5ModelFiles files, string? modelSha256, string workloadPath, string nativePackage)
    {
        var packages = BenchEnvironment.CommonPackages(nativePackage);
        packages["Starlings.Models.E5"] = BenchEnvironment.InformationalVersion(typeof(E5Encoder).Assembly);
        var fixtures = new JsonObject
        {
            ["embeddings.jsonl"] = BenchEnvironment.Sha256File(Path.Combine(E5TestAssets.CpuFixtureDirectory, "embeddings.jsonl")),
            ["tokenizer.jsonl"] = BenchEnvironment.Sha256File(Path.Combine(E5TestAssets.CpuFixtureDirectory, "tokenizer.jsonl")),
        };
        return BenchEnvironment.Artifacts(files.ModelPath, modelSha256, files.TokenizerPath, workloadPath, fixtures, packages);
    }
}
