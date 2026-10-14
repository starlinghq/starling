using System.Text.Json;

namespace Starlings.Benchmarks;

/// <summary>One workload of a model workload file (<c>ai-dotnet/benchmarks/workloads/*.json</c>); model fields stay in <see cref="Raw"/>.</summary>
internal sealed record WorkloadEntry(string Name, string Path, string Input, int Batch, int Samples, int[] Concurrency, int? WarmupCalls, JsonElement Raw);

/// <summary>A model workload file: warmup calls, the cold-start entry and the workloads, in file order.</summary>
internal sealed record WorkloadSpec(string File, int WarmupCalls, JsonElement Cold, IReadOnlyList<WorkloadEntry> Workloads)
{
    public static WorkloadSpec Load(string path)
    {
        using var doc = JsonDocument.Parse(System.IO.File.ReadAllBytes(path));
        var root = doc.RootElement.Clone();
        var workloads = root.GetProperty("workloads").EnumerateArray().Select(w => new WorkloadEntry(
            w.GetProperty("name").GetString()!,
            w.GetProperty("path").GetString()!,
            w.GetProperty("input").GetString()!,
            w.TryGetProperty("batch", out var b) ? b.GetInt32() : 1,
            w.GetProperty("samples").GetInt32(),
            w.TryGetProperty("concurrency", out var c) ? c.EnumerateArray().Select(e => e.GetInt32()).ToArray() : [1],
            w.TryGetProperty("warmup_calls", out var wc) ? wc.GetInt32() : null,
            w)).ToList();
        return new WorkloadSpec(path, root.GetProperty("warmup_calls").GetInt32(), root.GetProperty("cold"), workloads);
    }

    /// <summary>A string array property.</summary>
    public static string[] Strings(JsonElement element) => element.EnumerateArray().Select(e => e.GetString()!).ToArray();
}
