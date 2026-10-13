using System.Text.Json;

namespace Starlings.Runtime.Onnx;

/// <summary>Reads node placement from an ONNX Runtime profiler JSON file (Chrome trace format).</summary>
internal static class PlacementProfileParser
{
    private const string KernelTimeSuffix = "_kernel_time";

    public static OnnxPlacementReport Parse(Stream json)
    {
        using var document = JsonDocument.Parse(json);
        var nodes = new Dictionary<string, (string Provider, string Op)>(StringComparer.Ordinal);
        foreach (var e in document.RootElement.EnumerateArray())
        {
            if (!e.TryGetProperty("cat", out var cat) || cat.GetString() != "Node")
            {
                continue;
            }

            if (!e.TryGetProperty("args", out var args) ||
                !args.TryGetProperty("provider", out var provider) ||
                !args.TryGetProperty("op_name", out var op) ||
                !e.TryGetProperty("name", out var nameElement))
            {
                continue;
            }

            string name = nameElement.GetString() ?? string.Empty;
            if (!name.EndsWith(KernelTimeSuffix, StringComparison.Ordinal))
            {
                continue;
            }

            name = name[..^KernelTimeSuffix.Length];
            nodes[name] = (provider.GetString() ?? "unknown", op.GetString() ?? "unknown");
        }

        var byProvider = nodes.GroupBy(n => n.Value.Provider, StringComparer.Ordinal).ToList();
        return new OnnxPlacementReport(
            byProvider.ToDictionary(g => g.Key, g => g.Count(), StringComparer.Ordinal),
            byProvider.ToDictionary(
                g => g.Key,
                g => (IReadOnlyDictionary<string, int>)g.GroupBy(n => n.Value.Op, StringComparer.Ordinal)
                    .ToDictionary(o => o.Key, o => o.Count(), StringComparer.Ordinal),
                StringComparer.Ordinal),
            byProvider.ToDictionary(
                g => g.Key,
                g => (IReadOnlyList<string>)g.Select(n => n.Key).Order(StringComparer.Ordinal).ToList(),
                StringComparer.Ordinal));
    }
}
