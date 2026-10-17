using System.Text.Json.Nodes;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Benchmarks;

/// <summary>
/// Micro-batching helpers shared by the sweep targets of the e5, GLiNER2 and EmbeddingGemma 2 harnesses. Linked only into
/// those harnesses (and their .Cuda builds), which reference Starlings.Runtime.Onnx; the rest of Core does not.
/// </summary>
internal static class SweepTargets
{
    /// <summary>The micro-batcher's counters as the harness core records them, or null when micro-batching is off.</summary>
    public static SweepBatcherStats? Stats(MicroBatcherStatistics? statistics) =>
        statistics is not { } s
            ? null
            : new SweepBatcherStats(s.Batches, s.Items, s.TokenCells, s.Tokens, s.QueueTime.TotalMilliseconds, s.RunTime.TotalMilliseconds);

    /// <summary>The micro-batching options in effect for the run record's settings, or null when off.</summary>
    public static JsonObject? MicroBatching(MicroBatchingOptions? options) =>
        options is null
            ? null
            : new JsonObject
            {
                ["max_wait_ms"] = options.MaxWait.TotalMilliseconds,
                ["max_rows"] = options.MaxRows,
                ["max_tokens"] = options.MaxTokens,
                ["queue_capacity"] = options.QueueCapacity,
            };
}
