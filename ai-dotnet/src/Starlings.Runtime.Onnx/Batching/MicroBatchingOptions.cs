namespace Starlings.Runtime.Onnx.Batching;

/// <summary>
/// Limits of a <see cref="MicroBatcher{TItem, TResult}"/>: how long the oldest queued item may wait for company, how
/// many rows and padded tokens one dispatched batch may hold, and how many items may wait in the queue.
/// </summary>
/// <remarks>
/// Experimental (capability <c>extensions_ai.micro_batching</c> of the e5 and EmbeddingGemma 2 packages,
/// <c>micro_batching</c> of GLiNER2). With micro-batching, one caller's request runs in a padded batch with other
/// callers' requests, so its output depends on the concurrent traffic: equal to a single call within the profile's
/// padded-batch tolerance, not bit for bit reproducible.
/// </remarks>
public sealed record MicroBatchingOptions
{
    /// <summary>
    /// Longest time the oldest queued item waits for more items before its batch is dispatched anyway. Default
    /// <see cref="TimeSpan.Zero"/>: no added latency; a batch is dispatched as soon as the dispatcher is free and holds
    /// whatever queued while the previous batch ran (adaptive batching). A positive value trades up to that much
    /// latency for fuller batches under light load. Waits are timed with the operating system's timer resolution (about
    /// a millisecond). Must be zero or positive and at most <see cref="int.MaxValue"/> milliseconds.
    /// </summary>
    public TimeSpan MaxWait { get; init; } = TimeSpan.Zero;

    /// <summary>
    /// Most items in one dispatched batch. Default 8: in the e5 batching measurements (short texts) CUDA throughput
    /// saturated at about 8 rows and CPU throughput was highest at 8 and fell at 32. Provisional until the concurrency
    /// benchmarks decide. The model's own <c>MaxBatchSize</c> still splits a dispatched batch into ONNX Runtime calls.
    /// </summary>
    public int MaxRows { get; init; } = 8;

    /// <summary>
    /// Most padded token cells in one dispatched batch (rows x the longest row's token count), or null (default) for no
    /// token limit. An item whose own count exceeds the limit is dispatched alone.
    /// </summary>
    public int? MaxTokens { get; init; }

    /// <summary>
    /// Most items waiting to be dispatched (items in a running batch do not count). When the queue is full,
    /// <see cref="MicroBatcher{TItem, TResult}.EnqueueAsync"/> applies backpressure: its task completes only once a slot
    /// is free (the caller's token cancels the wait). Default 1,024.
    /// </summary>
    public int QueueCapacity { get; init; } = 1_024;

    /// <summary>Throws if a value is out of range.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A limit is out of range.</exception>
    public void Validate()
    {
        if (MaxWait < TimeSpan.Zero || MaxWait.TotalMilliseconds > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxWait), MaxWait, "MaxWait must be zero or positive and at most int.MaxValue milliseconds.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(MaxRows, 1, nameof(MaxRows));
        if (MaxTokens is int tokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(tokens, 1, nameof(MaxTokens));
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(QueueCapacity, 1, nameof(QueueCapacity));
    }
}

/// <summary>Counters of a <see cref="MicroBatcher{TItem, TResult}"/> since it was created.</summary>
/// <param name="Batches">Batches dispatched to the batch function.</param>
/// <param name="Items">Items in those batches (average batch size: <c>Items / Batches</c>).</param>
/// <param name="TokenCells">Padded token cells of those batches (rows x longest row), 0 without a token counter.</param>
/// <param name="Tokens">Token counts of those items, 0 without a token counter (padding efficiency: <c>Tokens / TokenCells</c>).</param>
/// <param name="QueueTime">Total time items waited between being queued and their batch being dispatched.</param>
/// <param name="RunTime">Total time spent in the batch function.</param>
public readonly record struct MicroBatcherStatistics(long Batches, long Items, long TokenCells, long Tokens, TimeSpan QueueTime, TimeSpan RunTime);
