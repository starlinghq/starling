namespace Starlings.Models.Kev;

/// <summary>
/// Budget of the experimental batched-row paths (<see cref="KevDecisionModel.RunBatched"/>,
/// <see cref="KevDecisionModel.RunUncachedBatched"/>). Internal: capability <c>batched_rows</c> stays Unsupported until
/// the maintainer decides (docs/evidence/kev-batched-rows.md).
/// </summary>
internal sealed record KevBatchOptions
{
    /// <summary>Most question rows in one graph run. Default 8.</summary>
    public int MaxRows { get; init; } = 8;

    /// <summary>
    /// Most padded token cells in one graph run, or null for no cap: rows times (past length plus the longest row's
    /// width). On the cached path the past is the state, tiled once per row, so the cap bounds the tiled cache
    /// (Kev-0.8B: about 25 KB of attention cache per past token per row, plus about 19 MB of DeltaNet state per row).
    /// A single row over the cap still runs, alone. Default 32,768.
    /// </summary>
    public int? MaxTokens { get; init; } = 32_768;

    /// <summary>
    /// Group rows by length (longest first) to reduce padding; results are scattered back to question order either way.
    /// Default false: rows are grouped in question order.
    /// </summary>
    public bool SortByLength { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxRows, 1);
        if (MaxTokens is { } tokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(tokens, 1, nameof(MaxTokens));
        }
    }
}

/// <summary>Splits question rows into graph runs under a <see cref="KevBatchOptions"/> budget.</summary>
internal static class KevBatchPlanner
{
    /// <summary>
    /// Groups of row indices, each one graph run. Every index appears exactly once. A group's cost is
    /// <c>count * (pastLength + max width)</c>; a group grows while it stays within both budgets.
    /// </summary>
    public static IReadOnlyList<int[]> Plan(IReadOnlyList<int> widths, int pastLength, KevBatchOptions options)
    {
        ArgumentNullException.ThrowIfNull(widths);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegative(pastLength);
        options.Validate();
        int[] order = [.. Enumerable.Range(0, widths.Count)];
        if (options.SortByLength)
        {
            // Stable: equal widths keep question order.
            order = [.. order.OrderByDescending(i => widths[i])];
        }

        var groups = new List<int[]>();
        var current = new List<int>();
        int width = 0;
        foreach (int i in order)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(widths[i], 1, nameof(widths));
            int nextWidth = Math.Max(width, widths[i]);
            bool fits = current.Count < options.MaxRows
                && (options.MaxTokens is not { } cap || (long)(current.Count + 1) * (pastLength + nextWidth) <= cap);
            if (current.Count > 0 && !fits)
            {
                groups.Add([.. current]);
                current.Clear();
                nextWidth = widths[i];
            }

            current.Add(i);
            width = nextWidth;
        }

        if (current.Count > 0)
        {
            groups.Add([.. current]);
        }

        return groups;
    }
}
