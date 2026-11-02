namespace Starlings.Models.Bert;

/// <summary>Token ids, mask, and segment ids for a single unpadded or right-padded BERT row.</summary>
/// <remarks>Caller-owned buffers must remain unchanged until inference returns.</remarks>
public sealed record BertInput(ReadOnlyMemory<long> InputIds, ReadOnlyMemory<long> AttentionMask, ReadOnlyMemory<long> TokenTypeIds)
{
    /// <summary>Sequence length including special tokens.</summary>
    public int Length => InputIds.Length;
}

/// <summary>Explicit query/document input to the reranker.</summary>
public sealed record BertPair(string Query, string Document);

/// <summary>A document's original index and raw relevance score; higher scores rank first.</summary>
public sealed record BertRankedDocument(int Index, string Document, float Score);

/// <summary>Feature states of the new CPU candidate route.</summary>
public static class BertCapabilities
{
    /// <summary>Text processing and inference are experimental development candidates.</summary>
    public const string Text = "experimental";
    /// <summary>Explicit padded batches are experimental.</summary>
    public const string Batch = "experimental";
    /// <summary>The optional token budget per batch call (<c>maxBatchTokens</c>) is experimental.</summary>
    public const string BatchTokenBudget = "experimental";
    /// <summary>Length-sorted chunking with results in input order (<c>sortByLength</c>) is experimental.</summary>
    public const string BatchSortByLength = "experimental";
    /// <summary>Cancellation is experimental, at native graph-node and batch-chunk boundaries.</summary>
    public const string Cancellation = "experimental";
    /// <summary>
    /// Text truncation (capability key <c>text.truncation</c>, #193): text is truncated on the right to the profile's
    /// limit, special tokens and any prompt included, as the pinned tokenizer does. Experimental, as the whole route is.
    /// </summary>
    public const string TextTruncation = "experimental";
    /// <summary>Token counting without embedding (<c>CountTokens</c>, <c>Validate</c>) and <c>TextChunker</c> over <c>ChunkingTokenizer</c> are experimental.</summary>
    public const string TokenCounting = "experimental";
    /// <summary>No incremental output contract.</summary>
    public const string Streaming = "unsupported";
    /// <summary>Model saving is not implemented.</summary>
    public const string Persistence = "unsupported";
    /// <summary>No training path.</summary>
    public const string Training = "unsupported";
    /// <summary>Native AOT has not been qualified.</summary>
    public const string Aot = "untested";

    /// <summary>
    /// The capability record of every registry profile this package serves (<see cref="ModelProfileCapabilities"/>),
    /// generated from the profiles' registry manifests by <c>model-pipeline catalog</c>, keyed by profile id. Each
    /// record's <see cref="ModelProfileCapabilities.Capabilities"/> is the manifest's <c>capabilities</c> map key for key.
    /// </summary>
    public static IReadOnlyDictionary<string, ModelProfileCapabilities> ProfileCapabilities => GeneratedProfileCapabilities.Profiles;
}
