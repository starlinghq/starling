using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Acquisition;

namespace Starlings.Models.Bert;

/// <summary>An explicitly supported task; embeddings and reranking are distinct APIs.</summary>
public enum BertTask
{
    /// <summary>A normalized text vector.</summary>
    Embeddings,
    /// <summary>A scalar raw relevance logit for a query/document pair.</summary>
    Reranking,
}

/// <summary>Pinned source identity and exact processing semantics of one experimental candidate.</summary>
public sealed class BertModelProfile
{
    /// <summary>Per-repository BERT processing semantics.</summary>
    private readonly record struct BertSpec(
        BertTask Task, int MaxTokens, string Pooling, int HiddenSize, string Normalize,
        int VocabularySize, int ClsTokenId, int SepTokenId, int PadTokenId,
        string OutputName, string TokenizerProfile, bool TokenTypeIds);

    /// <summary>Common BERT/WordPiece geometry; pass overrides for other tokenizers or geometries.</summary>
    private static BertSpec Bert(BertTask task, int maxTokens, string pooling, int hiddenSize, string normalize,
        int vocabularySize = 30522, int clsTokenId = 101, int sepTokenId = 102, int padTokenId = 0,
        string? outputName = null, string tokenizerProfile = "bert-wordpiece-uncased", bool tokenTypeIds = true)
    {
        outputName ??= task == BertTask.Embeddings ? "last_hidden_state" : "logits";
        return new BertSpec(task, maxTokens, pooling, hiddenSize, normalize,
            vocabularySize, clsTokenId, sepTokenId, padTokenId, outputName, tokenizerProfile, tokenTypeIds);
    }

    private BertModelProfile(string id, ModelAssetLock sourceLock, BertSpec spec,
        IReadOnlyDictionary<string, string> prompts)
    {
        Id = id;
        SourceLock = sourceLock;
        Task = spec.Task;
        MaxTokens = spec.MaxTokens;
        Pooling = spec.Pooling;
        HiddenSize = spec.HiddenSize;
        Normalize = spec.Normalize;
        VocabularySize = spec.VocabularySize;
        ClsTokenId = spec.ClsTokenId;
        SepTokenId = spec.SepTokenId;
        PadTokenId = spec.PadTokenId;
        OutputName = spec.OutputName;
        TokenizerProfile = spec.TokenizerProfile;
        TokenTypeIds = spec.TokenTypeIds;
        Prompts = new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(prompts));
        RuntimeLock = RuntimeLockFor(id, sourceLock);
    }

    /// <summary>Candidate profile identifier.</summary>
    public string Id { get; }
    /// <summary>Exact upstream file lock used by reference jobs.</summary>
    public ModelAssetLock SourceLock { get; }
    /// <summary>The graph and tokenizer files required for offline production inference.</summary>
    public ModelAssetLock RuntimeLock { get; }
    /// <summary>Declared task.</summary>
    public BertTask Task { get; }
    /// <summary>Maximum tokens including special tokens and an explicit prompt.</summary>
    public int MaxTokens { get; }
    /// <summary>Mean, CLS, or none (scalar reranking).</summary>
    public string Pooling { get; }
    /// <summary>Explicit named prefixes. None is always the empty string.</summary>
    public IReadOnlyDictionary<string, string> Prompts { get; }
    /// <summary>Embedding width of the profile (per-repository).</summary>
    public int HiddenSize { get; }
    /// <summary>Embedding normalization (l2 or none, per-repository).</summary>
    public string Normalize { get; }
    /// <summary>Right-padding id of the tokenizer.</summary>
    public int PadTokenId { get; }
    /// <summary>Exclusive token id upper bound.</summary>
    public int VocabularySize { get; }
    /// <summary>Classification (first) token id.</summary>
    public int ClsTokenId { get; }
    /// <summary>Separator (last) token id.</summary>
    public int SepTokenId { get; }
    /// <summary>Graph output name to pool (token-level for embeddings, scalar for reranking).</summary>
    public string OutputName { get; }
    /// <summary>Declared tokenizer profile.</summary>
    public string TokenizerProfile { get; }
    /// <summary>Whether the graph takes token_type_ids (BERT segment ids); false for XLM-R/ModernBERT.</summary>
    public bool TokenTypeIds { get; }
    /// <summary>Graph path inside the pinned snapshot.</summary>
    public const string GraphFile = "onnx/model.onnx";
    /// <summary>
    /// External weights of the graph, for graphs over the 2 GB protobuf limit. Part of <see cref="RuntimeLock"/> exactly
    /// when the source lock pins it.
    /// </summary>
    public const string GraphDataFile = "onnx/model.onnx_data";
    /// <summary>Owned tokenizer asset path.</summary>
    public const string TokenizerFile = "tokenizer.json";
    /// <summary>The optional BGE retrieval-query prefix; callers choose it explicitly.</summary>
    public const string BgeQueryInstruction = "Represent this sentence for searching relevant passages: ";

    /// <summary>Reads a portable candidate JSON contract; never executes or downloads source.</summary>
    public static BertModelProfile Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parses and rejects contracts that differ from the implemented task profiles.</summary>
    public static BertModelProfile Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (root.GetProperty("format_version").GetInt32() != 1)
        {
            throw new NotSupportedException("Only candidate format_version 1 is supported.");
        }

        string id = root.GetProperty("id").GetString()!;
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        var origin = root.GetProperty("origin");
        string repository = origin.GetProperty("repo").GetString()!;
        var spec = repository switch
        {
            "sentence-transformers/all-MiniLM-L6-v2" => Bert(BertTask.Embeddings, 256, "mean", 384, "l2"),
            "BAAI/bge-small-en-v1.5" => Bert(BertTask.Embeddings, 512, "cls", 384, "l2"),
            "BAAI/bge-base-en-v1.5" => Bert(BertTask.Embeddings, 512, "cls", 768, "l2"),
            "cross-encoder/ms-marco-MiniLM-L6-v2" => Bert(BertTask.Reranking, 512, "none", 384, "none"),
            "nomic-ai/nomic-embed-text-v1.5" => Bert(BertTask.Embeddings, 8192, "mean", 768, "none"),
            "BAAI/bge-m3" => Bert(BertTask.Embeddings, 8192, "cls", 1024, "l2",
                vocabularySize: 250002, clsTokenId: 0, sepTokenId: 2, padTokenId: 1,
                outputName: "token_embeddings", tokenizerProfile: "xlmr-unigram", tokenTypeIds: false),
            "Starlinghq/bge-reranker-v2-m3-onnx" => Bert(BertTask.Reranking, 8192, "none", 1024, "none",
                vocabularySize: 250002, clsTokenId: 0, sepTokenId: 2, padTokenId: 1,
                tokenizerProfile: "xlmr-unigram", tokenTypeIds: false),
            _ => throw new NotSupportedException($"Unsupported BERT repository '{repository}'."),
        };
        Require(root, "task", spec.Task == BertTask.Embeddings ? "embeddings" : "reranking");
        var runtime = root.GetProperty("runtime");
        Require(runtime, "backend", "onnx");
        Require(runtime, "version", "1.30.0");
        Require(runtime, "provider", "CPUExecutionProvider");
        Require(runtime, "precision", "fp32");
        var behavior = root.GetProperty("behavior");
        Require(behavior, "architecture", "bert");
        Require(behavior, "graph_file", GraphFile);
        Require(behavior, "tokenizer_file", TokenizerFile);
        Require(behavior, "padding_side", "right");
        Require(behavior, "truncation", "longest_first");
        Require(behavior, "tokenizer_profile", spec.TokenizerProfile);
        Require(behavior, "pooling", spec.Pooling);
        Require(behavior, "normalize", spec.Normalize);
        Require(behavior, "output_name", spec.OutputName);
        Require(behavior, "max_tokens", spec.MaxTokens);
        Require(behavior, "hidden_size", spec.HiddenSize);
        Require(behavior, "pad_token_id", spec.PadTokenId);
        Require(behavior, "vocabulary_size", spec.VocabularySize);
        if (spec.Task == BertTask.Reranking)
        {
            Require(behavior, "score_activation", "identity");
        }

        var prompts = behavior.GetProperty("prompts").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal);
        int expectedCount = repository is "BAAI/bge-small-en-v1.5" or "BAAI/bge-base-en-v1.5" ? 2 : 1;
        if (prompts.Count != expectedCount || !prompts.TryGetValue("none", out var none) || none != ""
            || (expectedCount == 2 && (!prompts.TryGetValue("query", out var query) || query != BgeQueryInstruction)))
        {
            throw new NotSupportedException("Prompt contract does not match the implemented BERT profile.");
        }

        var assetLock = new ModelAssetLock(id, repository, origin.GetProperty("revision").GetString()!,
            origin.GetProperty("files").EnumerateArray().Select(f => new ModelAssetFile(
                f.GetProperty("path").GetString()!, f.GetProperty("size").GetInt64(), f.GetProperty("sha256").GetString()!)));
        return new BertModelProfile(id, assetLock, spec, prompts);
    }

    /// <summary>The graph, its external data when the source pins one, and the tokenizer.</summary>
    internal static ModelAssetLock RuntimeLockFor(string id, ModelAssetLock sourceLock)
    {
        var files = new List<ModelAssetFile> { sourceLock.GetFile(GraphFile) };
        var data = sourceLock.Files.FirstOrDefault(f => string.Equals(f.Path, GraphDataFile, StringComparison.Ordinal));
        if (data is not null)
        {
            files.Add(data);
        }

        files.Add(sourceLock.GetFile(TokenizerFile));
        return new ModelAssetLock(id, sourceLock.Repository, sourceLock.Revision, files);
    }

    private static void Require(JsonElement parent, string key, string expected)
    {
        if (parent.GetProperty(key).GetString() != expected)
        {
            throw new NotSupportedException($"Unsupported BERT contract: {key} must be '{expected}'.");
        }
    }

    private static void Require(JsonElement parent, string key, int expected)
    {
        if (parent.GetProperty(key).GetInt32() != expected)
        {
            throw new NotSupportedException($"Unsupported BERT contract: {key} must be {expected}.");
        }
    }
}
