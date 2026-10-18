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
    private BertModelProfile(string id, ModelAssetLock sourceLock, BertTask task, int maxTokens,
        string pooling, int hiddenSize, string normalize, IReadOnlyDictionary<string, string> prompts)
    {
        Id = id;
        SourceLock = sourceLock;
        Task = task;
        MaxTokens = maxTokens;
        Pooling = pooling;
        HiddenSize = hiddenSize;
        Normalize = normalize;
        Prompts = new ReadOnlyDictionary<string, string>(new Dictionary<string, string>(prompts));
        RuntimeLock = new ModelAssetLock(id, sourceLock.Repository, sourceLock.Revision,
            [sourceLock.GetFile(GraphFile), sourceLock.GetFile(TokenizerFile)]);
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
    /// <summary>Right-padding id of the admitted uncased BERT tokenizer.</summary>
    public const int PadTokenId = 0;
    /// <summary>Exclusive token id upper bound.</summary>
    public const int VocabularySize = 30522;
    /// <summary>Graph path inside the pinned snapshot.</summary>
    public const string GraphFile = "onnx/model.onnx";
    /// <summary>Owned tokenizer asset path.</summary>
    public const string TokenizerFile = "tokenizer.json";
    /// <summary>The optional BGE retrieval-query prefix; callers choose it explicitly.</summary>
    public const string BgeQueryInstruction = "Represent this sentence for searching relevant passages: ";

    /// <summary>Reads a portable candidate JSON contract; never executes or downloads source.</summary>
    public static BertModelProfile Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parses and rejects contracts that differ from the five implemented task profiles.</summary>
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
        var (task, maxTokens, pooling, hiddenSize, normalize) = repository switch
        {
            "sentence-transformers/all-MiniLM-L6-v2" => (BertTask.Embeddings, 256, "mean", 384, "l2"),
            "BAAI/bge-small-en-v1.5" => (BertTask.Embeddings, 512, "cls", 384, "l2"),
            "BAAI/bge-base-en-v1.5" => (BertTask.Embeddings, 512, "cls", 768, "l2"),
            "cross-encoder/ms-marco-MiniLM-L6-v2" => (BertTask.Reranking, 512, "none", 384, "none"),
            "nomic-ai/nomic-embed-text-v1.5" => (BertTask.Embeddings, 8192, "mean", 768, "none"),
            _ => throw new NotSupportedException($"Unsupported BERT repository '{repository}'."),
        };
        Require(root, "task", task == BertTask.Embeddings ? "embeddings" : "reranking");
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
        Require(behavior, "tokenizer_profile", "bert-wordpiece-uncased");
        Require(behavior, "pooling", pooling);
        Require(behavior, "normalize", normalize);
        Require(behavior, "output_name", task == BertTask.Embeddings ? "last_hidden_state" : "logits");
        Require(behavior, "max_tokens", maxTokens);
        Require(behavior, "hidden_size", hiddenSize);
        Require(behavior, "pad_token_id", PadTokenId);
        Require(behavior, "vocabulary_size", VocabularySize);
        if (task == BertTask.Reranking)
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
        return new BertModelProfile(id, assetLock, task, maxTokens, pooling, hiddenSize, normalize, prompts);
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
