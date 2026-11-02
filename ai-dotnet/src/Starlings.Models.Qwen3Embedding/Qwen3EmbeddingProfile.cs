using Starlings.Acquisition;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>What a <see cref="Qwen3EmbeddingProfile"/> computes.</summary>
public enum Qwen3EmbeddingTask
{
    /// <summary>One L2-normalized vector per text (<see cref="Qwen3TextEmbedder"/>).</summary>
    Embedding,

    /// <summary>One relevance score per query-document pair (<see cref="Qwen3Reranker"/>).</summary>
    Reranking,
}

/// <summary>
/// A built-in, pinned Qwen3 Embedding series profile: the upstream checkpoint (repository and full commit), the ONNX graph
/// the export recipe in <c>model-pipeline/reference/qwen3-embedding/</c> builds from it (size and SHA-256 of
/// <c>model.onnx</c> and its external data <c>model.onnx.data</c>), the upstream files the released directory carries,
/// and the behavior constants (prompts, special token ids, token limit). Everything is compiled in, so a NuGet consumer
/// needs no repository file. Experimental (<see cref="Qwen3EmbeddingCapabilities"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Graphs.</b> The graphs are not hosted yet, so there is no download step: build them with the recipe
/// (<c>export_qwen3_onnx.py</c>, then <c>release_qwen3_onnx.py</c>, CPU, fp32) from the pinned snapshot. Two
/// independent exports give the same bytes, and <see cref="RuntimeLock"/> records those bytes; the embedder and the
/// reranker verify a directory against it before loading (<see cref="OpenDirectory"/>).
/// </para>
/// <para>
/// <b>Shared geometry.</b> Both checkpoints are a Qwen3 decoder used non-autoregressively: hidden size 1024, 28 layers,
/// vocabulary 151,669, tied embeddings. The graphs take <c>input_ids</c> and <c>attention_mask</c> (int64,
/// <c>[batch, sequence]</c>, right padding with <see cref="PadTokenId"/>) and have no KV cache.
/// </para>
/// </remarks>
public sealed class Qwen3EmbeddingProfile
{
    /// <summary>The graph in a released directory.</summary>
    public const string GraphFile = "model.onnx";

    /// <summary>The graph's external weights, next to <see cref="GraphFile"/>.</summary>
    public const string GraphDataFile = "model.onnx.data";

    /// <summary>The upstream <c>tokenizer.json</c>, unchanged.</summary>
    public const string TokenizerFile = "tokenizer.json";

    /// <summary><c>&lt;|endoftext|&gt;</c>: the embedding tokenizer appends it, and both profiles pad with it.</summary>
    public const int EndOfTextTokenId = 151643;

    /// <summary>The right-padding id of both graphs (<see cref="EndOfTextTokenId"/>, the upstream <c>pad_token</c>).</summary>
    public const int PadTokenId = EndOfTextTokenId;

    /// <summary>Token ids accepted by the graphs: the tokenizer vocabulary including added tokens (151,669).</summary>
    public const int VocabularySize = 151669;

    /// <summary>Hidden-state width of both checkpoints, and the embedding dimension.</summary>
    public const int HiddenSize = 1024;

    /// <summary>Tokens per input at most, special tokens, prompts and the reranker template included.</summary>
    public const int DefaultMaxTokens = 8192;

    /// <summary>The reranker's <c>"yes"</c> token: the score is <c>logit[yes] - logit[no]</c> at the last position.</summary>
    public const int YesTokenId = 9693;

    /// <summary>The reranker's <c>"no"</c> token.</summary>
    public const int NoTokenId = 2152;

    /// <summary>
    /// The query instruction of the upstream <c>config_sentence_transformers.json</c> (<c>prompts.query</c>), verbatim:
    /// no trailing space.
    /// </summary>
    public const string QueryPrompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

    /// <summary>
    /// The reranker's default instruction, the <c>default(...)</c> of the pinned <c>chat_template.jinja</c> and the card's
    /// <c>task</c>.
    /// </summary>
    public const string DefaultRerankInstruction = "Given a web search query, retrieve relevant passages that answer the query";

    /// <summary>
    /// The reranker's system and user-turn opening, as the model card's transformers snippet and the pinned
    /// <c>chat_template.jinja</c> write it. Tokenized on its own, without special tokens added.
    /// </summary>
    public const string RerankPrefix =
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. " +
        "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";

    /// <summary>The reranker's user-turn end and empty thinking block, as <see cref="RerankPrefix"/>.</summary>
    public const string RerankSuffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    private Qwen3EmbeddingProfile(string id, Qwen3EmbeddingTask task, string repository, string revision, string tokenizerProfile,
        ModelAssetFile[] files, IReadOnlyDictionary<string, string> prompts)
    {
        Id = id;
        Task = task;
        Repository = repository;
        Revision = revision;
        TokenizerProfile = tokenizerProfile;
        Prompts = prompts;
        RuntimeLock = new ModelAssetLock(id + "-onnx", repository, revision, files);
    }

    /// <summary>
    /// <c>Qwen/Qwen3-Embedding-0.6B</c> at <c>97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3</c>: last-token pooling, L2
    /// normalization, 1024 floats; tokenizer profile <c>qwen3-bpe-embedding</c> (appends <c>&lt;|endoftext|&gt;</c>).
    /// </summary>
    public static Qwen3EmbeddingProfile Embedding06B { get; } = new(
        "qwen3-embedding-0.6b",
        Qwen3EmbeddingTask.Embedding,
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "qwen3-bpe-embedding",
        [
            new(GraphFile, 337132, "de6805d74481fefc313ea470de3877f64fe4093c99ce9d1172a026128d90d864"),
            new(GraphDataFile, 2383077376, "7c7569e58783ee0ad8c5fb797d7944aa4f5928af53fb4c1f626f71885af22969"),
            new(TokenizerFile, 11423705, "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
            new("config.json", 727, "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
            new("tokenizer_config.json", 9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
            new("config_sentence_transformers.json", 215, "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961"),
        ],
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["none"] = string.Empty,
            ["document"] = string.Empty,
            ["query"] = QueryPrompt,
        }.AsReadOnly());

    /// <summary>
    /// <c>Qwen/Qwen3-Reranker-0.6B</c> at <c>e61197ed45024b0ed8a2d74b80b4d909f1255473</c>: the score is the raw log-odds
    /// <c>logit[yes] - logit[no]</c> at the last position (sentence-transformers <c>LogitScore</c>); tokenizer profile
    /// <c>qwen3-bpe-chat</c> (byte-identical to Qwen/Qwen3-0.6B's).
    /// </summary>
    public static Qwen3EmbeddingProfile Reranker06B { get; } = new(
        "qwen3-reranker-0.6b",
        Qwen3EmbeddingTask.Reranking,
        "Qwen/Qwen3-Reranker-0.6B",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "qwen3-bpe-chat",
        [
            new(GraphFile, 337355, "b162410db82c0e26f231cda7ddf1ec659ed6707c90e82e5556fb8f1076997581"),
            new(GraphDataFile, 2383085568, "fbfe2bddcffc9e8b4a1145bf02f4ebc99119e57e38d32bf7f04172a7d0cf9c7c"),
            new(TokenizerFile, 11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            new("config.json", 727, "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0"),
            new("tokenizer_config.json", 9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
            new("chat_template.jinja", 741, "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d"),
        ],
        new Dictionary<string, string>(StringComparer.Ordinal).AsReadOnly());

    /// <summary>Every built-in profile.</summary>
    public static IReadOnlyList<Qwen3EmbeddingProfile> BuiltIn { get; } = [Embedding06B, Reranker06B];

    /// <summary>Profile id (<c>qwen3-embedding-0.6b</c>, <c>qwen3-reranker-0.6b</c>).</summary>
    public string Id { get; }

    /// <summary>What the profile computes.</summary>
    public Qwen3EmbeddingTask Task { get; }

    /// <summary>Upstream Hub repository of the checkpoint.</summary>
    public string Repository { get; }

    /// <summary>Full upstream commit SHA.</summary>
    public string Revision { get; }

    /// <summary>The Starlings.Tokenizers profile that qualifies the pinned <c>tokenizer.json</c>.</summary>
    public string TokenizerProfile { get; }

    /// <summary>
    /// Named prompts prepended to the text before tokenization (embedding profile: <c>none</c> and <c>document</c> are
    /// empty, <c>query</c> is <see cref="QueryPrompt"/>); empty for the reranker.
    /// </summary>
    public IReadOnlyDictionary<string, string> Prompts { get; }

    /// <summary>
    /// Size and SHA-256 of every file of a released directory: the graph and its external data (from the export recipe)
    /// and the upstream files at <see cref="Revision"/>. <see cref="ModelAssetLock.Repository"/> and
    /// <see cref="ModelAssetLock.Revision"/> name the upstream checkpoint; the graph files are not in that repository and
    /// are not downloadable yet.
    /// </summary>
    public ModelAssetLock RuntimeLock { get; }

    /// <summary>Tokens per input at most (<see cref="DefaultMaxTokens"/>).</summary>
    public int MaxTokens { get; } = DefaultMaxTokens;

    /// <summary>
    /// Verifies a released directory against <see cref="RuntimeLock"/> (every file's size, and its SHA-256 by default).
    /// Never downloads.
    /// </summary>
    /// <param name="directory">The directory <c>release_qwen3_onnx.py</c> wrote.</param>
    /// <param name="verification">SHA-256 (default) or size only.</param>
    /// <exception cref="ModelAcquisitionException">A file is missing or differs from the lock.</exception>
    public ModelSnapshot OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
        ModelAssetStore.VerifyDirectory(RuntimeLock, directory, verification);

    /// <summary>Resolves a built-in profile id.</summary>
    /// <exception cref="NotSupportedException">Not a built-in profile.</exception>
    public static Qwen3EmbeddingProfile FromId(string id) =>
        BuiltIn.FirstOrDefault(p => string.Equals(p.Id, id, StringComparison.Ordinal))
        ?? throw new NotSupportedException($"Unknown Qwen3 Embedding series profile '{id}'; built in: {string.Join(", ", BuiltIn.Select(p => p.Id))}.");

    /// <inheritdoc />
    public override string ToString() => $"{Id} ({Repository}@{Revision})";
}
