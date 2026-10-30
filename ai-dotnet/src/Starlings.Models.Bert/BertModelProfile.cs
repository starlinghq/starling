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
    /// <summary>Per-repository BERT processing semantics: one entry of the route contract table.</summary>
    internal sealed record BertSpec(
        BertTask Task, int MaxTokens, string Pooling, int HiddenSize, string Normalize,
        int VocabularySize, int ClsTokenId, int SepTokenId, int PadTokenId,
        string OutputName, string TokenizerProfile, bool TokenTypeIds, IReadOnlyDictionary<string, string> Prompts);

    /// <summary>Manifest resource name of the embedded route contract table.</summary>
    internal const string ContractsResource = "Starlings.Models.Bert.Contracts.json";

    private const string ContractsFormat = "starlings.bert-route-contracts";
    private const string Route = "bert-onnx-cpu-v1";

    private static readonly string[] RequiredContractFields =
    [
        "task", "max_tokens", "pooling", "hidden_size", "normalize", "vocabulary_size", "cls_token_id", "sep_token_id",
        "pad_token_id", "output_name", "tokenizer_profile", "token_type_ids", "prompts",
    ];

    private static readonly Lazy<IReadOnlyDictionary<string, BertSpec>> EmbeddedContracts = new(LoadEmbeddedContracts);

    /// <summary>
    /// The repositories this route implements, in the order of the route contract table
    /// (<c>model-pipeline/routes/bert-onnx-cpu-v1/contracts.json</c>, embedded in the package). <see cref="Parse"/>
    /// rejects every other repository.
    /// </summary>
    public static IReadOnlyList<string> SupportedRepositories => [.. EmbeddedContracts.Value.Keys];

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

    private const string BuiltInResourcePrefix = "Starlings.Models.Bert.Profiles.";

    /// <summary>
    /// Names of the profiles built into the package, one per supported repository: the candidate directory names of
    /// <c>model-pipeline/candidates/</c> (<c>bge-base</c>, <c>bge-m3</c>, <c>bge-reranker-v2-m3</c>, <c>bge-small</c>,
    /// <c>minilm</c>, <c>ms-marco</c>, <c>nomic-embed</c>), in ordinal order.
    /// </summary>
    public static IReadOnlyList<string> BuiltInIds { get; } = Array.AsReadOnly(new[]
    {
        "bge-base", "bge-m3", "bge-reranker-v2-m3", "bge-small", "minilm", "ms-marco", "nomic-embed",
    });

    /// <summary>
    /// Parses a profile built into the package by its candidate name (<see cref="BuiltInIds"/>). The embedded JSON is
    /// the candidate contract <c>model-pipeline/candidates/&lt;name&gt;/profile.json</c> at the packed commit, so
    /// <c>BuiltIn(name)</c> equals <c>Load</c> of that file.
    /// </summary>
    /// <exception cref="ArgumentException">The name is not a built-in profile.</exception>
    public static BertModelProfile BuiltIn(string name) => Parse(BuiltInJson(name));

    /// <summary>The embedded candidate JSON of a built-in profile, byte for byte.</summary>
    internal static string BuiltInJson(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (!BuiltInIds.Contains(name, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                $"'{name}' is not a built-in BERT profile; expected one of: {string.Join(", ", BuiltInIds)}.", nameof(name));
        }

        using var stream = typeof(BertModelProfile).Assembly.GetManifestResourceStream(BuiltInResourcePrefix + name + ".json")
            ?? throw new InvalidOperationException($"The built-in profile resource for '{name}' is missing from the assembly.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

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
        if (!EmbeddedContracts.Value.TryGetValue(repository, out var spec))
        {
            throw new NotSupportedException($"Unsupported BERT repository '{repository}'.");
        }

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
        if (prompts.Count != spec.Prompts.Count
            || spec.Prompts.Any(p => !prompts.TryGetValue(p.Key, out var value) || value != p.Value))
        {
            throw new NotSupportedException("Prompt contract does not match the implemented BERT profile.");
        }

        var assetLock = new ModelAssetLock(id, repository, origin.GetProperty("revision").GetString()!,
            origin.GetProperty("files").EnumerateArray().Select(f => new ModelAssetFile(
                f.GetProperty("path").GetString()!, f.GetProperty("size").GetInt64(), f.GetProperty("sha256").GetString()!)));
        return new BertModelProfile(id, assetLock, spec, prompts);
    }

    /// <summary>The embedded route contract table, byte for byte.</summary>
    internal static string EmbeddedContractsJson()
    {
        using var stream = typeof(BertModelProfile).Assembly.GetManifestResourceStream(ContractsResource)
            ?? throw new InvalidOperationException($"The BERT route contract table '{ContractsResource}' is missing from the assembly.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static IReadOnlyDictionary<string, BertSpec> LoadEmbeddedContracts() => ParseContracts(EmbeddedContractsJson());

    /// <summary>
    /// Parses and validates the route contract table with the rules of the Python allow-list
    /// (<c>validate_contracts</c> in <c>onboarding_bert.py</c>). A table that breaks them is a packaging defect.
    /// </summary>
    /// <exception cref="InvalidDataException">The table is malformed or breaks a rule.</exception>
    internal static IReadOnlyDictionary<string, BertSpec> ParseContracts(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            var rootKeys = UniqueNames(root, "the table");
            if (!rootKeys.SetEquals(["format", "format_version", "route", "contracts"])
                || root.GetProperty("format").GetString() != ContractsFormat
                || root.GetProperty("format_version").GetInt32() != 1
                || root.GetProperty("route").GetString() != Route)
            {
                throw new InvalidDataException($"BERT route contracts: expected exactly {ContractsFormat} version 1 for {Route}.");
            }

            var contracts = root.GetProperty("contracts");
            UniqueNames(contracts, "contracts");
            var table = new Dictionary<string, BertSpec>(StringComparer.Ordinal);
            var order = new List<string>();
            foreach (var entry in contracts.EnumerateObject())
            {
                table.Add(entry.Name, ParseContract(entry.Name, entry.Value));
                order.Add(entry.Name);
            }

            if (table.Count == 0)
            {
                throw new InvalidDataException("BERT route contracts: the table is empty.");
            }

            // Ordered as in the file, so SupportedRepositories lists the table's order.
            return new OrderedReadOnly(order, table);
        }
        catch (Exception exception) when (exception is JsonException or KeyNotFoundException or InvalidOperationException
            or FormatException)
        {
            throw new InvalidDataException($"BERT route contracts: malformed table: {exception.Message}", exception);
        }
    }

    private static BertSpec ParseContract(string repository, JsonElement spec)
    {
        string where = $"BERT route contract '{repository}'";
        string[] parts = repository.Split('/');
        if (parts.Length != 2 || parts.Any(string.IsNullOrEmpty))
        {
            throw new InvalidDataException($"{where}: the key must be a Hugging Face repository id <owner>/<name>.");
        }

        var names = UniqueNames(spec, where);
        if (!names.IsSupersetOf(RequiredContractFields) || names.Count - RequiredContractFields.Length
            != (names.Contains("score_activation_source") ? 1 : 0))
        {
            throw new InvalidDataException($"{where}: fields must be {string.Join(", ", RequiredContractFields)}, optionally score_activation_source.");
        }

        string String(string key) => spec.GetProperty(key).ValueKind == JsonValueKind.String
            ? spec.GetProperty(key).GetString()!
            : throw new InvalidDataException($"{where}: {key} must be a string.");
        int Integer(string key) => spec.GetProperty(key).ValueKind == JsonValueKind.Number && spec.GetProperty(key).TryGetInt32(out int value)
            ? value
            : throw new InvalidDataException($"{where}: {key} must be an integer.");

        string task = String("task"), pooling = String("pooling"), normalize = String("normalize");
        BertTask parsedTask;
        if (task == "embeddings")
        {
            parsedTask = BertTask.Embeddings;
            if (pooling is not ("mean" or "cls") || normalize is not ("l2" or "none") || names.Contains("score_activation_source"))
            {
                throw new InvalidDataException($"{where}: embeddings need mean or cls pooling, l2 or none normalization and no score_activation_source.");
            }
        }
        else if (task == "reranking")
        {
            parsedTask = BertTask.Reranking;
            if (pooling != "none" || normalize != "none"
                || (names.Contains("score_activation_source") && String("score_activation_source") is not ("config" or "explicit")))
            {
                throw new InvalidDataException($"{where}: reranking needs pooling none, normalize none and a score_activation_source of config or explicit.");
            }
        }
        else
        {
            throw new InvalidDataException($"{where}: task must be embeddings or reranking.");
        }

        string tokenizerProfile = String("tokenizer_profile"), outputName = String("output_name");
        if (tokenizerProfile is not ("bert-wordpiece-uncased" or "xlmr-unigram") || outputName.Length == 0)
        {
            throw new InvalidDataException($"{where}: tokenizer_profile must be bert-wordpiece-uncased or xlmr-unigram and output_name must not be empty.");
        }

        int maxTokens = Integer("max_tokens"), hiddenSize = Integer("hidden_size"), vocabularySize = Integer("vocabulary_size");
        int cls = Integer("cls_token_id"), sep = Integer("sep_token_id"), pad = Integer("pad_token_id");
        if (maxTokens <= 0 || hiddenSize <= 0 || vocabularySize <= 0
            || new[] { cls, sep, pad }.Any(id => id < 0 || id >= vocabularySize))
        {
            throw new InvalidDataException($"{where}: sizes must be positive and the cls, sep and pad ids inside the vocabulary.");
        }

        var tokenTypeIds = spec.GetProperty("token_type_ids");
        if (tokenTypeIds.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new InvalidDataException($"{where}: token_type_ids must be a boolean.");
        }

        var promptsElement = spec.GetProperty("prompts");
        if (promptsElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{where}: prompts must be an object.");
        }

        UniqueNames(promptsElement, where + " prompts");
        var prompts = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var prompt in promptsElement.EnumerateObject())
        {
            prompts.Add(prompt.Name, prompt.Value.ValueKind == JsonValueKind.String
                ? prompt.Value.GetString()!
                : throw new InvalidDataException($"{where}: prompt '{prompt.Name}' must be a string."));
        }

        if (!prompts.TryGetValue("none", out var none) || none.Length != 0)
        {
            throw new InvalidDataException($"{where}: prompts must include none as the empty string.");
        }

        return new BertSpec(parsedTask, maxTokens, pooling, hiddenSize, normalize, vocabularySize, cls, sep, pad,
            outputName, tokenizerProfile, tokenTypeIds.GetBoolean(), new ReadOnlyDictionary<string, string>(prompts));
    }

    /// <summary>The property names of an object; JSON allows duplicates, the contract table does not.</summary>
    private static HashSet<string> UniqueNames(JsonElement element, string where)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"BERT route contracts: {where} must be an object.");
        }

        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in element.EnumerateObject())
        {
            if (!names.Add(property.Name))
            {
                throw new InvalidDataException($"BERT route contracts: duplicate key '{property.Name}' in {where}.");
            }
        }

        return names;
    }

    /// <summary>A read-only dictionary that enumerates in insertion order.</summary>
    private sealed class OrderedReadOnly(IReadOnlyList<string> order, Dictionary<string, BertSpec> table)
        : IReadOnlyDictionary<string, BertSpec>
    {
        public BertSpec this[string key] => table[key];
        public IEnumerable<string> Keys => order;
        public IEnumerable<BertSpec> Values => order.Select(key => table[key]);
        public int Count => table.Count;
        public bool ContainsKey(string key) => table.ContainsKey(key);
        public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out BertSpec value) =>
            table.TryGetValue(key, out value);
        public IEnumerator<KeyValuePair<string, BertSpec>> GetEnumerator() =>
            order.Select(key => new KeyValuePair<string, BertSpec>(key, table[key])).GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
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
