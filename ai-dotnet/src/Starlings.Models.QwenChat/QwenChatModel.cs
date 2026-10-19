using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat;

/// <summary>Text-only generation over admitted Qwen originals and a hash-verified experimental CPU fp32 bundle.</summary>
public sealed class QwenChatModel : IDisposable
{
    private readonly GenAIModel _backend;
    private readonly TextGenerator _generator;

    private QwenChatModel(QwenChatProfile profile, GenAIModel backend, Tokenizer tokenizer)
    {
        Profile = profile;
        _backend = backend;
        Tokenizer = tokenizer;
        _generator = new TextGenerator(backend, tokenizer);
        Info = new QwenChatModelInfo(profile.Id, profile.Repository, profile.Revision, profile.TokenizerSha256,
            profile.TemplateSha256, GenAIRuntimeInfo.ManagedVersion ?? "unknown", GenAIRuntimeInfo.NativePackage, backend.ContextLength);
    }

    /// <summary>Admitted original profile.</summary>
    public QwenChatProfile Profile { get; }
    /// <summary>Owned tokenizer; no foreign tokenizer is used in production.</summary>
    public Tokenizer Tokenizer { get; }
    /// <summary>Loaded identity and explicit experimental capabilities.</summary>
    public QwenChatModelInfo Info { get; }

    /// <summary>Verify the bundle against the profile's pinned lock and load CPU fp32. No implicit download occurs.</summary>
    public static QwenChatModel Load(QwenChatProfile profile, string bundleDirectory, QwenChatModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        options ??= new QwenChatModelOptions();
        if (options.IntraOpThreads is < 1 or > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "CPU profiles permit one or two native threads.");
        }

        string directory = Path.GetFullPath(bundleDirectory);
        VerifyBundle(profile, directory);
        Tokenizer tokenizer = profile.OpenTokenizer(Path.Combine(directory, "tokenizer.json"));

        var backend = GenAIModel.Load(directory, new GenAIModelOptions
        {
            Provider = GenAIExecutionProvider.Cpu,
            IntraOpThreads = options.IntraOpThreads,
            PadTokenId = 151935,
        });
        try
        {
            if (backend.ModelType != profile.ModelType || backend.VocabularySize != 151936 || backend.ContextLength != profile.ContextLength ||
                !backend.EosTokenIds.Order().SequenceEqual(profile.EosTokenIds.Order()))
            {
                throw new InvalidDataException("GenAI configuration differs from the admitted architecture, vocabulary, or EOS contract.");
            }

            return new QwenChatModel(profile, backend, tokenizer);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    /// <summary>Render with the owned text-only template and tokenize without added special tokens.</summary>
    public QwenChatPrompt CreatePrompt(IReadOnlyList<QwenChatMessage> messages)
    {
        string text = QwenChatTemplate.Render(Profile, messages);
        return new QwenChatPrompt(text, Tokenizer.Encode(text, addSpecialTokens: false).Ids);
    }

    /// <summary>Generate greedily in a fresh independent request.</summary>
    public GenerationResult Generate(IReadOnlyList<QwenChatMessage> messages, QwenChatOptions? options = null,
        CancellationToken cancellationToken = default) => Generate(CreatePrompt(messages), options, cancellationToken);

    /// <summary>Generate from a prepared prompt.</summary>
    public GenerationResult Generate(QwenChatPrompt prompt, QwenChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.Generate(prompt.TokenIds, (options ?? new QwenChatOptions()).ToGenerationOptions(), cancellationToken);
    }

    /// <summary>Stream with the same fresh-request, stop and usage contract as non-streaming generation.</summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(QwenChatPrompt prompt, QwenChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.GenerateStreaming(prompt.TokenIds, (options ?? new QwenChatOptions()).ToGenerationOptions(), cancellationToken);
    }

    /// <summary>Advanced diagnostic sequence; each call returns an independent disposable cache.</summary>
    public GenAISequence CreateSequence(GenAISequenceOptions options) => _backend.CreateSequence(options);

    /// <summary>Dispose the shared model; existing sequences retain native lifetime until disposed.</summary>
    public void Dispose() => _backend.Dispose();

    private static void VerifyBundle(QwenChatProfile profile, string directory)
    {
        // A local build writes bundle-lock.json; a download (QwenChatModelAssets) has none. Either way every file must
        // equal the profile's pin.
        string lockPath = Path.Combine(directory, "bundle-lock.json");
        if (File.Exists(lockPath))
        {
            VerifyBuildLock(profile, lockPath);
        }

        foreach (QwenChatBundleFile pinned in profile.BundleFiles)
        {
            string path = Path.Combine(directory, pinned.Path);
            if (!File.Exists(path) || new FileInfo(path).Length != pinned.Size || Sha256(path) != pinned.Sha256)
            {
                throw new InvalidDataException($"Bundle asset mismatch: {pinned.Path}.");
            }
        }

        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "genai_config.json")));
        JsonElement decoder = config.RootElement.GetProperty("model").GetProperty("decoder");
        if (decoder.GetProperty("hidden_size").GetInt32() != profile.HiddenSize ||
            decoder.GetProperty("num_hidden_layers").GetInt32() != profile.Layers ||
            decoder.GetProperty("num_attention_heads").GetInt32() != profile.AttentionHeads ||
            decoder.GetProperty("num_key_value_heads").GetInt32() != profile.KeyValueHeads ||
            decoder.GetProperty("head_size").GetInt32() != profile.HeadSize || decoder.GetProperty("filename").GetString() != "model.onnx")
        {
            throw new InvalidDataException("GenAI decoder geometry differs from the admitted original configuration.");
        }
    }

    private static void VerifyBuildLock(QwenChatProfile profile, string lockPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(lockPath));
        JsonElement root = document.RootElement;
        if (root.GetProperty("format_version").GetInt32() != 1 || root.GetProperty("profile").GetString() != profile.Id ||
            root.GetProperty("origin").GetProperty("repo").GetString() != profile.Repository ||
            root.GetProperty("origin").GetProperty("revision").GetString() != profile.Revision ||
            root.GetProperty("builder").GetProperty("version").GetString() != "0.17.1" ||
            root.GetProperty("builder").GetProperty("source_commit").GetString() != "83de55a1f3886cddbb32a5517e26adb3f9d65159" ||
            root.GetProperty("precision").GetString() != "fp32" || root.GetProperty("provider").GetString() != "CPUExecutionProvider")
        {
            throw new InvalidDataException("Bundle identity differs from the admitted original or pinned builder.");
        }

        var files = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonElement file in root.GetProperty("files").EnumerateArray())
        {
            string name = file.GetProperty("path").GetString() ?? throw new InvalidDataException("Null bundle path.");
            if (Path.GetFileName(name) != name || name.Contains('\\', StringComparison.Ordinal) || !files.Add(name))
            {
                throw new InvalidDataException("Unsafe or duplicate bundle path.");
            }

            QwenChatBundleFile pinned = profile.BundleFiles.FirstOrDefault(f => f.Path == name)
                ?? throw new InvalidDataException($"Bundle file not in the profile's lock: {name}.");
            if (file.GetProperty("size").GetInt64() != pinned.Size || file.GetProperty("sha256").GetString() != pinned.Sha256)
            {
                throw new InvalidDataException($"Bundle lock differs from the profile's pinned digest: {name}.");
            }
        }

        if (!files.SetEquals(profile.BundleFiles.Select(f => f.Path)))
        {
            throw new InvalidDataException("Bundle lock does not list exactly the profile's pinned files.");
        }
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
