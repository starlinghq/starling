using System.Collections.Immutable;
using System.Security.Cryptography;
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

namespace Starlings.Models.Qwen35;

/// <summary>
/// <c>Qwen/Qwen3.5-0.8B</c> chat generation: a conversation is rendered with the pinned chat template
/// (<see cref="Qwen35ChatTemplate"/>), tokenized with the owned tokenizer (profile <c>qwen35-bpe-chat</c>), and decoded
/// by <see cref="TextGenerator"/> on ONNX Runtime GenAI. Load once and reuse; thread-safe.
/// </summary>
/// <remarks>
/// <para>
/// Requests are independent: each one renders and prefills its whole conversation in a fresh generator, so concurrent
/// requests share the weights but no state. Continuing a conversation means sending the whole conversation again.
/// Reusing a generator's state for the next turn is not offered: the template drops the think block of assistant turns
/// before the last user query, so the earlier state is not a prefix of the next turn's rendering, and partial rewind is
/// unsupported on this hybrid recurrent model.
/// </para>
/// <para>
/// Fixed at load: TF32 off on CUDA, and <c>pad_token_id</c> overlaid to <see cref="PadTokenIdOverlay"/> (an embedding
/// row the tokenizer never produces), because ONNX Runtime GenAI masks prompt positions equal to the configured pad id
/// 248044, which is <c>&lt;|endoftext|&gt;</c> and can appear in user text.
/// </para>
/// </remarks>
public sealed class Qwen35ChatModel : IDisposable
{
    /// <summary><c>&lt;|im_end|&gt;</c>: ends an assistant turn.</summary>
    public const int ImEndTokenId = 248046;

    /// <summary><c>&lt;|endoftext|&gt;</c>.</summary>
    public const int EndOfTextTokenId = 248044;

    /// <summary>
    /// Pad id overlaid on <c>genai_config.json</c>: the last of the 248,320 embedding rows; the tokenizer's ids end at
    /// 248,076, so it never occurs in a prompt.
    /// </summary>
    public const int PadTokenIdOverlay = 248319;

    private const string ExpectedModelType = "qwen3_5_text";
    private const int ExpectedVocabularySize = 248320;

    private readonly GenAIModel? _backend;
    private readonly TextGenerator _generator;

    private Qwen35ChatModel(GenAIModel backend, Tokenizer tokenizer, Qwen35ModelInfo info)
        : this(backend, new TextGenerator(backend, tokenizer), tokenizer, info)
    {
    }

    private Qwen35ChatModel(GenAIModel? backend, TextGenerator generator, Tokenizer tokenizer, Qwen35ModelInfo info)
    {
        _backend = backend;
        Tokenizer = tokenizer;
        _generator = generator;
        Info = info with { StopTokenIds = _generator.StopTokenIds };
    }

    /// <summary>Identities of the loaded model, tokenizer, template, backend and provider.</summary>
    public Qwen35ModelInfo Info { get; }

    /// <summary>The owned tokenizer (profile <c>qwen35-bpe-chat</c>).</summary>
    public Tokenizer Tokenizer { get; }

    internal GenAIModel Backend => _backend ?? throw new InvalidOperationException("This instance has no native backend (test seam).");

    internal TextGenerator Generator => _generator;

    /// <summary>
    /// Test seam: a model over a decode loop without a native backend (for example <see cref="TextGenerator"/> over
    /// scripted sequences). <see cref="Dispose"/> releases nothing.
    /// </summary>
    internal static Qwen35ChatModel CreateWithoutBackend(TextGenerator generator, Qwen35ModelInfo info) =>
        new(null, generator, generator.Tokenizer, info);

    /// <summary>Loads a verified variant (see <see cref="Qwen35ModelAssets"/>).</summary>
    /// <exception cref="NotSupportedException">The variant and provider combination is unsupported (fp16x on CPU).</exception>
    /// <exception cref="InvalidDataException">The template, tokenizer or model configuration is not the pinned one.</exception>
    /// <exception cref="GenAIRuntimeUnavailableException">No ONNX Runtime GenAI native package.</exception>
    /// <exception cref="GenAIProviderUnavailableException">CUDA requested but unusable.</exception>
    public static Qwen35ChatModel Load(Qwen35ModelFiles files, Qwen35ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new Qwen35ModelOptions();
        if (files.Variant == Qwen35Variant.Fp16x && options.Provider == GenAIExecutionProvider.Cpu)
        {
            throw new NotSupportedException("The fp16x variant is built for the CUDA provider; on CPU use the fp32x variant.");
        }

        string templateSha = Sha256(files.ChatTemplatePath);
        if (templateSha != Qwen35ChatTemplate.TemplateSha256)
        {
            throw new InvalidDataException(
                $"{files.ChatTemplatePath} has SHA-256 {templateSha}; the C# renderer was ported from {Qwen35ChatTemplate.TemplateSha256}.");
        }

        Tokenizer tokenizer = Tokenizer.FromFile(files.TokenizerPath);
        CheckTokenizer(tokenizer);

        var backend = GenAIModel.Load(files.ModelDirectory, new GenAIModelOptions
        {
            Provider = options.Provider,
            CudaDeviceId = options.CudaDeviceId,
            UseTf32 = false,
            IntraOpThreads = options.IntraOpThreads,
            PadTokenId = PadTokenIdOverlay,
            SerializeNativeCalls = options.SerializeNativeCalls,
        });
        try
        {
            if (backend.ModelType != ExpectedModelType || backend.VocabularySize != ExpectedVocabularySize ||
                !backend.EosTokenIds.Order().SequenceEqual([EndOfTextTokenId, ImEndTokenId]))
            {
                throw new InvalidDataException(
                    $"{files.ModelDirectory} is not the pinned build: type {backend.ModelType}, vocab {backend.VocabularySize}, eos [{string.Join(", ", backend.EosTokenIds)}].");
            }

            var info = new Qwen35ModelInfo
            {
                Profile = $"qwen3.5-0.8b/{Qwen35ModelAssets.VariantDirectory(files.Variant)}",
                Checkpoint = $"{Qwen35ModelAssets.Repository}@{Qwen35ModelAssets.Revision}",
                Variant = files.Variant,
                Provider = options.Provider,
                GenAIVersion = GenAIRuntimeInfo.ManagedVersion ?? $"unknown (qualified with {GenAIRuntimeInfo.PinnedVersion})",
                GenAINativePackage = GenAIRuntimeInfo.NativePackage,
                Tokenizer = $"Starlings.Tokenizers profile {Qwen35ModelAssets.TokenizerProfile}, tokenizer.json {Qwen35ModelAssets.TokenizerSha256}",
                ChatTemplate = $"owned C# renderer of chat_template.jinja {Qwen35ChatTemplate.TemplateSha256} (text and tool-calling subset)",
                ModelDirectory = backend.ModelDirectory,
                ConfigOverlays = backend.AppliedOverlays,
                ProviderOptions = backend.ProviderOptions,
                SerializesNativeCalls = backend.SerializesNativeCalls,
                ContextLength = backend.ContextLength,
                StopTokenIds = [],
            };
            return new Qwen35ChatModel(backend, tokenizer, info);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    /// <summary>Renders <paramref name="messages"/> with the chat template and tokenizes the result with the owned tokenizer.</summary>
    /// <exception cref="Qwen35TemplateException">The conversation is invalid for the template.</exception>
    public Qwen35Prompt CreatePrompt(IReadOnlyList<Qwen35Message> messages, bool enableThinking = false) => CreatePrompt(messages, tools: null, enableThinking);

    /// <summary>
    /// Renders <paramref name="messages"/> and the tool definitions <paramref name="tools"/> with the chat template and
    /// tokenizes the result with the owned tokenizer.
    /// </summary>
    /// <exception cref="Qwen35TemplateException">The conversation or tool data is invalid for the template.</exception>
    public Qwen35Prompt CreatePrompt(IReadOnlyList<Qwen35Message> messages, IReadOnlyList<Qwen35Tool>? tools, bool enableThinking = false)
    {
        string text = Qwen35ChatTemplate.Render(messages, tools, enableThinking);
        return new Qwen35Prompt(text, Tokenizer.Encode(text, addSpecialTokens: false).Ids);
    }

    /// <summary>
    /// Generates the assistant reply to <paramref name="messages"/> (with <see cref="Qwen35ChatOptions.Tools"/> rendered
    /// into the prompt). The result's text is the raw completion; <see cref="Qwen35ToolCallParser.Parse(string, Qwen35ChatOptions)"/>
    /// splits it into text and tool calls.
    /// </summary>
    /// <exception cref="OperationCanceledException">Cancelled before the prefill or between decode steps.</exception>
    public GenerationResult Generate(IReadOnlyList<Qwen35Message> messages, Qwen35ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new Qwen35ChatOptions();
        return Generate(CreatePrompt(messages, options.Tools, options.EnableThinking), options, cancellationToken);
    }

    /// <summary>Generates from a prepared prompt (<see cref="Qwen35ChatOptions.EnableThinking"/> is a property of the prompt and is ignored here).</summary>
    public GenerationResult Generate(Qwen35Prompt prompt, Qwen35ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.Generate(prompt.TokenIds, (options ?? new Qwen35ChatOptions()).ToGenerationOptions(Info.Provider), cancellationToken);
    }

    /// <summary>
    /// Streams the assistant reply to <paramref name="messages"/>; see <see cref="TextGenerator.GenerateStreaming"/> for
    /// the update contract. The template is rendered and tokenized when this method is called.
    /// </summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(IReadOnlyList<Qwen35Message> messages, Qwen35ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new Qwen35ChatOptions();
        return GenerateStreaming(CreatePrompt(messages, options.Tools, options.EnableThinking), options, cancellationToken);
    }

    /// <summary>Streams from a prepared prompt.</summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(Qwen35Prompt prompt, Qwen35ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.GenerateStreaming(prompt.TokenIds, (options ?? new Qwen35ChatOptions()).ToGenerationOptions(Info.Provider), cancellationToken);
    }

    /// <summary>
    /// Stops new requests. Requests already streaming keep the native model alive until they finish or their
    /// enumerators are disposed. Idempotent.
    /// </summary>
    public void Dispose() => _backend?.Dispose();

    private static void CheckTokenizer(Tokenizer tokenizer)
    {
        if (tokenizer.TokenToId("<|im_end|>") != ImEndTokenId || tokenizer.TokenToId("<|endoftext|>") != EndOfTextTokenId ||
            tokenizer.TokenToId("<|im_start|>") != 248045 || tokenizer.IdToToken(PadTokenIdOverlay) is not null)
        {
            throw new InvalidDataException("tokenizer.json is not the qwen35-bpe-chat profile asset (special token ids differ).");
        }
    }

    private static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
