using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// One ONNX Runtime GenAI model directory (<c>genai_config.json</c>, the graph and its weights) loaded once with an
/// explicit execution provider. Thread-safe: any number of <see cref="GenAISequence"/> instances (one per request) may
/// run concurrently on one model.
/// </summary>
/// <remarks>
/// <para>
/// The ONNX Runtime GenAI C# API is a preview API; this class is the adapter that keeps its types out of the
/// Starling public surface (plan 5.2). Settings that change behavior (provider, TF32, pad id, threads) are
/// applied as overlays at load, never by editing <c>genai_config.json</c>, and are reported in
/// <see cref="AppliedOverlays"/> and <see cref="ProviderOptions"/>.
/// </para>
/// <para>
/// Disposal is reference-counted: <see cref="Dispose"/> stops new sequences; sequences already created keep the native
/// model alive until they are disposed, so native memory is never freed under an active request.
/// </para>
/// </remarks>
public sealed class GenAIModel : IDisposable
{
    private readonly Config _config;
    private readonly Model _model;
    private int _references = 1;
    private int _disposed;

    private GenAIModel(string directory, GenAIModelOptions options, Config config, Model model, GenAIConfigSummary summary,
        IReadOnlyList<string> overlays, IReadOnlyDictionary<string, string> providerOptions)
    {
        ModelDirectory = directory;
        Options = options;
        _config = config;
        _model = model;
        ModelType = summary.ModelType;
        ContextLength = summary.ContextLength;
        VocabularySize = summary.VocabularySize;
        EosTokenIds = summary.EosTokenIds;
        PadTokenId = options.PadTokenId ?? summary.PadTokenId;
        AppliedOverlays = overlays;
        ProviderOptions = providerOptions;
        NativeCallLock = options.EffectiveSerializeNativeCalls ? new Lock() : null;
    }

    /// <summary>Absolute path of the model directory.</summary>
    public string ModelDirectory { get; }

    /// <summary>The load-time settings.</summary>
    public GenAIModelOptions Options { get; }

    /// <summary>Execution provider the model runs on.</summary>
    public GenAIExecutionProvider Provider => Options.Provider;

    /// <summary><c>model.type</c> of <c>genai_config.json</c> (for example <c>qwen3_5_text</c>).</summary>
    public string ModelType { get; }

    /// <summary><c>model.context_length</c> of <c>genai_config.json</c>: the largest prompt plus generated length.</summary>
    public int ContextLength { get; }

    /// <summary><c>model.vocab_size</c>: width of the logits.</summary>
    public int VocabularySize { get; }

    /// <summary>End-of-sequence ids of <c>genai_config.json</c>; GenAI marks a sequence done after any of them.</summary>
    public ImmutableArray<int> EosTokenIds { get; }

    /// <summary>The effective pad id (after <see cref="GenAIModelOptions.PadTokenId"/>), or null when none is configured.</summary>
    public int? PadTokenId { get; }

    /// <summary>JSON overlays applied to the configuration at load, in order.</summary>
    public IReadOnlyList<string> AppliedOverlays { get; }

    /// <summary>Provider options set at load (CUDA only; empty for CPU).</summary>
    public IReadOnlyDictionary<string, string> ProviderOptions { get; }

    /// <summary>True when the native work of this model's sequences runs one call at a time (see <see cref="GenAIModelOptions.SerializeNativeCalls"/>).</summary>
    public bool SerializesNativeCalls => NativeCallLock is not null;

    internal Lock? NativeCallLock { get; }

    /// <summary>True after <see cref="Dispose"/>.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Loads a model directory.</summary>
    /// <param name="modelDirectory">Directory with <c>genai_config.json</c> and the files it names.</param>
    /// <param name="options">Load settings; null for CPU defaults.</param>
    /// <exception cref="GenAIRuntimeUnavailableException">
    /// No ONNX Runtime GenAI native library could be loaded, or the packages are not <see cref="GenAIRuntimeInfo.PinnedVersion"/>.
    /// </exception>
    /// <exception cref="GenAIProviderUnavailableException">The requested provider cannot be used.</exception>
    /// <exception cref="GenAIModelException">GenAI rejected the directory or its configuration.</exception>
    public static GenAIModel Load(string modelDirectory, GenAIModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        options ??= new GenAIModelOptions();
        options.Validate();
        string directory = Path.GetFullPath(modelDirectory);
        string configPath = Path.Combine(directory, "genai_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"genai_config.json not found in {directory}.", configPath);
        }

        GenAIConfigSummary summary = GenAIConfigSummary.Read(configPath);
        GenAIRuntimeInfo.CheckVersions();
        IReadOnlyList<string> overlays = options.BuildOverlays();
        var providerOptions = new Dictionary<string, string>(StringComparer.Ordinal);
        if (options.Provider == GenAIExecutionProvider.Cuda)
        {
            providerOptions["device_id"] = options.CudaDeviceId.ToString(CultureInfo.InvariantCulture);
            providerOptions["use_tf32"] = options.UseTf32 ? "1" : "0";
        }

        Config? config = null;
        try
        {
            config = new Config(directory);
            config.ClearProviders();
            if (options.Provider == GenAIExecutionProvider.Cuda)
            {
                config.AppendProvider("cuda");
                foreach (var (key, value) in providerOptions)
                {
                    config.SetProviderOption("cuda", key, value);
                }
            }

            foreach (string overlay in overlays)
            {
                config.Overlay(overlay);
            }

            var model = new Model(config);
            return new GenAIModel(directory, options, config, model, summary, overlays, providerOptions);
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException or TypeInitializationException)
        {
            config?.Dispose();
            throw new GenAIRuntimeUnavailableException(
                $"The ONNX Runtime GenAI native library could not be loaded. Reference exactly one native package in the application: " +
                $"Microsoft.ML.OnnxRuntimeGenAI {GenAIRuntimeInfo.PinnedVersion} (CPU) or Microsoft.ML.OnnxRuntimeGenAI.Cuda {GenAIRuntimeInfo.PinnedVersion} (CUDA).",
                ex);
        }
        catch (OnnxRuntimeGenAIException ex) when (options.Provider == GenAIExecutionProvider.Cuda && LooksLikeProviderFailure(ex.Message))
        {
            config?.Dispose();
            throw new GenAIProviderUnavailableException(
                $"The CUDA execution provider could not be used ({ex.Message}). It needs Microsoft.ML.OnnxRuntimeGenAI.Cuda " +
                $"{GenAIRuntimeInfo.PinnedVersion} (not the CPU package) and the CUDA 13 runtime and cuBLAS on the loader path. " +
                "There is no fallback to the CPU provider.",
                ex);
        }
        catch (OnnxRuntimeGenAIException ex)
        {
            config?.Dispose();
            throw new GenAIModelException($"ONNX Runtime GenAI could not load {directory}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Advanced: creates a generation sequence (one GenAI generator) for one request. The caller owns and disposes it.
    /// Most callers use <see cref="TextGenerator"/> instead.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The model was disposed.</exception>
    public GenAISequence CreateSequence(GenAISequenceOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (options.MaxLength > ContextLength)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxLength, $"MaxLength exceeds the context length {ContextLength}.");
        }

        Acquire();
        GeneratorParams? parameters = null;
        try
        {
            parameters = new GeneratorParams(_model);
            parameters.SetSearchOption("max_length", options.MaxLength);
            if (options.Sampling is { } sampling)
            {
                parameters.SetSearchOption("do_sample", true);
                parameters.SetSearchOption("temperature", sampling.Temperature);
                parameters.SetSearchOption("top_k", sampling.TopK);
                parameters.SetSearchOption("top_p", sampling.TopP);
                parameters.SetSearchOption("random_seed", sampling.Seed);
            }
            else
            {
                parameters.SetSearchOption("do_sample", false);
            }

            Generator generator;
            if (NativeCallLock is { } gate)
            {
                lock (gate)
                {
                    generator = new Generator(_model, parameters);
                }
            }
            else
            {
                generator = new Generator(_model, parameters);
            }

            return new GenAISequence(this, parameters, generator, VocabularySize);
        }
        catch (OnnxRuntimeGenAIException ex)
        {
            parameters?.Dispose();
            Release();
            throw new GenAIModelException("ONNX Runtime GenAI could not create a generator: " + ex.Message, ex);
        }
        catch
        {
            parameters?.Dispose();
            Release();
            throw;
        }
    }

    /// <summary>
    /// Stops new sequences. The native model is released when the last sequence created from it is disposed.
    /// Idempotent.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Release();
        }
    }

    internal void Release()
    {
        if (Interlocked.Decrement(ref _references) == 0)
        {
            _model.Dispose();
            _config.Dispose();
        }
    }

    private void Acquire()
    {
        while (true)
        {
            int current = Volatile.Read(ref _references);
            ObjectDisposedException.ThrowIf(current == 0 || IsDisposed, this);
            if (Interlocked.CompareExchange(ref _references, current + 1, current) == current)
            {
                break;
            }
        }

        if (IsDisposed)
        {
            Release();
            ObjectDisposedException.ThrowIf(true, this);
        }
    }

    private static bool LooksLikeProviderFailure(string message) =>
        message.Contains("cuda", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("provider", StringComparison.OrdinalIgnoreCase) ||
        message.Contains("libcu", StringComparison.OrdinalIgnoreCase);
}

/// <summary>The fields of <c>genai_config.json</c> the adapter reads before loading.</summary>
internal sealed record GenAIConfigSummary(string ModelType, int ContextLength, int VocabularySize, ImmutableArray<int> EosTokenIds, int? PadTokenId)
{
    public static GenAIConfigSummary Read(string path)
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            var model = doc.RootElement.GetProperty("model");
            var eos = model.GetProperty("eos_token_id");
            ImmutableArray<int> eosIds = eos.ValueKind == JsonValueKind.Array
                ? eos.EnumerateArray().Select(e => e.GetInt32()).ToImmutableArray()
                : [eos.GetInt32()];
            int? pad = model.TryGetProperty("pad_token_id", out var p) && p.ValueKind == JsonValueKind.Number ? p.GetInt32() : null;
            return new GenAIConfigSummary(
                model.GetProperty("type").GetString() ?? string.Empty,
                model.GetProperty("context_length").GetInt32(),
                model.GetProperty("vocab_size").GetInt32(),
                eosIds,
                pad);
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new GenAIModelException($"{path} is not a valid genai_config.json: {ex.Message}", ex);
        }
    }
}
