using System.Runtime.Versioning;
using Starlings.Runtime.OnnxWeb;

namespace Starlings.Models.E5.Browser;

/// <summary>What <see cref="E5BrowserEmbedder.LoadAsync"/> loads and how.</summary>
public sealed record E5BrowserLoadOptions
{
    /// <summary>The graph. Default <see cref="E5BrowserVariant.Fp32"/> (loads from the Hub).</summary>
    public E5BrowserVariant Variant { get; init; } = E5BrowserVariant.Fp32;

    /// <summary>
    /// Where to fetch the graph (relative to the page's base URI, or absolute), or <see langword="null"/> for the
    /// variant's <see cref="E5BrowserGraph.DefaultUrl"/>. Required for <see cref="E5BrowserVariant.Int8Embed"/>. The
    /// bytes must match the variant's pinned SHA-256 wherever they come from.
    /// </summary>
    public string? ModelUrl { get; init; }

    /// <summary>Where to fetch <c>tokenizer.json</c>. Default <see cref="E5BrowserAssets.TokenizerUrl"/> (the Hub). Its SHA-256 is enforced.</summary>
    public string TokenizerUrl { get; init; } = E5BrowserAssets.TokenizerUrl;

    /// <summary>Execution provider. Default <see cref="OnnxWebExecutionProvider.Wasm"/> (the qualified path).</summary>
    public OnnxWebExecutionProvider Provider { get; init; } = OnnxWebExecutionProvider.Wasm;

    /// <summary>
    /// onnxruntime-web settings, or <see langword="null"/> (default) to use the page's existing initialization, or the
    /// default options when nothing initialized it yet. A page initializes onnxruntime-web once: settings that differ
    /// from an earlier, successful initialization throw <see cref="InvalidOperationException"/>, so after the first
    /// load leave this <see langword="null"/>.
    /// </summary>
    public OnnxWebRuntimeOptions? Runtime { get; init; }

    /// <summary>Cache Storage settings for the graph and the tokenizer.</summary>
    public OnnxWebFetchOptions Fetch { get; init; } = new();

    /// <summary>Encoder options (batch size and the profile limits). <see cref="E5EncoderOptions.Session"/> is ignored.</summary>
    public E5EncoderOptions Encoder { get; init; } = new();
}

/// <summary>Which file <see cref="E5BrowserEmbedder.LoadAsync"/> is on.</summary>
public enum E5BrowserLoadStage
{
    /// <summary>Loading onnxruntime-web.</summary>
    Runtime,

    /// <summary>Downloading (or reading from Cache Storage) and verifying <c>tokenizer.json</c>.</summary>
    Tokenizer,

    /// <summary>Downloading (or reading from Cache Storage) and verifying the graph.</summary>
    Model,

    /// <summary>onnxruntime-web is creating the session.</summary>
    Session,

    /// <summary>Ready.</summary>
    Done,
}

/// <summary>Progress of <see cref="E5BrowserEmbedder.LoadAsync"/>.</summary>
/// <param name="Stage">Current stage.</param>
/// <param name="BytesReceived">Bytes of the current file so far (download stages), else 0.</param>
/// <param name="TotalBytes">Size of the current file when known.</param>
public readonly record struct E5BrowserLoadProgress(E5BrowserLoadStage Stage, long BytesReceived, long? TotalBytes);

/// <summary>
/// Text in, embeddings out, in the browser: <see cref="E5Tokenizer"/> (the owned tokenizer) followed by an
/// <see cref="E5BrowserEncoder"/>. The text is tokenized exactly as given; add <see cref="E5TextEmbedder.QueryPrefix"/>
/// or <see cref="E5TextEmbedder.PassagePrefix"/> yourself, or use <see cref="E5BrowserEmbeddingGenerator"/> with an
/// input type.
/// </summary>
public sealed class E5BrowserEmbedder : IDisposable
{
    private readonly IEncoderTokenizer _tokenizer;
    private readonly E5BrowserEncoder _encoder;
    private readonly bool _ownsEncoder;

    /// <summary>Combines a tokenizer and an encoder. Neither is owned or disposed by this instance.</summary>
    public E5BrowserEmbedder(IEncoderTokenizer tokenizer, E5BrowserEncoder encoder)
        : this(tokenizer, encoder, ownsEncoder: false)
    {
    }

    private E5BrowserEmbedder(IEncoderTokenizer tokenizer, E5BrowserEncoder encoder, bool ownsEncoder)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(encoder);
        _tokenizer = tokenizer;
        _encoder = encoder;
        _ownsEncoder = ownsEncoder;
    }

    /// <summary>The encoder (dimension, options, session diagnostics).</summary>
    public E5BrowserEncoder Encoder => _encoder;

    /// <summary>The tokenizer.</summary>
    public IEncoderTokenizer Tokenizer => _tokenizer;

    /// <summary>The variant loaded by <see cref="LoadAsync"/>, or <see langword="null"/> for an embedder built from parts.</summary>
    public E5BrowserGraph? Graph { get; private init; }

    /// <summary>The registry profile of the loaded variant and provider, or <see langword="null"/> for an embedder built from parts.</summary>
    public string? Profile { get; private init; }

    /// <summary>
    /// Initializes onnxruntime-web (if not yet), fetches and verifies the pinned tokenizer and graph (Cache Storage
    /// first), creates the session and returns an embedder that owns it.
    /// </summary>
    /// <exception cref="ArgumentException"><see cref="E5BrowserLoadOptions.ModelUrl"/> is missing for a variant without a default URL.</exception>
    /// <exception cref="InvalidDataException">A file does not match its pinned SHA-256.</exception>
    /// <exception cref="NotSupportedException">The provider or thread count is not available on this page.</exception>
    [SupportedOSPlatform("browser")]
    public static async Task<E5BrowserEmbedder> LoadAsync(
        E5BrowserLoadOptions? options = null,
        IProgress<E5BrowserLoadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new E5BrowserLoadOptions();
        ArgumentNullException.ThrowIfNull(options.Fetch);
        ArgumentNullException.ThrowIfNull(options.Encoder);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.TokenizerUrl, nameof(options));
        var graph = E5BrowserAssets.For(options.Variant);
        string modelUrl = options.ModelUrl ?? graph.DefaultUrl
            ?? throw new ArgumentException($"The {options.Variant} graph has no public URL; set ModelUrl to where the app hosts it (SHA-256 {graph.Sha256}).", nameof(options));

        progress?.Report(new E5BrowserLoadProgress(E5BrowserLoadStage.Runtime, 0, null));
        if (options.Runtime is not null || OnnxWebRuntime.Info is null)
        {
            await OnnxWebRuntime.InitializeAsync(options.Runtime).ConfigureAwait(false);
        }

        // Everything that can be rejected without the files is rejected before the downloads (up to 487 MB).
        OnnxWebRuntime.EnsureProviderAvailable(options.Provider);
        options.Encoder.Validate();

        E5Tokenizer tokenizer;
        using (var tokenizerAsset = await OnnxWebAssets.FetchAsync(
            options.TokenizerUrl, E5ModelAssets.TokenizerSha256, options.Fetch, Forward(progress, E5BrowserLoadStage.Tokenizer), cancellationToken).ConfigureAwait(false))
        {
            // Verified in JavaScript; E5Tokenizer checks the copy again, so a bad copy cannot pass either.
            tokenizer = E5Tokenizer.FromUtf8Json(tokenizerAsset.ToArray(), E5ModelAssets.TokenizerSha256, options.Encoder.MaxSequenceLength);
        }

        OnnxWebSession session;
        using (var modelAsset = await OnnxWebAssets.FetchAsync(
            modelUrl, graph.Sha256, options.Fetch, Forward(progress, E5BrowserLoadStage.Model), cancellationToken).ConfigureAwait(false))
        {
            progress?.Report(new E5BrowserLoadProgress(E5BrowserLoadStage.Session, 0, null));
            session = await OnnxWebSession.CreateAsync(modelAsset, new OnnxWebSessionOptions { Provider = options.Provider }, cancellationToken).ConfigureAwait(false);
        }

        try
        {
            var encoder = new E5BrowserEncoder(session, options.Encoder, ownsSession: true);
            progress?.Report(new E5BrowserLoadProgress(E5BrowserLoadStage.Done, 0, null));
            return new E5BrowserEmbedder(tokenizer, encoder, ownsEncoder: true)
            {
                Graph = graph,
                Profile = options.Provider == OnnxWebExecutionProvider.WebGpu ? graph.ProfileWebGpu : graph.ProfileWasm,
            };
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>Embeds one text, tokenized as given.</summary>
    public Task<float[]> EmbedAsync(string text, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _encoder.EmbedAsync(_tokenizer.Tokenize(text), cancellationToken);
    }

    /// <summary>Embeds texts in order, tokenized as given.</summary>
    public Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        EmbedBatchAsync(texts, tokenCounts: null, cancellationToken);

    /// <summary>
    /// Embeds texts in order, tokenized as given, and writes each text's token count (special tokens included, after
    /// truncation) to <paramref name="tokenCounts"/> when it is not <see langword="null"/>; it must then hold exactly
    /// <c>texts.Count</c> values.
    /// </summary>
    public Task<float[][]> EmbedBatchAsync(IReadOnlyList<string> texts, int[]? tokenCounts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (tokenCounts is not null && tokenCounts.Length != texts.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {texts.Count} texts need {texts.Count}.", nameof(tokenCounts));
        }

        var inputs = new EncoderInput[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = _tokenizer.Tokenize(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)));
            if (tokenCounts is not null)
            {
                tokenCounts[i] = inputs[i].Length;
            }
        }

        return _encoder.EmbedBatchAsync(inputs, cancellationToken);
    }

    /// <summary>Disposes the encoder and its session when this instance came from <see cref="LoadAsync"/>. Idempotent.</summary>
    public void Dispose()
    {
        if (_ownsEncoder)
        {
            _encoder.Dispose();
        }
    }

    private static DownloadProgress? Forward(IProgress<E5BrowserLoadProgress>? progress, E5BrowserLoadStage stage) =>
        progress is null ? null : new DownloadProgress(progress, stage);

    private sealed class DownloadProgress(IProgress<E5BrowserLoadProgress> target, E5BrowserLoadStage stage) : IProgress<OnnxWebDownloadProgress>
    {
        public void Report(OnnxWebDownloadProgress value) => target.Report(new E5BrowserLoadProgress(stage, value.BytesReceived, value.TotalBytes));
    }
}
