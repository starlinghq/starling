using System.Buffers;
using System.Globalization;
using Starlings.Runtime.Onnx;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Image, video and interleaved text/image/video embeddings for <c>google/embeddinggemma-2</c> (experimental; default
/// budget of 280 soft tokens per image; videos at the checkpoint's settings: 1 frame per second, at most 32 frames, 140
/// soft tokens per frame, no timestamps): the upstream processors restated from decoded RGB
/// (<see cref="EmbeddingGemma2ImageProcessor"/>, <see cref="EmbeddingGemma2VideoSampling"/>), Starling's vision graph
/// (pixels to soft tokens), the soft tokens in place of the <c>&lt;|image|&gt;</c> and <c>&lt;|video|&gt;</c> token
/// embeddings of the multimodal text graph, mean pooling over every token, L2 normalization and Matryoshka truncation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Videos.</b> An <see cref="EmbeddingGemma2Video"/> holds decoded RGB frames. Its frames are sampled as upstream
/// (<see cref="EmbeddingGemma2Video.GetSampledFrameIndices"/>), each sampled frame is preprocessed like an image at
/// <see cref="VideoFrameSoftTokens"/> (same resize kernel), all frames go through the vision graph in calls of at most
/// <see cref="EmbeddingGemma2MultimodalOptions.MaxVideoFramesPerVisionCall"/>, and each frame renders as
/// <c>&lt;|image&gt;</c> + n x <c>&lt;|video|&gt;</c> + <c>&lt;image|&gt;</c>. One video costs <c>frames x (n + 2)</c>
/// tokens, at most 4,544. Frames without timing are rejected unless
/// <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/> is set.
/// </para>
/// <para>
/// <b>Graphs.</b> Inputs with at least one image or video run on the vision graph
/// (<see cref="EmbeddingGemma2ModelAssets.VisionModelFile"/>) and the multimodal text graph
/// (<see cref="EmbeddingGemma2ModelAssets.MultimodalModelFile"/>). Text-only inputs run on the released text graph
/// (<see cref="EmbeddingGemma2ModelAssets.ModelFile"/>) and equal <see cref="EmbeddingGemma2TextEmbedder"/> bit for bit;
/// <see cref="Create(EmbeddingGemma2ImageModelFiles, EmbeddingGemma2MultimodalOptions?)"/> loads that graph on the first
/// text-only input. The multimodal graph gives the released graph's outputs bit for bit on text-only input as well.
/// </para>
/// <para>
/// <b>Rendering.</b> As upstream (sentence-transformers 6.1.0 message format, <c>EmbeddingGemma2Processor</c>): the
/// prompt text (if any), then the parts in order, each image expanded to <c>&lt;|image&gt;</c> + n x
/// <c>&lt;|image|&gt;</c> + <c>&lt;image|&gt;</c>, tokenized with <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>. The card
/// recommends no prompt for images.
/// </para>
/// <para>
/// <b>Limits.</b> Every input is validated, rendered and tokenized before any inference: inputs above
/// <see cref="EmbeddingGemma2MultimodalOptions.MaxSequenceLength"/> tokens throw
/// <see cref="EmbeddingGemma2InputTooLongException"/>; images above
/// <see cref="EmbeddingGemma2MultimodalOptions.MaxImagePixels"/> throw <see cref="EmbeddingGemma2ImageTooLargeException"/>.
/// </para>
/// <para>
/// <b>Batching and memory.</b> Images are preprocessed per text chunk and run on the vision graph in calls of at most
/// <see cref="EmbeddingGemma2MultimodalOptions.MaxVisionBatchSize"/> images (default 1, GPU-safe). Methods are
/// synchronous, safe to call concurrently, and check cancellation between graph calls and inside each run.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2MultimodalEmbedder : IDisposable
{
    /// <summary>Vision graph input: patch values, float32 [batch, patches, 768].</summary>
    public const string PixelValuesName = "pixel_values";

    /// <summary>Vision graph input: patch (x, y), int64 [batch, patches, 2]; padding rows (-1, -1).</summary>
    public const string PixelPositionIdsName = "pixel_position_ids";

    /// <summary>Vision graph output: soft tokens, float32 [batch, patches / 9, 512].</summary>
    public const string SoftTokensName = "soft_tokens";

    /// <summary>Vision graph output: true for the valid soft-token slots, bool [batch, patches / 9].</summary>
    public const string SoftTokenMaskName = "soft_token_mask";

    /// <summary>Multimodal text graph input: soft tokens at their sequence positions, float32 [batch, sequence, 512].</summary>
    public const string MultimodalEmbedsName = "multimodal_embeds";

    /// <summary>Multimodal text graph input: true where <see cref="MultimodalEmbedsName"/> replaces the token embedding, bool [batch, sequence].</summary>
    public const string MultimodalMaskName = "multimodal_mask";

    /// <summary>Width of a soft token (the text model's hidden size).</summary>
    public const int SoftTokenDimension = 512;

    /// <summary>Id of <c>&lt;|image&gt;</c>, which opens an image.</summary>
    public const long ImageStartTokenId = 255_999;

    /// <summary>Id of <c>&lt;|image|&gt;</c>, one per soft token.</summary>
    public const long ImageTokenId = 258_880;

    /// <summary>Id of <c>&lt;image|&gt;</c>, which closes an image or a video frame.</summary>
    public const long ImageEndTokenId = 258_882;

    /// <summary>Id of <c>&lt;|video|&gt;</c>, one per soft token of a video frame.</summary>
    public const long VideoTokenId = 258_884;

    /// <summary>
    /// Soft tokens per video frame at most (the checkpoint's video processor: <c>max_soft_tokens</c> 140, 1,260 patches).
    /// Fixed: other budgets are not offered for video.
    /// </summary>
    public const int VideoFrameSoftTokens = 140;

    /// <summary>
    /// Video frames preprocessed together (in parallel) ahead of their vision calls: 8 frames at 140 soft tokens hold at
    /// most about 31 MB of patch values (1,260 x 768 float32 each).
    /// </summary>
    internal const int VideoFramesPreprocessedAhead = 8;

    private static readonly string[] VisionInputNames = [PixelValuesName, PixelPositionIdsName];
    private static readonly string[] VisionOutputNames = [SoftTokensName, SoftTokenMaskName];
    private static readonly string[] TextInputNames = [EmbeddingGemma2Encoder.InputIdsName, EmbeddingGemma2Encoder.AttentionMaskName, MultimodalEmbedsName, MultimodalMaskName];
    private static readonly string[] TextOutputNames = [EmbeddingGemma2Encoder.TokenEmbeddingsName];

    private readonly IEmbeddingGemma2Tokenizer _tokenizer;
    private readonly OnnxModelSession _vision;
    private readonly OnnxModelSession _multimodal;
    private readonly Lazy<EmbeddingGemma2Encoder>? _textEncoder;
    private readonly bool _ownsSessions;
    private readonly EmbeddingGemma2BufferPool _buffers;
    private readonly EmbeddingGemma2MultimodalOptions _options;
    private int _disposed;

    /// <summary>
    /// Combines a tokenizer, a vision session, a multimodal text session and (optionally) an encoder over the released
    /// text graph for text-only inputs. Without <paramref name="textEncoder"/>, text-only inputs throw
    /// <see cref="InvalidOperationException"/>.
    /// </summary>
    /// <param name="tokenizer">The owned tokenizer (<see cref="EmbeddingGemma2Tokenizer"/>) or another exact implementation.</param>
    /// <param name="vision">A session over the vision graph.</param>
    /// <param name="multimodalText">A session over the multimodal text graph.</param>
    /// <param name="textEncoder">An encoder over the released text graph, or null. Its <see cref="EmbeddingGemma2EncoderOptions.Dimension"/> and <see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/> must equal the options'.</param>
    /// <param name="options">Options; <see cref="EmbeddingGemma2MultimodalOptions.Session"/> is ignored (the sessions are configured).</param>
    /// <param name="ownsSessions">When true, <see cref="Dispose"/> disposes the two sessions and the text encoder.</param>
    /// <exception cref="NotSupportedException">A session does not have the expected graph signature.</exception>
    /// <exception cref="ArgumentException"><paramref name="textEncoder"/> has another dimension or another sequence limit than <paramref name="options"/>.</exception>
    public EmbeddingGemma2MultimodalEmbedder(
        IEmbeddingGemma2Tokenizer tokenizer,
        OnnxModelSession vision,
        OnnxModelSession multimodalText,
        EmbeddingGemma2Encoder? textEncoder = null,
        EmbeddingGemma2MultimodalOptions? options = null,
        bool ownsSessions = false)
        : this(tokenizer, vision, multimodalText, textEncoder is null ? null : new Lazy<EmbeddingGemma2Encoder>(textEncoder), options, ownsSessions)
    {
    }

    private EmbeddingGemma2MultimodalEmbedder(
        IEmbeddingGemma2Tokenizer tokenizer,
        OnnxModelSession vision,
        OnnxModelSession multimodalText,
        Lazy<EmbeddingGemma2Encoder>? textEncoder,
        EmbeddingGemma2MultimodalOptions? options,
        bool ownsSessions)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(vision);
        ArgumentNullException.ThrowIfNull(multimodalText);
        options ??= new EmbeddingGemma2MultimodalOptions();
        options.Validate();
        ValidateVisionSignature(vision);
        ValidateMultimodalSignature(multimodalText);
        if (textEncoder is { IsValueCreated: true } && textEncoder.Value.Dimension != options.Dimension)
        {
            throw new ArgumentException($"The text encoder produces {textEncoder.Value.Dimension} dimensions; the options ask for {options.Dimension}.", nameof(textEncoder));
        }

        // Text-only rows are checked by the text encoder against its own limit, rows with media against
        // options.MaxSequenceLength; the two must agree or the limit would depend on the row's modality.
        if (textEncoder is { IsValueCreated: true } && textEncoder.Value.Options.MaxSequenceLength != options.MaxSequenceLength)
        {
            throw new ArgumentException($"The text encoder accepts at most {textEncoder.Value.Options.MaxSequenceLength} tokens; the options ask for {options.MaxSequenceLength}. Use the same MaxSequenceLength for both.", nameof(textEncoder));
        }

        if (options.AutoBatchLimits is { } auto)
        {
            // After load, so the free memory excludes the resident weights.
            int deviceId = vision.Diagnostics.ProviderOptions.TryGetValue("device_id", out string? id)
                && int.TryParse(id, NumberStyles.None, CultureInfo.InvariantCulture, out int parsed) ? parsed : 0;
            AutoBatchAvailableBytes = auto.ProbeFreeBytes(vision.Diagnostics.RequestedProvider, deviceId);
            options = auto.Apply(options, AutoBatchAvailableBytes);
            options.Validate();
        }

        _tokenizer = tokenizer;
        _vision = vision;
        _multimodal = multimodalText;
        _textEncoder = textEncoder;
        _options = options;
        _ownsSessions = ownsSessions;
        _buffers = new EmbeddingGemma2BufferPool(options.PoolBuffers);
    }

    /// <summary>Output dimension (<see cref="EmbeddingGemma2MultimodalOptions.Dimension"/>).</summary>
    public int Dimension => _options.Dimension;

    /// <summary>
    /// Options in effect: with <see cref="EmbeddingGemma2MultimodalOptions.AutoBatchLimits"/>, the vision limits chosen
    /// at load.
    /// </summary>
    public EmbeddingGemma2MultimodalOptions Options => _options;

    /// <summary>
    /// Free bytes the <see cref="EmbeddingGemma2MultimodalOptions.AutoBatchLimits"/> probe read at load (host RAM on
    /// CPU, device memory on CUDA); <see langword="null"/> when the option is off or the probe was unavailable (then the
    /// configured limits apply). Experimental.
    /// </summary>
    public long? AutoBatchAvailableBytes { get; }

    /// <summary>The tokenizer. Owned by the caller unless created by <see cref="Create(EmbeddingGemma2ImageModelFiles, EmbeddingGemma2MultimodalOptions?)"/>.</summary>
    public IEmbeddingGemma2Tokenizer Tokenizer => _tokenizer;

    /// <summary>The vision session, for diagnostics and placement.</summary>
    public OnnxModelSession VisionSession => _vision;

    /// <summary>The multimodal text session, for diagnostics and placement.</summary>
    public OnnxModelSession MultimodalSession => _multimodal;

    /// <summary>
    /// Loads the tokenizer (SHA-256 checked), the vision graph and the multimodal text graph from verified files
    /// (each graph's SHA-256 checked while loading when <see cref="EmbeddingGemma2MultimodalOptions.VerifyModelSha256"/>);
    /// the released text graph loads on the first text-only input. The embedder owns everything it loads.
    /// </summary>
    public static EmbeddingGemma2MultimodalEmbedder Create(EmbeddingGemma2ImageModelFiles files, EmbeddingGemma2MultimodalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new EmbeddingGemma2MultimodalOptions();
        options.Validate();
        bool verify = options.VerifyModelSha256;
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.Text.TokenizerPath);
        OnnxModelSession? vision = null;
        OnnxModelSession? multimodal = null;
        try
        {
            // On CPU, unless the caller chose: return the memory that creating each graph's session freed (about
            // 400 MiB of free chunks in the main glibc heap, varying from load to load) and, after each run, the
            // run's freed buffers (100 to 190 MiB of free chunks per run thread's heap, rising in steps over an
            // hour). Both otherwise stayed resident while loaded (docs/evidence/eg2-image-loaded-soak/).
            bool cpu = options.Session.Provider == OnnxExecutionProvider.Cpu;
            bool trimAfterLoad = options.Session.TrimNativeHeapAfterLoad ?? cpu;
            bool trimAfterRun = options.Session.TrimNativeHeapAfterRun ?? cpu;
            vision = OnnxModelSession.Create(files.VisionModelPath, EmbeddingGemma2Encoder.SessionDefaults(options.Session) with
            {
                ExpectedModelSha256 = verify ? EmbeddingGemma2ModelAssets.VisionModelSha256 : null,
                KnownModelSha256 = files.VisionModelSha256,
                TrimNativeHeapAfterLoad = trimAfterLoad,
                TrimNativeHeapAfterRun = trimAfterRun,
            });
            multimodal = OnnxModelSession.Create(files.MultimodalModelPath, EmbeddingGemma2Encoder.SessionDefaults(options.Session) with
            {
                ExpectedModelSha256 = verify ? EmbeddingGemma2ModelAssets.MultimodalModelSha256 : null,
                KnownModelSha256 = files.MultimodalModelSha256,
                TrimNativeHeapAfterLoad = trimAfterLoad,
                TrimNativeHeapAfterRun = trimAfterRun,
            });
            string textPath = files.Text.ModelPath;
            var textOptions = options.ToTextEncoderOptions(verify ? EmbeddingGemma2ModelAssets.ModelSha256 : null, files.Text.ModelSha256);
            var lazy = new Lazy<EmbeddingGemma2Encoder>(() => EmbeddingGemma2Encoder.Create(textPath, textOptions), LazyThreadSafetyMode.ExecutionAndPublication);
            return new EmbeddingGemma2MultimodalEmbedder(tokenizer, vision, multimodal, lazy, options, ownsSessions: true);
        }
        catch
        {
            multimodal?.Dispose();
            vision?.Dispose();
            throw;
        }
    }

    /// <summary>Embeds one image with no prompt (upstream: the image alone).</summary>
    public float[] Embed(EmbeddingGemma2Image image, CancellationToken cancellationToken = default) =>
        Embed(EmbeddingGemma2MultimodalInput.FromImage(image), EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds one video with no prompt (upstream: the video alone).</summary>
    public float[] Embed(EmbeddingGemma2Video video, CancellationToken cancellationToken = default) =>
        Embed(EmbeddingGemma2MultimodalInput.FromVideo(video), EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds one input with no prompt.</summary>
    public float[] Embed(EmbeddingGemma2MultimodalInput input, CancellationToken cancellationToken = default) =>
        Embed(input, EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>Embeds one input with <paramref name="prompt"/> rendered first.</summary>
    public float[] Embed(EmbeddingGemma2MultimodalInput input, EmbeddingGemma2Prompt prompt, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        return EmbedBatch([input], prompt, cancellationToken)[0];
    }

    /// <summary>Embeds images in order, each alone with no prompt.</summary>
    public float[][] EmbedBatch(IReadOnlyList<EmbeddingGemma2Image> images, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(images);
        var inputs = new EmbeddingGemma2MultimodalInput[images.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = EmbeddingGemma2MultimodalInput.FromImage(images[i] ?? throw new ArgumentException($"Image {i} is null.", nameof(images)));
        }

        return EmbedBatch(inputs, EmbeddingGemma2Prompt.None, cancellationToken);
    }

    /// <summary>Embeds inputs in order with no prompt.</summary>
    public float[][] EmbedBatch(IReadOnlyList<EmbeddingGemma2MultimodalInput> inputs, CancellationToken cancellationToken = default) =>
        EmbedBatch(inputs, EmbeddingGemma2Prompt.None, cancellationToken);

    /// <summary>
    /// Embeds inputs in order, each with <paramref name="prompt"/> rendered first. Text-only inputs run on the released
    /// text graph, inputs with images on the vision and multimodal graphs. All inputs are validated and tokenized before
    /// any inference runs.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<EmbeddingGemma2MultimodalInput> inputs, EmbeddingGemma2Prompt prompt, CancellationToken cancellationToken = default) =>
        EmbedBatch(inputs, prompt, default, cancellationToken);

    /// <summary>
    /// Like <see cref="EmbedBatch(IReadOnlyList{EmbeddingGemma2MultimodalInput}, EmbeddingGemma2Prompt, CancellationToken)"/>,
    /// and writes each input's token count (image tokens, special tokens and prompt included) to
    /// <paramref name="tokenCounts"/>, which must be empty or hold exactly <c>inputs.Count</c> values.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<EmbeddingGemma2MultimodalInput> inputs, EmbeddingGemma2Prompt prompt, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        var flat = new float[(inputs ?? throw new ArgumentNullException(nameof(inputs))).Count * Dimension];
        if (!tokenCounts.IsEmpty && tokenCounts.Length != inputs.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {inputs.Count} inputs need {inputs.Count}.", nameof(tokenCounts));
        }

        EmbedCore(inputs, prompt, flat, tokenCounts, null, cancellationToken);
        var result = new float[inputs.Count][];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = flat.AsSpan(i * Dimension, Dimension).ToArray();
        }

        return result;
    }

    /// <summary>
    /// The ids the text graph sees for <paramref name="input"/>: the rendered input (prompt, text parts, expanded image
    /// placeholders) tokenized with <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>. Its <see cref="EmbeddingGemma2Input.Length"/>
    /// is the token count checked against <see cref="EmbeddingGemma2MultimodalOptions.MaxSequenceLength"/>.
    /// </summary>
    public EmbeddingGemma2Input Tokenize(EmbeddingGemma2MultimodalInput input, EmbeddingGemma2Prompt? prompt = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        return Prepare(input, prompt ?? EmbeddingGemma2Prompt.None, 0).Tokens;
    }

    /// <summary>
    /// Disposes the sessions and the text encoder if this instance owns them, and releases the embedder's reusable request
    /// buffers. Idempotent.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _buffers.Dispose();
        if (_ownsSessions)
        {
            _multimodal.Dispose();
            _vision.Dispose();
            if (_textEncoder is { IsValueCreated: true })
            {
                _textEncoder.Value.Dispose();
            }
        }
    }

    /// <summary>
    /// Embeds into <paramref name="destination"/> (<c>inputs.Count x Dimension</c>). <paramref name="diagnostics"/>, when
    /// set, receives the soft tokens of every image and the token embeddings of every image row (tests).
    /// </summary>
    internal void EmbedCore(
        IReadOnlyList<EmbeddingGemma2MultimodalInput> inputs,
        EmbeddingGemma2Prompt prompt,
        Span<float> destination,
        Span<int> tokenCounts,
        IEmbeddingGemma2Diagnostics? diagnostics,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(prompt);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (destination.Length != inputs.Count * Dimension)
        {
            throw new ArgumentException($"Destination holds {destination.Length} floats; {inputs.Count} inputs need {inputs.Count * Dimension}.", nameof(destination));
        }

        var prepared = new PreparedInput[inputs.Count];
        var textRows = new List<int>();
        var imageRows = new List<int>();
        for (int i = 0; i < inputs.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            prepared[i] = Prepare(inputs[i] ?? throw new ArgumentException($"Input {i} is null.", nameof(inputs)), prompt, i);
            if (!tokenCounts.IsEmpty)
            {
                tokenCounts[i] = prepared[i].Tokens.Length;
            }

            (prepared[i].HasMedia ? imageRows : textRows).Add(i);
        }

        if (textRows.Count > 0)
        {
            if (_textEncoder is null)
            {
                throw new InvalidOperationException("Text-only inputs need the released text graph; this embedder was created without a text encoder.");
            }

            var encoder = _textEncoder.Value;
            var textInputs = textRows.Select(i => prepared[i].Tokens).ToArray();
            var textOut = new float[textInputs.Length * Dimension];
            encoder.EmbedBatch(textInputs, textOut, cancellationToken);
            for (int k = 0; k < textRows.Count; k++)
            {
                textOut.AsSpan(k * Dimension, Dimension).CopyTo(destination.Slice(textRows[k] * Dimension, Dimension));
            }
        }

        if (imageRows.Count == 0)
        {
            return;
        }

        var imageInputs = imageRows.Select(i => prepared[i].Tokens).ToArray();
        int start = 0;
        foreach (int count in EmbeddingGemma2Encoder.PlanChunks(imageInputs, _options.MaxBatchSize, _options.MaxBatchTokens))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var chunk = new PreparedInput[count];
            for (int r = 0; r < count; r++)
            {
                chunk[r] = prepared[imageRows[start + r]];
            }

            var chunkOut = new float[count * Dimension];
            EmbedImageChunk(chunk, chunkOut, diagnostics, cancellationToken);
            for (int r = 0; r < count; r++)
            {
                chunkOut.AsSpan(r * Dimension, Dimension).CopyTo(destination.Slice(imageRows[start + r] * Dimension, Dimension));
            }

            start += count;
        }
    }

    /// <summary>
    /// Vision-call plan over images of known patch counts, in order: a call grows while it has fewer than
    /// <paramref name="maxImages"/> images and its padded size (images x longest) stays within <paramref name="maxPatches"/>.
    /// </summary>
    internal static List<int> PlanVisionCalls(IReadOnlyList<int> patchCounts, int maxImages, int maxPatches)
    {
        var calls = new List<int>();
        int count = 0, width = 0;
        foreach (int patches in patchCounts)
        {
            int newWidth = Math.Max(width, patches);
            if (count > 0 && (count >= maxImages || (long)(count + 1) * newWidth > maxPatches))
            {
                calls.Add(count);
                count = 0;
                newWidth = patches;
            }

            count++;
            width = newWidth;
        }

        if (count > 0)
        {
            calls.Add(count);
        }

        return calls;
    }

    private PreparedInput Prepare(EmbeddingGemma2MultimodalInput input, EmbeddingGemma2Prompt prompt, int index) =>
        PrepareCore(_tokenizer, _options, input, prompt, index);

    /// <summary>Validates, samples, renders and tokenizes one input (no inference). Internal for tests.</summary>
    internal static PreparedInput PrepareCore(IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2MultimodalOptions options, EmbeddingGemma2MultimodalInput input, EmbeddingGemma2Prompt prompt, int index)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var images = new List<EmbeddingGemma2Image>();
        var counts = new List<int>();
        var videos = new List<PreparedVideo>();
        var media = new List<(int Frames, int SoftTokens)>();
        var blocks = new List<(long TokenId, int Count)>();
        foreach (var part in input.Parts)
        {
            switch (part)
            {
                case EmbeddingGemma2Image image:
                {
                    if (image.PixelCount > options.MaxImagePixels)
                    {
                        throw new EmbeddingGemma2ImageTooLargeException(image.Width, image.Height, options.MaxImagePixels);
                    }

                    int n = EmbeddingGemma2ImageProcessor.GetSoftTokenCount(image.Width, image.Height, options.MaxImageSoftTokens);
                    images.Add(image);
                    counts.Add(n);
                    media.Add((1, n));
                    blocks.Add((ImageTokenId, n));
                    break;
                }

                case EmbeddingGemma2Video video:
                {
                    var prepared = PrepareVideo(video, options, index, videos.Count);
                    videos.Add(prepared);
                    media.Add((prepared.Frames.Length, prepared.SoftTokensPerFrame));
                    for (int f = 0; f < prepared.Frames.Length; f++)
                    {
                        blocks.Add((VideoTokenId, prepared.SoftTokensPerFrame));
                    }

                    break;
                }
            }
        }

        // Cheap length check before tokenizing: every image and every video frame adds its soft tokens and two boundary tokens.
        long mediaTokens = blocks.Sum(b => (long)b.Count + 2);
        if (mediaTokens + 2 > options.MaxSequenceLength)
        {
            throw new EmbeddingGemma2InputTooLongException(index, (int)Math.Min(int.MaxValue, mediaTokens + 2), options.MaxSequenceLength);
        }

        var tokens = tokenizer.Tokenize(input.Render(prompt, media));
        if (tokens.Length > options.MaxSequenceLength)
        {
            throw new EmbeddingGemma2InputTooLongException(index, tokens.Length, options.MaxSequenceLength);
        }

        ValidateMediaTokens(tokens.InputIds.Span, blocks, index);
        return new PreparedInput(tokens, images, counts) { Index = index, Videos = videos };
    }

    /// <summary>
    /// The sampled frames of a video (upstream's sampling, or the pre-sampled frames), checked: one size for all, within
    /// the pixel limit. Reads only the sampled frames of <see cref="EmbeddingGemma2Video.Frames"/>.
    /// </summary>
    private static PreparedVideo PrepareVideo(EmbeddingGemma2Video video, EmbeddingGemma2MultimodalOptions options, int index, int videoIndex)
    {
        int[] indices = video.GetSampledFrameIndices(options.AllowVideosWithoutTiming);
        var frames = new EmbeddingGemma2Image[indices.Length];
        for (int f = 0; f < indices.Length; f++)
        {
            var frame = video.Frames[indices[f]] ?? throw new ArgumentException($"Input {index}, video {videoIndex}: frame {indices[f]} is null.");
            if (frame.PixelCount > options.MaxImagePixels)
            {
                throw new EmbeddingGemma2ImageTooLargeException(frame.Width, frame.Height, options.MaxImagePixels);
            }

            if (f > 0 && (frame.Width != frames[0].Width || frame.Height != frames[0].Height))
            {
                throw new ArgumentException($"Input {index}, video {videoIndex}: frame {indices[f]} is {frame.Width}x{frame.Height}; frame {indices[0]} is {frames[0].Width}x{frames[0].Height}. All frames of a video must have one size.");
            }

            frames[f] = frame;
        }

        int n = EmbeddingGemma2ImageProcessor.GetSoftTokenCount(frames[0].Width, frames[0].Height, VideoFrameSoftTokens);
        return new PreparedVideo(frames, indices, n);
    }

    /// <summary>
    /// The ids must hold exactly the expected media blocks, in order: each <c>&lt;|image&gt;</c> followed by the block's
    /// soft-token count of its token (<c>&lt;|image|&gt;</c> for an image, <c>&lt;|video|&gt;</c> for a video frame) and one
    /// <c>&lt;image|&gt;</c>, and no other media id.
    /// </summary>
    private static void ValidateMediaTokens(ReadOnlySpan<long> ids, List<(long TokenId, int Count)> blocks, int index)
    {
        int block = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            long id = ids[t];
            if (id == ImageStartTokenId)
            {
                if (block >= blocks.Count || t + blocks[block].Count + 1 >= ids.Length)
                {
                    throw new ArgumentException($"Input {index}: unexpected media start token at position {t}.");
                }

                var (tokenId, count) = blocks[block];
                for (int k = 1; k <= count; k++)
                {
                    if (ids[t + k] != tokenId)
                    {
                        throw new ArgumentException($"Input {index}: media block {block} expects {count} tokens of id {tokenId} after position {t}.");
                    }
                }

                if (ids[t + count + 1] != ImageEndTokenId)
                {
                    throw new ArgumentException($"Input {index}: media block {block} is not closed at position {t + count + 1}.");
                }

                t += count + 1;
                block++;
            }
            else if (id is 256_000 or (>= 258_880 and <= 258_884))
            {
                throw new ArgumentException($"Input {index} contains media token id {id} outside a media expansion (position {t}).");
            }
        }

        if (block != blocks.Count)
        {
            throw new ArgumentException($"Input {index}: {blocks.Count} media blocks expected but {block} expansions in the tokens.");
        }
    }

    private void EmbedImageChunk(PreparedInput[] chunk, Span<float> destination, IEmbeddingGemma2Diagnostics? diagnostics, CancellationToken cancellationToken)
    {
        int rows = chunk.Length;
        int width = chunk.Max(p => p.Tokens.Length);
        int cells = rows * width;

        // Soft tokens per image of the chunk, in row order.
        var images = new List<(int Row, int Index, EmbeddingGemma2Image Image, int Count)>();
        for (int r = 0; r < rows; r++)
        {
            for (int k = 0; k < chunk[r].Images.Count; k++)
            {
                images.Add((r, k, chunk[r].Images[k], chunk[r].SoftTokenCounts[k]));
            }
        }

        // Video frames of the chunk (all videos, row order).
        var frames = new List<EmbeddingGemma2Image>();
        foreach (var p in chunk)
        {
            foreach (var v in p.Videos)
            {
                frames.AddRange(v.Frames);
            }
        }

        // Every large buffer of the request comes from the embedder's pool and goes back to it (EmbeddingGemma2BufferPool).
        float[]? imageSoft = null;
        float[]? frameSoft = null;
        var assembled = new List<float[]>();
        long[]? ids = null;
        long[]? mask = null;
        bool[]? mmMask = null;
        float[]? mmEmbeds = null;
        float[]? tokens = null;
        try
        {
            int[] imageOffsets = [];
            if (images.Count > 0)
            {
                imageSoft = RunVision(images.Select(x => x.Image).ToArray(), _options.MaxImageSoftTokens, _options.MaxVisionBatchSize, _options.MaxVisionBatchSize, out imageOffsets, out _, cancellationToken);
            }

            if (diagnostics is not null)
            {
                for (int i = 0; i < images.Count; i++)
                {
                    diagnostics.OnSoftTokens(chunk[images[i].Row].Index, images[i].Index, imageSoft.AsSpan(imageOffsets[i], images[i].Count * SoftTokenDimension).ToArray());
                }
            }

            int[] frameOffsets = [];
            int[] frameCounts = [];
            if (frames.Count > 0)
            {
                frameSoft = RunVision(frames.ToArray(), VideoFrameSoftTokens, _options.MaxVideoFramesPerVisionCall, VideoFramesPreprocessedAhead, out frameOffsets, out frameCounts, cancellationToken);
            }

            // Each row's video soft tokens: its videos in order, each frame at a stride of SoftTokensPerFrame tokens. The
            // frames are contiguous in frameSoft in that order, so the row's region is a slice of it when every frame has
            // SoftTokensPerFrame tokens (always, for frames of one size); otherwise it is assembled with zero padding.
            var rowVideoSoft = new ReadOnlyMemory<float>[rows];
            int frameIndex = 0;
            for (int r = 0; r < rows; r++)
            {
                int firstFrame = frameIndex;
                int rowFloats = 0;
                bool contiguous = true;
                foreach (var video in chunk[r].Videos)
                {
                    for (int f = 0; f < video.Frames.Length; f++)
                    {
                        contiguous &= frameCounts[frameIndex + f] == video.SoftTokensPerFrame;
                    }

                    rowFloats += video.Frames.Length * video.SoftTokensPerFrame * SoftTokenDimension;
                    frameIndex += video.Frames.Length;
                }

                if (rowFloats == 0)
                {
                    continue;
                }

                if (contiguous)
                {
                    rowVideoSoft[r] = frameSoft.AsMemory(frameOffsets[firstFrame], rowFloats);
                }
                else
                {
                    float[] buffer = _buffers.Rent<float>(rowFloats);
                    assembled.Add(buffer);
                    Array.Clear(buffer, 0, rowFloats);
                    int at = 0;
                    int fi = firstFrame;
                    foreach (var video in chunk[r].Videos)
                    {
                        for (int f = 0; f < video.Frames.Length; f++, fi++)
                        {
                            if (frameCounts[fi] > video.SoftTokensPerFrame)
                            {
                                throw new InvalidOperationException($"A video frame produced {frameCounts[fi]} soft tokens; {video.SoftTokensPerFrame} expected.");
                            }

                            frameSoft.AsSpan(frameOffsets[fi], frameCounts[fi] * SoftTokenDimension)
                                .CopyTo(buffer.AsSpan(at + (f * video.SoftTokensPerFrame * SoftTokenDimension)));
                        }

                        at += video.Frames.Length * video.SoftTokensPerFrame * SoftTokenDimension;
                    }

                    rowVideoSoft[r] = buffer.AsMemory(0, rowFloats);
                }

                if (diagnostics is not null)
                {
                    int at = 0;
                    for (int v = 0; v < chunk[r].Videos.Count; v++)
                    {
                        var video = chunk[r].Videos[v];
                        int videoFloats = video.Frames.Length * video.SoftTokensPerFrame * SoftTokenDimension;
                        diagnostics.OnVideoSoftTokens(chunk[r].Index, v, rowVideoSoft[r].Slice(at, videoFloats).ToArray());
                        at += videoFloats;
                    }
                }
            }

            ids = _buffers.Rent<long>(cells);
            mask = _buffers.Rent<long>(cells);
            mmMask = _buffers.Rent<bool>(cells);
            mmEmbeds = _buffers.Rent<float>(checked(cells * SoftTokenDimension));
            tokens = _buffers.Rent<float>(checked(cells * EmbeddingGemma2Encoder.NativeDimension));
            EmbeddingGemma2Encoder.PackRightPadded(chunk.Select(p => p.Tokens).ToArray(), 0, rows, width, ids, mask);
            Array.Clear(mmMask, 0, cells);
            Array.Clear(mmEmbeds, 0, cells * SoftTokenDimension);
            int imageIndex = 0;
            for (int r = 0; r < rows; r++)
            {
                var rowIds = ids.AsSpan(r * width, chunk[r].Tokens.Length);
                int offset = 0;
                int count = 0;
                int videoOffset = 0;
                for (int t = 0; t < rowIds.Length; t++)
                {
                    if (rowIds[t] == VideoTokenId)
                    {
                        // Video soft tokens fill the <|video|> positions in order (upstream's masked_scatter).
                        rowVideoSoft[r].Span.Slice(videoOffset * SoftTokenDimension, SoftTokenDimension)
                            .CopyTo(mmEmbeds.AsSpan((r * width + t) * SoftTokenDimension, SoftTokenDimension));
                        mmMask[r * width + t] = true;
                        videoOffset++;
                        continue;
                    }

                    if (rowIds[t] != ImageTokenId)
                    {
                        continue;
                    }

                    if (offset == count)
                    {
                        count = images[imageIndex].Count;
                        offset = 0;
                        imageIndex++;
                    }

                    imageSoft.AsSpan(imageOffsets[imageIndex - 1] + (offset * SoftTokenDimension), SoftTokenDimension)
                        .CopyTo(mmEmbeds.AsSpan((r * width + t) * SoftTokenDimension, SoftTokenDimension));
                    mmMask[r * width + t] = true;
                    offset++;
                }
            }

            RunMultimodal(ids, mask, mmEmbeds, mmMask, tokens, rows, width, cancellationToken);
            for (int r = 0; r < rows; r++)
            {
                var rowTokens = tokens.AsSpan(r * width * EmbeddingGemma2Encoder.NativeDimension, width * EmbeddingGemma2Encoder.NativeDimension);
                diagnostics?.OnTokenEmbeddings(chunk[r].Index, rowTokens[..(chunk[r].Tokens.Length * EmbeddingGemma2Encoder.NativeDimension)]);
                EmbeddingGemma2Encoder.MeanPoolAndNormalize(
                    rowTokens,
                    mask.AsSpan(r * width, width),
                    EmbeddingGemma2Encoder.NativeDimension,
                    Dimension,
                    destination.Slice(r * Dimension, Dimension));
            }
        }
        finally
        {
            _buffers.Return(tokens);
            _buffers.Return(mmEmbeds);
            _buffers.Return(mmMask);
            _buffers.Return(mask);
            _buffers.Return(ids);
            foreach (var buffer in assembled)
            {
                _buffers.Return(buffer);
            }

            _buffers.Return(frameSoft);
            _buffers.Return(imageSoft);
        }
    }

    /// <summary>
    /// Preprocesses images (or video frames) at <paramref name="budget"/> soft tokens and runs them through the vision graph
    /// in planned calls of at most <paramref name="maxPerCall"/>. Returns one pooled buffer (the caller returns it) with
    /// each image's valid soft tokens at <paramref name="offsets"/> (in floats), <paramref name="counts"/> tokens each, in order.
    /// </summary>
    private float[] RunVision(EmbeddingGemma2Image[] images, int budget, int maxPerCall, int preprocessAhead, out int[] offsets, out int[] counts, CancellationToken cancellationToken)
    {
        int maxPatches = EmbeddingGemma2ImageProcessor.MaxPatches(budget);
        var patchCounts = new int[images.Length];
        offsets = new int[images.Length];
        counts = new int[images.Length];
        int total = 0;
        for (int i = 0; i < images.Length; i++)
        {
            var (tw, th) = EmbeddingGemma2ImageProcessor.GetTargetSize(images[i].Width, images[i].Height, budget);
            int patches = (tw / EmbeddingGemma2ImageProcessor.PatchSize) * (th / EmbeddingGemma2ImageProcessor.PatchSize);
            patchCounts[i] = _options.PadImagesToMaxPatches ? maxPatches : patches;
            counts[i] = patches / (EmbeddingGemma2ImageProcessor.PoolingKernelSize * EmbeddingGemma2ImageProcessor.PoolingKernelSize);
            offsets[i] = total;
            total = checked(total + (counts[i] * SoftTokenDimension));
        }

        float[] soft = _buffers.Rent<float>(total);

        // Images are preprocessed in parallel ahead of their vision calls, at least preprocessAhead at a time (video
        // frames: VideoFramesPreprocessedAhead; images: one call's worth), which bounds the preprocessed memory held.
        int start = 0;
        int ready = 0;
        var preprocessed = new EmbeddingGemma2ImagePatches?[images.Length];
        try
        {
            foreach (int count in PlanVisionCalls(patchCounts, maxPerCall, _options.MaxVisionBatchPatches))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (ready < start + count)
                {
                    int take = Math.Min(images.Length - ready, Math.Max(start + count - ready, preprocessAhead));
                    EmbeddingGemma2ImageProcessor.Preprocess(new ArraySegment<EmbeddingGemma2Image>(images, ready, take), budget, _buffers).CopyTo(preprocessed, ready);
                    ready += take;
                }

                var batch = preprocessed.AsSpan(start, count).ToArray()!;
                int rows = _options.PadImagesToMaxPatches ? maxPatches : batch.Max(p => p!.PatchCount);
                RunVisionCall(batch!, rows, soft, offsets.AsSpan(start, count), cancellationToken);
                foreach (var p in batch)
                {
                    p!.ReturnTo(_buffers);
                }

                preprocessed.AsSpan(start, count).Clear();
                start += count;
            }

            return soft;
        }
        catch
        {
            foreach (var p in preprocessed)
            {
                p?.ReturnTo(_buffers);
            }

            _buffers.Return(soft);
            throw;
        }
    }

    private void RunVisionCall(EmbeddingGemma2ImagePatches[] batch, int rows, float[] soft, ReadOnlySpan<int> offsets, CancellationToken cancellationToken)
    {
        int count = batch.Length;
        int slots = rows / 9;
        int pixelCells = checked(count * rows * EmbeddingGemma2ImageProcessor.PatchValues);
        float[] pixels = _buffers.Rent<float>(pixelCells);
        long[] positions = _buffers.Rent<long>(count * rows * 2);
        float[] softOut = _buffers.Rent<float>(checked(count * slots * SoftTokenDimension));
        bool[] maskOut = _buffers.Rent<bool>(count * slots);
        try
        {
            for (int b = 0; b < count; b++)
            {
                var p = batch[b];
                var dstPixels = pixels.AsSpan(b * rows * EmbeddingGemma2ImageProcessor.PatchValues, rows * EmbeddingGemma2ImageProcessor.PatchValues);
                p.PixelValues.Span.CopyTo(dstPixels);
                dstPixels[p.PixelValues.Length..].Clear();
                var dstPositions = positions.AsSpan(b * rows * 2, rows * 2);
                p.Positions.Span.CopyTo(dstPositions);
                dstPositions[p.Positions.Length..].Fill(-1);
            }

            var memoryInfo = OrtMemoryInfo.DefaultInstance;
            using (var pixelValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, pixels.AsMemory(0, pixelCells), [count, rows, EmbeddingGemma2ImageProcessor.PatchValues]))
            using (var positionValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, positions.AsMemory(0, count * rows * 2), [count, rows, 2]))
            using (var softValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, softOut.AsMemory(0, count * slots * SoftTokenDimension), [count, slots, SoftTokenDimension]))
            using (var maskValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, maskOut.AsMemory(0, count * slots), [count, slots]))
            {
                _vision.Run(VisionInputNames, [pixelValue, positionValue], VisionOutputNames, [softValue, maskValue], cancellationToken);
            }

            for (int b = 0; b < count; b++)
            {
                int n = batch[b].SoftTokenCount;
                var m = maskOut.AsSpan(b * slots, slots);
                for (int k = 0; k < slots; k++)
                {
                    if (m[k] != (k < n))
                    {
                        throw new InvalidOperationException($"The vision graph's soft-token mask is not the first {n} slots (slot {k}); the graph is not the expected EmbeddingGemma 2 vision graph.");
                    }
                }

                softOut.AsSpan(b * slots * SoftTokenDimension, n * SoftTokenDimension).CopyTo(soft.AsSpan(offsets[b]));
            }
        }
        finally
        {
            _buffers.Return(maskOut);
            _buffers.Return(softOut);
            _buffers.Return(positions);
            _buffers.Return(pixels);
        }
    }

    private void RunMultimodal(long[] ids, long[] mask, float[] mmEmbeds, bool[] mmMask, float[] tokens, int rows, int width, CancellationToken cancellationToken)
    {
        int cells = rows * width;
        long[] shape = [rows, width];
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        using var idsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, ids.AsMemory(0, cells), shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, mask.AsMemory(0, cells), shape);
        using var embedsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, mmEmbeds.AsMemory(0, cells * SoftTokenDimension), [rows, width, SoftTokenDimension]);
        using var mmMaskValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, mmMask.AsMemory(0, cells), shape);
        using var tokensValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, tokens.AsMemory(0, cells * EmbeddingGemma2Encoder.NativeDimension), [rows, width, EmbeddingGemma2Encoder.NativeDimension]);
        _multimodal.Run(TextInputNames, [idsValue, maskValue, embedsValue, mmMaskValue], TextOutputNames, [tokensValue], cancellationToken);
    }

    private static void ValidateVisionSignature(OnnxModelSession session)
    {
        var inputs = session.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        if (!inputs.TryGetValue(PixelValuesName, out var pv) || pv.ElementType != "Float" || pv.Dimensions.Count != 3 ||
            !inputs.TryGetValue(PixelPositionIdsName, out var pos) || pos.ElementType != "Int64" || pos.Dimensions.Count != 3 ||
            inputs.Count != 2)
        {
            throw new NotSupportedException($"The vision graph must take exactly '{PixelValuesName}' (float, rank 3) and '{PixelPositionIdsName}' (int64, rank 3); this is not the EmbeddingGemma 2 vision graph.");
        }

        var outputs = session.Outputs.ToDictionary(o => o.Name, StringComparer.Ordinal);
        if (!outputs.TryGetValue(SoftTokensName, out var st) || st.ElementType != "Float" || st.Dimensions.Count != 3 || st.Dimensions[2] != SoftTokenDimension ||
            !outputs.TryGetValue(SoftTokenMaskName, out var sm) || sm.ElementType != "Bool" || sm.Dimensions.Count != 2)
        {
            throw new NotSupportedException($"The vision graph must produce '{SoftTokensName}' (float, [batch, slots, {SoftTokenDimension}]) and '{SoftTokenMaskName}' (bool, [batch, slots]).");
        }
    }

    private static void ValidateMultimodalSignature(OnnxModelSession session)
    {
        var inputs = session.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        bool ok = inputs.Count == 4 &&
            inputs.TryGetValue(EmbeddingGemma2Encoder.InputIdsName, out var a) && a.ElementType == "Int64" && a.Dimensions.Count == 2 &&
            inputs.TryGetValue(EmbeddingGemma2Encoder.AttentionMaskName, out var b) && b.ElementType == "Int64" && b.Dimensions.Count == 2 &&
            inputs.TryGetValue(MultimodalEmbedsName, out var c) && c.ElementType == "Float" && c.Dimensions.Count == 3 &&
            inputs.TryGetValue(MultimodalMaskName, out var d) && d.ElementType == "Bool" && d.Dimensions.Count == 2;
        var output = session.Outputs.FirstOrDefault(o => o.Name == EmbeddingGemma2Encoder.TokenEmbeddingsName);
        if (!ok || output is null || output.ElementType != "Float" || output.Dimensions.Count != 3 || output.Dimensions[2] != EmbeddingGemma2Encoder.NativeDimension)
        {
            throw new NotSupportedException($"The multimodal text graph must take input_ids, attention_mask (int64), {MultimodalEmbedsName} (float) and {MultimodalMaskName} (bool) and produce token_embeddings (float, 768); this is not the EmbeddingGemma 2 multimodal text graph.");
        }
    }

    internal sealed record PreparedInput(EmbeddingGemma2Input Tokens, IReadOnlyList<EmbeddingGemma2Image> Images, IReadOnlyList<int> SoftTokenCounts)
    {
        public int Index { get; init; }

        /// <summary>The videos in order, with their sampled frames.</summary>
        public IReadOnlyList<PreparedVideo> Videos { get; init; } = [];

        /// <summary>True when the input has an image or a video.</summary>
        public bool HasMedia => Images.Count > 0 || Videos.Count > 0;
    }

    /// <summary>A video's sampled frames (indices into its frames) and soft tokens per frame.</summary>
    internal sealed record PreparedVideo(EmbeddingGemma2Image[] Frames, int[] FrameIndices, int SoftTokensPerFrame);
}

/// <summary>Receives intermediate tensors of media rows (parity tests).</summary>
internal interface IEmbeddingGemma2Diagnostics
{
    /// <summary>The valid soft tokens (count x 512) of image <paramref name="image"/> of input <paramref name="input"/>.</summary>
    void OnSoftTokens(int input, int image, float[] softTokens);

    /// <summary>The soft tokens (frames x count x 512, frames in order) of video <paramref name="video"/> of input <paramref name="input"/>.</summary>
    void OnVideoSoftTokens(int input, int video, float[] softTokens)
    {
    }

    /// <summary>The token embeddings (tokens x 768, unpadded) of input <paramref name="input"/>.</summary>
    void OnTokenEmbeddings(int input, ReadOnlySpan<float> tokenEmbeddings);
}
