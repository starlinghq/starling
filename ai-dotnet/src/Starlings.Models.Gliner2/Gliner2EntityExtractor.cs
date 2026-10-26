using Starlings.Batching;
using Starlings.Runtime.Onnx;
using Starlings.Runtime.Onnx.Batching;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Models.Gliner2;

/// <summary>
/// GLiNER2 entity extraction (<c>fastino/gliner2-base-v1</c>, GLiNER2 2.0.0) on ONNX Runtime: text and entity labels
/// in, labelled spans with offsets and confidences out, as <c>GLiNER2.extract_entities(text, labels, threshold,
/// include_spans=True, include_confidence=True)</c> computes them for English text.
/// </summary>
/// <remarks>
/// <para>
/// <b>Pipeline.</b> GLiNER2 text normalization, the Python-compatible word splitter and lower-casing, the schema
/// prefix encoded with the owned tokenizer (<see cref="Prepare"/>); one ONNX graph from token ids and routing positions
/// to count and span logits (<see cref="RunModel"/>); the count gate, float32 sigmoid, threshold, per-label greedy
/// overlap suppression and result order of GLiNER2's span decoder; offsets mapped onto the caller's text
/// (<see cref="Gliner2Entity"/>).
/// </para>
/// <para>
/// <b>Limits.</b> Texts above <see cref="Gliner2ExtractorOptions.MaxSequenceLength"/> tokens (default 512, schema
/// prefix included) are rejected with <see cref="Gliner2InputTooLongException"/>; nothing is truncated. One call uses
/// one label set: the graph's label projection depends on the whole label set, so a padded batch cannot mix label
/// sets. Classification, relations, structured extraction and GLiNER2's plain (de-duplicated, string-only) output are
/// not offered (<see cref="Gliner2Capabilities"/>).
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Methods are synchronous and safe to call concurrently. Cancellation follows E5 and
/// <see cref="OnnxModelSession"/>: checked while preparing each text, before each chunk (at most
/// <see cref="Gliner2ExtractorOptions.MaxBatchSize"/> texts and, when set,
/// <see cref="Gliner2ExtractorOptions.MaxBatchTokens"/> padded tokens) and inside a run (between graph nodes on CPU;
/// on CUDA when the run completes). A cancelled call
/// throws <see cref="OperationCanceledException"/> and returns nothing; the extractor stays usable.
/// </para>
/// <para>
/// <b>Micro-batching (Experimental, opt-in).</b> With <see cref="Gliner2ExtractorOptions.MicroBatching"/> set,
/// <see cref="ExtractAsync(string, IReadOnlyList{string}, IReadOnlyDictionary{string, string}?, float?, CancellationToken)"/>
/// prepares and length-checks the text on the calling thread (an invalid text fails only its own call), queues it, and
/// runs it in one padded batch with other callers' queued texts of the same schema on one dispatcher thread; each caller
/// decodes with its own threshold. On CUDA this replaces one-at-a-time serialized calls with batched ones. Cancelling a
/// queued call removes it; cancelling a dispatched one cancels its task at once without stopping the batch.
/// <see cref="Dispose"/> faults queued calls with <see cref="ObjectDisposedException"/> and waits for a running batch.
/// </para>
/// </remarks>
public sealed class Gliner2EntityExtractor : IDisposable
{
    /// <summary>Graph input: token ids, int64 [batch, sequence].</summary>
    public const string InputIdsName = "input_ids";

    /// <summary>Graph input: attention mask, int64 [batch, sequence].</summary>
    public const string AttentionMaskName = "attention_mask";

    /// <summary>Graph input: first-subword position of each word, int64 [batch, words].</summary>
    public const string WordIndexName = "word_index";

    /// <summary>Graph input: positions of <c>[P]</c> and each <c>[E]</c>, int64 [batch, 1 + labels].</summary>
    public const string SchemaIndexName = "schema_index";

    /// <summary>Graph output: count logits, float [batch, 20].</summary>
    public const string CountLogitsName = "count_logits";

    /// <summary>Graph output: span logits, float [batch, labels, words, max width].</summary>
    public const string SpanLogitsName = "span_logits";

    private readonly SemaphoreSlim? _runGate;

    private static readonly string[] InputNames = [InputIdsName, AttentionMaskName, WordIndexName, SchemaIndexName];
    private static readonly string[] OutputNames = [CountLogitsName, SpanLogitsName];

    private readonly OnnxModelSession _session;
    private readonly bool _ownsSession;
    private readonly Gliner2Preprocessor _preprocessor;
    private readonly Gliner2ExtractorOptions _options;
    private readonly MicroBatcher<Gliner2ModelInput, Gliner2ModelOutput>? _batcher;
    private int _disposed;

    private Gliner2EntityExtractor(OnnxModelSession session, bool ownsSession, Gliner2Preprocessor preprocessor, Gliner2ExtractorOptions options)
    {
        ValidateSignature(session);
        _session = session;
        _ownsSession = ownsSession;
        _preprocessor = preprocessor;
        _options = options;
        bool serialize = options.SerializeNativeCalls ?? session.Diagnostics.SessionProviders[0] == OnnxRuntimeInfo.CudaProviderName;
        _runGate = serialize ? new SemaphoreSlim(1, 1) : null;
        if (options.MicroBatching is { } microBatching)
        {
            _batcher = new MicroBatcher<Gliner2ModelInput, Gliner2ModelOutput>(
                (inputs, token) => RunModelCore(inputs, enforceLengthLimit: true, token),
                microBatching,
                static input => input.TokenCount,
                static input => new SchemaKey(input.SchemaTokens));
        }
    }

    /// <summary>
    /// True when this extractor runs one graph call at a time (<see cref="Gliner2ExtractorOptions.SerializeNativeCalls"/>;
    /// the default on CUDA, where concurrent runs on one session returned wrong entities).
    /// </summary>
    public bool SerializesNativeCalls => _runGate is not null;

    /// <summary>Options in effect.</summary>
    public Gliner2ExtractorOptions Options => _options;

    /// <summary>The underlying session, for diagnostics and placement.</summary>
    public OnnxModelSession Session => _session;

    /// <summary>
    /// Loads the verified files of <see cref="Gliner2ModelAssets.Lock"/> and owns the session. Unless
    /// <paramref name="options"/> sets <see cref="OnnxSessionOptions.ExpectedModelSha256"/>, the graph must match
    /// <see cref="Gliner2ModelAssets.ModelSha256"/>.
    /// </summary>
    public static Gliner2EntityExtractor Create(Gliner2ModelFiles files, Gliner2ExtractorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new Gliner2ExtractorOptions();
        if (options.Session is { ExpectedModelSha256: null } session)
        {
            options = options with { Session = session with { ExpectedModelSha256 = Gliner2ModelAssets.ModelSha256 } };
        }

        // A snapshot verified with SHA-256 already hashed the graph: pass the digest instead of reading the file again.
        if (options.Session is { KnownModelSha256: null } known && files.ModelSha256 is { } verified)
        {
            options = options with { Session = known with { KnownModelSha256 = verified } };
        }

        return Create(files.ModelPath, files.TokenizerPath, options);
    }

    /// <summary>
    /// Loads a graph and a <c>tokenizer.json</c> and owns the session. The tokenizer must be the pinned file
    /// (<see cref="Gliner2ModelAssets.TokenizerSha256"/>); the graph is checked only when
    /// <see cref="OnnxSessionOptions.ExpectedModelSha256"/> is set.
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer file is not the pinned one.</exception>
    /// <exception cref="NotSupportedException">The graph does not have the route A signature.</exception>
    public static Gliner2EntityExtractor Create(string modelPath, string tokenizerPath, Gliner2ExtractorOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        ArgumentException.ThrowIfNullOrEmpty(tokenizerPath);
        options ??= new Gliner2ExtractorOptions();
        options.Validate();
        var preprocessor = Gliner2Preprocessor.FromFile(tokenizerPath, Gliner2ModelAssets.TokenizerSha256);
        var session = OnnxModelSession.Create(modelPath, options.Session);
        try
        {
            return new Gliner2EntityExtractor(session, ownsSession: true, preprocessor, options);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Wraps an existing session over the route A graph; <paramref name="options"/>.<see cref="Gliner2ExtractorOptions.Session"/>
    /// is ignored because the session is already configured.
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer file is not the pinned one.</exception>
    /// <exception cref="NotSupportedException">The graph does not have the route A signature.</exception>
    public static Gliner2EntityExtractor Create(OnnxModelSession session, string tokenizerPath, Gliner2ExtractorOptions? options = null, bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentException.ThrowIfNullOrEmpty(tokenizerPath);
        options ??= new Gliner2ExtractorOptions();
        options.Validate();
        return new Gliner2EntityExtractor(session, ownsSession, Gliner2Preprocessor.FromFile(tokenizerPath, Gliner2ModelAssets.TokenizerSha256), options);
    }

    /// <summary>Extracts entities of the given labels from one text.</summary>
    /// <param name="text">English text (valid Unicode).</param>
    /// <param name="labels">Entity labels; duplicates are ignored after the first, order and case matter to the model.</param>
    /// <param name="threshold">Score threshold in [0, 1]; null for <see cref="Gliner2ExtractorOptions.Threshold"/>.</param>
    /// <param name="cancellationToken">Cancels before or during the run.</param>
    /// <exception cref="Gliner2InputTooLongException">The text needs more than <see cref="Gliner2ExtractorOptions.MaxSequenceLength"/> tokens.</exception>
    /// <exception cref="ArgumentException">No labels, an empty label, or text with an unpaired surrogate.</exception>
    public Gliner2ExtractionResult Extract(string text, IReadOnlyList<string> labels, float? threshold = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        return ExtractBatch([text], labels, threshold, cancellationToken)[0];
    }

    /// <summary>
    /// Extracts entities of one label set from several texts, in padded batches of
    /// <see cref="Gliner2ExtractorOptions.MaxBatchSize"/>. Every text is prepared and validated before any inference
    /// runs. Results are those of single calls (padding does not change valid logits; see the README for the evidence).
    /// </summary>
    /// <exception cref="Gliner2InputTooLongException">A text needs more than <see cref="Gliner2ExtractorOptions.MaxSequenceLength"/> tokens.</exception>
    public IReadOnlyList<Gliner2ExtractionResult> ExtractBatch(IReadOnlyList<string> texts, IReadOnlyList<string> labels, float? threshold = null, CancellationToken cancellationToken = default) =>
        ExtractBatch(texts, labels, null, threshold, cancellationToken);

    /// <summary>
    /// As <see cref="ExtractBatch(IReadOnlyList{string}, IReadOnlyList{string}, float?, CancellationToken)"/>, with label
    /// descriptions (label to description), which GLiNER2 adds to the schema prompt. Experimental: inputs, logits and
    /// entities equal the pinned GLiNER2 on 11 description cases (<see cref="Gliner2Capabilities"/>).
    /// </summary>
    public IReadOnlyList<Gliner2ExtractionResult> ExtractBatch(
        IReadOnlyList<string> texts, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions, float? threshold = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        float t = threshold ?? _options.Threshold;
        Gliner2ExtractorOptions.ValidateThreshold(t, nameof(threshold));
        var normalizedLabels = Gliner2Preprocessor.NormalizeLabels(labels, nameof(labels));
        var inputs = new Gliner2ModelInput[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string text = texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts));
            inputs[i] = _preprocessor.Build(text, normalizedLabels, descriptions);
            CheckLength(inputs[i], i);
        }

        var outputs = RunModel(inputs, cancellationToken);
        var results = new Gliner2ExtractionResult[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            results[i] = Decode(inputs[i], outputs[i], t);
        }

        return results;
    }

    /// <summary>
    /// Extracts entities of the given labels from one text, as a task. Runs inline unless a scheduler is set: without
    /// <see cref="Gliner2ExtractorOptions.MicroBatching"/> it runs on the calling thread exactly as
    /// <see cref="Extract"/> and returns a completed task. With it, the text is prepared and validated on the calling
    /// thread, then queued and run in a padded batch with other callers' texts of the same schema (Experimental; see the
    /// class remarks). With <see cref="Gliner2ExtractorOptions.ExecutionScheduler"/> set (Experimental), the work that
    /// would run on the calling thread runs as one task on that scheduler instead. Failures are never thrown
    /// synchronously: they are in the returned task.
    /// </summary>
    /// <param name="text">English text (valid Unicode).</param>
    /// <param name="labels">Entity labels; duplicates are ignored after the first, order and case matter to the model.</param>
    /// <param name="threshold">Score threshold in [0, 1]; null for <see cref="Gliner2ExtractorOptions.Threshold"/>.</param>
    /// <param name="cancellationToken">Cancels this call only: before or during its run, or while it is queued.</param>
    public Task<Gliner2ExtractionResult> ExtractAsync(string text, IReadOnlyList<string> labels, float? threshold = null, CancellationToken cancellationToken = default) =>
        ExtractAsync(text, labels, null, threshold, cancellationToken);

    /// <summary>
    /// As <see cref="ExtractAsync(string, IReadOnlyList{string}, float?, CancellationToken)"/>, with label descriptions
    /// (Experimental, <see cref="Gliner2Capabilities"/>). Only texts with the same labels and descriptions share a batch.
    /// </summary>
    /// <param name="text">English text (valid Unicode).</param>
    /// <param name="labels">Entity labels; duplicates are ignored after the first, order and case matter to the model.</param>
    /// <param name="descriptions">Label descriptions (label to description), or null.</param>
    /// <param name="threshold">Score threshold in [0, 1]; null for <see cref="Gliner2ExtractorOptions.Threshold"/>.</param>
    /// <param name="cancellationToken">Cancels this call only: before or during its run, or while it is queued.</param>
    public Task<Gliner2ExtractionResult> ExtractAsync(
        string text, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions, float? threshold = null, CancellationToken cancellationToken = default) =>
        _options.ExecutionScheduler is { } scheduler
            ? Task.Factory.StartNew(() => ExtractInlineAsync(text, labels, descriptions, threshold, cancellationToken), cancellationToken, TaskCreationOptions.DenyChildAttach, scheduler).Unwrap()
            : ExtractInlineAsync(text, labels, descriptions, threshold, cancellationToken);

    private Task<Gliner2ExtractionResult> ExtractInlineAsync(
        string text, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions, float? threshold, CancellationToken cancellationToken)
    {
        try
        {
            if (_batcher is null)
            {
                ArgumentNullException.ThrowIfNull(text);
                return Task.FromResult(ExtractBatch([text], labels, descriptions, threshold, cancellationToken)[0]);
            }

            return ExtractQueuedAsync(_batcher, text, labels, descriptions, threshold, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled<Gliner2ExtractionResult>(cancellationToken);
        }
#pragma warning disable CA1031 // Every failure is reported through the returned task, never thrown synchronously.
        catch (Exception ex)
#pragma warning restore CA1031
        {
            return Task.FromException<Gliner2ExtractionResult>(ex);
        }
    }

    /// <summary>
    /// Statistics of the micro-batcher (batches, items, padding, queue and run time), or null when micro-batching is off.
    /// Experimental.
    /// </summary>
    public MicroBatcherStatistics? MicroBatchingStatistics => _batcher?.GetStatistics();

    /// <summary>
    /// Stage 1: builds the model input GLiNER2 would build (normalization, words, schema prefix, token ids, routing
    /// positions). Does not check the length limit; <see cref="RunModel"/> does.
    /// </summary>
    public Gliner2ModelInput Prepare(string text, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _preprocessor.Build(text, Gliner2Preprocessor.NormalizeLabels(labels, nameof(labels)), descriptions);
    }

    /// <summary>
    /// Stage 2: runs the graph on prepared inputs in right-padded batches of <see cref="Gliner2ExtractorOptions.MaxBatchSize"/>
    /// (ids and mask padded with 0, word positions padded with 0). All inputs must share one label count. Outputs are in
    /// input order; <see cref="Gliner2ExtractorOptions.MaxBatchTokens"/> and
    /// <see cref="Gliner2ExtractorOptions.SortByLength"/> change only how inputs are grouped into calls.
    /// </summary>
    /// <exception cref="Gliner2InputTooLongException">An input exceeds <see cref="Gliner2ExtractorOptions.MaxSequenceLength"/>.</exception>
    /// <exception cref="ArgumentException">Inputs with different label counts.</exception>
    public IReadOnlyList<Gliner2ModelOutput> RunModel(IReadOnlyList<Gliner2ModelInput> inputs, CancellationToken cancellationToken = default) =>
        RunModelCore(inputs, enforceLengthLimit: true, cancellationToken);

    /// <summary>
    /// <see cref="Prepare"/> with GLiNER2's <c>max_len</c> word cap, for fidelity evidence on fixture cases outside the
    /// contract. Not public: the word cap is unsupported.
    /// </summary>
    internal Gliner2ModelInput PrepareOutsideContract(string text, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions, int? maxWords)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _preprocessor.Build(text, Gliner2Preprocessor.NormalizeLabels(labels, nameof(labels)), descriptions, maxWords);
    }

    /// <summary>
    /// <see cref="RunModel"/> without the length limit, for fidelity evidence on fixture inputs outside the contract
    /// (the graph itself accepts any length). Not public: the profile rejects such inputs.
    /// </summary>
    internal IReadOnlyList<Gliner2ModelOutput> RunModelOutsideContract(IReadOnlyList<Gliner2ModelInput> inputs, CancellationToken cancellationToken = default) =>
        RunModelCore(inputs, enforceLengthLimit: false, cancellationToken);

    private Gliner2ModelOutput[] RunModelCore(IReadOnlyList<Gliner2ModelInput> inputs, bool enforceLengthLimit, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        for (int i = 0; i < inputs.Count; i++)
        {
            ArgumentNullException.ThrowIfNull(inputs[i], nameof(inputs));
            if (enforceLengthLimit)
            {
                CheckLength(inputs[i], i);
            }

            if (inputs[i].Labels.Count != inputs[0].Labels.Count)
            {
                throw new ArgumentException(
                    $"Input {i} has {inputs[i].Labels.Count} labels, input 0 has {inputs[0].Labels.Count}; one call runs one label set.", nameof(inputs));
            }
        }

        var outputs = new Gliner2ModelOutput[inputs.Count];
        var plan = Plan(inputs, _options);
        foreach (var chunk in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            RunChunk(inputs, plan.Rows(chunk).ToArray(), outputs, cancellationToken);
        }

        return outputs;
    }

    /// <summary>The chunk plan <see cref="RunModel"/> runs, over token counts (schema prefix included).</summary>
    internal static BatchPlan Plan(IReadOnlyList<Gliner2ModelInput> inputs, Gliner2ExtractorOptions options)
    {
        var lengths = new int[inputs.Count];
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = inputs[i].TokenCount;
        }

        return BatchPlan.Create(lengths, options.MaxBatchSize, options.MaxBatchTokens, options.SortByLength);
    }

    /// <summary>Stage 3: GLiNER2's span decoding of one input's graph outputs, mapped onto the input text.</summary>
    /// <param name="input">The prepared input.</param>
    /// <param name="output">Its graph outputs.</param>
    /// <param name="threshold">Score threshold in [0, 1]; null for <see cref="Gliner2ExtractorOptions.Threshold"/>.</param>
    public Gliner2ExtractionResult Decode(Gliner2ModelInput input, Gliner2ModelOutput output, float? threshold = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(output);
        float t = threshold ?? _options.Threshold;
        Gliner2ExtractorOptions.ValidateThreshold(t, nameof(threshold));
        if (output.LabelCount != input.Labels.Count || output.WordCount != input.WordCount)
        {
            throw new ArgumentException("The output does not belong to this input (label or word count differs).", nameof(output));
        }

        return Gliner2Decoder.ToResult(input, Gliner2Decoder.DecodeUpstream(input, output, t));
    }

    /// <summary>Disposes the session if this extractor owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Queued calls fault; a running batch finishes before the session goes.
            _batcher?.Dispose();
            if (_ownsSession)
            {
                _session.Dispose();
            }
        }
    }

    private async Task<Gliner2ExtractionResult> ExtractQueuedAsync(
        MicroBatcher<Gliner2ModelInput, Gliner2ModelOutput> batcher,
        string text,
        IReadOnlyList<string> labels,
        IReadOnlyDictionary<string, string>? descriptions,
        float? threshold,
        CancellationToken cancellationToken)
    {
        // Everything ExtractBatch checks before inference is checked here, so an invalid text fails this call alone and
        // never the batch it would have joined.
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        float t = threshold ?? _options.Threshold;
        Gliner2ExtractorOptions.ValidateThreshold(t, nameof(threshold));
        var input = _preprocessor.Build(text, Gliner2Preprocessor.NormalizeLabels(labels, nameof(labels)), descriptions);
        CheckLength(input, 0);
        var output = await batcher.EnqueueAsync(input, cancellationToken).ConfigureAwait(false);
        return Decode(input, output, t);
    }

    private static void ValidateSignature(OnnxModelSession session)
    {
        var inputs = session.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        foreach (var name in InputNames)
        {
            if (!inputs.TryGetValue(name, out var info) || info.ElementType != "Int64" || info.Dimensions.Count != 2)
            {
                throw new NotSupportedException($"Graph input '{name}' (int64, rank 2) is missing or has another type; this is not the GLiNER2 route A graph.");
            }
        }

        var unknown = inputs.Keys.Except(InputNames, StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            throw new NotSupportedException($"Graph has inputs this extractor does not feed: {string.Join(", ", unknown)}.");
        }

        var outputs = session.Outputs.ToDictionary(o => o.Name, StringComparer.Ordinal);
        if (!outputs.TryGetValue(CountLogitsName, out var count) || count.ElementType != "Float" || count.Dimensions.Count != 2)
        {
            throw new NotSupportedException($"Graph output '{CountLogitsName}' (float, rank 2) is missing.");
        }

        if (!outputs.TryGetValue(SpanLogitsName, out var span) || span.ElementType != "Float" || span.Dimensions.Count != 4)
        {
            throw new NotSupportedException($"Graph output '{SpanLogitsName}' (float, rank 4) is missing.");
        }
    }

    private void CheckLength(Gliner2ModelInput input, int index)
    {
        if (input.TokenCount > _options.MaxSequenceLength)
        {
            throw new Gliner2InputTooLongException(
                $"Text {index} needs {input.TokenCount} tokens (schema prefix included); the maximum is {_options.MaxSequenceLength}. " +
                "Nothing is truncated: split or shorten the text.")
            {
                InputIndex = index,
                TokenCount = input.TokenCount,
                MaxTokens = _options.MaxSequenceLength,
            };
        }
    }

    /// <summary>Runs the inputs at <paramref name="rows"/> as one padded call; writes <c>outputs[rows[r]]</c>.</summary>
    private void RunChunk(IReadOnlyList<Gliner2ModelInput> inputs, int[] rows, Gliner2ModelOutput[] outputs, CancellationToken cancellationToken)
    {
        int count = rows.Length;
        int sequence = 0;
        int words = 0;
        int slots = inputs[rows[0]].SchemaIndex.Length;
        for (int r = 0; r < count; r++)
        {
            sequence = Math.Max(sequence, inputs[rows[r]].TokenCount);
            words = Math.Max(words, inputs[rows[r]].WordCount);
        }

        var ids = new long[count * sequence];
        var mask = new long[count * sequence];
        var wordIndex = new long[count * words];
        var schemaIndex = new long[count * slots];
        for (int r = 0; r < count; r++)
        {
            var input = inputs[rows[r]];
            input.InputIds.CopyTo(ids.AsSpan(r * sequence));
            mask.AsSpan(r * sequence, input.TokenCount).Fill(1);
            input.WordIndex.CopyTo(wordIndex.AsSpan(r * words));
            input.SchemaIndex.CopyTo(schemaIndex.AsSpan(r * slots));
        }

        using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, [count, sequence]);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, [count, sequence]);
        using var wordValue = OrtValue.CreateTensorValueFromMemory(wordIndex, [count, words]);
        using var schemaValue = OrtValue.CreateTensorValueFromMemory(schemaIndex, [count, slots]);
        // Concurrent runs on one CUDA session returned wrong entities in 6 of 8 test runs (four threads, ORT 1.30.0 CUDA
        // EP); on CUDA the run and the read of its outputs happen one call at a time. A waiting call stays cancellable.
        _runGate?.Wait(cancellationToken);
        try
        {
            ReadOutputs(inputs, rows, outputs, words, slots, idsValue, maskValue, wordValue, schemaValue, cancellationToken);
        }
        finally
        {
            _runGate?.Release();
        }
    }

    private void ReadOutputs(
        IReadOnlyList<Gliner2ModelInput> inputs, int[] rows, Gliner2ModelOutput[] outputs, int words, int slots,
        OrtValue idsValue, OrtValue maskValue, OrtValue wordValue, OrtValue schemaValue, CancellationToken cancellationToken)
    {
        int count = rows.Length;
        using var results = _session.Run(InputNames, [idsValue, maskValue, wordValue, schemaValue], OutputNames, cancellationToken);

        var countShape = results[0].GetTensorTypeAndShape().Shape;
        var spanShape = results[1].GetTensorTypeAndShape().Shape;
        int countWidth = (int)countShape[1];
        int labels = (int)spanShape[1];
        int maxWidth = (int)spanShape[3];
        if (spanShape[0] != count || spanShape[2] != words || labels != slots - 1 || countShape[0] != count)
        {
            throw new InvalidOperationException(
                $"Unexpected graph output shapes: count_logits [{string.Join(", ", countShape)}], span_logits [{string.Join(", ", spanShape)}].");
        }

        var countData = results[0].GetTensorDataAsSpan<float>();
        var spanData = results[1].GetTensorDataAsSpan<float>();
        for (int r = 0; r < count; r++)
        {
            var input = inputs[rows[r]];
            int n = input.WordCount;
            var countLogits = countData.Slice(r * countWidth, countWidth).ToArray();
            var spans = new float[labels * n * maxWidth];
            for (int label = 0; label < labels; label++)
            {
                var source = spanData.Slice(((r * labels + label) * words) * maxWidth, n * maxWidth);
                source.CopyTo(spans.AsSpan(label * n * maxWidth));
            }

            outputs[rows[r]] = new Gliner2ModelOutput(countLogits, spans, labels, n, maxWidth);
        }
    }

    /// <summary>
    /// Micro-batching group key: the schema tokens of a prepared input (labels in order and the description prompt).
    /// The threshold is not part of it: it only affects decoding, which each caller does for itself.
    /// </summary>
    private sealed class SchemaKey(IReadOnlyList<string> tokens) : IEquatable<SchemaKey>
    {
        private readonly IReadOnlyList<string> _tokens = tokens;
        private readonly int _hash = Hash(tokens);

        public bool Equals(SchemaKey? other) =>
            other is not null && (ReferenceEquals(this, other) || (_hash == other._hash && _tokens.SequenceEqual(other._tokens, StringComparer.Ordinal)));

        public override bool Equals(object? obj) => Equals(obj as SchemaKey);

        public override int GetHashCode() => _hash;

        private static int Hash(IReadOnlyList<string> tokens)
        {
            var hash = default(HashCode);
            foreach (string token in tokens)
            {
                hash.Add(token, StringComparer.Ordinal);
            }

            return hash.ToHashCode();
        }
    }
}
