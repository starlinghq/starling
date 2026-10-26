using Starlings.Batching;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.E5;

/// <summary>
/// Sentence embeddings for <c>intfloat/multilingual-e5-small</c> (<c>onnx/model.onnx</c>) from already-tokenized
/// input: right-pads each batch with <see cref="E5EncoderOptions.PadTokenId"/>, runs the graph, mean-pools the last
/// hidden state over the attention mask (double accumulation) and L2-normalizes, as the sentence-transformers
/// modules of the pinned revision do.
/// </summary>
/// <remarks>
/// <para>
/// <b>Prefixes.</b> e5 expects <c>"query: "</c> or <c>"passage: "</c> at the start of the text (model card). The
/// encoder never sees text; whoever tokenizes adds the prefix to the text before tokenizing.
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Methods are synchronous and safe to call concurrently; each call uses its own pooled
/// buffers. Cancellation follows <see cref="OnnxModelSession"/>: checked before each chunk (at most
/// <see cref="E5EncoderOptions.MaxBatchSize"/> rows and, when set, <see cref="E5EncoderOptions.MaxBatchTokens"/> padded
/// cells) and inside a run (between graph nodes on CPU; on CUDA when the run completes). A cancelled call throws
/// <see cref="OperationCanceledException"/> and returns nothing; the <see cref="Span{T}"/> overload may already have
/// written the rows of the chunks that completed before cancellation (a prefix of the rows by default; with
/// <see cref="E5EncoderOptions.SortByLength"/>, the shortest rows, wherever they are in the input).
/// </para>
/// <para>
/// <b>Chunking.</b> By default a batch runs as consecutive chunks of <see cref="E5EncoderOptions.MaxBatchSize"/> rows
/// in input order, which is the qualified behavior. <see cref="E5EncoderOptions.MaxBatchTokens"/> also closes a chunk
/// before its padded size (rows x longest row) would exceed the budget, and
/// <see cref="E5EncoderOptions.SortByLength"/> chunks rows in ascending length order and writes each result back to
/// its input position (both experimental).
/// </para>
/// </remarks>
public sealed class E5Encoder : IDisposable
{
    /// <summary>Graph input name for token ids.</summary>
    public const string InputIdsName = "input_ids";

    /// <summary>Graph input name for the attention mask.</summary>
    public const string AttentionMaskName = "attention_mask";

    /// <summary>Graph input name for token type ids (all zero for e5).</summary>
    public const string TokenTypeIdsName = "token_type_ids";

    /// <summary>Graph output name for the last hidden state.</summary>
    public const string LastHiddenStateName = "last_hidden_state";

    private readonly OnnxModelSession _session;
    private readonly bool _ownsSession;
    private readonly E5EncoderOptions _options;
    private readonly EncoderCore _core;
    private int _disposed;

    /// <summary>
    /// Wraps an existing session. <paramref name="options"/>.<see cref="E5EncoderOptions.Session"/> is ignored because
    /// the session is already configured.
    /// </summary>
    /// <param name="session">A session over the e5 graph.</param>
    /// <param name="options">Encoder options; defaults to the e5 profile.</param>
    /// <param name="ownsSession">When <see langword="true"/>, <see cref="Dispose"/> also disposes the session.</param>
    /// <exception cref="NotSupportedException">The graph signature is not the e5 encoder signature.</exception>
    public E5Encoder(OnnxModelSession session, E5EncoderOptions? options = null, bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        options ??= new E5EncoderOptions();
        options.Validate();
        _session = session;
        _ownsSession = ownsSession;
        _options = options;
        (Dimension, bool hasTokenTypeIds) = ValidateSignature(session);
        _core = new EncoderCore(session, InputIdsName, AttentionMaskName, hasTokenTypeIds ? TokenTypeIdsName : null, LastHiddenStateName);
    }

    /// <summary>Embedding dimension read from the graph (384 for e5-small).</summary>
    public int Dimension { get; }

    /// <summary>Encoder options in effect.</summary>
    public E5EncoderOptions Options => _options;

    /// <summary>The underlying session, for diagnostics and placement.</summary>
    public OnnxModelSession Session => _session;

    /// <summary>
    /// The library's session defaults: on CPU, runs execute on the runtime's dedicated run threads unless
    /// <see cref="OnnxSessionOptions.DedicatedRunThreads"/> is set. Each run allocates and frees memory with
    /// <c>malloc</c> on the thread that calls ONNX Runtime; glibc keeps freed memory resident at the top of that
    /// thread's heap, and callers from a rotating thread pool spread it over more and more heaps (hosted one-hour soak:
    /// 1 to 8 heaps holding 4 to 139 MiB after unloading, docs/evidence/e5-soak-reliability/). Outputs are unchanged.
    /// ONNX Runtime's CPU arena keeps its default (on).
    /// </summary>
    internal static OnnxSessionOptions SessionDefaults(OnnxSessionOptions session) =>
        session.Provider != OnnxExecutionProvider.Cpu
            ? session
            : session with { DedicatedRunThreads = session.DedicatedRunThreads ?? true };

    /// <summary>Loads <c>onnx/model.onnx</c> and owns the resulting session.</summary>
    public static E5Encoder Create(string modelPath, E5EncoderOptions? options = null)
    {
        options ??= new E5EncoderOptions();
        options.Validate();
        var session = OnnxModelSession.Create(modelPath, SessionDefaults(options.Session));
        try
        {
            return new E5Encoder(session, options, ownsSession: true);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>Embeds one sequence. Returns a new L2-normalized vector of <see cref="Dimension"/> floats.</summary>
    public float[] Embed(EncoderInput input, CancellationToken cancellationToken = default)
    {
        var result = new float[Dimension];
        EmbedBatch([input], result, cancellationToken);
        return result;
    }

    /// <summary>Embeds a batch. Returns one new vector per input, in input order.</summary>
    public float[][] EmbedBatch(IReadOnlyList<EncoderInput> inputs, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var flat = new float[inputs.Count * Dimension];
        EmbedBatch(inputs, flat, cancellationToken);
        var result = new float[inputs.Count][];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = flat.AsSpan(i * Dimension, Dimension).ToArray();
        }

        return result;
    }

    /// <summary>
    /// Embeds a batch into <paramref name="destination"/>, row-major, <c>inputs.Count x Dimension</c> floats.
    /// All inputs are validated before any inference runs.
    /// </summary>
    /// <exception cref="InputTooLongException">An input has more ids than <see cref="E5EncoderOptions.MaxSequenceLength"/> (an <see cref="ArgumentException"/>).</exception>
    /// <exception cref="ArgumentException">An input is empty, has an invalid mask or id, or the destination size is wrong.</exception>
    public void EmbedBatch(IReadOnlyList<EncoderInput> inputs, Span<float> destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (destination.Length != inputs.Count * Dimension)
        {
            throw new ArgumentException($"Destination holds {destination.Length} floats; {inputs.Count} inputs need {inputs.Count * Dimension}.", nameof(destination));
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            Validate(inputs[i], i);
        }

        var plan = Plan(inputs, _options);
        foreach (var chunk in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EmbedChunk(inputs, plan.Rows(chunk), chunk.Width, destination, cancellationToken);
        }
    }

    /// <summary>The chunk plan <see cref="EmbedBatch(IReadOnlyList{EncoderInput}, Span{float}, CancellationToken)"/> runs.</summary>
    internal static BatchPlan Plan(IReadOnlyList<EncoderInput> inputs, E5EncoderOptions options)
    {
        Span<int> lengths = inputs.Count <= 256 ? stackalloc int[inputs.Count] : new int[inputs.Count];
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = inputs[i].Length;
        }

        return BatchPlan.Create(lengths, options.MaxBatchSize, options.MaxBatchTokens, options.SortByLength);
    }

    /// <summary>
    /// Checks one input as <see cref="EmbedBatch(IReadOnlyList{EncoderInput}, Span{float}, CancellationToken)"/> does
    /// before any inference, and throws what a one-input call would throw (the message names it input 0). Lets a caller
    /// reject an invalid input on its own before queueing it next to others (micro-batching).
    /// </summary>
    /// <exception cref="InputTooLongException">The input has more ids than <see cref="E5EncoderOptions.MaxSequenceLength"/> (an <see cref="ArgumentException"/>).</exception>
    /// <exception cref="ArgumentException">The input is empty, or has an invalid mask or id.</exception>
    public void Validate(EncoderInput input) => Validate(input, 0);

    /// <summary>Disposes the session if this encoder owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsSession)
        {
            _session.Dispose();
        }
    }

    /// <summary>
    /// Mean over positions with mask 1 (sum / max(count, 1e-9)), then x / max(||x||2, 1e-12), accumulated in double
    /// and rounded to float once. Matches sentence-transformers Pooling(mean) + Normalize. The arithmetic is
    /// <see cref="EncoderCore.Pool"/>'s (shared with the other encoders).
    /// </summary>
    internal static void MeanPoolAndNormalize(ReadOnlySpan<float> hidden, ReadOnlySpan<long> mask, int dimension, Span<float> destination) =>
        EncoderCore.Pool(hidden, mask, dimension, dimension, EncoderPooling.Mean, normalize: true, destination);

    /// <summary>
    /// Writes rows <paramref name="start"/>..start+count into row-major <c>[count, width]</c> id and mask buffers,
    /// right-padding each row with <paramref name="padId"/> and mask 0.
    /// </summary>
    internal static void PackRightPadded(IReadOnlyList<EncoderInput> inputs, int start, int count, int width, long padId, Span<long> ids, Span<long> mask)
    {
        Span<int> rows = count <= 256 ? stackalloc int[count] : new int[count];
        for (int r = 0; r < count; r++)
        {
            rows[r] = start + r;
        }

        PackRightPadded(inputs, rows, width, padId, ids, mask);
    }

    /// <summary>
    /// Writes the inputs at <paramref name="rows"/>, in that order, into row-major <c>[rows.Length, width]</c> id and
    /// mask buffers, right-padding each row with <paramref name="padId"/> and mask 0.
    /// </summary>
    internal static void PackRightPadded(IReadOnlyList<EncoderInput> inputs, ReadOnlySpan<int> rows, int width, long padId, Span<long> ids, Span<long> mask) =>
        EncoderCore.PackRightPadded(inputs, rows, width, padId, ReadRow, ids, mask, tokenTypes: default);

    /// <summary>An input as <see cref="EncoderCore"/> packs it: ids and mask; token type ids are all zero for e5.</summary>
    internal static EncoderRow ReadRow(EncoderInput input) => new(input.InputIds, input.AttentionMask, default);

    private static (int Dimension, bool HasTokenTypeIds) ValidateSignature(OnnxModelSession session) =>
        ValidateSignature(session.Inputs, session.Outputs);

    /// <summary>The e5 graph signature checks on ONNX Runtime's tensor descriptions (shared with the browser encoder).</summary>
    internal static (int Dimension, bool HasTokenTypeIds) ValidateSignature(IReadOnlyList<OnnxTensorInfo> graphInputs, IReadOnlyList<OnnxTensorInfo> graphOutputs)
    {
        var inputs = graphInputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        foreach (var name in new[] { InputIdsName, AttentionMaskName })
        {
            if (!inputs.TryGetValue(name, out var info) || info.ElementType != "Int64" || info.Dimensions.Count != 2)
            {
                throw new NotSupportedException($"Graph input '{name}' (int64, rank 2) is missing or has another type; this is not the e5 encoder signature.");
            }
        }

        bool hasTypes = inputs.TryGetValue(TokenTypeIdsName, out var types);
        if (hasTypes && (types!.ElementType != "Int64" || types.Dimensions.Count != 2))
        {
            throw new NotSupportedException($"Graph input '{TokenTypeIdsName}' must be int64 of rank 2.");
        }

        var unknown = inputs.Keys.Except([InputIdsName, AttentionMaskName, TokenTypeIdsName], StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            throw new NotSupportedException($"Graph has inputs this encoder does not feed: {string.Join(", ", unknown)}.");
        }

        var output = graphOutputs.FirstOrDefault(o => o.Name == LastHiddenStateName);
        if (output is null || output.ElementType != "Float" || output.Dimensions.Count != 3 || output.Dimensions[2] <= 0)
        {
            throw new NotSupportedException($"Graph output '{LastHiddenStateName}' (float, [batch, sequence, hidden] with fixed hidden size) is missing.");
        }

        return ((int)output.Dimensions[2], hasTypes);
    }

    private void Validate(EncoderInput input, int index) => Validate(input, index, _options);

    /// <summary>The input checks of <see cref="Validate(EncoderInput)"/> against <paramref name="options"/> (shared with the browser encoder).</summary>
    internal static void Validate(EncoderInput input, int index, E5EncoderOptions options)
    {
        var ids = input.InputIds.Span;
        if (ids.Length == 0)
        {
            throw new ArgumentException($"Input {index} has no tokens.", nameof(input));
        }

        if (ids.Length > options.MaxSequenceLength)
        {
            throw new InputTooLongException($"Input {index} has {ids.Length} tokens; the maximum is {options.MaxSequenceLength}. Truncate in the tokenizer.", nameof(input), index, ids.Length, options.MaxSequenceLength);
        }

        foreach (long id in ids)
        {
            if (id < 0 || id >= options.VocabularySize)
            {
                throw new ArgumentException($"Input {index} contains token id {id}, outside [0, {options.VocabularySize}).", nameof(input));
            }
        }

        if (input.AttentionMask.IsEmpty)
        {
            return;
        }

        var mask = input.AttentionMask.Span;
        if (mask.Length != ids.Length)
        {
            throw new ArgumentException($"Input {index} has {ids.Length} ids but {mask.Length} mask values.", nameof(input));
        }

        bool any = false;
        foreach (long m in mask)
        {
            if (m is not (0 or 1))
            {
                throw new ArgumentException($"Input {index} has attention mask value {m}; expected 0 or 1.", nameof(input));
            }

            any |= m == 1;
        }

        if (!any)
        {
            throw new ArgumentException($"Input {index} has no attended position (attention mask all zero).", nameof(input));
        }
    }

    /// <summary>
    /// Runs the inputs at <paramref name="rows"/> as one call padded to <paramref name="width"/> and writes row
    /// <c>rows[r]</c>'s embedding at <c>destination[rows[r] * Dimension]</c>: mean pooling and L2 normalization of the
    /// last hidden state (<see cref="EncoderCore.EmbedChunk"/>; token type ids, when the graph has them, are zero).
    /// </summary>
    private void EmbedChunk(IReadOnlyList<EncoderInput> inputs, ReadOnlySpan<int> rows, int width, Span<float> destination, CancellationToken cancellationToken) =>
        _core.EmbedChunk(inputs, rows, width, ReadRow, _options.PadTokenId, Dimension, EncoderPooling.Mean, normalize: true, Dimension, destination, cancellationToken);
}
