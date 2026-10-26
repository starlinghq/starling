using Starlings.Batching;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Text embeddings for <c>google/embeddinggemma-2</c> from already-tokenized input, on Starling's text-path ONNX
/// graph (<see cref="EmbeddingGemma2ModelAssets.ModelFile"/>): right-pads each chunk with <c>&lt;pad&gt;</c> (0), runs
/// the graph to the 768-wide token embeddings, mean-pools them over the attention mask (every attended token, prompt
/// tokens included; double accumulation), L2-normalizes, and for a Matryoshka size keeps the leading dimensions and
/// normalizes again: the sentence-transformers modules (Transformer, Pooling, Normalize) of the pinned revision.
/// </summary>
/// <remarks>
/// <para>
/// <b>Text only.</b> Ids of the image, audio and video placeholder and boundary tokens
/// (<see cref="MediaTokenIds"/>) are rejected: the graph has no vision or audio tower, and embedding a placeholder
/// without its media would silently mean something else.
/// </para>
/// <para>
/// <b>Length.</b> Inputs longer than <see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/> (at most 8,192, the
/// card's validated context) throw <see cref="EmbeddingGemma2InputTooLongException"/>; nothing is truncated.
/// sentence-transformers would run them silently.
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Methods are synchronous and safe to call concurrently; each call uses its own pooled
/// buffers. All inputs are validated before any inference runs. Cancellation follows <see cref="OnnxModelSession"/>:
/// checked before each chunk and inside a run (between graph nodes on CPU; on CUDA when the run completes). A
/// cancelled call throws
/// <see cref="OperationCanceledException"/> and returns nothing; the <see cref="Span{T}"/> overload may already have
/// written the rows of the chunks that completed before cancellation (a prefix of the rows by default; with
/// <see cref="EmbeddingGemma2EncoderOptions.SortByLength"/>, the shortest rows, wherever they are in the input).
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2Encoder : IDisposable
{
    /// <summary>Graph input name for token ids.</summary>
    public const string InputIdsName = "input_ids";

    /// <summary>Graph input name for the attention mask.</summary>
    public const string AttentionMaskName = "attention_mask";

    /// <summary>Graph output name for the 768-wide token embeddings (after <c>embedding_projection</c>).</summary>
    public const string TokenEmbeddingsName = "token_embeddings";

    /// <summary>Graph output name for the 512-wide last hidden state (after the final norm, before the projection; diagnostic).</summary>
    public const string LastHiddenStateName = "last_hidden_state";

    /// <summary>Width of the token embeddings and of the full embedding.</summary>
    public const int NativeDimension = 768;

    /// <summary>The longest accepted input, special tokens and prompt included (the model card's context window).</summary>
    public const int MaxSupportedSequenceLength = 8_192;

    /// <summary>Id of <c>&lt;pad&gt;</c>, used to right-pad shorter rows.</summary>
    public const long PadTokenId = 0;

    /// <summary>Id of <c>&lt;eos&gt;</c>.</summary>
    public const long EosTokenId = 1;

    /// <summary>Id of <c>&lt;bos&gt;</c>.</summary>
    public const long BosTokenId = 2;

    /// <summary>Exclusive upper bound for token ids (vocabulary size 262,144).</summary>
    public const long VocabularySize = 262_144;

    private readonly OnnxModelSession _session;
    private readonly bool _ownsSession;
    private readonly EmbeddingGemma2EncoderOptions _options;
    private readonly EncoderCore _core;
    private int _disposed;

    /// <summary>
    /// Wraps an existing session. <paramref name="options"/>.<see cref="EmbeddingGemma2EncoderOptions.Session"/> is
    /// ignored because the session is already configured.
    /// </summary>
    /// <param name="session">A session over the EmbeddingGemma 2 text-path graph.</param>
    /// <param name="options">Encoder options; defaults to the profile.</param>
    /// <param name="ownsSession">When <see langword="true"/>, <see cref="Dispose"/> also disposes the session.</param>
    /// <exception cref="NotSupportedException">The graph signature is not the EmbeddingGemma 2 text-path signature.</exception>
    public EmbeddingGemma2Encoder(OnnxModelSession session, EmbeddingGemma2EncoderOptions? options = null, bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        options ??= new EmbeddingGemma2EncoderOptions();
        options.Validate();
        ValidateSignature(session);
        _session = session;
        _core = new EncoderCore(session, InputIdsName, AttentionMaskName, tokenTypeIdsName: null, TokenEmbeddingsName);
        _ownsSession = ownsSession;
        _options = options;
    }

    /// <summary>The sizes an embedding can have: 768 (native) and the Matryoshka sizes 512, 256 and 128.</summary>
    public static IReadOnlyList<int> SupportedDimensions { get; } = [768, 512, 256, 128];

    /// <summary>
    /// Ids of the multimodal placeholder and boundary tokens, which a text-only embedding rejects: <c>&lt;|image&gt;</c>
    /// (255999), <c>&lt;|audio&gt;</c> (256000), <c>&lt;|image|&gt;</c> (258880), <c>&lt;|audio|&gt;</c> (258881),
    /// <c>&lt;image|&gt;</c> (258882), <c>&lt;audio|&gt;</c> (258883), <c>&lt;|video|&gt;</c> (258884).
    /// </summary>
    public static IReadOnlyList<long> MediaTokenIds { get; } = [255_999, 256_000, 258_880, 258_881, 258_882, 258_883, 258_884];

    /// <summary>Output dimension (<see cref="EmbeddingGemma2EncoderOptions.Dimension"/>).</summary>
    public int Dimension => _options.Dimension;

    /// <summary>Encoder options in effect.</summary>
    public EmbeddingGemma2EncoderOptions Options => _options;

    /// <summary>The underlying session, for diagnostics and placement.</summary>
    public OnnxModelSession Session => _session;

    /// <summary>True for 768, 512, 256 and 128.</summary>
    public static bool IsSupportedDimension(int dimension) => dimension is 768 or 512 or 256 or 128;

    /// <summary>
    /// The library's session defaults: on CPU, ONNX Runtime's CPU memory arena is off unless
    /// <see cref="OnnxSessionOptions.CpuMemoryArena"/> is set. With the arena, the resident memory of a loaded model
    /// kept climbing for an hour on 4-CPU hosts (video: about 2.2 GB/h with one session kept loaded), and its regions of
    /// up to 32 MiB, freed at unload, stayed resident in glibc's per-thread heaps (docs/evidence/eg2-hosted-soak/).
    /// Outputs are unchanged. Runs execute on the runtime's dedicated run threads unless
    /// <see cref="OnnxSessionOptions.DedicatedRunThreads"/> is set: without the arena, each run's buffers are freed into
    /// the calling thread's glibc heap, which keeps them resident, and callers from a rotating thread pool spread that
    /// memory over more and more heaps (docs/evidence/eg2-soak-reliability/).
    /// </summary>
    internal static OnnxSessionOptions SessionDefaults(OnnxSessionOptions session) =>
        session.Provider != OnnxExecutionProvider.Cpu
            ? session
            : session with
            {
                CpuMemoryArena = session.CpuMemoryArena ?? false,
                DedicatedRunThreads = session.DedicatedRunThreads ?? true,
            };

    /// <summary>Loads the text-path graph and owns the resulting session.</summary>
    public static EmbeddingGemma2Encoder Create(string modelPath, EmbeddingGemma2EncoderOptions? options = null)
    {
        options ??= new EmbeddingGemma2EncoderOptions();
        options.Validate();
        var session = OnnxModelSession.Create(modelPath, SessionDefaults(options.Session));
        try
        {
            return new EmbeddingGemma2Encoder(session, options, ownsSession: true);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>
    /// A Matryoshka truncation of an L2-normalized embedding: the first <paramref name="dimension"/> values,
    /// L2-normalized again (double accumulation). <paramref name="dimension"/> must be a supported size no larger than
    /// the embedding.
    /// </summary>
    public static float[] Truncate(ReadOnlySpan<float> embedding, int dimension)
    {
        if (!IsSupportedDimension(dimension))
        {
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Supported dimensions are 768, 512, 256 and 128.");
        }

        if (dimension > embedding.Length)
        {
            throw new ArgumentException($"Cannot truncate a {embedding.Length}-dimensional embedding to {dimension} dimensions.", nameof(dimension));
        }

        var result = new float[dimension];
        double norm = 0;
        for (int d = 0; d < dimension; d++)
        {
            norm += (double)embedding[d] * embedding[d];
        }

        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        for (int d = 0; d < dimension; d++)
        {
            result[d] = (float)(embedding[d] / norm);
        }

        return result;
    }

    /// <summary>Embeds one sequence. Returns a new L2-normalized vector of <see cref="Dimension"/> floats.</summary>
    public float[] Embed(EmbeddingGemma2Input input, CancellationToken cancellationToken = default)
    {
        var result = new float[Dimension];
        EmbedBatch([input], result, cancellationToken);
        return result;
    }

    /// <summary>Embeds a batch. Returns one new vector per input, in input order.</summary>
    public float[][] EmbedBatch(IReadOnlyList<EmbeddingGemma2Input> inputs, CancellationToken cancellationToken = default)
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
    /// <exception cref="EmbeddingGemma2InputTooLongException">An input has more tokens than <see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/>.</exception>
    /// <exception cref="ArgumentException">An input is empty, has an invalid mask, an out-of-range or media token id, or the destination size is wrong.</exception>
    public void EmbedBatch(IReadOnlyList<EmbeddingGemma2Input> inputs, Span<float> destination, CancellationToken cancellationToken = default)
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

        var plan = Plan(inputs, _options.MaxBatchSize, _options.MaxBatchTokens, _options.SortByLength);
        foreach (var chunk in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EmbedChunk(inputs, plan.Rows(chunk), chunk.Width, destination, cancellationToken);
        }
    }

    /// <summary>
    /// Checks one input as <see cref="EmbedBatch(IReadOnlyList{EmbeddingGemma2Input}, Span{float}, CancellationToken)"/>
    /// does before any inference, and throws what a one-input call would throw (the input index is 0). Lets a caller
    /// reject an invalid input on its own before queueing it next to others (micro-batching).
    /// </summary>
    /// <exception cref="EmbeddingGemma2InputTooLongException">The input has more tokens than <see cref="EmbeddingGemma2EncoderOptions.MaxSequenceLength"/>.</exception>
    /// <exception cref="ArgumentException">The input is empty, has an invalid mask, or an out-of-range or media token id.</exception>
    public void Validate(EmbeddingGemma2Input input) => Validate(input, 0);

    /// <summary>Disposes the session if this encoder owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsSession)
        {
            _session.Dispose();
        }
    }

    /// <summary>
    /// Consecutive chunk sizes in input order: a chunk grows while it has fewer than <paramref name="maxRows"/> rows and
    /// its padded size (rows x longest row) stays within <paramref name="maxCells"/>; every chunk has at least one row.
    /// </summary>
    internal static List<int> PlanChunks(IReadOnlyList<EmbeddingGemma2Input> inputs, int maxRows, int maxCells) =>
        Plan(inputs, maxRows, maxCells, sortByLength: false).Chunks.Select(c => c.Count).ToList();

    /// <summary>
    /// The chunk plan of a batch: <see cref="PlanChunks"/>, over rows in ascending length order (ties in input order)
    /// when <paramref name="sortByLength"/> is set.
    /// </summary>
    internal static BatchPlan Plan(IReadOnlyList<EmbeddingGemma2Input> inputs, int maxRows, int maxCells, bool sortByLength)
    {
        Span<int> lengths = inputs.Count <= 256 ? stackalloc int[inputs.Count] : new int[inputs.Count];
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = inputs[i].Length;
        }

        return BatchPlan.Create(lengths, maxRows, maxCells, sortByLength);
    }

    /// <summary>
    /// Mean over positions with mask 1 (sum / max(count, 1e-9)), accumulated in double; then the first
    /// <paramref name="outputDimension"/> values divided by their L2 norm (max(norm, 1e-12)), rounded to float once.
    /// For 768 this is sentence-transformers Pooling(mean) + Normalize; for a Matryoshka size it equals normalizing,
    /// truncating and normalizing again (normalization is scale-invariant).
    /// </summary>
    internal static void MeanPoolAndNormalize(ReadOnlySpan<float> tokens, ReadOnlySpan<long> mask, int width, int outputDimension, Span<float> destination) =>
        EncoderCore.Pool(tokens, mask, width, outputDimension, EncoderPooling.Mean, normalize: true, destination);

    /// <summary>
    /// Writes rows <paramref name="start"/>..start+count into row-major <c>[count, width]</c> id and mask buffers,
    /// right-padding each row with <see cref="PadTokenId"/> and mask 0.
    /// </summary>
    internal static void PackRightPadded(IReadOnlyList<EmbeddingGemma2Input> inputs, int start, int count, int width, Span<long> ids, Span<long> mask)
    {
        Span<int> rows = count <= 256 ? stackalloc int[count] : new int[count];
        for (int r = 0; r < count; r++)
        {
            rows[r] = start + r;
        }

        PackRightPadded(inputs, rows, width, ids, mask);
    }

    /// <summary>
    /// Writes the inputs at <paramref name="rows"/>, in that order, into row-major <c>[rows.Length, width]</c> id and
    /// mask buffers, right-padding each row with <see cref="PadTokenId"/> and mask 0.
    /// </summary>
    internal static void PackRightPadded(IReadOnlyList<EmbeddingGemma2Input> inputs, ReadOnlySpan<int> rows, int width, Span<long> ids, Span<long> mask) =>
        EncoderCore.PackRightPadded(inputs, rows, width, PadTokenId, ReadRow, ids, mask, tokenTypes: default);

    /// <summary>An input as <see cref="EncoderCore"/> packs it: ids and mask (the graph has no token type ids).</summary>
    private static EncoderRow ReadRow(EmbeddingGemma2Input input) => new(input.InputIds, input.AttentionMask, default);

    private static void ValidateSignature(OnnxModelSession session)
    {
        var inputs = session.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        foreach (var name in new[] { InputIdsName, AttentionMaskName })
        {
            if (!inputs.TryGetValue(name, out var info) || info.ElementType != "Int64" || info.Dimensions.Count != 2)
            {
                throw new NotSupportedException($"Graph input '{name}' (int64, rank 2) is missing or has another type; this is not the EmbeddingGemma 2 text-path signature.");
            }
        }

        var unknown = inputs.Keys.Except([InputIdsName, AttentionMaskName], StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            throw new NotSupportedException($"Graph has inputs this encoder does not feed: {string.Join(", ", unknown)}.");
        }

        var output = session.Outputs.FirstOrDefault(o => o.Name == TokenEmbeddingsName);
        if (output is null || output.ElementType != "Float" || output.Dimensions.Count != 3 || output.Dimensions[2] != NativeDimension)
        {
            throw new NotSupportedException($"Graph output '{TokenEmbeddingsName}' (float, [batch, sequence, {NativeDimension}]) is missing; this is not the EmbeddingGemma 2 text-path graph.");
        }
    }

    private void Validate(EmbeddingGemma2Input input, int index)
    {
        var ids = input.InputIds.Span;
        if (ids.Length == 0)
        {
            throw new ArgumentException($"Input {index} has no tokens.", nameof(input));
        }

        if (ids.Length > _options.MaxSequenceLength)
        {
            throw new EmbeddingGemma2InputTooLongException(index, ids.Length, _options.MaxSequenceLength);
        }

        foreach (long id in ids)
        {
            if (id < 0 || id >= VocabularySize)
            {
                throw new ArgumentException($"Input {index} contains token id {id}, outside [0, {VocabularySize}).", nameof(input));
            }

            if (id >= 255_999 && IsMediaToken(id))
            {
                throw new ArgumentException($"Input {index} contains token id {id}, an image, audio or video placeholder or boundary token. This package embeds text only.", nameof(input));
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

    private static bool IsMediaToken(long id) => id is 255_999 or 256_000 or (>= 258_880 and <= 258_884);

    /// <summary>
    /// Runs the inputs at <paramref name="rows"/> as one call padded to <paramref name="width"/> and writes row
    /// <c>rows[r]</c>'s embedding at <c>destination[rows[r] * Dimension]</c>: mean pooling of the 768-wide token
    /// embeddings, L2 normalization of the leading <see cref="Dimension"/> values (<see cref="EncoderCore.EmbedChunk"/>).
    /// </summary>
    private void EmbedChunk(IReadOnlyList<EmbeddingGemma2Input> inputs, ReadOnlySpan<int> rows, int width, Span<float> destination, CancellationToken cancellationToken) =>
        _core.EmbedChunk(inputs, rows, width, ReadRow, PadTokenId, NativeDimension, EncoderPooling.Mean, normalize: true, Dimension, destination, cancellationToken);
}
