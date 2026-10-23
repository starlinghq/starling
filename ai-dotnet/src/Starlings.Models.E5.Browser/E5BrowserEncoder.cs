using System.Buffers;
using System.Runtime.Versioning;
using Starlings.Batching;
using Starlings.Runtime.OnnxWeb;

namespace Starlings.Models.E5.Browser;

/// <summary>
/// <c>intfloat/multilingual-e5-small</c> sentence embeddings from already-tokenized input on onnxruntime-web. The
/// browser counterpart of <see cref="E5Encoder"/>: the same input checks, chunk plan, right padding, mean pooling over
/// the attention mask (double accumulation) and L2 normalization (the same code), with the graph run through an
/// <see cref="OnnxWebSession"/>. Methods are asynchronous because onnxruntime-web is.
/// </summary>
/// <remarks>
/// <para>
/// <b>Options.</b> <see cref="E5EncoderOptions"/> applies as for <see cref="E5Encoder"/> (<c>MaxBatchSize</c>,
/// <c>MaxBatchTokens</c>, <c>SortByLength</c>, the id and length limits), except <see cref="E5EncoderOptions.Session"/>,
/// which is ignored: the session is an <see cref="OnnxWebSession"/>.
/// </para>
/// <para>
/// <b>Concurrency and cancellation.</b> Calls may overlap; their graph runs are serialized by the session. The token
/// is checked before each chunk and before each run starts; a run already started in onnxruntime-web completes, then
/// the call throws <see cref="OperationCanceledException"/>. A cancelled call returns nothing; the <see cref="Memory{T}"/>
/// overload may already have written the rows of completed chunks.
/// </para>
/// </remarks>
public sealed class E5BrowserEncoder : IDisposable
{
    private readonly IE5GraphRunner _runner;
    private readonly E5EncoderOptions _options;
    private int _disposed;

    internal E5BrowserEncoder(IE5GraphRunner runner, E5EncoderOptions? options)
    {
        ArgumentNullException.ThrowIfNull(runner);
        options ??= new E5EncoderOptions();
        options.Validate();
        _runner = runner;
        _options = options;
        Dimension = runner.Dimension;
    }

    /// <summary>
    /// Wraps a session over the e5 graph (fp32 or int8-embed).
    /// </summary>
    /// <param name="session">The session.</param>
    /// <param name="options">Encoder options; defaults to the e5 profile. <see cref="E5EncoderOptions.Session"/> is ignored.</param>
    /// <param name="ownsSession">When <see langword="true"/>, <see cref="Dispose"/> also disposes the session.</param>
    /// <exception cref="NotSupportedException">The graph signature is not the e5 encoder signature.</exception>
    [SupportedOSPlatform("browser")]
    public E5BrowserEncoder(OnnxWebSession session, E5EncoderOptions? options = null, bool ownsSession = false)
        : this(new OnnxWebE5GraphRunner(session ?? throw new ArgumentNullException(nameof(session)), ownsSession), options)
    {
        Session = session;
    }

    /// <summary>Embedding dimension read from the graph (384 for e5-small).</summary>
    public int Dimension { get; }

    /// <summary>Encoder options in effect.</summary>
    public E5EncoderOptions Options => _options;

    /// <summary>The underlying session, for diagnostics.</summary>
    [SupportedOSPlatform("browser")]
    public OnnxWebSession? Session { get; }

    /// <summary>Embeds one sequence. Returns a new L2-normalized vector of <see cref="Dimension"/> floats.</summary>
    public async Task<float[]> EmbedAsync(EncoderInput input, CancellationToken cancellationToken = default)
    {
        var result = new float[Dimension];
        await EmbedBatchAsync([input], result, cancellationToken).ConfigureAwait(false);
        return result;
    }

    /// <summary>Embeds a batch. Returns one new vector per input, in input order.</summary>
    public async Task<float[][]> EmbedBatchAsync(IReadOnlyList<EncoderInput> inputs, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var flat = new float[inputs.Count * Dimension];
        await EmbedBatchAsync(inputs, flat, cancellationToken).ConfigureAwait(false);
        var result = new float[inputs.Count][];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = flat.AsSpan(i * Dimension, Dimension).ToArray();
        }

        return result;
    }

    /// <summary>
    /// Embeds a batch into <paramref name="destination"/>, row-major, <c>inputs.Count x Dimension</c> floats. All
    /// inputs are validated before any inference runs.
    /// </summary>
    /// <exception cref="ArgumentException">An input is empty, too long, has an invalid mask or id, or the destination size is wrong.</exception>
    public async Task EmbedBatchAsync(IReadOnlyList<EncoderInput> inputs, Memory<float> destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (destination.Length != inputs.Count * Dimension)
        {
            throw new ArgumentException($"Destination holds {destination.Length} floats; {inputs.Count} inputs need {inputs.Count * Dimension}.", nameof(destination));
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            E5Encoder.Validate(inputs[i], i, _options);
        }

        var plan = E5Encoder.Plan(inputs, _options);
        foreach (var chunk in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] rows = plan.Rows(chunk).ToArray();
            await EmbedChunkAsync(inputs, rows, chunk.Width, destination, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Checks one input as <see cref="EmbedBatchAsync(IReadOnlyList{EncoderInput}, Memory{float}, CancellationToken)"/>
    /// does before any inference, and throws what a one-input call would throw.
    /// </summary>
    public void Validate(EncoderInput input) => E5Encoder.Validate(input, 0, _options);

    /// <summary>Disposes the session if this encoder owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _runner.Dispose();
        }
    }

    private async Task EmbedChunkAsync(IReadOnlyList<EncoderInput> inputs, int[] rows, int width, Memory<float> destination, CancellationToken cancellationToken)
    {
        int count = rows.Length;
        int cells = count * width;
        int hiddenCells = checked(cells * Dimension);
        long[] ids = ArrayPool<long>.Shared.Rent(cells);
        long[] mask = ArrayPool<long>.Shared.Rent(cells);
        long[]? types = _runner.HasTokenTypeIds ? ArrayPool<long>.Shared.Rent(cells) : null;
        float[] hidden = ArrayPool<float>.Shared.Rent(hiddenCells);
        try
        {
            E5Encoder.PackRightPadded(inputs, rows, width, _options.PadTokenId, ids, mask);
            if (types is not null)
            {
                Array.Clear(types, 0, cells);
            }

            await _runner.RunAsync(ids, mask, types, count, width, hidden.AsMemory(0, hiddenCells), cancellationToken).ConfigureAwait(false);

            var output = destination.Span;
            for (int r = 0; r < count; r++)
            {
                E5Encoder.MeanPoolAndNormalize(
                    hidden.AsSpan(r * width * Dimension, width * Dimension),
                    mask.AsSpan(r * width, width),
                    Dimension,
                    output.Slice(rows[r] * Dimension, Dimension));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hidden);
            if (types is not null)
            {
                ArrayPool<long>.Shared.Return(types);
            }

            ArrayPool<long>.Shared.Return(mask);
            ArrayPool<long>.Shared.Return(ids);
        }
    }
}
