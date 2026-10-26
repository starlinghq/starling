using Microsoft.ML.OnnxRuntimeGenAI;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>Search settings of one <see cref="GenAISequence"/>.</summary>
public sealed record GenAISequenceOptions
{
    /// <summary>Total length (prompt plus generated tokens) the sequence may reach. GenAI sizes its caches for it.</summary>
    public required int MaxLength { get; init; }

    /// <summary>Sampling settings, or null for greedy search (argmax at every step).</summary>
    public SamplingOptions? Sampling { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxLength, nameof(MaxLength));
        Sampling?.Validate();
    }
}

/// <summary>
/// Advanced: one ONNX Runtime GenAI generator, owned by one request. Not thread-safe; dispose it when the request ends.
/// Holds a reference on its <see cref="GenAIModel"/>, so the model's native memory stays alive until this is disposed.
/// </summary>
/// <remarks>
/// <see cref="Dispose"/> is the way to release the generator. A sequence that becomes unreachable without it (for
/// example inside a streaming enumerator that was dropped without disposal) is released after a garbage collection
/// finds it: the finalizer queues the same release as <see cref="Dispose"/> (the native generator under the model's
/// native-call lock, then the model reference) to a thread-pool thread, never running it on the finalizer thread.
/// </remarks>
public sealed class GenAISequence : IGenerationSequence
{
    private readonly GenAIModel _owner;
    private readonly Generator _generator;
    private readonly NativeLifetime _lifetime;
    private float[]? _logitsRow;
    private int _disposed;

    internal GenAISequence(GenAIModel owner, GeneratorParams parameters, Generator generator, int vocabularySize)
    {
        _owner = owner;
        _generator = generator;
        VocabularySize = vocabularySize;

        // GenAI's own finalizers would destroy these on the finalizer thread, outside the native-call lock and the
        // model's reference count; NativeLifetime owns their release instead (see its remarks).
        NativeLifetime.TakeOverFinalization(generator);
        NativeLifetime.TakeOverFinalization(parameters);
        _lifetime = new NativeLifetime(() => ReleaseNative(owner, parameters, generator));
    }

    /// <summary>Width of the logits.</summary>
    public int VocabularySize { get; }

    /// <summary>Tokens in the sequence (prompt and generated).</summary>
    public int TokenCount
    {
        get
        {
            ThrowIfDisposed();
            long count = 0;
            Invoke(() => count = (long)_generator.TokenCount());
            return checked((int)count);
        }
    }

    /// <summary>True once GenAI considers the sequence finished (an end-of-sequence id or the maximum length).</summary>
    public bool IsDone
    {
        get
        {
            ThrowIfDisposed();
            bool done = false;
            Invoke(() => done = _generator.IsDone());
            return done;
        }
    }

    /// <summary>
    /// Appends token ids and runs the model over them (the prefill for a prompt). Blocking native work that cannot be
    /// interrupted. Ids must be smaller than <see cref="VocabularySize"/>.
    /// </summary>
    public void AppendTokens(ReadOnlySpan<int> tokenIds)
    {
        ThrowIfDisposed();
        if (tokenIds.IsEmpty)
        {
            throw new ArgumentException("At least one token id is required.", nameof(tokenIds));
        }

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)VocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIds), id, $"Token id outside the vocabulary [0, {VocabularySize}).");
            }
        }

        int[] ids = tokenIds.ToArray();
        Invoke(() => _generator.AppendTokens(ids));
    }

    /// <summary>Runs one decode step and returns the selected token id (greedy or sampled per the options).</summary>
    public int GenerateNextToken()
    {
        ThrowIfDisposed();
        int next = 0;
        Invoke(() =>
        {
            _generator.GenerateNextToken();
            next = _generator.GetNextTokens()[0];
        });
        return next;
    }

    /// <summary>
    /// Copies the most recent logits into <paramref name="destination"/> as float32 (fp16 logits are widened): after
    /// <see cref="AppendTokens"/>, those for the last appended position (the step-0 logits of a prompt); after
    /// <see cref="GenerateNextToken"/>, those that selected the token it returned (GenAI runs the model on a selected
    /// token only at the next step; measured with ONNX Runtime GenAI 0.17.1).
    /// </summary>
    public void CopyLastLogits(Span<float> destination)
    {
        ThrowIfDisposed();
        if (destination.Length != VocabularySize)
        {
            throw new ArgumentException($"Destination length must be the vocabulary size {VocabularySize}.", nameof(destination));
        }

        // Reused across calls: per-step logprobs read one row per decode step.
        float[] row = _logitsRow ??= new float[VocabularySize];
        Invoke(() =>
        {
            using var tensor = _generator.GetOutput("logits");
            long total = tensor.NumElements();
            int vocab = VocabularySize;
            int start = checked((int)(total - vocab));
            switch (tensor.Type())
            {
                case ElementType.float32:
                    tensor.GetData<float>().Slice(start, vocab).CopyTo(row);
                    break;
                case ElementType.float16:
                    {
                        var half = tensor.GetData<Half>().Slice(start, vocab);
                        for (int i = 0; i < vocab; i++)
                        {
                            row[i] = (float)half[i];
                        }

                        break;
                    }

                default:
                    throw new GenAIModelException($"Unsupported logits element type {tensor.Type()}.");
            }
        });
        row.CopyTo(destination);
    }

    /// <summary>Releases the generator and its reference on the model, on the calling thread. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _lifetime.Dispose();
        }
    }

    // The release of Dispose and of the finalizer path (NativeLifetime): the generator under the native-call lock, its
    // parameters, then the model reference (which frees the native model after its last sequence).
    private static void ReleaseNative(GenAIModel owner, GeneratorParams parameters, Generator generator)
    {
        try
        {
            if (owner.NativeCallLock is { } gate)
            {
                lock (gate)
                {
                    generator.Dispose();
                }
            }
            else
            {
                generator.Dispose();
            }

            parameters.Dispose();
        }
        finally
        {
            owner.Release();
        }
    }

    private void Invoke(Action action)
    {
        try
        {
            if (_owner.NativeCallLock is { } gate)
            {
                lock (gate)
                {
                    action();
                }
            }
            else
            {
                action();
            }
        }
        catch (OnnxRuntimeGenAIException ex)
        {
            throw new GenAIModelException("ONNX Runtime GenAI failed: " + ex.Message, ex);
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
}

/// <summary>What the decode loop needs from a sequence; implemented by <see cref="GenAISequence"/> and by test fakes.</summary>
internal interface IGenerationSequence : IDisposable
{
    int VocabularySize { get; }

    bool IsDone { get; }

    void AppendTokens(ReadOnlySpan<int> tokenIds);

    int GenerateNextToken();

    void CopyLastLogits(Span<float> destination);
}
