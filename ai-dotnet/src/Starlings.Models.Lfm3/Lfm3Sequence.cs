using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Starlings.Generation;

namespace Starlings.Models.Lfm3;

/// <summary>
/// One request's decoding state on an <see cref="Lfm3Model"/>: the convolution and attention caches, the number of
/// tokens the graph has seen, and the logits of the last position. Not thread-safe; one instance per request, disposed
/// when the request ends.
/// </summary>
/// <remarks>
/// <para>A token selected by <see cref="Step"/> is not run through the graph until the next <see cref="Step"/> or
/// <see cref="AppendTokens"/> (as ONNX Runtime GenAI does), so <see cref="CopyLastLogits"/> after a step returns the
/// logits that selected it, and a request that stops at that token never pays for its forward pass.</para>
/// <para>Each call runs the graph with <c>logits_to_keep</c> 1: only the last position's logits are computed.
/// <see cref="Step"/> is greedy (the first maximum); sampling is not offered.</para>
/// </remarks>
public sealed class Lfm3Sequence : IGenerationSequence
{
    private readonly Lfm3Model _model;
    private readonly float[] _logits = new float[Lfm3Checkpoint.VocabularySize];

    // The intervention strength of this request: fixed at creation, never written afterwards, owned by this sequence
    // (no tensor is shared between sequences). Null on the original graph, where no extra input exists.
    private readonly float[]? _strength;
    private OrtValue[] _past;
    private IDisposable? _pastOwner;
    private int _processed;
    private int _pending = -1;
    private bool _hasLogits;
    private bool _disposed;

    internal Lfm3Sequence(Lfm3Model model, float? strength = null)
    {
        _model = model;
        _strength = strength is { } value ? [value] : null;
        _past = new OrtValue[model.PastNames.Length];
        TensorElementType type = model.CacheElementType;
        try
        {
            for (int i = 0; i < _past.Length; i++)
            {
                _past[i] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, type, model.InitialPastShapes[i]);
                _past[i].GetTensorMutableRawData().Clear();
            }
        }
        catch
        {
            DisposePast();
            throw;
        }
    }

    /// <inheritdoc />
    public int VocabularySize => Lfm3Checkpoint.VocabularySize;

    /// <summary>The intervention strength fixed for this request; null when the graph has no intervention.</summary>
    internal float? Strength => _strength?[0];

    /// <summary>Tokens in the sequence: those run through the graph plus a selected token not yet run.</summary>
    public int Length => _processed + (_pending >= 0 ? 1 : 0);

    /// <summary>True once the sequence holds <see cref="Lfm3Model.ContextLength"/> tokens.</summary>
    public bool IsDone => Length >= _model.ContextLength;

    /// <summary>
    /// Appends token ids (after a token selected by <see cref="Step"/>, if any) and runs the graph over them in one call.
    /// </summary>
    /// <exception cref="ArgumentException">No ids, or an id outside the vocabulary.</exception>
    /// <exception cref="InvalidOperationException">The ids would exceed <see cref="Lfm3Model.ContextLength"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled before or during the run.</exception>
    public void AppendTokens(ReadOnlySpan<int> tokenIds, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokenIds.IsEmpty)
        {
            throw new ArgumentException("At least one token id is needed.", nameof(tokenIds));
        }

        foreach (int id in tokenIds)
        {
            if ((uint)id >= Lfm3Checkpoint.VocabularySize)
            {
                throw new ArgumentException($"Token id {id} is outside the vocabulary (0 to {Lfm3Checkpoint.VocabularySize - 1}).", nameof(tokenIds));
            }
        }

        if ((long)Length + tokenIds.Length > _model.ContextLength)
        {
            throw new InvalidOperationException(
                $"{Length} tokens plus {tokenIds.Length} exceed the context length {_model.ContextLength}.");
        }

        int[] ids;
        if (_pending >= 0)
        {
            ids = new int[tokenIds.Length + 1];
            ids[0] = _pending;
            tokenIds.CopyTo(ids.AsSpan(1));
        }
        else
        {
            ids = tokenIds.ToArray();
        }

        Run(ids, cancellationToken);
    }

    /// <summary>
    /// Runs a selected token not yet run (if any), then selects the next token greedily from the last logits and
    /// returns it. The selected token is run by the next call.
    /// </summary>
    /// <exception cref="InvalidOperationException">No tokens were appended yet, or the sequence is full.</exception>
    /// <exception cref="OperationCanceledException">Cancelled before or during the run.</exception>
    public int Step(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_hasLogits)
        {
            throw new InvalidOperationException("Append the prompt before the first step.");
        }

        if (IsDone)
        {
            throw new InvalidOperationException($"The sequence holds {Length} tokens, the context length.");
        }

        if (_pending >= 0)
        {
            int id = _pending;
            _pending = -1;
            Run([id], cancellationToken);
        }

        _pending = ArgMax(_logits);
        return _pending;
    }

    /// <summary>
    /// Copies the most recent logits (float32): after <see cref="AppendTokens"/>, those of the last appended position;
    /// after <see cref="Step"/>, those that selected the token it returned.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is not <see cref="VocabularySize"/> long.</exception>
    /// <exception cref="InvalidOperationException">No tokens were appended yet.</exception>
    public void CopyLastLogits(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length != _logits.Length)
        {
            throw new ArgumentException($"The destination must hold {_logits.Length} values.", nameof(destination));
        }

        if (!_hasLogits)
        {
            throw new InvalidOperationException("Append the prompt first.");
        }

        _logits.CopyTo(destination);
    }

    /// <inheritdoc />
    ValueTask IGenerationSequence.AppendTokensAsync(ReadOnlyMemory<int> tokenIds, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AppendTokens(tokenIds.Span, cancellationToken);
        return ValueTask.CompletedTask;
    }

    /// <inheritdoc />
    ValueTask<int[]> IGenerationSequence.StepAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult<int[]>([Step(cancellationToken)]);
    }

    /// <inheritdoc />
    ValueTask IGenerationSequence.CopyLastLogitsAsync(Memory<float> destination, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        CopyLastLogits(destination.Span);
        return ValueTask.CompletedTask;
    }

    /// <summary>Releases the caches. Idempotent.</summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            DisposePast();
        }
    }

    internal static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestValue)
            {
                bestValue = values[i];
                best = i;
            }
        }

        return best;
    }

    private void Run(int[] ids, CancellationToken cancellationToken)
    {
        int n = ids.Length;
        int past = _processed;
        var inputIds = new long[n];
        var positions = new long[n];
        for (int i = 0; i < n; i++)
        {
            inputIds[i] = ids[i];
            positions[i] = past + i;
        }

        var mask = new long[past + n];
        Array.Fill(mask, 1L);
        using var idsValue = OrtValue.CreateTensorValueFromMemory(inputIds, [1, n]);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, [1, past + n]);
        using var positionsValue = OrtValue.CreateTensorValueFromMemory(positions, [1, n]);
        using var keepValue = OrtValue.CreateTensorValueFromMemory(new long[] { 1 }, [1]);
        // The strength tensor wraps this sequence's own array and lives until the native call returns.
        using OrtValue? strengthValue = _strength is null ? null : OrtValue.CreateTensorValueFromMemory(_strength, [1]);
        OrtValue[] inputs = strengthValue is null
            ? [idsValue, maskValue, positionsValue, keepValue, .. _past]
            : [idsValue, maskValue, positionsValue, keepValue, .. _past, strengthValue];
        var outputs = _model.Session.Run(_model.InputNames, inputs, _model.OutputNames, cancellationToken);
        OrtValue[] values = [.. outputs];
        try
        {
            ReadOnlySpan<float> logits = values[0].GetTensorDataAsSpan<float>();
            if (logits.Length != _logits.Length)
            {
                throw new InvalidDataException($"The graph returned {logits.Length} logits, expected {_logits.Length}.");
            }

            logits.CopyTo(_logits);
        }
        catch
        {
            outputs.Dispose();
            throw;
        }

        DisposePast();
        _past = values[1..];
        _pastOwner = outputs;
        _processed = past + n;
        _hasLogits = true;
    }

    private void DisposePast()
    {
        if (_pastOwner is not null)
        {
            _pastOwner.Dispose();
            _pastOwner = null;
        }
        else
        {
            foreach (OrtValue? value in _past)
            {
                value?.Dispose();
            }
        }
    }
}
