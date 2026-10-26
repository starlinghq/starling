using System.Collections.Immutable;
using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Kev;

/// <summary>
/// A state that has run through the backbone once (<see cref="KevDecisionModel.CreateState"/>), with its cache
/// (attention keys and values, DeltaNet convolution and recurrent states) kept in host memory, so later question sets
/// on the same document skip the state pass
/// (<see cref="KevDecisionModel.Decide(KevState, IEnumerable{KeyValuePair{string, KevQuestion}}, CancellationToken)"/>).
/// </summary>
/// <remarks>
/// <para>
/// The cache is the same tensors <see cref="KevDecisionModel.Decide(KevRequest, CancellationToken)"/> computes and
/// reuses for every question of one request; no graph run modifies it, so answers from a state equal answers from a
/// request with the same state text and questions, and several calls may use one state at the same time.
/// </para>
/// <para>
/// <b>Memory.</b> The cache grows with the state. From the graph's cache shapes (fp32): about 24 KB of attention
/// cache per state token plus 18 MB of DeltaNet state for <see cref="KevModel.Kev08B"/>, about 64 KB per token plus
/// 48 MB for <see cref="KevModel.Kev4B"/> (plus the small convolution states). It is released by <see cref="Dispose"/>,
/// or when the last call using it returns if one is running. Dispose every state before its model.
/// </para>
/// </remarks>
public sealed class KevState : IDisposable
{
    private readonly IDisposableReadOnlyCollection<OrtValue> _cache;
    private readonly OrtValue[] _past;

    // One reference held by the owner plus one per call using the cache. The cache is released at zero.
    private int _references = 1;
    private int _disposed;

    internal KevState(KevDecisionModel model, ImmutableArray<int> stateIds, IDisposableReadOnlyCollection<OrtValue> cache)
    {
        Model = model;
        StateIds = stateIds;
        _cache = cache;
        _past = [.. cache];
    }

    /// <summary><c>&lt;state&gt;</c> followed by the state's tokens.</summary>
    public ImmutableArray<int> StateIds { get; }

    /// <summary>Tokens of the state, <c>&lt;state&gt;</c> included.</summary>
    public int TokenCount => StateIds.Length;

    /// <summary><see langword="true"/> after <see cref="Dispose"/>.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    internal KevDecisionModel Model { get; }

    internal OrtValue[] Past => _past;

    /// <summary>Releases the cache once no call is using it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Release();
        }
    }

    /// <summary>Takes a reference for one call; throws when the state is disposed.</summary>
    internal void Acquire()
    {
        while (true)
        {
            int current = Volatile.Read(ref _references);
            ObjectDisposedException.ThrowIf(current == 0 || IsDisposed, this);
            if (Interlocked.CompareExchange(ref _references, current + 1, current) == current)
            {
                return;
            }
        }
    }

    internal void Release()
    {
        if (Interlocked.Decrement(ref _references) == 0)
        {
            _cache.Dispose();
        }
    }
}

/// <summary>Where a <see cref="KevDecisionModel"/> runs (<see cref="KevDecisionModel.Placement"/>).</summary>
/// <param name="Provider">The execution provider of the session.</param>
/// <param name="CudaDeviceId">The CUDA device ordinal on <see cref="OnnxExecutionProvider.Cuda"/>; null on the CPU.</param>
/// <param name="ProfileId">The registry profile this model runs (<see cref="KevDecisionModel.ProfileId"/>).</param>
/// <param name="SerializesNativeCalls">Whether requests run their graph calls one at a time (<see cref="KevDecisionModel.SerializesNativeCalls"/>).</param>
/// <param name="Nodes">
/// Node placement of the first run when <see cref="OnnxSessionOptions.CapturePlacement"/> is set
/// (<see cref="OnnxModelSession.Placement"/>); otherwise, or before that run, null.
/// </param>
public sealed record KevPlacement(
    OnnxExecutionProvider Provider, int? CudaDeviceId, string ProfileId, bool SerializesNativeCalls, OnnxPlacementReport? Nodes)
{
    /// <summary>
    /// True: the state cache always lives in host memory between graph runs. The qualified bundles are the builder's
    /// CPU export; inside a CUDA session ONNX Runtime keeps their <c>GroupQueryAttention</c> nodes on the CPU provider and
    /// the cache passes through host memory per row (package README, "CUDA").
    /// </summary>
    public bool StateCacheInHostMemory { get; init; } = true;
}
