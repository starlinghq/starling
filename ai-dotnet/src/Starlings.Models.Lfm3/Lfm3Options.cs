using Starlings.Runtime.Onnx;

namespace Starlings.Models.Lfm3;

/// <summary>Load options for <see cref="Lfm3Model"/>.</summary>
public sealed record Lfm3Options
{
    /// <summary>
    /// Largest prompt plus completion length accepted (at least 1, at most <see cref="Lfm3Checkpoint.ContextLength"/>).
    /// Default 32,768: the attention caches of the 11 attention layers grow by 44 KB per token in fp32 (1.4 GB at the
    /// default), and the checkpoint's 262,144 positions are untested.
    /// </summary>
    public int ContextLength { get; init; } = 32768;

    /// <summary>ONNX Runtime session options (provider, threads, model digest).</summary>
    public OnnxSessionOptions Session { get; init; } = new();

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(ContextLength, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(ContextLength, Lfm3Checkpoint.ContextLength);
        ArgumentNullException.ThrowIfNull(Session);
    }
}
