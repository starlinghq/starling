namespace Starlings.Runtime.Onnx;

/// <summary>Execution provider requested for a session.</summary>
public enum OnnxExecutionProvider
{
    /// <summary>ONNX Runtime's CPU execution provider. Available in every ONNX Runtime native package.</summary>
    Cpu,

    /// <summary>
    /// NVIDIA CUDA execution provider, fp32 with TF32 disabled (<c>use_tf32=0</c>). Requires the
    /// <c>Microsoft.ML.OnnxRuntime.Gpu.Linux</c> native package plus the CUDA 13 runtime and cuDNN 9 on the
    /// loader path. Session creation fails when any of these is missing; there is no silent CPU fallback.
    /// ONNX Runtime still assigns some shape-path nodes to the CPU provider by design; enable
    /// <see cref="OnnxSessionOptions.CapturePlacement"/> to see the actual node placement.
    /// </summary>
    Cuda,
}

/// <summary>
/// Immutable session configuration. Every setting that changes numerical results or native resource use is
/// explicit here; settings ONNX Runtime would accept but this profile does not support are not exposed.
/// </summary>
public sealed record OnnxSessionOptions
{
    /// <summary>Execution provider. Default <see cref="OnnxExecutionProvider.Cpu"/>.</summary>
    public OnnxExecutionProvider Provider { get; init; } = OnnxExecutionProvider.Cpu;

    /// <summary>
    /// Threads used inside one operator (<c>intra_op_num_threads</c>). <see langword="null"/> keeps ONNX
    /// Runtime's default (one per physical core). Must be positive when set.
    /// </summary>
    public int? IntraOpNumThreads { get; init; }

    /// <summary>
    /// Threads used across independent operators (<c>inter_op_num_threads</c>). Only relevant with parallel
    /// execution, which this component does not enable (execution mode is sequential); exposed so the value is
    /// recorded. <see langword="null"/> keeps ONNX Runtime's default. Must be positive when set.
    /// </summary>
    public int? InterOpNumThreads { get; init; }

    /// <summary>
    /// CUDA device ordinal as the CUDA runtime enumerates it (after <c>CUDA_VISIBLE_DEVICES</c> and
    /// <c>CUDA_DEVICE_ORDER</c>). Ignored for CPU. Default 0.
    /// </summary>
    public int CudaDeviceId { get; init; }

    /// <summary>
    /// Records which execution provider ran each node. Enables ONNX Runtime profiling for the first
    /// <see cref="OnnxModelSession.Run(IReadOnlyCollection{string}, IReadOnlyCollection{Microsoft.ML.OnnxRuntime.OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
    /// only, then stops it and publishes <see cref="OnnxModelSession.Placement"/>. The first run is serialized
    /// with other runs while profiling is active. Default <see langword="false"/>.
    /// </summary>
    public bool CapturePlacement { get; init; }

    /// <summary>
    /// Expected lowercase hex SHA-256 of the model bytes. When set, the model is hashed before loading and a
    /// mismatch throws <see cref="OnnxModelIntegrityException"/>.
    /// </summary>
    public string? ExpectedModelSha256 { get; init; }

    /// <summary>
    /// Hash the model bytes and report the digest in <see cref="OnnxSessionDiagnostics.ModelSha256"/>.
    /// Default <see langword="true"/>. Costs one sequential read of the model file at load
    /// (about 1 s for a 470 MB file). Always on when <see cref="ExpectedModelSha256"/> is set.
    /// </summary>
    public bool ComputeModelSha256 { get; init; } = true;

    internal void Validate()
    {
        if (!Enum.IsDefined(Provider))
        {
            throw new ArgumentOutOfRangeException(nameof(Provider), Provider, "Unknown execution provider.");
        }

        if (IntraOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(IntraOpNumThreads), IntraOpNumThreads, "Must be positive, or null for the ONNX Runtime default.");
        }

        if (InterOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(InterOpNumThreads), InterOpNumThreads, "Must be positive, or null for the ONNX Runtime default.");
        }

        if (CudaDeviceId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(CudaDeviceId), CudaDeviceId, "Must not be negative.");
        }

        if (ExpectedModelSha256 is { } sha && (sha.Length != 64 || !sha.All(char.IsAsciiHexDigitLower)))
        {
            throw new ArgumentException("Expected a 64-character lowercase hex SHA-256.", nameof(ExpectedModelSha256));
        }
    }
}
