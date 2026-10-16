namespace Starlings.Runtime.Onnx;

/// <summary>A graph input or output as ONNX Runtime reports it.</summary>
/// <param name="Name">Tensor name.</param>
/// <param name="ElementType">ONNX element type, for example <c>Int64</c> or <c>Float</c>.</param>
/// <param name="Dimensions">Dimensions; -1 for a symbolic or unknown dimension.</param>
/// <param name="SymbolicDimensions">Symbolic dimension names, empty string where the dimension is fixed.</param>
public sealed record OnnxTensorInfo(
    string Name,
    string ElementType,
    IReadOnlyList<long> Dimensions,
    IReadOnlyList<string> SymbolicDimensions);

/// <summary>Model-level metadata stored in the ONNX file.</summary>
public sealed record OnnxModelMetadata(
    string ProducerName,
    string GraphName,
    string Domain,
    string Description,
    long Version,
    IReadOnlyDictionary<string, string> CustomMetadata);

/// <summary>
/// Identity and configuration of one loaded session: enough to name the exact runtime profile
/// (plan section 9) without exposing inputs.
/// </summary>
public sealed record OnnxSessionDiagnostics
{
    /// <summary>Native ONNX Runtime version.</summary>
    public required string OrtVersion { get; init; }

    /// <summary>Managed binding version.</summary>
    public required string ManagedBindingVersion { get; init; }

    /// <summary>Providers compiled into the loaded native library.</summary>
    public required IReadOnlyList<string> AvailableProviders { get; init; }

    /// <summary>Provider requested in <see cref="OnnxSessionOptions.Provider"/>.</summary>
    public required OnnxExecutionProvider RequestedProvider { get; init; }

    /// <summary>
    /// Providers registered on the session in priority order. ONNX Runtime always appends the CPU provider, which
    /// runs nodes the preferred provider cannot or should not run; see <see cref="OnnxModelSession.Placement"/>.
    /// </summary>
    public required IReadOnlyList<string> SessionProviders { get; init; }

    /// <summary>Options of the non-CPU provider as ONNX Runtime reports them after configuration; empty for CPU.</summary>
    public required IReadOnlyDictionary<string, string> ProviderOptions { get; init; }

    /// <summary>
    /// Intra-op threads configured on the session: the requested count, or the container-aware default
    /// (<see cref="OnnxSessionOptions.IntraOpNumThreads"/>); <see langword="null"/> means the ONNX Runtime default.
    /// </summary>
    public int? IntraOpNumThreads { get; init; }

    /// <summary>Configured inter-op threads; <see langword="null"/> means the ONNX Runtime default.</summary>
    public int? InterOpNumThreads { get; init; }

    /// <summary>
    /// Whether releasing this session returns freed native heap memory to the operating system
    /// (<see cref="OnnxSessionOptions.TrimNativeHeapOnRelease"/>, Linux with glibc, process switch not set).
    /// </summary>
    public bool TrimNativeHeapOnRelease { get; init; }

    /// <summary>Whether ONNX Runtime's CPU memory arena is on for this session (<see cref="OnnxSessionOptions.CpuMemoryArena"/>).</summary>
    public bool CpuMemoryArena { get; init; }

    /// <summary>Whether runs execute on the package's dedicated run threads (<see cref="OnnxSessionOptions.DedicatedRunThreads"/>).</summary>
    public bool DedicatedRunThreads { get; init; }

    /// <summary>Graph optimization level applied at session creation.</summary>
    public required string GraphOptimizationLevel { get; init; }

    /// <summary>Model file path, or <c>&lt;memory&gt;</c> for a model loaded from bytes.</summary>
    public required string ModelSource { get; init; }

    /// <summary>Model size in bytes.</summary>
    public required long ModelSizeBytes { get; init; }

    /// <summary>Lowercase hex SHA-256 of the model bytes, or <see langword="null"/> when not computed.</summary>
    public string? ModelSha256 { get; init; }

    /// <summary>Metadata stored in the model.</summary>
    public required OnnxModelMetadata Model { get; init; }

    /// <summary>Graph inputs.</summary>
    public required IReadOnlyList<OnnxTensorInfo> Inputs { get; init; }

    /// <summary>Graph outputs.</summary>
    public required IReadOnlyList<OnnxTensorInfo> Outputs { get; init; }
}

/// <summary>
/// Which execution provider ran which nodes during one profiled run, read from ONNX Runtime's profiler output.
/// Counts are distinct nodes of the optimized graph, so fused operators appear under their fused names.
/// </summary>
/// <param name="NodesByProvider">Distinct node count per provider name.</param>
/// <param name="OperatorsByProvider">Per provider, the distinct node count per operator type.</param>
/// <param name="NodeNamesByProvider">Per provider, the node names (sorted ordinally).</param>
public sealed record OnnxPlacementReport(
    IReadOnlyDictionary<string, int> NodesByProvider,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, int>> OperatorsByProvider,
    IReadOnlyDictionary<string, IReadOnlyList<string>> NodeNamesByProvider);
