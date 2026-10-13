using System.Globalization;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx;

/// <summary>
/// One loaded ONNX model and the native resources behind it (inference session, session options, provider
/// options). Create it once and reuse it: loading is expensive and <see cref="Run(IReadOnlyCollection{string}, IReadOnlyCollection{OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
/// is safe to call concurrently from multiple threads.
/// </summary>
/// <remarks>
/// <para>
/// <b>Ownership.</b> The instance owns its native session. <see cref="Dispose"/> is idempotent; calls made after it
/// throw <see cref="ObjectDisposedException"/>. A run that is already executing when <see cref="Dispose"/> is called
/// completes normally, and the native session is released when the last such run returns, so native memory is never
/// freed under an active run. Output values returned by <see cref="Run(IReadOnlyCollection{string}, IReadOnlyCollection{OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
/// belong to the caller, who must dispose them (before disposing the session).
/// </para>
/// <para>
/// <b>Cancellation.</b> Runs are synchronous; there is no asynchronous API, because ONNX Runtime's C# binding has no
/// truly asynchronous run for these providers and wrapping a blocking call in a task would not make it cancellable.
/// A token that is already cancelled prevents the run. A token cancelled during a run sets ONNX Runtime's terminate
/// flag, which the executor checks between graph nodes: the operator currently executing finishes first (and on CUDA,
/// kernels already queued on the device complete), then the run stops and throws
/// <see cref="OperationCanceledException"/>. Cancellation latency is therefore bounded by the longest single node.
/// The session stays usable after a cancelled run.
/// </para>
/// </remarks>
public sealed class OnnxModelSession : IDisposable
{
    private const string MemorySource = "<memory>";

    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private readonly OrtCUDAProviderOptions? _cudaOptions;
    private readonly object _placementGate = new();
    private volatile bool _placementPending;
    private OnnxPlacementReport? _placement;

    // One reference held by the owner plus one per in-flight run. The native session is released at zero.
    private int _references = 1;
    private int _disposed;

    private OnnxModelSession(
        InferenceSession session,
        SessionOptions sessionOptions,
        OrtCUDAProviderOptions? cudaOptions,
        OnnxSessionDiagnostics diagnostics,
        bool capturePlacement)
    {
        _session = session;
        _sessionOptions = sessionOptions;
        _cudaOptions = cudaOptions;
        Diagnostics = diagnostics;
        _placementPending = capturePlacement;
    }

    /// <summary>Identity, configuration and graph signature of this session.</summary>
    public OnnxSessionDiagnostics Diagnostics { get; }

    /// <summary>Graph inputs.</summary>
    public IReadOnlyList<OnnxTensorInfo> Inputs => Diagnostics.Inputs;

    /// <summary>Graph outputs.</summary>
    public IReadOnlyList<OnnxTensorInfo> Outputs => Diagnostics.Outputs;

    /// <summary>
    /// Node placement observed on the first completed run when <see cref="OnnxSessionOptions.CapturePlacement"/>
    /// is set; otherwise, or before that run, <see langword="null"/>.
    /// </summary>
    public OnnxPlacementReport? Placement => Volatile.Read(ref _placement);

    /// <summary><see langword="true"/> after <see cref="Dispose"/>.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Loads a model file.</summary>
    /// <exception cref="OnnxRuntimeUnavailableException">The native runtime is missing or mismatched.</exception>
    /// <exception cref="OnnxProviderUnavailableException">The requested provider cannot be used.</exception>
    /// <exception cref="OnnxModelIntegrityException">The file does not match <see cref="OnnxSessionOptions.ExpectedModelSha256"/>.</exception>
    public static OnnxModelSession Create(string modelPath, OnnxSessionOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        options ??= new OnnxSessionOptions();
        options.Validate();
        var file = new FileInfo(modelPath);
        if (!file.Exists)
        {
            throw new FileNotFoundException("Model file not found.", modelPath);
        }

        string? sha = null;
        if (options.ComputeModelSha256 || options.ExpectedModelSha256 is not null)
        {
            using var stream = file.OpenRead();
            sha = Convert.ToHexStringLower(SHA256.HashData(stream));
        }

        return CreateCore(options, file.FullName, file.Length, sha, so => new InferenceSession(file.FullName, so));
    }

    /// <summary>Loads a model from memory. The bytes are copied by ONNX Runtime and not retained.</summary>
    /// <exception cref="OnnxRuntimeUnavailableException">The native runtime is missing or mismatched.</exception>
    /// <exception cref="OnnxProviderUnavailableException">The requested provider cannot be used.</exception>
    /// <exception cref="OnnxModelIntegrityException">The bytes do not match <see cref="OnnxSessionOptions.ExpectedModelSha256"/>.</exception>
    public static OnnxModelSession Create(byte[] model, OnnxSessionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        options ??= new OnnxSessionOptions();
        options.Validate();
        string? sha = options.ComputeModelSha256 || options.ExpectedModelSha256 is not null
            ? Convert.ToHexStringLower(SHA256.HashData(model))
            : null;
        return CreateCore(options, MemorySource, model.LongLength, sha, so => new InferenceSession(model, so));
    }

    /// <summary>
    /// Runs the model and returns outputs allocated by ONNX Runtime. The caller owns and must dispose the result.
    /// Input values stay owned by the caller and must remain alive for the duration of the call.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public IDisposableReadOnlyCollection<OrtValue> Run(
        IReadOnlyCollection<string> inputNames,
        IReadOnlyCollection<OrtValue> inputValues,
        IReadOnlyCollection<string> outputNames,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        ArgumentNullException.ThrowIfNull(outputNames);
        IDisposableReadOnlyCollection<OrtValue>? result = null;
        Execute(runOptions => result = _session.Run(runOptions, inputNames, inputValues, outputNames), cancellationToken);
        return result!;
    }

    /// <summary>
    /// Runs the model writing into caller-allocated output values (for example tensors over pooled buffers),
    /// avoiding per-call output allocation. Shapes must match what the graph produces for these inputs.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public void Run(
        IReadOnlyCollection<string> inputNames,
        IReadOnlyCollection<OrtValue> inputValues,
        IReadOnlyCollection<string> outputNames,
        IReadOnlyCollection<OrtValue> outputValues,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(outputValues);
        Execute(runOptions => _session.Run(runOptions, inputNames, inputValues, outputNames, outputValues), cancellationToken);
    }

    /// <summary>
    /// Releases the native session once no run is executing. Idempotent. Runs already executing complete first.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Release();
        }
    }

    private static OnnxModelSession CreateCore(
        OnnxSessionOptions options,
        string source,
        long size,
        string? sha,
        Func<SessionOptions, InferenceSession> load)
    {
        if (options.ExpectedModelSha256 is { } expected && !string.Equals(expected, sha, StringComparison.Ordinal))
        {
            throw new OnnxModelIntegrityException($"Model {source} has SHA-256 {sha}; expected {expected}.");
        }

        var env = OnnxRuntimeInfo.Environment;
        var available = env.GetAvailableProviders();
        bool cuda = options.Provider == OnnxExecutionProvider.Cuda;
        if (cuda && !available.Contains(OnnxRuntimeInfo.CudaProviderName, StringComparer.Ordinal))
        {
            throw new OnnxProviderUnavailableException(
                $"The loaded ONNX Runtime {env.GetVersionString()} native library has no CUDA execution provider (available: {string.Join(", ", available)}). " +
                $"Reference Microsoft.ML.OnnxRuntime.Gpu.Linux {OnnxRuntimeInfo.ManagedVersion} instead of Microsoft.ML.OnnxRuntime.");
        }

        var sessionOptions = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        OrtCUDAProviderOptions? cudaOptions = null;
        InferenceSession? session = null;
        try
        {
            if (options.IntraOpNumThreads is int intra)
            {
                sessionOptions.IntraOpNumThreads = intra;
            }

            if (options.InterOpNumThreads is int inter)
            {
                sessionOptions.InterOpNumThreads = inter;
            }

            if (options.CapturePlacement)
            {
                sessionOptions.EnableProfiling = true;
                sessionOptions.ProfileOutputPathPrefix = Path.Combine(Path.GetTempPath(), "starlings-ort-placement-" + Guid.NewGuid().ToString("N"));
            }

            var providerOptions = new Dictionary<string, string>(StringComparer.Ordinal);
            if (cuda)
            {
                // Creating or updating the provider options already loads libonnxruntime_providers_cuda and its CUDA
                // dependencies, so failures can surface at any of these three calls.
                try
                {
                    cudaOptions = new OrtCUDAProviderOptions();
                    cudaOptions.UpdateOptions(new Dictionary<string, string>
                    {
                        ["device_id"] = options.CudaDeviceId.ToString(CultureInfo.InvariantCulture),

                        // fp32 profile: TF32 rounds matmul inputs to a 10-bit mantissa (docs/m0/e5-gpu.md). ORT 1.30 defaults to 1.
                        ["use_tf32"] = "0",
                    });
                    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                }
                catch (OnnxRuntimeException ex)
                {
                    throw CudaUnavailable(ex);
                }

                providerOptions = ParseProviderOptions(cudaOptions.GetOptions());
                if (!providerOptions.TryGetValue("use_tf32", out var tf32) || tf32 != "0")
                {
                    throw new InvalidOperationException($"CUDA provider reports use_tf32={tf32 ?? "<missing>"} after configuration; the fp32 profile requires 0.");
                }
            }

            try
            {
                session = load(sessionOptions);
            }
            catch (OnnxRuntimeException ex) when (cuda)
            {
                throw CudaUnavailable(ex);
            }

            var diagnostics = new OnnxSessionDiagnostics
            {
                OrtVersion = env.GetVersionString(),
                ManagedBindingVersion = OnnxRuntimeInfo.ManagedVersion,
                AvailableProviders = available,
                RequestedProvider = options.Provider,
                SessionProviders = cuda
                    ? [OnnxRuntimeInfo.CudaProviderName, OnnxRuntimeInfo.CpuProviderName]
                    : [OnnxRuntimeInfo.CpuProviderName],
                ProviderOptions = providerOptions,
                IntraOpNumThreads = options.IntraOpNumThreads,
                InterOpNumThreads = options.InterOpNumThreads,
                GraphOptimizationLevel = sessionOptions.GraphOptimizationLevel.ToString(),
                ModelSource = source,
                ModelSizeBytes = size,
                ModelSha256 = sha,
                Model = ReadModelMetadata(session),
                Inputs = ReadTensorInfo(session.InputMetadata),
                Outputs = ReadTensorInfo(session.OutputMetadata),
            };
            return new OnnxModelSession(session, sessionOptions, cudaOptions, diagnostics, options.CapturePlacement);
        }
        catch
        {
            session?.Dispose();
            cudaOptions?.Dispose();
            sessionOptions.Dispose();
            throw;
        }
    }

    private static OnnxProviderUnavailableException CudaUnavailable(OnnxRuntimeException ex) => new(
        "The CUDA execution provider could not be initialized: " + ex.Message +
        " It needs libonnxruntime_providers_cuda from Microsoft.ML.OnnxRuntime.Gpu.Linux next to libonnxruntime, the CUDA 13 runtime " +
        "(libcudart, libcublas, libcublasLt) and cuDNN 9 on the native loader path (LD_LIBRARY_PATH or the ldconfig cache on Linux), " +
        "a compatible NVIDIA driver, and a visible device. No CPU fallback was attempted.",
        ex);

    private static Dictionary<string, string> ParseProviderOptions(string options)
    {
        // OrtCUDAProviderOptions.GetOptions returns "key=value;key=value".
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var pair in options.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = pair.IndexOf('=', StringComparison.Ordinal);
            if (eq > 0)
            {
                result[pair[..eq]] = pair[(eq + 1)..];
            }
        }

        return result;
    }

    private static OnnxModelMetadata ReadModelMetadata(InferenceSession session)
    {
        var m = session.ModelMetadata;
        return new OnnxModelMetadata(
            m.ProducerName,
            m.GraphName,
            m.Domain,
            m.Description,
            m.Version,
            new Dictionary<string, string>(m.CustomMetadataMap, StringComparer.Ordinal));
    }

    private static List<OnnxTensorInfo> ReadTensorInfo(IReadOnlyDictionary<string, NodeMetadata> metadata) =>
        metadata.Select(kv => new OnnxTensorInfo(
            kv.Key,
            kv.Value.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR
                ? kv.Value.ElementDataType.ToString()
                : kv.Value.OnnxValueType.ToString(),
            kv.Value.Dimensions.Select(d => (long)d).ToArray(),
            kv.Value.SymbolicDimensions.ToArray()))
        .ToList();

    private void Execute(Action<RunOptions> run, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Acquire();
        try
        {
            if (_placementPending)
            {
                lock (_placementGate)
                {
                    if (_placementPending)
                    {
                        RunOnce(run, cancellationToken);
                        CapturePlacement();
                        return;
                    }
                }
            }

            RunOnce(run, cancellationToken);
        }
        finally
        {
            Release();
        }
    }

    private static void RunOnce(Action<RunOptions> run, CancellationToken cancellationToken)
    {
        // RunOptions is per call: its terminate flag must not leak into concurrent runs.
        using var runOptions = new RunOptions();
        using var registration = cancellationToken.CanBeCanceled
            ? cancellationToken.UnsafeRegister(static state => ((RunOptions)state!).Terminate = true, runOptions)
            : default;
        try
        {
            run(runOptions);
        }
        catch (OnnxRuntimeException ex) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException("The ONNX Runtime run was cancelled at a node boundary.", ex, cancellationToken);
        }
    }

    private void CapturePlacement()
    {
        string path = _session.EndProfiling();
        _placementPending = false;
        try
        {
            using var stream = File.OpenRead(path);
            Volatile.Write(ref _placement, PlacementProfileParser.Parse(stream));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private void Acquire()
    {
        while (true)
        {
            int current = Volatile.Read(ref _references);
            ObjectDisposedException.ThrowIf(current == 0 || IsDisposed, this);
            if (Interlocked.CompareExchange(ref _references, current + 1, current) == current)
            {
                break;
            }
        }

        if (IsDisposed)
        {
            Release();
            ObjectDisposedException.ThrowIf(true, this);
        }
    }

    private void Release()
    {
        if (Interlocked.Decrement(ref _references) == 0)
        {
            _session.Dispose();
            _cudaOptions?.Dispose();
            _sessionOptions.Dispose();
        }
    }
}
