using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.JavaScript;
using System.Runtime.Versioning;
using System.Text.Json;

namespace Starlings.Runtime.OnnxWeb;

/// <summary>onnxruntime-web graph optimization level (<c>graphOptimizationLevel</c>).</summary>
public enum OnnxWebGraphOptimizationLevel
{
    /// <summary>No graph optimizations.</summary>
    Disabled,

    /// <summary>Basic optimizations.</summary>
    Basic,

    /// <summary>Extended optimizations.</summary>
    Extended,

    /// <summary>All optimizations (default; what the browser evidence used).</summary>
    All,
}

/// <summary>Options of <see cref="OnnxWebSession.CreateAsync"/>.</summary>
public sealed record OnnxWebSessionOptions
{
    /// <summary>Execution provider. Default <see cref="OnnxWebExecutionProvider.Wasm"/>. Never falls back to another one.</summary>
    public OnnxWebExecutionProvider Provider { get; init; } = OnnxWebExecutionProvider.Wasm;

    /// <summary>Graph optimization level. Default <see cref="OnnxWebGraphOptimizationLevel.All"/>.</summary>
    public OnnxWebGraphOptimizationLevel GraphOptimizationLevel { get; init; } = OnnxWebGraphOptimizationLevel.All;

    internal void Validate()
    {
        if (!Enum.IsDefined(Provider))
        {
            throw new ArgumentOutOfRangeException(nameof(Provider), Provider, "Unknown execution provider.");
        }

        if (!Enum.IsDefined(GraphOptimizationLevel))
        {
            throw new ArgumentOutOfRangeException(nameof(GraphOptimizationLevel), GraphOptimizationLevel, "Unknown optimization level.");
        }
    }

    internal string ProviderName => Provider == OnnxWebExecutionProvider.WebGpu ? "webgpu" : "wasm";

    internal string OptimizationName => GraphOptimizationLevel switch
    {
        OnnxWebGraphOptimizationLevel.Disabled => "disabled",
        OnnxWebGraphOptimizationLevel.Basic => "basic",
        OnnxWebGraphOptimizationLevel.Extended => "extended",
        _ => "all",
    };
}

/// <summary>A graph input or output as onnxruntime-web reports it.</summary>
/// <param name="Name">Name in the graph.</param>
/// <param name="ElementType">onnxruntime-web element type (<c>int64</c>, <c>float32</c>, ...), or <see langword="null"/> for a non-tensor value.</param>
/// <param name="Dimensions">Shape, -1 for a symbolic dimension, or <see langword="null"/> when not reported.</param>
public sealed record OnnxWebTensorInfo(string Name, string? ElementType, IReadOnlyList<long>? Dimensions)
{
    internal static (List<OnnxWebTensorInfo> Inputs, List<OnnxWebTensorInfo> Outputs) ParseMetadata(string json)
    {
        using var document = JsonDocument.Parse(json);
        return (Read(document.RootElement.GetProperty("inputs")), Read(document.RootElement.GetProperty("outputs")));

        static List<OnnxWebTensorInfo> Read(JsonElement array)
        {
            var result = new List<OnnxWebTensorInfo>(array.GetArrayLength());
            foreach (var item in array.EnumerateArray())
            {
                var type = item.GetProperty("type");
                var shape = item.GetProperty("shape");
                result.Add(new OnnxWebTensorInfo(
                    item.GetProperty("name").GetString() ?? string.Empty,
                    type.ValueKind == JsonValueKind.String ? type.GetString() : null,
                    shape.ValueKind == JsonValueKind.Array ? [.. shape.EnumerateArray().Select(d => d.GetInt64())] : null));
            }

            return result;
        }
    }
}

/// <summary>One named input tensor of <see cref="OnnxWebSession.RunAsync"/>, int64 or float32, row-major.</summary>
public readonly struct OnnxWebInput
{
    private readonly ReadOnlyMemory<long> _int64;
    private readonly ReadOnlyMemory<float> _float32;

    private OnnxWebInput(string name, string elementType, ReadOnlyMemory<long> int64, ReadOnlyMemory<float> float32, int[] dimensions)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(dimensions);
        long count = 1;
        foreach (int d in dimensions)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(d, nameof(dimensions));
            count *= d;
        }

        int length = elementType == "int64" ? int64.Length : float32.Length;
        if (count != length)
        {
            throw new ArgumentException($"Input '{name}' has {length} values but shape [{string.Join(", ", dimensions)}] needs {count}.", nameof(dimensions));
        }

        Name = name;
        ElementType = elementType;
        _int64 = int64;
        _float32 = float32;
        Dimensions = dimensions;
    }

    /// <summary>Graph input name.</summary>
    public string Name { get; }

    /// <summary><c>int64</c> or <c>float32</c>.</summary>
    public string ElementType { get; }

    /// <summary>Shape.</summary>
    public IReadOnlyList<int> Dimensions { get; }

    /// <summary>An int64 tensor. The memory is copied when the run starts.</summary>
    public static OnnxWebInput FromInt64(string name, ReadOnlyMemory<long> data, params int[] dimensions) => new(name, "int64", data, default, dimensions);

    /// <summary>A float32 tensor. The memory is copied when the run starts.</summary>
    public static OnnxWebInput FromFloat32(string name, ReadOnlyMemory<float> data, params int[] dimensions) => new(name, "float32", default, data, dimensions);

    internal ReadOnlySpan<byte> Bytes => ElementType == "int64" ? MemoryMarshal.AsBytes(_int64.Span) : MemoryMarshal.AsBytes(_float32.Span);
}

/// <summary>
/// An onnxruntime-web inference session created from an <see cref="OnnxWebAsset"/>. Runs are asynchronous and
/// serialized per session (onnxruntime-web does not run one session re-entrantly); concurrent callers wait in order.
/// </summary>
/// <remarks>
/// <para>
/// <b>Cancellation.</b> onnxruntime-web cannot interrupt a run. The token is checked while waiting for the session and
/// before the run starts; a run already started completes, then the call throws <see cref="OperationCanceledException"/>
/// without writing the destination.
/// </para>
/// <para>
/// <b>Lifetime.</b> <see cref="DisposeAsync"/> waits for a run in progress and releases the session. <see cref="Dispose"/>
/// does the same without waiting (the browser's main thread cannot block). Later calls throw
/// <see cref="ObjectDisposedException"/>.
/// </para>
/// </remarks>
[SupportedOSPlatform("browser")]
public sealed class OnnxWebSession : IAsyncDisposable, IDisposable
{
    private readonly JSObject _session;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private int _disposed;
    private Task? _release;

    private OnnxWebSession(JSObject session, OnnxWebExecutionProvider provider, string? modelSha256, long modelBytes, TimeSpan createTime)
    {
        _session = session;
        Provider = provider;
        ModelSha256 = modelSha256;
        ModelBytes = modelBytes;
        CreateTime = createTime;
        (var inputs, var outputs) = OnnxWebTensorInfo.ParseMetadata(OnnxWebInterop.SessionMetadata(session));
        Inputs = inputs;
        Outputs = outputs;
    }

    /// <summary>The execution provider the session runs on.</summary>
    public OnnxWebExecutionProvider Provider { get; }

    /// <summary>SHA-256 of the model bytes the session was created from.</summary>
    public string? ModelSha256 { get; }

    /// <summary>Size of the model in bytes.</summary>
    public long ModelBytes { get; }

    /// <summary>Time onnxruntime-web took to create the session (after the fetch).</summary>
    public TimeSpan CreateTime { get; }

    /// <summary>Graph inputs.</summary>
    public IReadOnlyList<OnnxWebTensorInfo> Inputs { get; }

    /// <summary>Graph outputs.</summary>
    public IReadOnlyList<OnnxWebTensorInfo> Outputs { get; }

    /// <summary>
    /// Creates a session from <paramref name="model"/>, which it consumes. <see cref="OnnxWebRuntime.InitializeAsync"/>
    /// must have completed.
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// WebGPU was requested but the runtime's bundle has no WebGPU provider, the browser does not expose WebGPU, or the
    /// proxy worker is on.
    /// </exception>
    /// <exception cref="OnnxWebException">onnxruntime-web could not create the session (an unsupported operator, for example).</exception>
    public static async Task<OnnxWebSession> CreateAsync(OnnxWebAsset model, OnnxWebSessionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        options ??= new OnnxWebSessionOptions();
        options.Validate();
        OnnxWebRuntime.EnsureProviderAvailable(options.Provider);

        cancellationToken.ThrowIfCancellationRequested();
        var handle = model.Handle;
        var stopwatch = Stopwatch.StartNew();
        JSObject session;
        try
        {
            session = await OnnxWebInterop.CreateSession(handle, options.ProviderName, options.OptimizationName).ConfigureAwait(false);
        }
        catch (JSException ex)
        {
            throw new OnnxWebException($"onnxruntime-web could not create a {options.ProviderName} session from {model.Url}: {ex.Message}", ex);
        }

        stopwatch.Stop();
        return new OnnxWebSession(session, options.Provider, model.Sha256, model.ByteLength, stopwatch.Elapsed);
    }

    /// <summary>
    /// Runs the graph and copies the float32 output <paramref name="outputName"/> into <paramref name="destination"/>,
    /// which must hold exactly its element count. Returns the output shape.
    /// </summary>
    /// <exception cref="ArgumentException">The output is unknown, not float32, or its size differs from the destination.</exception>
    /// <exception cref="OnnxWebException">onnxruntime-web failed the run.</exception>
    public async Task<int[]> RunAsync(IReadOnlyList<OnnxWebInput> inputs, string outputName, Memory<float> destination, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentException.ThrowIfNullOrEmpty(outputName);
        ThrowIfDisposed();
        if (!Outputs.Any(o => o.Name == outputName))
        {
            throw new ArgumentException($"The graph has no output '{outputName}'.", nameof(outputName));
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        var tensors = new JSObject[inputs.Count];
        var names = new string[inputs.Count];
        JSObject? output = null;
        try
        {
            ThrowIfDisposed();
            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                names[i] = input.Name;
                tensors[i] = CreateTensor(input);
            }

            try
            {
                output = await OnnxWebInterop.Run(_session, names, tensors, outputName).ConfigureAwait(false);
            }
            catch (JSException ex)
            {
                throw new OnnxWebException($"onnxruntime-web run failed: {ex.Message}", ex);
            }

            cancellationToken.ThrowIfCancellationRequested();
            string type = OnnxWebInterop.TensorType(output);
            if (type != "float32")
            {
                throw new ArgumentException($"Output '{outputName}' is {type}; RunAsync copies float32 outputs.", nameof(outputName));
            }

            int[] dims = OnnxWebInterop.TensorDims(output);
            long count = 1;
            foreach (int d in dims)
            {
                count *= d;
            }

            if (count != destination.Length)
            {
                throw new ArgumentException($"Output '{outputName}' has shape [{string.Join(", ", dims)}] ({count} values); the destination holds {destination.Length}.", nameof(destination));
            }

            OnnxWebInterop.CopyTensor(output, MemoryMarshal.AsBytes(destination.Span));
            return dims;
        }
        finally
        {
            if (output is not null)
            {
                OnnxWebInterop.DisposeTensor(output);
                output.Dispose();
            }

            foreach (var tensor in tensors)
            {
                if (tensor is not null)
                {
                    OnnxWebInterop.DisposeTensor(tensor);
                    tensor.Dispose();
                }
            }

            _gate.Release();
        }
    }

    /// <summary>Waits for a run in progress, then releases the session. Idempotent.</summary>
    public async ValueTask DisposeAsync()
    {
        Dispose();
        await (_release ?? Task.CompletedTask).ConfigureAwait(false);
    }

    /// <summary>Releases the session once a run in progress has finished, without waiting. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _release = ReleaseAsync();
        }
    }

    private static JSObject CreateTensor(OnnxWebInput input)
    {
        // The memory view is valid for this synchronous call only; createTensor copies it.
        ReadOnlySpan<byte> bytes = input.Bytes;
        var span = MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(bytes), bytes.Length);
        return OnnxWebInterop.CreateTensor(input.ElementType, span, [.. input.Dimensions]);
    }

    private async Task ReleaseAsync()
    {
        await _gate.WaitAsync().ConfigureAwait(false);
        try
        {
            await OnnxWebInterop.ReleaseSession(_session).ConfigureAwait(false);
        }
        finally
        {
            _session.Dispose();
            _gate.Release();
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
}

/// <summary>onnxruntime-web reported an error (session creation or run). The inner exception carries the JavaScript error.</summary>
public sealed class OnnxWebException : Exception
{
    /// <summary>Creates the exception.</summary>
    public OnnxWebException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public OnnxWebException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the JavaScript error.</summary>
    public OnnxWebException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
