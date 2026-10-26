using System.Globalization;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx;

/// <summary>How the CUDA provider's device memory arena grows (<c>arena_extend_strategy</c>).</summary>
public enum OnnxArenaExtendStrategy
{
    /// <summary>Each extension doubles the previous one (<c>kNextPowerOfTwo</c>, ONNX Runtime's default).</summary>
    NextPowerOfTwo,

    /// <summary>Each extension is the size of the request (<c>kSameAsRequested</c>); less over-allocation, more extensions.</summary>
    SameAsRequested,
}

/// <summary>How the CUDA provider picks a cuDNN convolution algorithm (<c>cudnn_conv_algo_search</c>).</summary>
public enum OnnxCudnnConvolutionAlgorithmSearch
{
    /// <summary>Benchmark every algorithm on first use (<c>EXHAUSTIVE</c>, ONNX Runtime's default).</summary>
    Exhaustive,

    /// <summary>cuDNN's heuristic choice without benchmarking (<c>HEURISTIC</c>).</summary>
    Heuristic,

    /// <summary>cuDNN's default algorithm (<c>DEFAULT</c>).</summary>
    Default,
}

/// <summary>
/// Typed CUDA execution provider options passed to ONNX Runtime when <see cref="OnnxSessionOptions.Provider"/> is
/// <see cref="OnnxExecutionProvider.Cuda"/>. Every member is optional: <see langword="null"/> keeps ONNX Runtime's
/// default. The values are validated by <see cref="OnnxSessionOptions"/> on any build, CPU-only included; whether a
/// CUDA device accepts them is only known when the provider is created. TF32 stays off (<c>use_tf32=0</c>) and is not
/// configurable here. Status: untested on CUDA (no GPU run with these options); see the package README.
/// </summary>
public sealed record OnnxCudaProviderOptions
{
    /// <summary>
    /// Upper bound in bytes of the CUDA provider's device memory arena (<c>gpu_mem_limit</c>). Must be positive when
    /// set. <see langword="null"/> keeps ONNX Runtime's default (no limit). A run that needs more fails; it does not fall
    /// back to the CPU.
    /// </summary>
    public long? GpuMemoryLimitBytes { get; init; }

    /// <summary>Growth policy of the device memory arena (<c>arena_extend_strategy</c>). <see langword="null"/> keeps ONNX Runtime's default.</summary>
    public OnnxArenaExtendStrategy? ArenaExtendStrategy { get; init; }

    /// <summary>
    /// cuDNN convolution algorithm search (<c>cudnn_conv_algo_search</c>). <see langword="null"/> keeps ONNX Runtime's
    /// default (exhaustive). Another algorithm can change convolution results in the last bits; the qualified CUDA
    /// profiles were measured with the default.
    /// </summary>
    public OnnxCudnnConvolutionAlgorithmSearch? CudnnConvolutionAlgorithmSearch { get; init; }

    /// <summary>
    /// Capture the graph as a CUDA graph on the first run and replay it afterwards (<c>enable_cuda_graph</c>).
    /// <see langword="null"/> keeps ONNX Runtime's default (off). ONNX Runtime requires every node on the CUDA provider,
    /// fixed input shapes and inputs and outputs bound at fixed device addresses with
    /// <see cref="OnnxModelSession.CreateIoBinding"/> and <see cref="OnnxModelSession.RunWithBinding"/>; a graph that does not meet
    /// these fails at session creation or at the first run.
    /// </summary>
    public bool? EnableCudaGraph { get; init; }

    internal void Validate()
    {
        if (GpuMemoryLimitBytes is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(GpuMemoryLimitBytes), GpuMemoryLimitBytes, "Must be positive, or null for the ONNX Runtime default.");
        }

        if (ArenaExtendStrategy is { } arena && !Enum.IsDefined(arena))
        {
            throw new ArgumentOutOfRangeException(nameof(ArenaExtendStrategy), arena, "Unknown arena extend strategy.");
        }

        if (CudnnConvolutionAlgorithmSearch is { } search && !Enum.IsDefined(search))
        {
            throw new ArgumentOutOfRangeException(nameof(CudnnConvolutionAlgorithmSearch), search, "Unknown cuDNN convolution algorithm search.");
        }
    }

    /// <summary>
    /// The ONNX Runtime CUDA provider option strings for a device and these options. Always includes <c>device_id</c>
    /// and <c>use_tf32=0</c>.
    /// </summary>
    internal static Dictionary<string, string> ToProviderOptions(int deviceId, OnnxCudaProviderOptions? options)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["device_id"] = deviceId.ToString(CultureInfo.InvariantCulture),

            // fp32 profile: TF32 rounds matmul inputs to a 10-bit mantissa (docs/m0/e5-gpu.md). ORT 1.30 defaults to 1.
            ["use_tf32"] = "0",
        };
        if (options is null)
        {
            return result;
        }

        if (options.GpuMemoryLimitBytes is long limit)
        {
            result["gpu_mem_limit"] = limit.ToString(CultureInfo.InvariantCulture);
        }

        if (options.ArenaExtendStrategy is { } arena)
        {
            result["arena_extend_strategy"] = arena == OnnxArenaExtendStrategy.SameAsRequested ? "kSameAsRequested" : "kNextPowerOfTwo";
        }

        if (options.CudnnConvolutionAlgorithmSearch is { } search)
        {
            result["cudnn_conv_algo_search"] = search switch
            {
                OnnxCudnnConvolutionAlgorithmSearch.Heuristic => "HEURISTIC",
                OnnxCudnnConvolutionAlgorithmSearch.Default => "DEFAULT",
                _ => "EXHAUSTIVE",
            };
        }

        if (options.EnableCudaGraph is bool graph)
        {
            result["enable_cuda_graph"] = graph ? "1" : "0";
        }

        return result;
    }

    /// <summary>
    /// Creates the native CUDA provider options. Creating or updating them already loads
    /// <c>libonnxruntime_providers_cuda</c> and its CUDA dependencies, so a missing library fails here.
    /// </summary>
    /// <exception cref="OnnxProviderUnavailableException">The provider library or its dependencies cannot be loaded, or an option is rejected.</exception>
    internal static OrtCUDAProviderOptions CreateNative(int deviceId, OnnxCudaProviderOptions? options)
    {
        OrtCUDAProviderOptions? native = null;
        try
        {
            native = new OrtCUDAProviderOptions();
            native.UpdateOptions(ToProviderOptions(deviceId, options));
            return native;
        }
        catch (OnnxRuntimeException ex)
        {
            native?.Dispose();
            throw CudaUnavailable(ex);
        }
    }

    internal static OnnxProviderUnavailableException CudaUnavailable(OnnxRuntimeException ex) => new(
        "The CUDA execution provider could not be initialized: " + ex.Message +
        " It needs libonnxruntime_providers_cuda from Microsoft.ML.OnnxRuntime.Gpu.Linux next to libonnxruntime, the CUDA 13 runtime " +
        "(libcudart, libcublas, libcublasLt) and cuDNN 9 on the native loader path (LD_LIBRARY_PATH or the ldconfig cache on Linux), " +
        "a compatible NVIDIA driver, and a visible device. No CPU fallback was attempted.",
        ex);
}
