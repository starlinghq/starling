using System.Globalization;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>Execution provider for an ONNX Runtime GenAI model.</summary>
public enum GenAIExecutionProvider
{
    /// <summary>The CPU execution provider (any native package).</summary>
    Cpu,

    /// <summary>The CUDA execution provider (needs <c>Microsoft.ML.OnnxRuntimeGenAI.Cuda</c> and the CUDA 13 runtime).</summary>
    Cuda,
}

/// <summary>
/// Immutable load-time settings of a <see cref="GenAIModel"/>. Request settings are separate
/// (<see cref="GenerationOptions"/>).
/// </summary>
public sealed record GenAIModelOptions
{
    /// <summary>Execution provider. Default CPU. A provider that cannot be used fails the load; there is no fallback.</summary>
    public GenAIExecutionProvider Provider { get; init; } = GenAIExecutionProvider.Cpu;

    /// <summary>CUDA device ordinal among the visible devices (<c>CUDA_VISIBLE_DEVICES</c>). Default 0.</summary>
    public int CudaDeviceId { get; init; }

    /// <summary>
    /// TF32 matrix math on the CUDA provider. Default false: ONNX Runtime enables TF32 by default, and with it the
    /// fp32 pilot LLM diverged from the reference at a large logit margin (docs/m0/generation-backend-check.md).
    /// </summary>
    public bool UseTf32 { get; init; }

    /// <summary>
    /// Intra-op thread count of the decoder session. Null selects the default: when a CPU quota or affinity mask
    /// limits the process (<see cref="Environment.ProcessorCount"/> below the host's online CPUs, as in most
    /// containers), that processor count; otherwise the GenAI default (one per physical core of the host).
    /// </summary>
    public int? IntraOpThreads { get; init; }

    /// <summary>
    /// Overrides <c>model.pad_token_id</c> of <c>genai_config.json</c> (an overlay; the file is not edited), or null to
    /// keep it. ONNX Runtime GenAI masks prompt positions whose id equals the pad id, so a model whose pad id is a
    /// token users can type (Qwen3.5: <c>&lt;|endoftext|&gt;</c>) needs a pad id the tokenizer never produces.
    /// </summary>
    public int? PadTokenId { get; init; }

    /// <summary>
    /// Advanced: further JSON overlays applied to <c>genai_config.json</c> in order, after the settings above. Every
    /// overlay is recorded in <see cref="GenAIModel.AppliedOverlays"/>.
    /// </summary>
    public IReadOnlyList<string> ConfigOverlays { get; init; } = [];

    /// <summary>
    /// Run the native work of all sequences of the model one at a time (requests still interleave step by step), or
    /// null (default) for the provider default: serialized on CUDA, concurrent on CPU. With ONNX Runtime GenAI 0.17.1
    /// on the CUDA provider, two or more generators stepping concurrently on one model produced wrong tokens (divergences
    /// from the reference at logit margins up to 6.2, in 15 of 48 requests on two threads); serialized, none did.
    /// Concurrent CPU generators matched the reference.
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }

    internal bool EffectiveSerializeNativeCalls => SerializeNativeCalls ?? Provider == GenAIExecutionProvider.Cuda;

    internal void Validate()
    {
        if (!Enum.IsDefined(Provider))
        {
            throw new ArgumentOutOfRangeException(nameof(Provider), Provider, "Unknown execution provider.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(CudaDeviceId, nameof(CudaDeviceId));
        if (IntraOpThreads is int threads)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads, nameof(IntraOpThreads));
        }

        if (PadTokenId is int pad)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(pad, nameof(PadTokenId));
        }

        if (Provider == GenAIExecutionProvider.Cpu && UseTf32)
        {
            throw new ArgumentException("UseTf32 applies to the CUDA provider only.", nameof(UseTf32));
        }

        ArgumentNullException.ThrowIfNull(ConfigOverlays, nameof(ConfigOverlays));
    }

    internal IReadOnlyList<string> BuildOverlays()
    {
        var overlays = new List<string>();
        if (Starlings.Threading.IntraOpThreadDefaults.Resolve(IntraOpThreads) is int threads)
        {
            overlays.Add(string.Create(CultureInfo.InvariantCulture, $"{{\"model\":{{\"decoder\":{{\"session_options\":{{\"intra_op_num_threads\":{threads}}}}}}}}}"));
        }

        if (PadTokenId is int pad)
        {
            overlays.Add(string.Create(CultureInfo.InvariantCulture, $"{{\"model\":{{\"pad_token_id\":{pad}}}}}"));
        }

        overlays.AddRange(ConfigOverlays);
        return overlays;
    }
}
