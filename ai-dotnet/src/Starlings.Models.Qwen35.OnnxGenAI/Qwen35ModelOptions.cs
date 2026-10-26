using System.Collections.Immutable;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.Qwen35;

/// <summary>Immutable load-time settings of a <see cref="Qwen35ChatModel"/>.</summary>
public sealed record Qwen35ModelOptions
{
    /// <summary>
    /// Execution provider. CPU runs <see cref="Qwen35Variant.Fp32x"/>; CUDA runs <see cref="Qwen35Variant.Fp32x"/> (TF32
    /// always off) or <see cref="Qwen35Variant.Fp16x"/>. A provider that cannot be used fails the load; no fallback.
    /// </summary>
    public GenAIExecutionProvider Provider { get; init; } = GenAIExecutionProvider.Cpu;

    /// <summary>CUDA device ordinal among the visible devices. Default 0.</summary>
    public int CudaDeviceId { get; init; }

    /// <summary>Intra-op thread count of the decoder session, or null for the GenAI default.</summary>
    public int? IntraOpThreads { get; init; }

    /// <summary>
    /// Advanced: see <see cref="GenAIModelOptions.SerializeNativeCalls"/>. Null (default) serializes the native work of
    /// concurrent requests on CUDA, where concurrent generators produced wrong tokens, and lets it run concurrently on CPU.
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }
}

/// <summary>Identity of a loaded model (plan 5.1: inspect model, tokenizer, backend, provider and artifacts).</summary>
public sealed record Qwen35ModelInfo
{
    /// <summary>Profile id, for example <c>qwen3.5-0.8b/ortgenai-fp32x</c>.</summary>
    public required string Profile { get; init; }

    /// <summary>Hub repository and commit of the checkpoint.</summary>
    public required string Checkpoint { get; init; }

    /// <summary>The built variant.</summary>
    public required Qwen35Variant Variant { get; init; }

    /// <summary>Execution provider.</summary>
    public required GenAIExecutionProvider Provider { get; init; }

    /// <summary>ONNX Runtime GenAI managed binding version (from the dependency manifest).</summary>
    public required string GenAIVersion { get; init; }

    /// <summary>The ONNX Runtime GenAI native package and version in the dependency manifest, or null when unknown.</summary>
    public string? GenAINativePackage { get; init; }

    /// <summary>Owned tokenizer profile and the SHA-256 of its <c>tokenizer.json</c>.</summary>
    public required string Tokenizer { get; init; }

    /// <summary>Chat template renderer and the SHA-256 of the pinned template.</summary>
    public required string ChatTemplate { get; init; }

    /// <summary>Model directory loaded.</summary>
    public required string ModelDirectory { get; init; }

    /// <summary>Configuration overlays applied at load.</summary>
    public required IReadOnlyList<string> ConfigOverlays { get; init; }

    /// <summary>True when concurrent requests' native work runs one call at a time (the CUDA default).</summary>
    public required bool SerializesNativeCalls { get; init; }

    /// <summary>Provider options applied at load.</summary>
    public required IReadOnlyDictionary<string, string> ProviderOptions { get; init; }

    /// <summary>
    /// Longest accepted sequence (prompt plus completion tokens): the smaller of the model's context length and
    /// <see cref="Qwen35ChatModel.MaxContextTokens"/>.
    /// </summary>
    public required int ContextLength { get; init; }

    /// <summary>The model's own context length (<c>genai_config.json</c>, 262,144); longer than the accepted <see cref="ContextLength"/>.</summary>
    public int ModelContextLength { get; init; }

    /// <summary>Stop token ids applied to every request.</summary>
    public required ImmutableArray<int> StopTokenIds { get; init; }

    /// <summary>
    /// Experimental: the files guided decoding (<see cref="Qwen35ChatOptions.Guidance"/>) needs that the model directory
    /// lacks (<see cref="GenAIModel.GuidanceMissingFiles"/>); empty when guided requests can be served. The published
    /// bundle's variant directories have neither <c>tokenizer.json</c> nor <c>tokenizer_config.json</c>. Defaults to both
    /// file names for an instance that <see cref="Qwen35ChatModel.Load"/> did not create.
    /// </summary>
    public IReadOnlyList<string> GuidanceMissingFiles { get; init; } = GenAIModel.GuidanceFiles;
}

/// <summary>ONNX Runtime GenAI view of <see cref="Qwen35ChatOptions"/>.</summary>
internal static class Qwen35ChatOptionsGenAI
{
    /// <summary>
    /// The decode-loop options of a request on <paramref name="provider"/>: a null
    /// <see cref="Qwen35ChatOptions.PrefillChunkTokens"/> becomes <see cref="Qwen35ChatOptions.DefaultCudaPrefillChunkTokens"/>
    /// on CUDA and <see cref="Qwen35ChatOptions.DefaultCpuPrefillChunkTokens"/> on CPU.
    /// </summary>
    public static GenerationOptions ToGenerationOptions(this Qwen35ChatOptions options, GenAIExecutionProvider provider = GenAIExecutionProvider.Cpu) =>
        options.ToGenerationOptionsCore(provider == GenAIExecutionProvider.Cuda
            ? Qwen35ChatOptions.DefaultCudaPrefillChunkTokens
            : Qwen35ChatOptions.DefaultCpuPrefillChunkTokens);
}
