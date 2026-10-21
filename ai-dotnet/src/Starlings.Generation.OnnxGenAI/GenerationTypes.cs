using System.Collections.Immutable;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>Per-request generation settings. Immutable; reuse freely.</summary>
public sealed record GenerationOptions
{
    /// <summary>Default <see cref="MaxNewTokens"/>.</summary>
    public const int DefaultMaxNewTokens = 512;

    /// <summary>
    /// Maximum completion tokens (at least 1). A stop token id never counts: generation that reaches this many
    /// completion tokens ends with <see cref="GenerationFinishReason.Length"/> without running another step.
    /// </summary>
    public int MaxNewTokens { get; init; } = DefaultMaxNewTokens;

    /// <summary>
    /// Strings that end generation when they appear in the completion text. The text is cut before the first match and
    /// the match is not returned; the token ids that produced it are kept in <see cref="GenerationResult.TokenIds"/> and
    /// counted as completion tokens (a match need not align with token boundaries). Empty strings are rejected.
    /// </summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>
    /// Token ids that end generation in addition to the model's end-of-sequence ids. A stop id is not part of the
    /// completion (not in the text, the ids or the usage count).
    /// </summary>
    public IReadOnlyCollection<int> AdditionalStopTokenIds { get; init; } = [];

    /// <summary>Sampling settings, or null (default) for greedy decoding.</summary>
    public SamplingOptions? Sampling { get; init; }

    /// <summary>
    /// Feeds the prompt to the model in chunks of at most this many tokens (at least 1), or null (default) to prefill
    /// it in one call. ONNX Runtime GenAI computes logits for every position it is given, so a one-call prefill
    /// allocates prompt length x vocabulary size logits (Qwen3.5: about 1 MB per prompt token in fp32); chunks bound
    /// that buffer, and cancellation is also checked between chunks.
    /// </summary>
    public int? PrefillChunkTokens { get; init; }

    /// <summary>
    /// Spike S5 (experimental): constrains decoding with ONNX Runtime GenAI's llguidance integration, or null (default)
    /// for unconstrained decoding. Needs <c>tokenizer.json</c> in the model directory (GenAI reads it from there).
    /// </summary>
    public GenerationGuidance? Guidance { get; init; }

    internal void Validate()
    {
        if (PrefillChunkTokens is int chunk)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunk, nameof(PrefillChunkTokens));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxNewTokens, nameof(MaxNewTokens));
        ArgumentNullException.ThrowIfNull(StopSequences, nameof(StopSequences));
        ArgumentNullException.ThrowIfNull(AdditionalStopTokenIds, nameof(AdditionalStopTokenIds));
        foreach (string s in StopSequences)
        {
            if (string.IsNullOrEmpty(s))
            {
                throw new ArgumentException("Stop sequences must be non-empty strings.", nameof(StopSequences));
            }
        }

        foreach (int id in AdditionalStopTokenIds)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(id, nameof(AdditionalStopTokenIds));
        }

        Sampling?.Validate();
    }
}

/// <summary>
/// Spike S5 (experimental): an llguidance constraint passed to <c>GeneratorParams.SetGuidance</c>. The token mask is
/// applied inside GenAI's <c>GenerateNextToken</c>; fast-forward tokens are off, so every step still yields one token.
/// </summary>
/// <param name="Type"><c>json_schema</c>, <c>regex</c> or <c>lark_grammar</c>.</param>
/// <param name="Data">The schema, regex or grammar text.</param>
public sealed record GenerationGuidance(string Type, string Data)
{
    /// <summary>A Lark grammar (llguidance dialect).</summary>
    public static GenerationGuidance Lark(string grammar) => new("lark_grammar", grammar);

    /// <summary>A JSON Schema.</summary>
    public static GenerationGuidance JsonSchema(string schema) => new("json_schema", schema);

    /// <summary>A regular expression.</summary>
    public static GenerationGuidance Regex(string pattern) => new("regex", pattern);
}

/// <summary>
/// Sampling through ONNX Runtime GenAI's sampler: temperature, then top-k, then top-p, with an explicit seed. The same
/// seed, prompt, settings, model and device reproduce the same tokens; outputs are not expected to match across
/// devices, versions or the reference implementation (plan 10.1).
/// </summary>
public sealed record SamplingOptions
{
    /// <summary>Softmax temperature, greater than 0. Default 1.</summary>
    public double Temperature { get; init; } = 1.0;

    /// <summary>Keep the k most likely tokens (at least 1). Default 50 (the GenAI default).</summary>
    public int TopK { get; init; } = 50;

    /// <summary>Nucleus probability mass in (0, 1]. Default 1 (off).</summary>
    public double TopP { get; init; } = 1.0;

    /// <summary>Random seed (non-negative). Required so that a run is reproducible.</summary>
    public required int Seed { get; init; }

    internal void Validate()
    {
        if (!(Temperature > 0) || double.IsInfinity(Temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature), Temperature, "Temperature must be a finite number greater than 0.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(TopK, nameof(TopK));
        if (!(TopP > 0 && TopP <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(TopP), TopP, "TopP must be in (0, 1].");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(Seed, nameof(Seed));
    }
}

/// <summary>Why generation ended. Cancellation is not a finish reason: it throws <see cref="OperationCanceledException"/>.</summary>
public enum GenerationFinishReason
{
    /// <summary>An end-of-sequence or stop token id was generated (<see cref="GenerationResult.StopTokenId"/>).</summary>
    Stop,

    /// <summary>A stop sequence appeared in the text (<see cref="GenerationResult.StopSequence"/>).</summary>
    StopSequence,

    /// <summary><see cref="GenerationOptions.MaxNewTokens"/> completion tokens were generated.</summary>
    Length,
}

/// <summary>Token accounting of one request.</summary>
/// <param name="PromptTokens">Token ids fed to the model as the prompt.</param>
/// <param name="CompletionTokens">Generated token ids returned as the completion; a stop token id is excluded.</param>
public readonly record struct GenerationUsage(int PromptTokens, int CompletionTokens)
{
    /// <summary>Prompt plus completion tokens.</summary>
    public int TotalTokens => PromptTokens + CompletionTokens;
}

/// <summary>Wall-clock stage timings of one request (plan 5.3 diagnostics).</summary>
/// <param name="Prefill">Running the model over the prompt (not cancellable).</param>
/// <param name="Decode">Decode steps, detokenization and stop matching after the prefill.</param>
public readonly record struct GenerationTimings(TimeSpan Prefill, TimeSpan Decode);

/// <summary>The completed output of one request.</summary>
public sealed record GenerationResult
{
    /// <summary>Completion text: the decoded <see cref="TokenIds"/>, cut before a stop sequence when one matched.</summary>
    public required string Text { get; init; }

    /// <summary>Completion token ids (a stop token id excluded).</summary>
    public required ImmutableArray<int> TokenIds { get; init; }

    /// <summary>Why generation ended.</summary>
    public required GenerationFinishReason FinishReason { get; init; }

    /// <summary>The stop token id generated, when <see cref="FinishReason"/> is <see cref="GenerationFinishReason.Stop"/>.</summary>
    public int? StopTokenId { get; init; }

    /// <summary>The stop sequence matched, when <see cref="FinishReason"/> is <see cref="GenerationFinishReason.StopSequence"/>.</summary>
    public string? StopSequence { get; init; }

    /// <summary>Token accounting.</summary>
    public required GenerationUsage Usage { get; init; }

    /// <summary>Stage timings.</summary>
    public required GenerationTimings Timings { get; init; }
}

/// <summary>
/// One streamed piece of a completion. Concatenating <see cref="Text"/> over all updates gives
/// <see cref="GenerationResult.Text"/>, and concatenating <see cref="TokenIds"/> gives
/// <see cref="GenerationResult.TokenIds"/>. Every update but the last carries non-empty text; the last carries
/// <see cref="Result"/> (finish reason and usage) and possibly the final text.
/// </summary>
public sealed record GenerationUpdate
{
    /// <summary>New text since the previous update (complete UTF-16 text; never half a code point).</summary>
    public required string Text { get; init; }

    /// <summary>Completion token ids since the previous update.</summary>
    public required ImmutableArray<int> TokenIds { get; init; }

    /// <summary>The completed result on the last update; null on every other update.</summary>
    public GenerationResult? Result { get; init; }

    /// <summary>True on the last update.</summary>
    public bool IsFinal => Result is not null;
}
