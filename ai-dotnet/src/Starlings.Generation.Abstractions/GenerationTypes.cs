using System.Collections.Immutable;

namespace Starlings.Generation;

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

    /// <summary>Largest accepted <see cref="TopLogprobs"/>.</summary>
    public const int MaxTopLogprobs = 20;

    /// <summary>
    /// Opt-in: return the log probability of every completion token in <see cref="GenerationUpdate.Logprobs"/> and
    /// <see cref="GenerationResult.Logprobs"/>. Default false. Each step then copies the step's logits row (vocabulary
    /// size x 4 bytes: about 1 MB for Qwen3.5) from the native generator and computes a log-softmax over it, so it costs
    /// time on every step. The values come from the model's raw logits for the step that selected the token: before
    /// temperature, top-k and top-p, so under <see cref="Sampling"/> they describe the model's distribution, not the
    /// truncated one the sample was drawn from.
    /// </summary>
    public bool ReturnLogprobs { get; init; }

    /// <summary>
    /// With <see cref="ReturnLogprobs"/>: how many of the most likely tokens of each step to return as
    /// <see cref="TokenLogprob.TopLogprobs"/>, from 0 (default: only the chosen token) to <see cref="MaxTopLogprobs"/>.
    /// A value above 0 without <see cref="ReturnLogprobs"/> is rejected.
    /// </summary>
    public int TopLogprobs { get; init; }

    /// <summary>
    /// Experimental, opt-in: a constraint on the completion (JSON Schema, Lark grammar or regex) applied by ONNX Runtime
    /// GenAI's llguidance integration, or null (default) for unconstrained decoding, which is unchanged by this option.
    /// Needs <c>tokenizer.json</c> and <c>tokenizer_config.json</c> in the model directory
    /// (<c>GenAIModel.GuidanceMissingFiles</c> in Starlings.Generation.OnnxGenAI); without them the request fails with
    /// <see cref="NotSupportedException"/>. Cannot be combined with <see cref="ReturnLogprobs"/> (how GenAI's token mask
    /// shows in the logits it returns is untested). See <see cref="GenerationGuidance"/>.
    /// </summary>
    public GenerationGuidance? Guidance { get; init; }

    /// <summary>
    /// Checks the settings as a decode loop does before it starts a request: <see cref="MaxNewTokens"/> and
    /// <see cref="PrefillChunkTokens"/> positive, stop sequences non-empty, stop ids non-negative, <see cref="TopLogprobs"/>
    /// in range and only with <see cref="ReturnLogprobs"/>, no <see cref="Guidance"/> with <see cref="ReturnLogprobs"/>,
    /// and valid <see cref="Sampling"/>.
    /// </summary>
    /// <exception cref="ArgumentException">A setting is out of range or the combination is rejected.</exception>
    public void Validate()
    {
        if (TopLogprobs is < 0 or > MaxTopLogprobs)
        {
            throw new ArgumentOutOfRangeException(nameof(TopLogprobs), TopLogprobs, $"TopLogprobs must be in [0, {MaxTopLogprobs}].");
        }

        if (TopLogprobs > 0 && !ReturnLogprobs)
        {
            throw new ArgumentException("TopLogprobs needs ReturnLogprobs = true.", nameof(TopLogprobs));
        }

        if (Guidance is not null && ReturnLogprobs)
        {
            throw new ArgumentException("ReturnLogprobs cannot be combined with Guidance.", nameof(ReturnLogprobs));
        }

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
/// Sampling settings: temperature, then top-k, then top-p, with an explicit seed (the order of ONNX Runtime GenAI's
/// sampler, which Starlings.Generation.OnnxGenAI uses). The same
/// seed, prompt, settings, model and device reproduce the same tokens; outputs are not expected to match across
/// devices, versions or the reference implementation (plan 10.1).
/// </summary>
public sealed record SamplingOptions
{
    /// <summary>
    /// Softmax temperature, greater than 0. Default 1. Setting only the temperature (or only <see cref="TopP"/>) does not
    /// turn <see cref="TopK"/> off: its default of 50 still applies, so the sample is drawn from the 50 most likely tokens.
    /// </summary>
    public double Temperature { get; init; } = 1.0;

    /// <summary>
    /// Keep the k most likely tokens (at least 1). Default 50 (the GenAI default), applied whenever this property is not
    /// set, including a request that sets only <see cref="Temperature"/> or <see cref="TopP"/>; there is no "off" value
    /// other than a k of at least the vocabulary size.
    /// </summary>
    public int TopK { get; init; } = 50;

    /// <summary>Nucleus probability mass in (0, 1]. Default 1 (off).</summary>
    public double TopP { get; init; } = 1.0;

    /// <summary>Random seed (non-negative). Required so that a run is reproducible.</summary>
    public required int Seed { get; init; }

    /// <summary>Checks the settings: temperature finite and above 0, top-k at least 1, top-p in (0, 1], seed non-negative.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is out of range.</exception>
    public void Validate()
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

    /// <summary>
    /// With <see cref="GenerationOptions.ReturnLogprobs"/>: one entry per id of <see cref="TokenIds"/>, in order (a stop
    /// token id has none). Empty otherwise.
    /// </summary>
    public ImmutableArray<TokenLogprob> Logprobs { get; init; } = [];
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

    /// <summary>
    /// With <see cref="GenerationOptions.ReturnLogprobs"/>: one entry per id of <see cref="TokenIds"/>, in order. Empty
    /// otherwise. Concatenated over all updates they equal <see cref="GenerationResult.Logprobs"/>.
    /// </summary>
    public ImmutableArray<TokenLogprob> Logprobs { get; init; } = [];

    /// <summary>The completed result on the last update; null on every other update.</summary>
    public GenerationResult? Result { get; init; }

    /// <summary>True on the last update.</summary>
    public bool IsFinal => Result is not null;
}
