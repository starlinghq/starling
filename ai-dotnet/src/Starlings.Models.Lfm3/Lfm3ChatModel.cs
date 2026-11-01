using System.Collections.Immutable;
using System.Diagnostics;
using System.Text;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.Lfm3;

/// <summary>
/// LFM3-1.7B chat on ONNX Runtime: renders text turns with <see cref="Lfm3ChatTemplate"/>, tokenizes them with the
/// owned tokenizer, and runs a greedy decode loop over <see cref="Lfm3Sequence"/>. One sequence per request, so
/// concurrent requests share the weights but no generation state. Thread-safe.
/// </summary>
/// <remarks>
/// <para>Per request: check cancellation, prefill the prompt (in chunks of
/// <see cref="GenerationOptions.PrefillChunkTokens"/> when set), then one greedy step at a time: check cancellation,
/// stop at <see cref="GenerationOptions.MaxNewTokens"/> completion tokens, stop at <c>&lt;|im_end|&gt;</c> or another
/// stop id (not part of the completion), detokenize incrementally (never half a character), and stop at a stop sequence
/// (cut before the match).</para>
/// <para>Not offered (rejected with <see cref="NotSupportedException"/>): <see cref="GenerationOptions.Sampling"/>,
/// <see cref="GenerationOptions.ReturnLogprobs"/> and <see cref="GenerationOptions.Guidance"/>. The completion text keeps
/// the reasoning the model writes before <c>&lt;/think&gt;</c>; parsing it out is the caller's.</para>
/// </remarks>
public sealed class Lfm3ChatModel : IDisposable
{
    private readonly bool _ownsModel;
    private int _disposed;

    /// <summary>Creates a chat model over a loaded graph and the owned tokenizer.</summary>
    /// <param name="model">The graph. Owned (disposed with this instance) when <paramref name="ownsModel"/> is true.</param>
    /// <param name="tokenizer">The owned tokenizer of profile <c>lfm3-bpe-chat</c>.</param>
    /// <param name="ownsModel">Dispose <paramref name="model"/> with this instance.</param>
    public Lfm3ChatModel(Lfm3Model model, Tokenizer tokenizer, bool ownsModel = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        Model = model;
        Tokenizer = tokenizer;
        _ownsModel = ownsModel;
    }

    /// <summary>The graph.</summary>
    public Lfm3Model Model { get; }

    /// <summary>The owned tokenizer.</summary>
    public Tokenizer Tokenizer { get; }

    /// <summary>Loads the graph and the tokenizer of verified files; the returned instance owns the graph.</summary>
    public static Lfm3ChatModel Create(Lfm3ModelFiles files, Lfm3Options? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        Tokenizer tokenizer = Tokenizer.FromFile(files.TokenizerPath);
        return new Lfm3ChatModel(Lfm3Model.Create(files, options), tokenizer, ownsModel: true);
    }

    /// <summary>Renders and tokenizes a conversation with the generation prompt (see <see cref="Lfm3ChatTemplate.Render"/>).</summary>
    public int[] EncodePrompt(IReadOnlyList<ChatTurn> turns, bool enableThinking = true, bool preserveThinking = false) =>
        Tokenizer.Encode(Lfm3ChatTemplate.Render(turns, enableThinking, preserveThinking), addSpecialTokens: false).Ids.ToArray();

    /// <summary>Generates the assistant's reply to <paramref name="turns"/>.</summary>
    public GenerationResult Chat(
        IReadOnlyList<ChatTurn> turns, GenerationOptions? options = null, bool enableThinking = true, CancellationToken cancellationToken = default) =>
        Generate(EncodePrompt(turns, enableThinking), options, cancellationToken);

    /// <summary>Generates a completion for <paramref name="promptIds"/>.</summary>
    /// <exception cref="OperationCanceledException">Cancelled before the prefill, between prefill chunks or between steps.</exception>
    /// <exception cref="GenerationContextLengthException">Prompt plus <see cref="GenerationOptions.MaxNewTokens"/> exceed <see cref="Lfm3Model.ContextLength"/>.</exception>
    /// <exception cref="NotSupportedException">Sampling, logprobs or guidance was requested.</exception>
    public GenerationResult Generate(IReadOnlyList<int> promptIds, GenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        GenerationResult? result = null;
        foreach (var update in GenerateStreaming(promptIds, options, cancellationToken))
        {
            result = update.Result;
        }

        return result!;
    }

    /// <summary>
    /// Streams a completion for <paramref name="promptIds"/>. Arguments are validated when this method is called;
    /// generation starts on the first <c>MoveNext</c>. Disposing the enumerator releases the request's caches.
    /// </summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(IReadOnlyList<int> promptIds, GenerationOptions? options = null, CancellationToken cancellationToken = default) =>
        GenerateStreamingCore(promptIds, options, null, cancellationToken);

    /// <summary>As <see cref="Generate"/>, with an intervention strength vector fixed for the request (see <see cref="Lfm3Model.CreateSequence(float[])"/>).</summary>
    internal GenerationResult GenerateWithStrength(IReadOnlyList<int> promptIds, GenerationOptions? options, float[]? strengths, CancellationToken cancellationToken)
    {
        GenerationResult? result = null;
        foreach (var update in GenerateStreamingCore(promptIds, options, null, strengths, cancellationToken))
        {
            result = update.Result;
        }

        return result!;
    }

    /// <summary>Disposes the graph when this instance owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsModel)
        {
            Model.Dispose();
        }
    }

    /// <summary>As <see cref="GenerateStreaming"/>; <paramref name="step0Logits"/> receives the logits after the prefill (fidelity evidence).</summary>
    internal IEnumerable<GenerationUpdate> GenerateStreamingCore(
        IReadOnlyList<int> promptIds, GenerationOptions? options, Action<float[]>? step0Logits, CancellationToken cancellationToken) =>
        GenerateStreamingCore(promptIds, options, step0Logits, (float[]?)null, cancellationToken);

    /// <summary>
    /// As <see cref="GenerateStreaming"/>, with an intervention strength vector fixed for the request. The vector is
    /// checked against the loaded graph and copied when this method is called, before generation starts.
    /// </summary>
    internal IEnumerable<GenerationUpdate> GenerateStreamingCore(
        IReadOnlyList<int> promptIds, GenerationOptions? options, Action<float[]>? step0Logits, float[]? strengths, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        options ??= new GenerationOptions();
        options.Validate();
        if (options.Sampling is not null)
        {
            throw new NotSupportedException("Sampling is not supported by the LFM3 decode loop; it is greedy.");
        }

        if (options.ReturnLogprobs)
        {
            throw new NotSupportedException("Logprobs are not supported by the LFM3 decode loop.");
        }

        if (options.Guidance is not null)
        {
            throw new NotSupportedException("Guided decoding is not supported by the LFM3 decode loop.");
        }

        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt needs at least one token id.", nameof(promptIds));
        }

        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int[] prompt = [.. promptIds];
        long total = (long)prompt.Length + options.MaxNewTokens;
        if (total > Model.ContextLength)
        {
            throw new GenerationContextLengthException(
                $"{prompt.Length} prompt tokens plus {options.MaxNewTokens} new tokens exceed the context length {Model.ContextLength}.")
            {
                PromptTokens = prompt.Length,
                MaxNewTokens = options.MaxNewTokens,
                ContextLength = Model.ContextLength,
            };
        }

        Model.CheckStrengths(strengths);
        float[]? fixedStrengths = strengths is null ? null : [.. strengths];
        HashSet<int> stopIds = [Lfm3Checkpoint.ImEndTokenId, .. options.AdditionalStopTokenIds];
        return Run(prompt, options, stopIds, step0Logits, fixedStrengths, cancellationToken);
    }

    private IEnumerable<GenerationUpdate> Run(
        int[] prompt, GenerationOptions options, HashSet<int> stopIds, Action<float[]>? step0Logits, float[]? strengths, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using Lfm3Sequence sequence = Model.CreateSequence(strengths);
        var clock = Stopwatch.StartNew();
        int chunk = options.PrefillChunkTokens ?? prompt.Length;
        for (int start = 0; start < prompt.Length; start += chunk)
        {
            if (start > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            sequence.AppendTokens(prompt.AsSpan(start, Math.Min(chunk, prompt.Length - start)), cancellationToken);
        }

        TimeSpan prefill = clock.Elapsed;
        if (step0Logits is not null)
        {
            float[] logits = new float[sequence.VocabularySize];
            sequence.CopyLastLogits(logits);
            step0Logits(logits);
        }

        var detokenizer = new IncrementalDetokenizer(Tokenizer);
        var stops = new StopSequenceMatcher(options.StopSequences);
        var all = ImmutableArray.CreateBuilder<int>();
        var pendingIds = ImmutableArray.CreateBuilder<int>();
        var text = new StringBuilder();
        GenerationFinishReason finish;
        int? stopTokenId = null;
        string? stopSequence = null;
        string finalText = string.Empty;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (all.Count == options.MaxNewTokens)
            {
                finish = GenerationFinishReason.Length;
                break;
            }

            int id = sequence.Step(cancellationToken);
            if (stopIds.Contains(id))
            {
                finish = GenerationFinishReason.Stop;
                stopTokenId = id;
                break;
            }

            all.Add(id);
            pendingIds.Add(id);
            var (emit, matched) = stops.Push(detokenizer.Push(id));
            if (matched is not null)
            {
                finish = GenerationFinishReason.StopSequence;
                stopSequence = matched;
                finalText = emit;
                break;
            }

            if (emit.Length > 0)
            {
                text.Append(emit);
                yield return new GenerationUpdate { Text = emit, TokenIds = pendingIds.ToImmutable() };
                pendingIds.Clear();
            }
        }

        if (finish != GenerationFinishReason.StopSequence)
        {
            var (emit, matched) = stops.Push(detokenizer.Flush());
            if (matched is not null)
            {
                finish = GenerationFinishReason.StopSequence;
                stopSequence = matched;
                stopTokenId = null;
                finalText = emit;
            }
            else
            {
                finalText = emit + stops.Flush();
            }
        }

        text.Append(finalText);
        var result = new GenerationResult
        {
            Text = text.ToString(),
            TokenIds = all.ToImmutable(),
            FinishReason = finish,
            StopTokenId = stopTokenId,
            StopSequence = stopSequence,
            Usage = new GenerationUsage(prompt.Length, all.Count),
            Timings = new GenerationTimings(prefill, clock.Elapsed - prefill),
        };
        yield return new GenerationUpdate { Text = finalText, TokenIds = pendingIds.ToImmutable(), Result = result };
    }
}
