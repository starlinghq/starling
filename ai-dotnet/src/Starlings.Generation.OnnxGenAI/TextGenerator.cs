using System.Collections.Immutable;
using System.Diagnostics;
using System.Text;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// The owned decode loop over ONNX Runtime GenAI: pre-tokenized prompt ids in, completion ids and text out. One
/// <see cref="GenAISequence"/> per request, so concurrent requests on one <see cref="TextGenerator"/> share the model
/// weights but no generation state. Thread-safe.
/// </summary>
/// <remarks>
/// <para>
/// Per request: check cancellation, run the prompt through the model in one prefill (blocking, not interruptible), then
/// one decode step at a time: check cancellation, stop at <see cref="GenerationOptions.MaxNewTokens"/> completion
/// tokens, generate a token (greedy unless <see cref="GenerationOptions.Sampling"/> is set), stop at a stop token id
/// (not part of the completion), detokenize incrementally with the owned tokenizer (never half a character), and stop
/// at a stop sequence (cut before the match).
/// </para>
/// <para>
/// With <see cref="GenerationOptions.Guidance"/> (experimental, opt-in), GenAI masks the tokens the constraint does not
/// allow before it selects each one. With fast-forward guidance a step can also append the tokens the constraint
/// forces; the loop takes them in order, before the step's selected token, through the same checks (maximum tokens,
/// stop ids, detokenization, stop sequences). Without guidance the loop is unchanged.
/// </para>
/// <para>
/// Cancellation is checked before the prefill and between decode steps. A token passed in cancels at the next check:
/// the call throws <see cref="OperationCanceledException"/> and the native generator is released. It cannot interrupt
/// a prefill or a step already running.
/// </para>
/// <para>
/// The APIs are synchronous on purpose: every step is blocking native work, and wrapping it in a task would only move
/// it to another thread (plan 5.1). Callers that need an asynchronous stream run the enumeration on a worker thread.
/// </para>
/// </remarks>
public sealed class TextGenerator
{
    private readonly HashSet<int> _stopIds;
    private readonly Func<GenAISequenceOptions, IDecodeSequence> _createSequence;
    private readonly int? _padTokenId;
    private readonly int _contextLength;

    /// <summary>Creates a generator over a loaded model and the owned tokenizer that produced its prompt ids.</summary>
    /// <param name="model">The loaded model. Not owned: dispose it separately after the generator's last request.</param>
    /// <param name="tokenizer">The owned tokenizer; its decoder turns completion ids into text.</param>
    /// <param name="additionalStopTokenIds">Stop ids in addition to <see cref="GenAIModel.EosTokenIds"/>.</param>
    public TextGenerator(GenAIModel model, Tokenizer tokenizer, IEnumerable<int>? additionalStopTokenIds = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        Model = model;
        Tokenizer = tokenizer;
        _createSequence = model.CreateSequence;
        _padTokenId = model.PadTokenId;
        _contextLength = model.ContextLength;
        _stopIds = [.. model.EosTokenIds, .. additionalStopTokenIds ?? []];
        StopTokenIds = [.. _stopIds.Order()];
    }

    /// <summary>Test seam: the decode loop over any sequence implementation (no native model).</summary>
    internal TextGenerator(Func<GenAISequenceOptions, IDecodeSequence> createSequence, Tokenizer tokenizer, IEnumerable<int> eosTokenIds, int contextLength, int? padTokenId)
    {
        Model = null!;
        Tokenizer = tokenizer;
        _createSequence = createSequence;
        _padTokenId = padTokenId;
        _contextLength = contextLength;
        _stopIds = [.. eosTokenIds];
        StopTokenIds = [.. _stopIds.Order()];
    }

    /// <summary>The model.</summary>
    public GenAIModel Model { get; }

    /// <summary>The owned tokenizer used for detokenization.</summary>
    public Tokenizer Tokenizer { get; }

    /// <summary>Stop token ids applied to every request (the model's end-of-sequence ids and the constructor's).</summary>
    public ImmutableArray<int> StopTokenIds { get; }

    /// <summary>Generates a completion for <paramref name="promptIds"/>.</summary>
    /// <exception cref="OperationCanceledException">Cancelled before the prefill or between decode steps.</exception>
    /// <exception cref="GenerationContextLengthException">Prompt plus <see cref="GenerationOptions.MaxNewTokens"/> exceed the context length.</exception>
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
    /// Streams a completion for <paramref name="promptIds"/>: lazily, one or more decode steps per update. Arguments are
    /// validated when this method is called; generation starts on the first <c>MoveNext</c>. Disposing the enumerator
    /// early releases the native generator.
    /// </summary>
    /// <remarks>
    /// Dispose the enumerator (<c>foreach</c> does): that releases the native generator and its reference on the model at
    /// once, on the disposing thread. An enumerator abandoned after its first <c>MoveNext</c> without disposal is a
    /// safety-net case: its <see cref="GenAISequence"/> is released on a thread-pool thread after a garbage collection
    /// finds it unreachable, so it no longer pins the model, but the native memory stays allocated until then.
    /// </remarks>
    public IEnumerable<GenerationUpdate> GenerateStreaming(IReadOnlyList<int> promptIds, GenerationOptions? options = null, CancellationToken cancellationToken = default) =>
        GenerateStreamingCore(promptIds, options, null, cancellationToken);

    /// <summary>As <see cref="GenerateStreaming"/>; <paramref name="step0Logits"/> receives the logits after the prefill (fidelity evidence).</summary>
    internal IEnumerable<GenerationUpdate> GenerateStreamingCore(
        IReadOnlyList<int> promptIds, GenerationOptions? options, Action<float[]>? step0Logits, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        options ??= new GenerationOptions();
        options.Validate();
        if (promptIds.Count == 0)
        {
            throw new ArgumentException("The prompt needs at least one token id.", nameof(promptIds));
        }

        int[] prompt = [.. promptIds];
        if (_padTokenId is int pad && Array.IndexOf(prompt, pad) >= 0)
        {
            // GenAI masks prompt positions equal to the pad id, which would silently change the output.
            throw new ArgumentException(
                $"The prompt contains token id {pad}, the model's pad id, which ONNX Runtime GenAI masks out of attention. " +
                "Load the model with a GenAIModelOptions.PadTokenId the tokenizer never produces.",
                nameof(promptIds));
        }

        int headroom = options.Guidance?.MaxFastForwardTokens ?? 0;
        long total = (long)prompt.Length + options.MaxNewTokens + headroom;
        if (total > _contextLength)
        {
            throw new GenerationContextLengthException(
                headroom == 0
                    ? $"{prompt.Length} prompt tokens plus {options.MaxNewTokens} new tokens exceed the context length {_contextLength}."
                    : $"{prompt.Length} prompt tokens plus {options.MaxNewTokens} new tokens plus {headroom} tokens of fast-forward headroom exceed the context length {_contextLength}.")
            {
                PromptTokens = prompt.Length,
                MaxNewTokens = options.MaxNewTokens,
                ContextLength = _contextLength,
            };
        }

        HashSet<int> stopIds = options.AdditionalStopTokenIds.Count == 0 ? _stopIds : [.. _stopIds, .. options.AdditionalStopTokenIds];
        return Run(prompt, options, stopIds, step0Logits, cancellationToken);
    }

    // An iterator: the sequence below is released by its using when the enumeration completes or the enumerator is
    // disposed, so a caller must dispose the enumerator (see GenerateStreaming).
    private IEnumerable<GenerationUpdate> Run(int[] prompt, GenerationOptions options, HashSet<int> stopIds, Action<float[]>? step0Logits, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // Fast-forward guidance appends forced tokens without checking GenAI's maximum length, so the sequence gets room
        // for one step's forced tokens beyond MaxNewTokens; the loop never returns more than MaxNewTokens tokens.
        int fastForwardHeadroom = options.Guidance?.MaxFastForwardTokens ?? 0;
        Queue<int>? forced = options.Guidance is { FastForward: true } ? new Queue<int>() : null;
        using IDecodeSequence sequence = _createSequence(new GenAISequenceOptions
        {
            MaxLength = prompt.Length + options.MaxNewTokens + fastForwardHeadroom,
            Sampling = options.Sampling,
            Guidance = options.Guidance,
        });

        var clock = Stopwatch.StartNew();
        int chunk = options.PrefillChunkTokens ?? prompt.Length;
        for (int start = 0; start < prompt.Length; start += chunk)
        {
            // Between prefill chunks too: a long prompt can be cancelled at chunk boundaries.
            if (start > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            sequence.AppendTokens(prompt.AsSpan(start, Math.Min(chunk, prompt.Length - start)));
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
        LogprobCalculator? logprobs = options.ReturnLogprobs ? new LogprobCalculator(Tokenizer, sequence.VocabularySize, options.TopLogprobs) : null;
        var allLogprobs = ImmutableArray.CreateBuilder<TokenLogprob>();
        var pendingLogprobs = ImmutableArray.CreateBuilder<TokenLogprob>();
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

            int id;
            if (forced is { Count: > 0 })
            {
                // The rest of a fast-forward step: its tokens go through the same checks one at a time, the selected
                // token last.
                id = forced.Dequeue();
            }
            else
            {
                if (sequence.IsDone)
                {
                    // GenAI ends a sequence only at its end-of-sequence ids (handled below) or at its maximum length, which
                    // is the prompt plus MaxNewTokens plus the fast-forward headroom (handled above).
                    throw new GenAIModelException($"ONNX Runtime GenAI ended the sequence unexpectedly after {all.Count} tokens.");
                }

                if (forced is null)
                {
                    id = sequence.GenerateNextToken();
                }
                else
                {
                    int[] appended = sequence.GenerateNextTokens();
                    if (appended.Length == 0 || appended.Length > fastForwardHeadroom + 1)
                    {
                        // GenAI has already written the tokens; the bound is the caller's (GenerationGuidance.WithFastForward).
                        throw new GenAIModelException(
                            $"A fast-forward step appended {appended.Length} tokens; the guidance allows at most {fastForwardHeadroom} forced tokens and the selected one.");
                    }

                    id = appended[0];
                    for (int i = 1; i < appended.Length; i++)
                    {
                        forced.Enqueue(appended[i]);
                    }
                }
            }

            if (stopIds.Contains(id))
            {
                finish = GenerationFinishReason.Stop;
                stopTokenId = id;
                break;
            }

            all.Add(id);
            pendingIds.Add(id);
            if (logprobs is not null)
            {
                // GenAI runs the model on a selected token only at the next step, so the logits read now are those of
                // the step that selected id (GenAISequence.CopyLastLogits; measured with ONNX Runtime GenAI 0.17.1).
                sequence.CopyLastLogits(logprobs.Logits);
                TokenLogprob entry = logprobs.Compute(id);
                allLogprobs.Add(entry);
                pendingLogprobs.Add(entry);
            }

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
                yield return new GenerationUpdate { Text = emit, TokenIds = pendingIds.ToImmutable(), Logprobs = pendingLogprobs.ToImmutable() };
                pendingIds.Clear();
                pendingLogprobs.Clear();
            }
        }

        if (finish != GenerationFinishReason.StopSequence)
        {
            // Text held back for an incomplete UTF-8 sequence or a possible stop-sequence start is released now; a stop
            // sequence can still complete in it.
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
            Logprobs = allLogprobs.ToImmutable(),
        };
        yield return new GenerationUpdate { Text = finalText, TokenIds = pendingIds.ToImmutable(), Logprobs = pendingLogprobs.ToImmutable(), Result = result };
    }
}
