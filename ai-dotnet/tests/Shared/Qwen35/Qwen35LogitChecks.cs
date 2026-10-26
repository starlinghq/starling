using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Xunit;

namespace Starlings.TestSupport;

/// <summary>
/// Logit-level checks on the native sequence (plan 10.1 LLM list: initial logits, cached vs uncached decoding) and the
/// additional stop token id, shared by the CPU and CUDA test projects. Uses Qwen35ChatModel internals.
/// </summary>
internal static class Qwen35LogitChecks
{
    /// <summary>
    /// Initial logits: the step-0 log-softmax after the prefill against the reference's 64 most likely ids. The
    /// full-vocabulary bound (<c>step0_logprob_max_abs</c>) is applied to those 64 ids (the full reference vectors are
    /// not in the repository), the top-5 bound to the reference's 5 best, and the argmax must be equal.
    /// </summary>
    public static void Step0LogitsMatchTheReference(Qwen35ChatModel model, GenerationCase c, Qwen35Tolerances tolerances, int? prefillChunkTokens = null)
    {
        using var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = c.PromptIds.Length + 1 });
        Append(seq, c.PromptIds, prefillChunkTokens);
        var lp = LogSoftmax(LastLogits(seq));
        Assert.All(lp, v => Assert.True(double.IsFinite(v), "NaN or Inf in the step-0 logits"));
        Assert.Equal(c.Step0Top[0].Id, ArgMax(lp));
        double all = c.Step0Top.Max(t => Math.Abs(lp[t.Id] - t.Logprob));
        double top5 = c.Step0Top.Take(5).Max(t => Math.Abs(lp[t.Id] - t.Logprob));
        Assert.True(all <= tolerances.Step0LogprobMaxAbs, $"{c.Id}: step-0 max abs {all} over the reference top 64 > {tolerances.Step0LogprobMaxAbs} ({tolerances.Source})");
        Assert.True(top5 <= tolerances.Step0Top5LogprobMaxAbs, $"{c.Id}: step-0 top-5 max abs {top5} > {tolerances.Step0Top5LogprobMaxAbs} ({tolerances.Source})");
    }

    /// <summary>
    /// Cached versus uncached decoding: the logits that selected each checked token during cached greedy decoding
    /// against a fresh prefill of the same prefix (<c>cached_vs_uncached_logprob_max_abs</c>), with the greedy tokens
    /// equal to the reference. GenAI reports, after GenerateNextToken, the logits that selected the returned token.
    /// </summary>
    public static double CachedDecodingEqualsAFreshPrefill(Qwen35ChatModel model, GenerationCase c, int steps, Qwen35Tolerances tolerances, int? prefillChunkTokens = null)
    {
        steps = Math.Min(steps, c.ContentIds.Length);
        var gen = new List<int>();
        var selecting = new List<double[]>();
        using (var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = c.PromptIds.Length + steps }))
        {
            Append(seq, c.PromptIds, prefillChunkTokens);
            for (int k = 0; k < steps; k++)
            {
                gen.Add(seq.GenerateNextToken());
                selecting.Add(LogSoftmax(LastLogits(seq)));
            }
        }

        Assert.Equal(c.ContentIds.Take(steps), gen);
        double worst = 0;
        foreach (int k in new[] { 1, steps / 2, steps - 1 }.Where(k => k >= 1).Distinct())
        {
            using var fresh = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = c.PromptIds.Length + k + 1 });
            Append(fresh, [.. c.PromptIds, .. gen.Take(k)], prefillChunkTokens);
            var lp = LogSoftmax(LastLogits(fresh));
            Assert.Equal(gen[k], ArgMax(lp));
            double d = 0;
            for (int i = 0; i < lp.Length; i++)
            {
                d = Math.Max(d, Math.Abs(lp[i] - selecting[k][i]));
            }

            Assert.True(d <= tolerances.CachedVsUncachedLogprobMaxAbs, $"{c.Id} step {k}: cached vs uncached {d} > {tolerances.CachedVsUncachedLogprobMaxAbs} ({tolerances.Source})");
            worst = Math.Max(worst, d);
        }

        return worst;
    }

    /// <summary>An additional stop token id ends generation at its first occurrence in the reference output.</summary>
    public static void AdditionalStopTokenIdEndsAtItsFirstOccurrence(Qwen35ChatModel model, GenerationCase c, int index)
    {
        int stopId = c.ContentIds[index];
        int first = Array.IndexOf(c.ContentIds, stopId);
        var r = model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = index + 8, AdditionalStopTokenIds = [stopId] });
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        Assert.Equal(stopId, r.StopTokenId);
        Assert.Equal(c.ContentIds[..first], r.TokenIds);
        Assert.Equal(first, r.Usage.CompletionTokens);
    }

    /// <summary>
    /// The public asynchronous seam (<see cref="IGenerationSequence"/>) over a GenAI sequence: the same greedy ids and
    /// bit-identical step-0 and selecting logits as the synchronous calls, and a cancelled token stops each call before
    /// it starts, leaving the sequence usable.
    /// </summary>
    public static async Task AsyncSequenceSeamEqualsTheSynchronousCalls(Qwen35ChatModel model, GenerationCase c, int steps)
    {
        steps = Math.Min(steps, c.ContentIds.Length);
        var options = new GenAISequenceOptions { MaxLength = c.PromptIds.Length + steps };
        var syncIds = new List<int>();
        var syncLogits = new List<float[]>();
        using (var seq = model.Backend.CreateSequence(options))
        {
            seq.AppendTokens(c.PromptIds);
            syncLogits.Add(LastLogits(seq));
            for (int k = 0; k < steps; k++)
            {
                syncIds.AddRange(seq.GenerateNextTokens());
                syncLogits.Add(LastLogits(seq));
            }
        }

        Assert.Equal(c.ContentIds.Take(steps), syncIds);

        using GenAISequence native = model.Backend.CreateSequence(options);
        var (asyncIds, asyncLogits) = await DriveAsync(native, c.PromptIds, steps);
        Assert.Equal(syncIds, asyncIds);
        Assert.Equal(syncLogits.Count, asyncLogits.Count);
        for (int k = 0; k < syncLogits.Count; k++)
        {
            Assert.True(syncLogits[k].AsSpan().SequenceEqual(asyncLogits[k]), $"{c.Id}: logits {k} differ between the synchronous and asynchronous calls");
        }
    }

    // Through the interface only, as a backend-neutral decode loop would drive it.
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1859", Justification = "The check is of the interface seam.")]
    private static async Task<(List<int> Ids, List<float[]> Logits)> DriveAsync(IGenerationSequence sequence, int[] promptIds, int steps)
    {
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        var buffer = new float[sequence.VocabularySize];
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => sequence.AppendTokensAsync(promptIds, cancelled.Token).AsTask());

        var ids = new List<int>();
        var logits = new List<float[]>();
        await sequence.AppendTokensAsync(promptIds, TestContext.Current.CancellationToken);
        await sequence.CopyLastLogitsAsync(buffer, TestContext.Current.CancellationToken);
        logits.Add([.. buffer]);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => sequence.StepAsync(cancelled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => sequence.CopyLastLogitsAsync(buffer, cancelled.Token).AsTask());
        for (int k = 0; k < steps; k++)
        {
            ids.AddRange(await sequence.StepAsync(TestContext.Current.CancellationToken));
            await sequence.CopyLastLogitsAsync(buffer, TestContext.Current.CancellationToken);
            logits.Add([.. buffer]);
        }

        return (ids, logits);
    }

    private static void Append(GenAISequence seq, int[] ids, int? chunk)
    {
        int size = chunk ?? ids.Length;
        for (int i = 0; i < ids.Length; i += size)
        {
            seq.AppendTokens(ids.AsSpan(i, Math.Min(size, ids.Length - i)));
        }
    }

    private static float[] LastLogits(GenAISequence seq)
    {
        var l = new float[seq.VocabularySize];
        seq.CopyLastLogits(l);
        return l;
    }

    private static double[] LogSoftmax(float[] logits)
    {
        double max = logits.Max();
        double sum = 0;
        foreach (float v in logits)
        {
            sum += Math.Exp(v - max);
        }

        double lse = max + Math.Log(sum);
        return [.. logits.Select(v => v - lse)];
    }

    private static int ArgMax(double[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }

        return best;
    }
}
