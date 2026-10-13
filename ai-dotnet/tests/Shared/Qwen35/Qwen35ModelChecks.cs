using System.Collections.Immutable;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Xunit;

namespace Starlings.TestSupport;

/// <summary>Model checks shared by the CPU and CUDA test projects (plan 10.1 LLM list).</summary>
internal static class Qwen35ModelChecks
{
    /// <summary>
    /// Greedy generation from the reference prompt ids. With <paramref name="nearTieMargin"/> null the ids must equal the
    /// reference exactly; otherwise they must equal it up to the first divergence, which must happen at a step whose
    /// reference top-1 minus top-2 logit margin is below the bound (fp16 variants; the bound is a proposal recorded in
    /// docs/evidence/p016-qwen35/README.md, not a protected tolerance).
    /// </summary>
    public static GenerationResult GreedyFromReferenceIds(Qwen35ChatModel model, GenerationCase c, int? maxNewTokens = null, double? nearTieMargin = null, int? prefillChunkTokens = null)
    {
        int max = maxNewTokens ?? c.MaxNewTokens;
        var r = model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = max, PrefillChunkTokens = prefillChunkTokens });
        int[] expected = c.OutputIds.Length > max ? c.OutputIds[..max] : c.OutputIds;
        int[] actual = r.StopTokenId is int stop ? [.. r.TokenIds, stop] : [.. r.TokenIds];
        if (nearTieMargin is null)
        {
            Assert.Equal(expected, actual);
            Assert.Equal(c.FinishReason == "eos" && expected.Length == c.OutputIds.Length ? GenerationFinishReason.Stop : GenerationFinishReason.Length, r.FinishReason);
        }
        else
        {
            int n = 0;
            while (n < expected.Length && n < actual.Length && expected[n] == actual[n])
            {
                n++;
            }

            if (n < expected.Length)
            {
                Assert.True(c.StepMargin[n] < nearTieMargin, $"{c.Id}: diverges at step {n} where the reference margin is {c.StepMargin[n]}");
            }
        }

        Assert.Equal(c.PromptIds.Length, r.Usage.PromptTokens);
        Assert.Equal(r.TokenIds.Length, r.Usage.CompletionTokens);
        return r;
    }

    /// <summary>The text path: messages through the owned template and tokenizer equal the reference prompt ids and the start of its output.</summary>
    public static void TextPathMatches(Qwen35ChatModel model, GenerationCase c, int maxNewTokens)
    {
        var prompt = model.CreatePrompt(c.Messages!, c.EnableThinking);
        Assert.Equal(c.Rendered, prompt.Text);
        Assert.Equal(c.PromptIds, prompt.TokenIds);
        var r = model.Generate(c.Messages!, new Qwen35ChatOptions { MaxNewTokens = maxNewTokens, EnableThinking = c.EnableThinking });
        Assert.Equal(c.ContentIds.Take(maxNewTokens), r.TokenIds);
    }

    /// <summary>Streaming reassembly: updates concatenate to the result, and the text equals one decode of the ids.</summary>
    public static void StreamingReassembles(Qwen35ChatModel model, GenerationCase c, int maxNewTokens)
    {
        var updates = model.GenerateStreaming(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = maxNewTokens }).ToList();
        var result = updates[^1].Result!;
        Assert.Equal(result.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(result.TokenIds, updates.SelectMany(u => u.TokenIds).ToImmutableArray());
        Assert.Equal(model.Tokenizer.Decode(result.TokenIds.AsSpan(), false), result.Text);
        Assert.Equal(c.ContentIds.Take(maxNewTokens), result.TokenIds);
        Assert.All(updates.Take(updates.Count - 1), u => Assert.NotEmpty(u.Text));
    }

    /// <summary>
    /// Concurrent requests on one model each match the reference (no state leakage), over several rounds: a single round
    /// on two threads missed the CUDA concurrency defect that 8 rounds of 6 requests found.
    /// </summary>
    public static void ConcurrentRequestsAreIndependent(Qwen35ChatModel model, IReadOnlyList<GenerationCase> cases, int maxNewTokens, int rounds = 1, int threads = 2)
    {
        for (int round = 0; round < rounds; round++)
        {
            var results = new ImmutableArray<int>[cases.Count];
            Parallel.For(0, cases.Count, new ParallelOptions { MaxDegreeOfParallelism = threads }, i =>
                results[i] = model.Generate(Qwen35Prompt.FromTokenIds(cases[i].PromptIds), new Qwen35ChatOptions { MaxNewTokens = maxNewTokens }).TokenIds);
            for (int i = 0; i < cases.Count; i++)
            {
                Assert.Equal(cases[i].ContentIds.Take(maxNewTokens), results[i]);
            }
        }
    }

    /// <summary>Cancellation between decode steps, then the model serves the next request correctly.</summary>
    public static void CancellationThenReuse(Qwen35ChatModel model, GenerationCase longCase, GenerationCase shortCase)
    {
        using var cts = new CancellationTokenSource();
        int seen = 0;
        Assert.ThrowsAny<OperationCanceledException>(() =>
        {
            foreach (var u in model.GenerateStreaming(Qwen35Prompt.FromTokenIds(longCase.PromptIds), new Qwen35ChatOptions { MaxNewTokens = 64 }, cts.Token))
            {
                if (++seen == 3)
                {
                    cts.Cancel();
                }
            }
        });
        Assert.Equal(3, seen);
        GreedyFromReferenceIds(model, shortCase);
    }
}
