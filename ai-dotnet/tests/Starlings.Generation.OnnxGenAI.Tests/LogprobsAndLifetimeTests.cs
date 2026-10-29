using System.Runtime.CompilerServices;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>
/// Per-token log probabilities of the owned decode loop (scripted sequences and logits, the real qwen35-bpe-chat
/// tokenizer) and the release of abandoned native sequences (no native model).
/// </summary>
public sealed class LogprobsTests
{
    private const int ImEnd = 248046;
    private const int EndOfText = 248044;
    private const int Vocabulary = 248320;
    private static readonly int[] Eos = [ImEnd, EndOfText];
    private static readonly int[] Prompt = [248045, 846, 198];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Tokenizer RequireTokenizer()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        return tokenizer!;
    }

    private static (TextGenerator Generator, List<FakeSequence> Sequences) Create(Tokenizer tokenizer, IReadOnlyList<int> script)
    {
        var sequences = new List<FakeSequence>();
        var generator = new TextGenerator(
            o =>
            {
                var s = new FakeSequence(o, script, Eos) { ScriptedLogits = true };
                sequences.Add(s);
                return s;
            },
            tokenizer,
            Eos,
            1000,
            null);
        return (generator, sequences);
    }

    // log-softmax of the scripted row: PeakLogit at the chosen id, RunnerUpLogit at id 0, 0 elsewhere.
    private static double Normalizer => Math.Log(Math.Exp(FakeSequence.PeakLogit) + Math.Exp(FakeSequence.RunnerUpLogit) + (Vocabulary - 2));

    [Fact]
    public void OffByDefaultNothingIsReturnedAndNoLogitsAreCopied()
    {
        var tok = RequireTokenizer();
        int[] content = [.. tok.Encode("Paris is the capital.", addSpecialTokens: false).Ids];
        var (gen, seqs) = Create(tok, [.. content, ImEnd]);
        var updates = gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 32 }, Ct).ToList();
        Assert.All(updates, u => Assert.True(u.Logprobs.IsEmpty));
        Assert.True(updates[^1].Result!.Logprobs.IsEmpty);
        Assert.Empty(seqs[0].LogitCopiesAtStep);
    }

    [Fact]
    public void EveryCompletionTokenGetsTheLogprobOfTheStepThatSelectedIt()
    {
        var tok = RequireTokenizer();
        int[] content = [.. tok.Encode("Paris is the capital, 你好.", addSpecialTokens: false).Ids];
        Assert.DoesNotContain(0, content);
        var (gen, seqs) = Create(tok, [.. content, ImEnd]);
        var options = new GenerationOptions { MaxNewTokens = 64, ReturnLogprobs = true, TopLogprobs = 3 };
        var updates = gen.GenerateStreaming(Prompt, options, Ct).ToList();
        var result = updates[^1].Result!;

        // One entry per completion id (the stop token has none), in order, and per update aligned with its ids.
        Assert.Equal(result.TokenIds, result.Logprobs.Select(l => l.TokenId));
        Assert.All(updates, u => Assert.Equal(u.TokenIds, u.Logprobs.Select(l => l.TokenId)));
        Assert.Equal(result.Logprobs, updates.SelectMany(u => u.Logprobs));

        // Read after each step (GenAI's logits after a step are the ones that selected its token), once per kept token.
        Assert.Equal(Enumerable.Range(1, content.Length), seqs[0].LogitCopiesAtStep);

        foreach (TokenLogprob entry in result.Logprobs)
        {
            Assert.Equal(FakeSequence.PeakLogit - Normalizer, entry.Logprob, 1e-9);
            Assert.Equal(tok.Decode([entry.TokenId], false), entry.Text);
            Assert.Equal(3, entry.TopLogprobs.Length);
            Assert.Equal(entry.TokenId, entry.TopLogprobs[0].TokenId);
            Assert.Equal(entry.Logprob, entry.TopLogprobs[0].Logprob);
            Assert.Equal(0, entry.TopLogprobs[1].TokenId);
            Assert.Equal(FakeSequence.RunnerUpLogit - Normalizer, entry.TopLogprobs[1].Logprob, 1e-9);

            // Ties (all other logits are 0) go to the lower id.
            Assert.Equal(entry.TokenId == 1 ? 2 : 1, entry.TopLogprobs[2].TokenId);
            Assert.Equal(-Normalizer, entry.TopLogprobs[2].Logprob, 1e-9);
        }

        // The probabilities of a step sum to 1.
        double total = Math.Exp(FakeSequence.PeakLogit - Normalizer) + Math.Exp(FakeSequence.RunnerUpLogit - Normalizer) + (Vocabulary - 2) * Math.Exp(-Normalizer);
        Assert.Equal(1.0, total, 1e-9);
    }

    [Fact]
    public void TopLogprobsZeroReturnsOnlyTheChosenToken()
    {
        var tok = RequireTokenizer();
        int[] content = [.. tok.Encode("Hello.", addSpecialTokens: false).Ids];
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var result = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 16, ReturnLogprobs = true }, Ct);
        Assert.Equal(content.Length, result.Logprobs.Length);
        Assert.All(result.Logprobs, l => Assert.True(l.TopLogprobs.IsEmpty));
    }

    [Fact]
    public void StopSequencesAndTheLengthLimitKeepOneEntryPerReturnedId()
    {
        var tok = RequireTokenizer();
        int[] content = [.. tok.Encode("one two three four five six", addSpecialTokens: false).Ids];
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var stopped = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 64, StopSequences = [" four"], ReturnLogprobs = true }, Ct);
        Assert.Equal(GenerationFinishReason.StopSequence, stopped.FinishReason);
        Assert.Equal(stopped.TokenIds, stopped.Logprobs.Select(l => l.TokenId));

        var (gen2, _) = Create(tok, [.. content, ImEnd]);
        var limited = gen2.Generate(Prompt, new GenerationOptions { MaxNewTokens = 2, ReturnLogprobs = true }, Ct);
        Assert.Equal(GenerationFinishReason.Length, limited.FinishReason);
        Assert.Equal(2, limited.Logprobs.Length);
    }

    [Theory]
    [InlineData(true, -1)]
    [InlineData(true, 21)]
    [InlineData(false, 1)]
    public void InvalidLogprobOptionsAreRejected(bool returnLogprobs, int top)
    {
        var tok = RequireTokenizer();
        var (gen, _) = Create(tok, [ImEnd]);
        Assert.ThrowsAny<ArgumentException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { ReturnLogprobs = returnLogprobs, TopLogprobs = top }, Ct));
    }

    [Fact]
    public void TopSelectionOrdersByLogitThenIdAndSkipsNaN()
    {
        float[] logits = [0.5f, float.NaN, 2f, 2f, -1f, 3f];
        int[] ids = new int[4];
        Assert.Equal(4, LogprobCalculator.SelectTop(logits, ids));
        Assert.Equal([5, 2, 3, 0], ids);

        int[] wide = new int[10];
        Assert.Equal(5, LogprobCalculator.SelectTop(logits, wide));
        Assert.Equal([5, 2, 3, 0, 4], wide[..5]);
        Assert.Equal(0, LogprobCalculator.SelectTop(logits, Span<int>.Empty));
    }
}

/// <summary>The release owner of native sequences and models (<see cref="NativeLifetime"/>).</summary>
public sealed class NativeLifetimeTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void DisposeRunsTheReleaseOnceOnTheCallingThread()
    {
        int runs = 0;
        int thread = -1;
        var lifetime = new NativeLifetime(() =>
        {
            runs++;
            thread = Environment.CurrentManagedThreadId;
        });
        Assert.False(lifetime.IsReleased);
        lifetime.Dispose();
        lifetime.Dispose();
        Assert.True(lifetime.IsReleased);
        Assert.Equal(1, runs);
        Assert.Equal(Environment.CurrentManagedThreadId, thread);
    }

    [Fact]
    public void AnUnreachableOwnerQueuesItsReleaseToTheThreadPool()
    {
        using var released = new ManualResetEventSlim();
        bool onThreadPool = false;
        long before = NativeLifetime.FinalizerReleases;
        CreateAndDrop(() =>
        {
            onThreadPool = Thread.CurrentThread.IsThreadPoolThread;
            released.Set();
        });

        Collect();
        Assert.True(released.Wait(TimeSpan.FromSeconds(30), Ct), "The release of an unreachable owner did not run.");
        Assert.True(onThreadPool, "The release ran on the finalizer thread instead of a thread-pool thread.");
        Assert.True(NativeLifetime.FinalizerReleases > before);
    }

    [Fact]
    public void AnAbandonedStreamingEnumeratorNoLongerPinsItsSequence()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        using var released = new ManualResetEventSlim();
        var generator = new TextGenerator(
            o => new FakeSequence(o, [11, 12, 13, 248046], [248046]) { Lifetime = new NativeLifetime(released.Set) },
            tokenizer!,
            [248046],
            1000,
            null);

        StartAndAbandon(generator);
        Collect();
        Assert.True(released.Wait(TimeSpan.FromSeconds(30), Ct), "The abandoned enumerator's sequence was not released.");
    }

    [Fact]
    public void ADisposedEnumeratorReleasesAtOnce()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        bool released = false;
        var generator = new TextGenerator(
            o => new FakeSequence(o, [11, 12, 13, 248046], [248046]) { Lifetime = new NativeLifetime(() => released = true) },
            tokenizer!,
            [248046],
            1000,
            null);
        using (var enumerator = generator.GenerateStreaming([1, 2, 3], new GenerationOptions { MaxNewTokens = 8 }, Ct).GetEnumerator())
        {
            Assert.True(enumerator.MoveNext());
        }

        Assert.True(released);
    }

    private static void Collect()
    {
        for (int i = 0; i < 3; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void CreateAndDrop(Action release) => _ = new NativeLifetime(release);

    // Starts a stream (the sequence is created and prefilled) and drops the enumerator without disposing it.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void StartAndAbandon(TextGenerator generator)
    {
        var enumerator = generator.GenerateStreaming([1, 2, 3], new GenerationOptions { MaxNewTokens = 8 }, CancellationToken.None).GetEnumerator();
        Assert.True(enumerator.MoveNext());
    }
}
