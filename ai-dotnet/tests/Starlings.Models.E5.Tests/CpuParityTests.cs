using System.Diagnostics;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

public sealed class E5CpuFixture() : E5ModelFixture(new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cpu });

/// <summary>
/// Profile <c>multilingual-e5-small/onnx-fp32-cpu</c>: the real model against the protected embeddings, using the
/// fixture ids and the binding tolerances. Skips when the model is not present (it is not in Git).
/// </summary>
public class CpuParityTests(E5CpuFixture fixture, ITestOutputHelper output) : IClassFixture<E5CpuFixture>
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CpuFixtureDirectory);

    [Fact]
    public void DiagnosticsMatchTheManifest()
    {
        var encoder = fixture.Encoder;
        var d = encoder.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion} (managed {d.ManagedBindingVersion}); providers {string.Join(", ", d.AvailableProviders)}; session {string.Join(", ", d.SessionProviders)}");
        output.WriteLine($"model {d.ModelSource} sha256 {d.ModelSha256} ({d.ModelSizeBytes} bytes); producer {d.Model.ProducerName}; graph {d.Model.GraphName}");
        output.WriteLine("inputs: " + string.Join("; ", d.Inputs.Select(i => $"{i.Name} {i.ElementType} [{string.Join(",", i.SymbolicDimensions.Zip(i.Dimensions, (s, n) => n >= 0 ? n.ToString(System.Globalization.CultureInfo.InvariantCulture) : s))}]")));
        output.WriteLine("outputs: " + string.Join("; ", d.Outputs.Select(i => $"{i.Name} {i.ElementType} [{string.Join(",", i.SymbolicDimensions.Zip(i.Dimensions, (s, n) => n >= 0 ? n.ToString(System.Globalization.CultureInfo.InvariantCulture) : s))}]")));

        Assert.Equal("1.30.0", d.OrtVersion);
        Assert.Equal(E5TestAssets.ModelSha256, d.ModelSha256);
        Assert.Equal(470_268_510, d.ModelSizeBytes);
        Assert.Equal(["input_ids", "attention_mask", "token_type_ids"], d.Inputs.Select(i => i.Name));
        Assert.All(d.Inputs, i => Assert.Equal("Int64", i.ElementType));
        var hidden = Assert.Single(d.Outputs);
        Assert.Equal("last_hidden_state", hidden.Name);
        Assert.Equal(384, hidden.Dimensions[2]);
        Assert.Equal(384, encoder.Dimension);
    }

    [Fact]
    public void ParityWithProtectedEmbeddings()
    {
        var encoder = fixture.Encoder;
        var singles = E5ParityChecks.SingleInputs(encoder, Tolerances, output);
        E5ParityChecks.PaddedBatchFixture(encoder, Tolerances, singles, output);
        E5ParityChecks.AllCasesInBatches(encoder, 16, Tolerances, singles, output);

        // One call with all 46 rows, split by MaxBatchSize = 5 into padded chunks.
        using var chunked = new E5Encoder(encoder.Session, new E5EncoderOptions { MaxBatchSize = 5 });
        E5ParityChecks.AllCasesInBatches(chunked, 46, Tolerances, singles, output);
    }

    /// <summary>
    /// The experimental batching options against the binding bounds: length-sorted chunks, a token budget, and both,
    /// over random orders of the fixture cases (lengths up to 512 interleaved).
    /// </summary>
    [Fact]
    public void SortedAndTokenBudgetedBatchesStayWithinThePaddedBatchBound()
    {
        var encoder = fixture.Encoder;
        var singles = E5Fixtures.Cases.ToDictionary(c => c.Id, c => encoder.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken), StringComparer.Ordinal);
        int[] seeds = [1, 2, 3];
        using (var sorted = new E5Encoder(encoder.Session, new E5EncoderOptions { MaxBatchSize = 8, SortByLength = true }))
        {
            E5ParityChecks.ShuffledBatches(sorted, seeds, Tolerances, singles, output);
        }

        using (var budgeted = new E5Encoder(encoder.Session, new E5EncoderOptions { MaxBatchTokens = 1024 }))
        {
            E5ParityChecks.ShuffledBatches(budgeted, seeds, Tolerances, singles, output);
        }

        using var both = new E5Encoder(encoder.Session, new E5EncoderOptions { MaxBatchSize = 16, MaxBatchTokens = 2048, SortByLength = true });
        E5ParityChecks.ShuffledBatches(both, seeds, Tolerances, singles, output);

        // Padding efficiency (real tokens / padded cells) of the plans on one random order: informational.
        var order = Enumerable.Range(0, E5Fixtures.Cases.Count).ToArray();
        new Random(1).Shuffle(order);
        var inputs = order.Select(i => new EncoderInput(E5Fixtures.Cases[i].InputIds)).ToList();
        long tokens = inputs.Sum(i => (long)i.Length);
        foreach (var options in new[] { new E5EncoderOptions { MaxBatchSize = 8 }, new E5EncoderOptions { MaxBatchSize = 8, SortByLength = true } })
        {
            var plan = E5Encoder.Plan(inputs, options);
            output.WriteLine($"plan MaxBatchSize 8, SortByLength {options.SortByLength}: {plan.Chunks.Count} calls, {plan.PaddedCells} padded cells for {tokens} tokens, efficiency {(double)tokens / plan.PaddedCells:P1}");
        }
    }

    [Fact]
    public void AllNodesRunOnCpu()
    {
        var encoder = fixture.Encoder;
        encoder.Embed(new EncoderInput(E5Fixtures.Cases[0].InputIds), TestContext.Current.CancellationToken);
        var placement = encoder.Session.Placement;
        Assert.NotNull(placement);
        output.WriteLine("placement: " + string.Join(", ", placement.NodesByProvider.Select(kv => $"{kv.Key} {kv.Value}")));
        Assert.Equal(OnnxRuntimeInfo.CpuProviderName, Assert.Single(placement.NodesByProvider).Key);
    }

    [Fact]
    public void OneThreadMatchesDefaultThreads()
    {
        var encoder = fixture.Encoder;
        using var oneThread = E5Encoder.Create(fixture.ModelPath!, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1, ComputeModelSha256 = false },
        });
        var stats = new ErrorStats();
        foreach (var c in E5Fixtures.Cases.Where(c => c.InputIds.Length <= 64))
        {
            stats.Add(oneThread.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken), encoder.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken));
        }

        output.WriteLine($"1 intra-op thread vs ORT default threads (cases up to 64 tokens): {stats}");
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs);
    }

    [Fact]
    public void ConcurrentCallsOnSharedEncoderMatchSequentialResults()
    {
        var encoder = fixture.Encoder;
        var cases = E5Fixtures.Cases.Where(c => c.InputIds.Length <= 128).ToList();
        var sequential = cases.Select(c => encoder.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken)).ToList();
        var stats = new ErrorStats[4];
        Parallel.For(0, stats.Length, t =>
        {
            stats[t] = new ErrorStats();
            for (int i = 0; i < cases.Count; i++)
            {
                int k = (i + (t * 7)) % cases.Count;
                stats[t].Add(encoder.Embed(new EncoderInput(cases[k].InputIds)), sequential[k]);
            }
        });

        double max = stats.Max(s => s.Max);
        output.WriteLine($"4 threads x {cases.Count} cases on one encoder vs sequential: max abs {max:E3}");
        Assert.True(max <= Tolerances.PaddedBatchVsSingleMaxAbs);
    }

    [Fact]
    public void CancellationInterruptsALargeBatchAndEncoderStaysUsable()
    {
        var encoder = fixture.Encoder;
        var longCase = E5Fixtures.Cases.Single(c => c.Id == "long-english");
        Assert.Equal(512, longCase.InputIds.Length);
        var inputs = Enumerable.Repeat(new EncoderInput(longCase.InputIds), 32).ToList();

        using var cts = new CancellationTokenSource();

        // The cancellation time is read on the cancelling thread before Cancel() and read here after joining it, not
        // from a callback registered on the token: callbacks run newest first, so the run's own (later) cancellation
        // callback can stop it before a recorder registered here has run.
        long start = Stopwatch.GetTimestamp();
        var canceller = OperationalChecks.CancelAfter(cts, TimeSpan.FromMilliseconds(200));
        long returned;
        try
        {
            Assert.ThrowsAny<OperationCanceledException>(() => encoder.EmbedBatch(inputs, cts.Token));
        }
        finally
        {
            returned = Stopwatch.GetTimestamp();
            canceller.Join();
        }

        var cancelledAt = TimeSpan.FromMilliseconds(canceller.MillisecondsSince(start));
        var latency = Stopwatch.GetElapsedTime(start, returned) - cancelledAt;
        output.WriteLine($"32 x 512 batch: cancelled at {cancelledAt.TotalMilliseconds:F0} ms, threw {latency.TotalMilliseconds:F1} ms later (an uncancelled 32 x 512 CPU run took ~10 s in docs/m0/e5-gpu.md)");
        Assert.True(latency < TimeSpan.FromSeconds(2), $"cancellation latency {latency.TotalMilliseconds:F0} ms");

        var empty = E5Fixtures.Cases[0];
        var stats = new ErrorStats();
        stats.Add(encoder.Embed(new EncoderInput(empty.InputIds), TestContext.Current.CancellationToken), empty.Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"after cancellation: {stats}");
    }
}
