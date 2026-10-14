using System.Diagnostics;
using System.Globalization;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

public sealed class EmbeddingGemma2CpuFixture() : EmbeddingGemma2ModelFixture(new OnnxSessionOptions());

/// <summary>
/// Profile <c>embeddinggemma-2/onnx-fp32-cpu</c>: the locked text-path graph on the CPU provider, from fixture ids,
/// against the protected PyTorch fp32 development fixtures (PR #29) with the profile's protected tolerances.yaml.
/// Skips when the graph is absent (CI).
/// </summary>
public class CpuParityTests(EmbeddingGemma2CpuFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2CpuFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cpu");

    [Fact]
    public void SessionIsTheLockedGraphOnCpu()
    {
        var d = fixture.Encoder.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; session {string.Join(", ", d.SessionProviders)}; model sha256 {d.ModelSha256}");
        Assert.Equal(OnnxRuntimeInfo.CpuProviderName, d.SessionProviders[0]);
        Assert.Equal(EmbeddingGemma2ModelAssets.ModelSha256, d.ModelSha256);
        Assert.Equal(["input_ids", "attention_mask"], d.Inputs.Select(i => i.Name));
        Assert.Equal(["last_hidden_state", "token_embeddings"], d.Outputs.Select(o => o.Name));
    }

    [Fact]
    public void ParityWithTheProtectedReference()
    {
        var encoder = fixture.Encoder;
        var singles = EmbeddingGemma2ParityChecks.SingleRows(encoder, fixture.Rows, Tolerances, output);
        EmbeddingGemma2ParityChecks.PaddedBatchFixture(encoder, fixture.Rows, Tolerances, singles, output);
        EmbeddingGemma2ParityChecks.AllRowsInBatches(encoder, fixture.Rows, 8, Tolerances, singles, output);
        EmbeddingGemma2ParityChecks.Matryoshka(encoder, fixture.Rows, Tolerances, singles, output);
    }

    [Fact]
    public void BatchIndependenceAcrossTheSlidingWindow() =>
        EmbeddingGemma2ParityChecks.SlidingWindowBatchIndependence(fixture.Encoder, fixture.Rows, Tolerances, output);

    [Fact]
    public void InputsOver8192TokensAreRejected() =>
        EmbeddingGemma2ParityChecks.InputsOver8192TokensAreRejected(fixture.Encoder, fixture.Rows, output);

    [Fact]
    public void AnInputOfExactly8192TokensRuns()
    {
        // Smoke only: 2,050 to 8,192 tokens has no reference evidence (capability input.2050_to_8192_tokens: untested).
        var seq = EmbeddingGemma2ParityChecks.SyntheticSequence(fixture.Rows, 8192);
        var sw = Stopwatch.StartNew();
        var embedding = fixture.Encoder.Embed(new EmbeddingGemma2Input(seq), TestContext.Current.CancellationToken);
        using var process = Process.GetCurrentProcess();
        output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"8,192 tokens: {sw.Elapsed.TotalSeconds:F1} s; norm {EmbeddingGemma2ParityChecks.Norm(embedding):F7}; peak working set {process.PeakWorkingSet64 / 1048576.0:F0} MB"));
        Assert.All(embedding, x => Assert.True(float.IsFinite(x)));
        Assert.InRange(EmbeddingGemma2ParityChecks.Norm(embedding), 0.9999, 1.0001);
    }

    [Fact]
    public void ConcurrentCallsGiveTheSequentialResults()
    {
        var encoder = fixture.Encoder;
        var rows = fixture.Rows.Where(r => r.InputIds.Length < 200).Take(12).ToList();
        var ct = TestContext.Current.CancellationToken;
        var sequential = rows.Select(r => encoder.Embed(new EmbeddingGemma2Input(r.InputIds), ct)).ToList();
        var parallel = new float[rows.Count][];
        Parallel.For(0, rows.Count, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, i =>
            parallel[i] = encoder.Embed(new EmbeddingGemma2Input(rows[i].InputIds), ct));
        for (int i = 0; i < rows.Count; i++)
        {
            Assert.Equal(sequential[i], parallel[i]);
        }
    }

    [Fact]
    public void CancellationDuringARunThrowsAndTheEncoderStaysUsable()
    {
        var encoder = fixture.Encoder;
        var longRow = fixture.Rows.MaxBy(r => r.InputIds.Length)!;
        var inputs = Enumerable.Repeat(new EmbeddingGemma2Input(longRow.InputIds), 8).ToList();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));
        var sw = Stopwatch.StartNew();
        Assert.ThrowsAny<OperationCanceledException>(() => encoder.EmbedBatch(inputs, cts.Token));
        output.WriteLine($"cancelled after {sw.ElapsedMilliseconds} ms (token fired at 100 ms)");
        var after = encoder.Embed(new EmbeddingGemma2Input(fixture.Rows[5].InputIds), TestContext.Current.CancellationToken);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add(fixture.Rows[5].Key, after, fixture.Rows[5].Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs);
    }

    [Fact]
    public void TextParityWithTheOwnedTokenizer() =>
        EmbeddingGemma2ParityChecks.TextRows(fixture.Encoder, EmbeddingGemma2TestAssets.Tokenizer, fixture.Rows, Tolerances, output);

    /// <summary>
    /// Capability <c>prompts_task</c>: every one of the 20 named prompts on four public texts (proposed fixture from
    /// sentence-transformers, judged with this profile's protected text bounds).
    /// </summary>
    [Fact]
    public void TaskPromptParityWithTheOwnedTokenizer()
    {
        var rows = EmbeddingGemma2Fixtures.LoadTaskPromptRows();
        Assert.Equal(80, rows.Count);
        Assert.Equal(EmbeddingGemma2Prompt.Named.Keys.Order(StringComparer.Ordinal), rows.Select(r => r.Prompt).Distinct().Order(StringComparer.Ordinal));
        Assert.All(rows, r => Assert.Equal(EmbeddingGemma2Prompt.Named[r.Prompt], r.PromptText));
        EmbeddingGemma2ParityChecks.TextRows(fixture.Encoder, EmbeddingGemma2TestAssets.Tokenizer, rows, Tolerances, output);
    }
}
