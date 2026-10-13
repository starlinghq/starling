using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

public sealed class Gliner2CpuFixture() : Gliner2ModelFixture(new OnnxSessionOptions());

/// <summary>
/// Profile <c>gliner2-base-v1/onnx-fp32-cpu</c>: the converted graph on the CPU provider, from text, against the
/// protected PyTorch fp32 development fixtures (accepted 2026-10-10, PR #19) with the profile's protected tolerances.yaml.
/// Skips when the graph or the fixtures are absent (CI).
/// </summary>
public class CpuParityTests(Gliner2CpuFixture fixture, ITestOutputHelper output) : IClassFixture<Gliner2CpuFixture>
{
    private static readonly Gliner2Tolerances Tolerances = Gliner2Tolerances.Load("onnx-fp32-cpu");

    [Fact]
    public void SessionIsTheLockedGraphOnCpu()
    {
        var d = fixture.Extractor.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; session {string.Join(", ", d.SessionProviders)}; model sha256 {d.ModelSha256}");
        Assert.Equal(OnnxRuntimeInfo.CpuProviderName, d.SessionProviders[0]);
        Assert.Equal(Gliner2ModelAssets.ModelSha256, d.ModelSha256);
        Assert.Equal(["input_ids", "attention_mask", "word_index", "schema_index"], d.Inputs.Select(i => i.Name));
    }

    [Fact]
    public void ParityWithTheProtectedReference()
    {
        var summary = Gliner2ModelParity.SingleCases(fixture.Extractor, fixture.Cases, Tolerances, output);
        Assert.Equal(54, summary.Cases);
        Assert.Equal(summary.Cases, summary.InputsExact);
        Gliner2ModelParity.PaddedBatches(fixture.Extractor, fixture.Cases, summary.Outputs, Tolerances, output);
    }

    [Fact]
    public void InputsOver512TokensAreRejected() =>
        Gliner2ModelParity.LongInputsAreRejected(fixture.Extractor, fixture.Cases, output);

    [Fact]
    public void PublicOffsetsStayInsideTheInputAtEveryThreshold()
    {
        var extractor = fixture.Extractor;
        int entities = 0;
        foreach (var c in fixture.Cases.Where(c => c.OutOfContract is null))
        {
            foreach (float threshold in new[] { 0f, c.Threshold })
            {
                var result = extractor.Extract(c.Text, c.LabelsRequested, threshold, TestContext.Current.CancellationToken);
                foreach (var e in result.Entities)
                {
                    Assert.InRange(e.Start, 0, c.Text.Length - 1);
                    Assert.InRange(e.End, e.Start + 1, c.Text.Length);
                    Assert.Equal(c.Text[e.Start..e.End], e.Text);
                    entities++;
                }
            }
        }

        output.WriteLine($"{entities} entities at the case thresholds and at 0.0: every offset inside the input, Text == input[Start..End]");
    }

    [Fact]
    public void ConcurrentExtractionGivesTheSequentialResults()
    {
        var extractor = fixture.Extractor;
        var cases = fixture.Cases.Where(c => c.OutOfContract is null).Take(16).ToList();
        var ct = TestContext.Current.CancellationToken;
        var sequential = cases.Select(c => extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, ct).Entities).ToList();
        var parallel = new IReadOnlyList<Gliner2Entity>[cases.Count];
        Parallel.For(0, cases.Count, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, i =>
            parallel[i] = extractor.Extract(cases[i].Text, cases[i].LabelsRequested, cases[i].Threshold, ct).Entities);
        for (int i = 0; i < cases.Count; i++)
        {
            Assert.Equal(sequential[i], parallel[i]);
        }
    }

    [Fact]
    public void CancellationDuringARunThrowsAndTheExtractorStaysUsable()
    {
        var extractor = fixture.Extractor;
        var c = fixture.Cases.First(c => c.Id == "basic-multi-label");
        var texts = Enumerable.Repeat(c.Text, 64).ToList();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        Assert.ThrowsAny<OperationCanceledException>(() => extractor.ExtractBatch(texts, c.LabelsRequested, c.Threshold, cts.Token));
        var after = extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, TestContext.Current.CancellationToken);
        Assert.Equal(c.Entities.Count, after.Entities.Count);
    }
}
