using System.Collections.Concurrent;
using System.Text.Json;
using Starlings.Models.Kev;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The model against Kev's PyTorch fp32 path (Data/probs-*.json, make_dotnet_cases.py probs) on the request goldens:
/// probabilities within the profile bound (2e-4, protected/kev-0.8b/onnx-fp32/tolerances.yaml; proposed for 4B), no
/// argmax flip, cached equal to uncached within 1e-5. Skips without a bundle (see <see cref="KevTestAssets"/>).
/// </summary>
public sealed class ModelParityTests
{
    private const double ProbabilityBound = 2.0e-4;
    private const double CachedBound = 1.0e-5;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly ConcurrentDictionary<KevModel, Lazy<(KevDecisionModel? Model, string Reason)>> Models = new();

    internal static (KevDecisionModel? Model, string Reason) Load(KevModel model) => Models.GetOrAdd(model, m => new(() =>
    {
        var (files, reason) = KevTestAssets.FindBundle(m);
        if (files is null)
        {
            return (null, reason);
        }

        // The bundle was size-checked; the session checks the graph digest against the lock.
        return (KevDecisionModel.Create(files, new KevOptions { Session = new OnnxSessionOptions { IntraOpNumThreads = 4 } }), string.Empty);
    })).Value;

    public static TheoryData<KevModel, string> Cases()
    {
        var data = new TheoryData<KevModel, string>();
        foreach (var model in new[] { KevModel.Kev08B, KevModel.Kev4B })
        {
            foreach (var c in KevTestAssets.ReadData("requests.json").EnumerateArray())
            {
                data.Add(model, c.GetProperty("id").GetString()!);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesPyTorchReference(KevModel model, string id)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        string probsFile = $"probs-{KevModelProfile.For(model).Id.Split('/')[0]}.json";
        Assert.SkipUnless(File.Exists(Path.Combine(KevTestAssets.DataDirectory, probsFile)), $"{probsFile} not generated");
        var request = KevTestAssets.ReadData("requests.json").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);
        var expected = KevTestAssets.ReadData(probsFile).GetProperty("cases").EnumerateArray()
            .Single(c => c.GetProperty("id").GetString() == id).GetProperty("probs").EnumerateArray()
            .Select(q => q.EnumerateArray().Select(p => p.GetDouble()).ToArray()).ToArray();

        var decision = kev!.Decide(KevRequest.ParseSystemOne(request.GetProperty("request").GetString()!), Ct);
        Assert.Equal(expected.Length, decision.Answers.Count);
        for (int q = 0; q < expected.Length; q++)
        {
            var actual = decision.Answers[q].ProbabilityList;
            Assert.Equal(expected[q].Length, actual.Count);
            double max = expected[q].Zip(actual, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(max <= ProbabilityBound, $"{id} question {q}: max abs {max:G3} > {ProbabilityBound:G3}");
            int refArg = Array.IndexOf(expected[q], expected[q].Max());
            var sorted = expected[q].OrderDescending().ToArray();
            if (sorted.Length < 2 || sorted[0] - sorted[1] >= 1e-4)
            {
                Assert.Equal(refArg, actual.ToList().IndexOf(actual.Max()));
            }
        }
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CachedEqualsUncached(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var request = KevTestAssets.ReadData("requests.json").EnumerateArray().Single(c => c.GetProperty("id").GetString() == "many-questions");
        var encoding = kev!.Encode(KevRequest.ParseSystemOne(request.GetProperty("request").GetString()!).ToRecord());
        var cached = kev.Run(encoding, Ct);
        var uncached = kev.RunUncached(encoding, Ct);
        double max = cached.Zip(uncached, (a, b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Max()).Max();
        Assert.True(max <= CachedBound, $"cached vs uncached {max:G3} > {CachedBound:G3}");
        // The cache is read, never written: running again from the same encoding gives the same bits.
        Assert.Equal(cached.SelectMany(p => p), kev.Run(encoding, Ct).SelectMany(p => p));
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void QuestionsAreIsolated(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var q1 = KevQuestion.Noul("Is this about billing?");
        var q2 = KevQuestion.Choice("Which team?", "billing", "shipping", "returns");
        const string state = "I was charged twice for order 1182.";
        var alone = kev!.Decide(new KevRequest(state, [new("a", q1)]), Ct);
        var together = kev.Decide(new KevRequest(state, [new("b", q2), new("a", q1)]), Ct);
        Assert.Equal(alone["a"].ProbabilityList, together["a"].ProbabilityList);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CancelledCallThrowsAndModelStaysUsable(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var request = new KevRequest("text", [new("q", KevQuestion.Noul("Is it text?"))]);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => kev!.Decide(request, cts.Token));
        Assert.Equal(2, kev!.Decide(request, Ct).Answers[0].ProbabilityList.Count);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void RawTemperatureSharpensOrFlattensButKeepsOrder(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var (files, _) = KevTestAssets.FindBundle(model);
        using var raw = KevDecisionModel.Create(files!, new KevOptions { Temperature = 1.0, Session = new OnnxSessionOptions { IntraOpNumThreads = 4 } });
        var request = new KevRequest("I was charged twice for order 1182.", [new("t", KevQuestion.Choice("Which team?", "billing", "shipping", "returns"))]);
        var calibrated = kev!.Decide(request, Ct)["t"];
        var uncalibrated = raw.Decide(request, Ct)["t"];
        Assert.Equal(1.0, raw.Temperature);
        Assert.Equal(calibrated.Choice, uncalibrated.Choice);
        Assert.True(uncalibrated.ProbabilityList.Max() >= calibrated.ProbabilityList.Max()); // T > 1 flattens
    }
}
