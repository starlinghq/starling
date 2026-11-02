using System.Collections.Concurrent;
using Starlings.Models.Kev;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.Kev.Tests.Cuda;

internal static class CudaGate
{
    public const string OptIn = "STARLING_CUDA_TESTS";

    /// <summary>Null when CUDA tests should run; otherwise the skip reason. Throws on an unsafe opt-in.</summary>
    public static string? Check()
    {
        if (Environment.GetEnvironmentVariable(OptIn) != "1")
        {
            return $"CUDA tests are opt-in: set {OptIn}=1, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the GPU to use.";
        }

        // Never let the CUDA runtime pick a default device on a shared workstation.
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")) ||
            Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER") != "PCI_BUS_ID")
        {
            throw new InvalidOperationException($"{OptIn}=1 requires CUDA_DEVICE_ORDER=PCI_BUS_ID and an explicit CUDA_VISIBLE_DEVICES.");
        }

        return null;
    }
}

/// <summary>
/// The Kev bundles on the CUDA execution provider (<c>use_tf32=0</c>, forced by Runtime.Onnx): profiles
/// <c>kev-0.8b/onnx-fp32-cuda</c> and <c>kev-4b/onnx-fp32-cuda</c>, against Kev's PyTorch fp32 probabilities with the
/// CPU profiles' bounds (the protected 0.8B tolerances state they apply to the CUDA provider with TF32 off). One model
/// is loaded at a time: the 4B weights take about 17 GB of device memory.
/// </summary>
public sealed class CudaParityTests(ITestOutputHelper output)
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly ConcurrentDictionary<KevModel, Lazy<(KevDecisionModel? Model, string Reason)>> Models = new();
    private static readonly Lock LoadLock = new();

    private static (KevDecisionModel? Model, string Reason) Load(KevModel model)
    {
        if (CudaGate.Check() is { } skip)
        {
            return (null, skip);
        }

        lock (LoadLock)
        {
            // Release the other size first so both never hold device memory at once.
            foreach (var other in Models.Where(kv => kv.Key != model && kv.Value.IsValueCreated))
            {
                other.Value.Value.Model?.Dispose();
                Models.TryRemove(other.Key, out _);
            }

            return Models.GetOrAdd(model, m => new(() =>
            {
                var (files, reason) = KevTestAssets.FindBundle(m);
                return files is null
                    ? (null, reason)
                    : (KevDecisionModel.Create(files, new KevOptions { Session = CudaSession }), string.Empty);
            })).Value;
        }
    }

    private static OnnxSessionOptions CudaSession => new() { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };

    public static TheoryData<KevModel> Sizes() => [KevModel.Kev08B, KevModel.Kev4B];

    [Theory]
    [MemberData(nameof(Sizes))]
    public void SessionUsesCudaWithTf32DisabledAndSerializes(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var d = kev!.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; session {string.Join(", ", d.SessionProviders)}; CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
        output.WriteLine("provider options: " + string.Join(", ", d.ProviderOptions.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}")));
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
        Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        Assert.Equal(KevModelAssets.ModelSha256(model), d.ModelSha256);
        Assert.Equal(KevModelProfile.For(model).CudaId, kev.ProfileId);
        Assert.True(kev.SerializesNativeCalls);
    }

    /// <summary>Every request golden within the profile bound and with the reference argmax (outside near ties).</summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void MatchesPyTorchReference(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        string? missing = KevParity.MissingReference(model);
        Assert.SkipWhen(missing is not null, missing ?? string.Empty);
        double worst = 0;
        int cases = 0;
        foreach (string id in KevParity.CaseIds())
        {
            double max = KevParity.AssertMatchesReference(kev!, id, Ct);
            output.WriteLine($"{id}: max abs {max:G3}");
            worst = Math.Max(worst, max);
            cases++;
        }

        output.WriteLine($"{KevModelProfile.For(model).CudaId}: {cases} goldens, worst max abs {worst:G3} (bound {KevParity.ProbabilityBound:G3})");
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void CachedEqualsUncached(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        output.WriteLine($"cached vs uncached max abs {KevParity.AssertCachedEqualsUncached(kev!, Ct):G3}");
    }

    [Theory]
    [MemberData(nameof(Sizes))]
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

    /// <summary>Capability <c>concurrent_calls</c> on CUDA (serialized, the default): four callers, bit-identical to sequential.</summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void ConcurrentSerializedCallsEqualSequential(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        int n = KevParity.AssertConcurrentEqualsSequential(kev!, callers: 4, rounds: 2, Ct);
        output.WriteLine($"{n} concurrent decisions equal to sequential");
    }

    /// <summary>
    /// Capability <c>state_reuse_across_calls.cuda</c>: on one CUDA session, every request golden decided from a
    /// <see cref="KevState"/> equals <c>Decide(request)</c> on the same session bit for bit, twice in a row and question by
    /// question (the CPU test <c>ApiTests.StateReuseEqualsDecideBitForBit</c> on the CUDA provider).
    /// </summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void StateReuseEqualsDecideBitForBit(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        int requests = 0, compared = 0;
        foreach (string id in KevParity.CaseIds())
        {
            var request = KevParity.Request(id);
            var expected = kev!.Decide(request, Ct);
            using var state = kev.CreateState(request.State, Ct);
            Assert.Equal(expected.StateTokens, state.TokenCount);
            for (int round = 0; round < 2; round++)
            {
                var actual = kev.Decide(state, request.Questions, Ct);
                Assert.Equal(expected.StateTokens, actual.StateTokens);
                Assert.Equal(expected.BranchTokens, actual.BranchTokens);
                Assert.Equal(expected.Answers.Select(a => a.Id), actual.Answers.Select(a => a.Id));
                for (int i = 0; i < expected.Answers.Count; i++)
                {
                    Assert.Equal(expected.Answers[i].ProbabilityList, actual.Answers[i].ProbabilityList);
                    compared++;
                }
            }

            // A different question set on the same state: each question alone.
            foreach (var question in request.Questions)
            {
                var alone = kev.Decide(state, [question], Ct);
                Assert.Equal(expected[question.Key].ProbabilityList, alone[question.Key].ProbabilityList);
                compared++;
            }

            requests++;
        }

        output.WriteLine($"{KevModelProfile.For(model).CudaId}: {requests} goldens from a KevState, {compared} answers bit-identical to Decide(request) on the same session");
    }

    /// <summary>
    /// One <see cref="KevState"/> shared by four concurrent callers (serialized, the CUDA default) gives the sequential
    /// answers bit for bit, a cancelled call leaves it usable, and it is refused after dispose.
    /// </summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void StateIsSharedByConcurrentCallsAndRefusedAfterDispose(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        const string text = "I was charged twice for order 1182. Please refund one of the charges.";
        KeyValuePair<string, KevQuestion>[] questions =
        [
            new("team", KevQuestion.Choice("Which team should handle this?", "billing", "shipping", "returns")),
            new("urgent", KevQuestion.Noul("Does this need a reply today?")),
            new("severity", KevQuestion.Score("How severe is the problem?", "minor", "moderate", "serious")),
        ];
        var expected = kev!.Decide(new KevRequest(text, questions), Ct);
        var state = kev.CreateState(text, Ct);
        var results = new KevDecision[8];
        Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = Ct }, i =>
            results[i] = kev.Decide(state, questions, Ct));
        Assert.All(results, r =>
        {
            for (int i = 0; i < questions.Length; i++)
            {
                Assert.Equal(expected.Answers[i].ProbabilityList, r.Answers[i].ProbabilityList);
            }
        });

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => kev.Decide(state, questions, cts.Token));
        Assert.Equal(expected["urgent"].ProbabilityList, kev.Decide(state, [questions[1]], Ct)["urgent"].ProbabilityList);

        state.Dispose();
        state.Dispose();
        Assert.True(state.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => kev.Decide(state, questions, Ct));
        output.WriteLine($"{results.Length} decisions from one state on 4 callers bit-identical to Decide(request) on the same session");
    }

    /// <summary>
    /// Cancelled calls leave no device work queued (<c>docs/evidence/gliner2-cuda-wrong-output/</c>), and the model
    /// answers the goldens within the bound afterwards.
    /// </summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void CancelledCallsLeaveNoQueuedWorkAndModelStaysCorrect(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var request = KevParity.Request("many-questions");
        var probe = CudaQueueProbe.CancelledCalls(token => kev!.Decide(request, token), cancellations: 10);
        output.WriteLine(probe.ToString());
        Assert.NotEmpty(probe.Drains);
        Assert.Equal(0, probe.Pending);
        foreach (string id in KevParity.CaseIds())
        {
            KevParity.AssertMatchesReference(kev!, id, Ct);
        }
    }
}
