using Starlings.TestSupport;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>
/// The opt-in <c>mallopt(M_MMAP_THRESHOLD, 65536)</c> (issue #178). On Linux CI (glibc) the first test exercises the
/// real <c>mallopt</c> call and the loud diagnostic; elsewhere the opt-in must be a documented no-op. Allocator
/// behavior itself (whether growth flattens) is judged by the soak rule, not by tests.
/// </summary>
public class GlibcMallocThresholdTests
{
    [Fact]
    public void OptInIsOffByDefaultAndLeavesNoDiagnostic()
    {
        Assert.False(new OnnxSessionOptions().ApplyGlibcMmapThreshold);

        using var session = OnnxModelSession.Create(TestModels.Add);
        Assert.Empty(session.Diagnostics.GlibcMmapThresholdDiagnostic);
    }

    [Fact]
    public void OptInAppliesOncePerProcessWithTheLoudDiagnostic()
    {
        // The opt-in is cached per process, so another test may have applied it first; only a fresh process logs.
        bool wasFresh = GlibcMallocThreshold.Applied is null;
        var logged = new List<string>();
        var outcome = GlibcMallocThreshold.Apply(logged.Add);

        if (OperatingSystem.IsLinux())
        {
            // glibc: the call was made (musl would leave Applied false with an empty description).
            Assert.True(outcome.Applied, outcome.Description);
            Assert.StartsWith("[Starlings.Runtime.Onnx]", outcome.Description, StringComparison.Ordinal);
            Assert.Contains("M_MMAP_THRESHOLD", outcome.Description, StringComparison.Ordinal);
            Assert.Contains("65536", outcome.Description, StringComparison.Ordinal);
            if (wasFresh)
            {
                Assert.Equal(outcome.Description, Assert.Single(logged));
            }

            // Idempotent: later calls return the first outcome and log nothing further.
            var second = GlibcMallocThreshold.Apply(logged.Add);
            Assert.Equal(outcome, second);
            Assert.Equal(wasFresh ? 1 : 0, logged.Count);
            Assert.True(GlibcMallocThreshold.Applied);
        }
        else
        {
            // Documented no-op off glibc.
            Assert.False(outcome.Applied);
            Assert.Empty(outcome.Description);
            Assert.Empty(logged);
        }
    }

    [Fact]
    public void OptInFlowsThroughSessionCreationAndIsReported()
    {
        var options = new OnnxSessionOptions
        {
            ApplyGlibcMmapThreshold = true,
            IntraOpNumThreads = 1,
        };
        using var session = OnnxModelSession.Create(TestModels.Add, options);
        string diagnostic = session.Diagnostics.GlibcMmapThresholdDiagnostic;

        if (OperatingSystem.IsLinux())
        {
            Assert.NotEmpty(diagnostic);
            Assert.Contains("M_MMAP_THRESHOLD", diagnostic, StringComparison.Ordinal);
        }
        else
        {
            Assert.Empty(diagnostic);
        }

        // A session without the option never reports a diagnostic, whatever the process state.
        using var plain = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { IntraOpNumThreads = 1 });
        Assert.Empty(plain.Diagnostics.GlibcMmapThresholdDiagnostic);
    }
}
