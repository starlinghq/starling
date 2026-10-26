using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// <see cref="Gliner2ExtractorOptions.ExecutionScheduler"/> (Experimental) on the weight-free GLiNER2 graph with the
/// pinned tokenizer.json (skips without it): null keeps <c>ExtractAsync</c> inline, a scheduler runs it there with the
/// same entities.
/// </summary>
public sealed class ExecutionSchedulerTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };
    private static readonly string[] PersonCity = ["person", "city"];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Gliner2EntityExtractor Fake(TaskScheduler? scheduler)
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        return Gliner2EntityExtractor.Create(
            OnnxModelSession.Create(TestModels.FakeGliner2, Fast),
            tokenizer!,
            new Gliner2ExtractorOptions { ExecutionScheduler = scheduler },
            ownsSession: true);
    }

    [Fact]
    public async Task ExtractAsyncRunsOnTheSchedulerWithTheSameEntities()
    {
        Assert.Null(new Gliner2ExtractorOptions().ExecutionScheduler);
        var scheduler = new CountingTaskScheduler();
        using var inline = Fake(null);
        using var scheduled = Fake(scheduler);
        const string text = "Ann met Bob in Paris.";

        var inlineTask = inline.ExtractAsync(text, PersonCity, cancellationToken: Ct);
        Assert.True(inlineTask.IsCompletedSuccessfully);
        var result = await scheduled.ExtractAsync(text, PersonCity, cancellationToken: Ct);
        Assert.Equal(1, scheduler.Count);
        Assert.Equal((await inlineTask).Entities, result.Entities);

        // Failures and cancellation are in the task.
        Assert.True(await scheduled.ExtractAsync("x", ["", "a"], cancellationToken: Ct).ContinueWith(t => t.IsFaulted, TaskScheduler.Default));
        var cancelled = scheduled.ExtractAsync(text, PersonCity, cancellationToken: new CancellationToken(canceled: true));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        Assert.True(cancelled.IsCanceled);
    }

    [Fact]
    public void TheCapabilityIsExperimental() =>
        Assert.Equal(Status.Experimental, Gliner2Capabilities.Statuses["execution_scheduler"]);
}
