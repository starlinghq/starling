using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>
/// <see cref="MicroBatcher{TItem, TResult}"/> without a model. Synchronization is explicit: a batch function that blocks
/// on a gate holds the dispatcher, so the items queued meanwhile form the next batch deterministically. No test relies on
/// a sleep for ordering.
/// </summary>
public class MicroBatcherTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(30);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    /// <summary>
    /// A batch function that records each batch (item lists in dispatch order), maps item x to "r:x", and blocks every
    /// batch that contains a held item until <see cref="Release"/>.
    /// </summary>
    private sealed class Recorder
    {
        private readonly TaskCompletionSource _gate = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly HashSet<string> _held;
        private readonly Lock _lock = new();
        private readonly List<string[]> _batches = [];
        private readonly TaskCompletionSource _entered = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Recorder(params string[] held) => _held = [.. held];

        public Func<string, bool>? Fail { get; init; }

        public bool HeldBatchSawCancellation { get; private set; }

        /// <summary>Completes when a held batch has entered the batch function.</summary>
        public Task Entered => _entered.Task;

        public IReadOnlyList<string[]> Batches
        {
            get
            {
                lock (_lock)
                {
                    return [.. _batches];
                }
            }
        }

        public void Release() => _gate.TrySetResult();

        public IReadOnlyList<string> Run(IReadOnlyList<string> items, CancellationToken token)
        {
            lock (_lock)
            {
                _batches.Add([.. items]);
            }

            if (items.Any(_held.Contains))
            {
                _entered.TrySetResult();
                try
                {
                    _gate.Task.Wait(Timeout, token);
                }
                catch (OperationCanceledException)
                {
                    HeldBatchSawCancellation = true;
                    throw;
                }
            }

            if (Fail is not null && items.Any(Fail))
            {
                throw new InvalidDataException("batch failed");
            }

            return [.. items.Select(x => "r:" + x)];
        }
    }

    private static MicroBatcher<string, string> Create(Recorder recorder, MicroBatchingOptions? options = null, Func<string, int>? tokens = null, Func<string, object?>? key = null) =>
        new(recorder.Run, options ?? new MicroBatchingOptions(), tokens, key);

    private static async Task<T> Within<T>(Task<T> task) => await task.WaitAsync(Timeout, Ct);

    private static async Task Within(Task task) => await task.WaitAsync(Timeout, Ct);

    [Fact]
    public void OptionsHaveTheContractDefaultsAndAreValidated()
    {
        var defaults = new MicroBatchingOptions();
        Assert.Equal(TimeSpan.Zero, defaults.MaxWait);
        Assert.Equal(8, defaults.MaxRows);
        Assert.Null(defaults.MaxTokens);
        Assert.Equal(1_024, defaults.QueueCapacity);
        defaults.Validate();

        Assert.Throws<ArgumentOutOfRangeException>(() => new MicroBatchingOptions { MaxWait = TimeSpan.FromTicks(-1) }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MicroBatchingOptions { MaxWait = TimeSpan.FromDays(30) }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MicroBatchingOptions { MaxRows = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MicroBatchingOptions { MaxTokens = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new MicroBatchingOptions { QueueCapacity = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new MicroBatcher<string, string>((items, _) => items, new MicroBatchingOptions { MaxTokens = 10 }));
    }

    [Fact]
    public async Task ItemsQueuedWhileABatchRunsFormTheNextBatchInOrder()
    {
        var recorder = new Recorder("a");
        using var batcher = Create(recorder);
        var a = batcher.EnqueueAsync("a", Ct);
        await Within(recorder.Entered);
        string[] queued = ["b", "c", "d", "e", "f"];
        var rest = queued.Select(x => batcher.EnqueueAsync(x, Ct)).ToArray();
        Assert.Equal(5, batcher.QueuedCount);
        recorder.Release();

        Assert.Equal("r:a", await Within(a));
        Assert.Equal(["r:b", "r:c", "r:d", "r:e", "r:f"], await Within(Task.WhenAll(rest)));
        Assert.Equal([["a"], ["b", "c", "d", "e", "f"]], recorder.Batches);
        var stats = batcher.GetStatistics();
        Assert.Equal(2, stats.Batches);
        Assert.Equal(6, stats.Items);
        Assert.True(stats.RunTime > TimeSpan.Zero);
    }

    [Fact]
    public async Task MaxRowsSplitsTheQueueInOrder()
    {
        var recorder = new Recorder("hold");
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxRows = 4 });
        var hold = batcher.EnqueueAsync("hold", Ct);
        await Within(recorder.Entered);
        var items = Enumerable.Range(0, 10).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        var tasks = items.Select(x => batcher.EnqueueAsync(x, Ct)).ToArray();
        recorder.Release();

        Assert.Equal(items.Select(x => "r:" + x), await Within(Task.WhenAll(tasks)));
        await Within(hold);
        Assert.Equal([["hold"], ["0", "1", "2", "3"], ["4", "5", "6", "7"], ["8", "9"]], recorder.Batches);
    }

    [Fact]
    public async Task MaxTokensBoundsPaddedCellsAndAnOversizedItemRunsAlone()
    {
        // Token count = item length; padded cells = rows x longest row.
        var recorder = new Recorder("h");
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxTokens = 10 }, tokens: x => x.Length);
        var hold = batcher.EnqueueAsync("h", Ct);
        await Within(recorder.Entered);
        string[] items = ["xxx", "yyy", "zzzzz", "w", new string('o', 20), "v"];
        var tasks = items.Select(x => batcher.EnqueueAsync(x, Ct)).ToArray();
        recorder.Release();

        await Within(Task.WhenAll(tasks));
        await Within(hold);
        // xxx+yyy = 2x3 = 6; +zzzzz = 3x5 = 15 > 10. zzzzz+w = 2x5 = 10. The 20-token item exceeds the budget alone.
        Assert.Equal([["h"], ["xxx", "yyy"], ["zzzzz", "w"], [new string('o', 20)], ["v"]], recorder.Batches);
        var stats = batcher.GetStatistics();
        Assert.Equal(1 + 3 + 3 + 5 + 1 + 20 + 1, stats.Tokens);
        Assert.Equal(1 + 6 + 10 + 20 + 1, stats.TokenCells);
    }

    [Fact]
    public async Task MaxWaitHoldsTheFirstItemUntilTheBatchIsFull()
    {
        // With a wait far longer than the test, the batch is dispatched because it is full, not because time passed.
        var recorder = new Recorder();
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxWait = TimeSpan.FromHours(1), MaxRows = 3 });
        var first = batcher.EnqueueAsync("1", Ct);
        var second = batcher.EnqueueAsync("2", Ct);
        Assert.False(first.IsCompleted);
        var third = batcher.EnqueueAsync("3", Ct);

        Assert.Equal(["r:1", "r:2", "r:3"], await Within(Task.WhenAll(first, second, third)));
        Assert.Equal([["1", "2", "3"]], recorder.Batches);
    }

    [Fact]
    public async Task AnUnfilledBatchIsDispatchedWhenMaxWaitElapses()
    {
        var recorder = new Recorder();
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxWait = TimeSpan.FromMilliseconds(20), MaxRows = 8 });
        Assert.Equal("r:1", await Within(batcher.EnqueueAsync("1", Ct)));
        Assert.True(batcher.GetStatistics().QueueTime >= TimeSpan.FromMilliseconds(20));
    }

    [Fact]
    public async Task OnlyItemsWithEqualGroupKeysShareABatch()
    {
        var recorder = new Recorder("hold");
        using var batcher = Create(recorder, key: x => x[0]);
        var hold = batcher.EnqueueAsync("hold", Ct);
        await Within(recorder.Entered);
        string[] items = ["a1", "b1", "a2", "b2", "a3", "c1"];
        var tasks = items.Select(x => batcher.EnqueueAsync(x, Ct)).ToArray();
        recorder.Release();

        Assert.Equal(items.Select(x => "r:" + x), await Within(Task.WhenAll(tasks)));
        await Within(hold);
        // The oldest queued item leads each batch; other keys keep their place.
        Assert.Equal([["hold"], ["a1", "a2", "a3"], ["b1", "b2"], ["c1"]], recorder.Batches);
    }

    [Fact]
    public async Task CancellingAQueuedItemRemovesItAndLeavesTheOthers()
    {
        var recorder = new Recorder("hold");
        using var batcher = Create(recorder);
        var hold = batcher.EnqueueAsync("hold", Ct);
        await Within(recorder.Entered);
        using var cts = new CancellationTokenSource();
        var b = batcher.EnqueueAsync("b", cts.Token);
        var c = batcher.EnqueueAsync("c", Ct);
        Assert.Equal(2, batcher.QueuedCount);

        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(b));
        Assert.Equal(1, batcher.QueuedCount);
        recorder.Release();

        Assert.Equal("r:c", await Within(c));
        await Within(hold);
        Assert.Equal([["hold"], ["c"]], recorder.Batches);
        Assert.False(recorder.HeldBatchSawCancellation);
    }

    [Fact]
    public async Task CancellingADispatchedItemCancelsOnlyThatCaller()
    {
        var recorder = new Recorder("a");
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxWait = TimeSpan.FromHours(1), MaxRows = 2 });
        using var cts = new CancellationTokenSource();
        var a = batcher.EnqueueAsync("a", cts.Token);
        var b = batcher.EnqueueAsync("b", Ct);
        await Within(recorder.Entered);

        await cts.CancelAsync();
        // The cancelled caller's task completes at once, before the batch it is part of.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(a));
        Assert.False(b.IsCompleted);
        recorder.Release();

        Assert.Equal("r:b", await Within(b));
        Assert.False(recorder.HeldBatchSawCancellation);
        Assert.Equal([["a", "b"]], recorder.Batches);
    }

    [Fact]
    public async Task TheBatchTokenIsCancelledOnlyWhenEveryCallerOfTheBatchHasCancelled()
    {
        var recorder = new Recorder("a");
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxWait = TimeSpan.FromHours(1), MaxRows = 2 });
        using var ctsA = new CancellationTokenSource();
        using var ctsB = new CancellationTokenSource();
        var a = batcher.EnqueueAsync("a", ctsA.Token);
        var b = batcher.EnqueueAsync("b", ctsB.Token);
        await Within(recorder.Entered);
        await ctsA.CancelAsync();
        await ctsB.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(a));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(b));

        // The dispatcher stays usable after the cancelled run.
        Assert.Equal(["r:c", "r:d"], await Within(Task.WhenAll(batcher.EnqueueAsync("c", Ct), batcher.EnqueueAsync("d", Ct))));
        Assert.True(recorder.HeldBatchSawCancellation);
    }

    [Fact]
    public async Task AFullQueueAppliesBackpressureUntilASlotFrees()
    {
        var recorder = new Recorder("hold");
        using var batcher = Create(recorder, new MicroBatchingOptions { QueueCapacity = 2 });
        var hold = batcher.EnqueueAsync("hold", Ct);
        await Within(recorder.Entered);
        var b = batcher.EnqueueAsync("b", Ct);
        var c = batcher.EnqueueAsync("c", Ct);
        using var cts = new CancellationTokenSource();
        var waitingThenCancelled = batcher.EnqueueAsync("x", cts.Token);
        var waiting = batcher.EnqueueAsync("d", Ct);

        Assert.Equal(2, batcher.QueuedCount);
        Assert.False(waiting.IsCompleted);
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(waitingThenCancelled));
        recorder.Release();

        Assert.Equal(["r:b", "r:c", "r:d"], await Within(Task.WhenAll(b, c, waiting)));
        await Within(hold);
        Assert.DoesNotContain(recorder.Batches, batch => batch.Contains("x"));
        Assert.Equal(["hold", "b", "c", "d"], recorder.Batches.SelectMany(x => x));
    }

    [Fact]
    public async Task AFailedBatchFaultsOnlyItsOwnCallers()
    {
        var recorder = new Recorder("hold") { Fail = x => x == "bad" };
        using var batcher = Create(recorder, new MicroBatchingOptions { MaxRows = 2 });
        var hold = batcher.EnqueueAsync("hold", Ct);
        await Within(recorder.Entered);
        var bad = batcher.EnqueueAsync("bad", Ct);
        var neighbour = batcher.EnqueueAsync("n", Ct);
        var later = batcher.EnqueueAsync("later", Ct);
        recorder.Release();

        Assert.Equal("r:hold", await Within(hold));
        var error = await Assert.ThrowsAsync<InvalidDataException>(() => Within(bad));
        Assert.Same(error, await Assert.ThrowsAsync<InvalidDataException>(() => Within(neighbour)));
        Assert.Equal("r:later", await Within(later));
        Assert.Equal([["hold"], ["bad", "n"], ["later"]], recorder.Batches);
    }

    [Fact]
    public async Task AWrongResultCountFaultsTheBatch()
    {
        using var batcher = new MicroBatcher<string, string>((items, _) => [.. items.Skip(1)], new MicroBatchingOptions());
        await Assert.ThrowsAsync<InvalidOperationException>(() => Within(batcher.EnqueueAsync("a", Ct)));
    }

    [Fact]
    public async Task FailuresBeforeQueueingAreInTheTask()
    {
        using var batcher = new MicroBatcher<string, string>((items, _) => items, new MicroBatchingOptions(), tokenCount: x => x == "neg" ? -1 : throw new FormatException());
        await Assert.ThrowsAsync<FormatException>(() => batcher.EnqueueAsync("a", Ct));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => batcher.EnqueueAsync("neg", Ct));
        using var cancelled = new CancellationTokenSource();
        await cancelled.CancelAsync();
        Assert.True(batcher.EnqueueAsync("a", cancelled.Token).IsCanceled);
    }

    [Fact]
    public async Task DisposeFaultsQueuedItemsAndLetsTheRunningBatchFinish()
    {
        var recorder = new Recorder("a");
        var batcher = Create(recorder, new MicroBatchingOptions { QueueCapacity = 2 });
        var a = batcher.EnqueueAsync("a", Ct);
        await Within(recorder.Entered);
        var b = batcher.EnqueueAsync("b", Ct);
        var c = batcher.EnqueueAsync("c", Ct);
        var waitingForSlot = batcher.EnqueueAsync("d", Ct);

        var disposing = Task.Run(batcher.Dispose, Ct);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Within(b));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Within(c));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Within(waitingForSlot));
        Assert.False(disposing.IsCompleted);
        Assert.False(a.IsCompleted);

        recorder.Release();
        await Within(disposing);
        Assert.Equal("r:a", await Within(a));
        Assert.Equal([["a"]], recorder.Batches);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => batcher.EnqueueAsync("e", Ct));
        batcher.Dispose();
        await batcher.DisposeAsync();
    }

    [Fact]
    public async Task DisposeWithoutTrafficAndFromTheBatchFunctionReturns()
    {
        new MicroBatcher<string, string>((items, _) => items, new MicroBatchingOptions()).Dispose();
        await new MicroBatcher<string, string>((items, _) => items, new MicroBatchingOptions()).DisposeAsync();

        MicroBatcher<string, string>? self = null;
        self = new MicroBatcher<string, string>(
            (items, _) =>
            {
                self!.Dispose();
                return items;
            },
            new MicroBatchingOptions());
        Assert.Equal("a", await Within(self.EnqueueAsync("a", Ct)));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => self.EnqueueAsync("b", Ct));
    }

    [Fact]
    public async Task ConcurrentCallersEachGetTheirOwnResult()
    {
        using var batcher = new MicroBatcher<int, int>(
            (items, _) => [.. items.Select(x => x * 2)],
            new MicroBatchingOptions { MaxRows = 16, MaxWait = TimeSpan.FromMilliseconds(1) });
        var tasks = Enumerable.Range(0, 2_000).Select(i => Task.Run(() => batcher.EnqueueAsync(i, Ct), Ct)).ToArray();
        var results = await Within(Task.WhenAll(tasks));
        Assert.Equal(Enumerable.Range(0, 2_000).Select(i => i * 2), results);
        var stats = batcher.GetStatistics();
        Assert.Equal(2_000, stats.Items);
        Assert.InRange(stats.Batches, 2_000 / 16, 2_000);
    }
}
