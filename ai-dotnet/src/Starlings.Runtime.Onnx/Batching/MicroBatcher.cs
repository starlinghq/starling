using System.Diagnostics;

namespace Starlings.Runtime.Onnx.Batching;

/// <summary>
/// Cross-request micro-batching: items queued by concurrent callers are coalesced into batches of at most
/// <see cref="MicroBatchingOptions.MaxRows"/> items and <see cref="MicroBatchingOptions.MaxTokens"/> padded token cells,
/// one batch function call runs each batch, and every caller's task completes with its own item's result.
/// Experimental.
/// </summary>
/// <typeparam name="TItem">One caller's input, already validated (see remarks).</typeparam>
/// <typeparam name="TResult">The result of one item.</typeparam>
/// <remarks>
/// <para>
/// <b>Dispatch.</b> One dispatcher thread, started on the first <see cref="EnqueueAsync"/>, runs one batch at a time.
/// A batch starts with the oldest queued item and takes, in queue order, the later items with an equal group key (see
/// the constructor) until <see cref="MicroBatchingOptions.MaxRows"/> items or the token budget is reached; items with
/// another key keep their place for a later batch. A batch is dispatched as soon as it is full, or when the oldest
/// item has waited <see cref="MicroBatchingOptions.MaxWait"/>. Items queued while a batch runs form the next one.
/// </para>
/// <para>
/// <b>Validation belongs before the queue.</b> An exception from the batch function faults every caller of that batch
/// (with the same exception instance); the batch is not retried item by item. Callers therefore validate each item
/// (token limits, ids) before enqueueing it, so an invalid request fails on its own and never fails its neighbours.
/// The model adapters do that. Results are matched to items by position; a batch function that returns a different
/// number of results faults the batch with <see cref="InvalidOperationException"/>.
/// </para>
/// <para>
/// <b>Cancellation.</b> The batch function receives a token owned by the batcher, never a caller's. A caller whose
/// token is cancelled while its item is queued has the item removed; once its batch is dispatched, its task is
/// cancelled at once and its result discarded when the batch completes. Other callers in the batch are unaffected:
/// the batch token is cancelled only when every caller of the batch has cancelled.
/// </para>
/// <para>
/// <b>Backpressure.</b> At most <see cref="MicroBatchingOptions.QueueCapacity"/> items wait in the queue. When it is
/// full, <see cref="EnqueueAsync"/> returns a task that completes only after a slot frees up and the item has run; its
/// cancellation token also cancels that wait.
/// </para>
/// <para>
/// <b>Disposal.</b> <see cref="Dispose"/> and <see cref="DisposeAsync"/> fault every queued item and every caller
/// still waiting for a queue slot with <see cref="ObjectDisposedException"/>, let a running batch finish (its callers
/// get their results), and return when the dispatcher thread has exited. Later calls to <see cref="EnqueueAsync"/>
/// fault with <see cref="ObjectDisposedException"/>. Disposal is idempotent. Disposing from inside the batch function
/// does not wait for the dispatcher.
/// </para>
/// <para>
/// <b>Reproducibility.</b> A caller's item runs padded next to whatever other traffic shares its batch, so its result
/// depends on that traffic within the model's padded-batch tolerance and is not bit for bit reproducible.
/// </para>
/// </remarks>
public sealed class MicroBatcher<TItem, TResult> : IDisposable, IAsyncDisposable
{
    private readonly Func<IReadOnlyList<TItem>, CancellationToken, IReadOnlyList<TResult>> _runBatch;
    private readonly Func<TItem, int>? _tokenCount;
    private readonly Func<TItem, object?>? _groupKey;
    private readonly MicroBatchingOptions _options;
    private readonly long _maxWaitTicks;
    private readonly object _gate = new();
    private readonly LinkedList<Entry> _queue = new();
    private readonly SemaphoreSlim _capacity;
    private readonly CancellationTokenSource _disposing = new();
    private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private Thread? _thread;
    private bool _disposed;
    private long _batches;
    private long _items;
    private long _tokenCells;
    private long _tokens;
    private long _queueTicks;
    private long _runTicks;

    /// <summary>Creates a batcher. Nothing runs until the first item is queued.</summary>
    /// <param name="runBatch">
    /// Runs one batch: receives the items in queue order and a batcher-owned token, returns one result per item in the
    /// same order. Called on the dispatcher thread, one batch at a time.
    /// </param>
    /// <param name="options">Limits; validated here.</param>
    /// <param name="tokenCount">
    /// Token count of an item, for <see cref="MicroBatchingOptions.MaxTokens"/> and the statistics; null counts 0 per
    /// item (then <see cref="MicroBatchingOptions.MaxTokens"/> must be null). Called once per item, on the caller's thread.
    /// </param>
    /// <param name="groupKey">
    /// Key of an item; only items with equal keys (<see cref="object.Equals(object, object)"/>) share a batch. Null:
    /// every item may share a batch with any other. Called once per item, on the caller's thread.
    /// </param>
    /// <exception cref="ArgumentException"><see cref="MicroBatchingOptions.MaxTokens"/> is set without a token counter.</exception>
    public MicroBatcher(
        Func<IReadOnlyList<TItem>, CancellationToken, IReadOnlyList<TResult>> runBatch,
        MicroBatchingOptions options,
        Func<TItem, int>? tokenCount = null,
        Func<TItem, object?>? groupKey = null)
    {
        ArgumentNullException.ThrowIfNull(runBatch);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        if (options.MaxTokens is not null && tokenCount is null)
        {
            throw new ArgumentException("MaxTokens needs a token counter.", nameof(tokenCount));
        }

        _runBatch = runBatch;
        _options = options;
        _tokenCount = tokenCount;
        _groupKey = groupKey;
        _maxWaitTicks = (long)(options.MaxWait.Ticks * (Stopwatch.Frequency / (double)TimeSpan.TicksPerSecond));
        _capacity = new SemaphoreSlim(options.QueueCapacity, options.QueueCapacity);
    }

    /// <summary>Limits in effect.</summary>
    public MicroBatchingOptions Options => _options;

    /// <summary>Items currently waiting to be dispatched.</summary>
    public int QueuedCount
    {
        get
        {
            lock (_gate)
            {
                return _queue.Count;
            }
        }
    }

    /// <summary>Counters since creation (batches, items, padding, queue and run time).</summary>
    public MicroBatcherStatistics GetStatistics()
    {
        lock (_gate)
        {
            return new MicroBatcherStatistics(
                _batches, _items, _tokenCells, _tokens, Stopwatch.GetElapsedTime(0, _queueTicks), Stopwatch.GetElapsedTime(0, _runTicks));
        }
    }

    /// <summary>
    /// Queues one item and returns a task for its result. Never throws synchronously: failures (a disposed batcher, a
    /// throwing token counter or group key, a failed batch) are in the task, and a cancelled token cancels it.
    /// </summary>
    /// <param name="item">The item, already validated.</param>
    /// <param name="cancellationToken">Cancels this caller only: removes a queued item, or discards the result of a dispatched one.</param>
    public Task<TResult> EnqueueAsync(TItem item, CancellationToken cancellationToken = default)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled<TResult>(cancellationToken);
        }

        Entry entry;
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            int tokens = _tokenCount?.Invoke(item) ?? 0;
            ArgumentOutOfRangeException.ThrowIfNegative(tokens, "tokenCount(item)");
            entry = new Entry(item, tokens, _groupKey?.Invoke(item));
        }
#pragma warning disable CA1031 // Every failure is reported through the returned task.
        catch (Exception ex)
#pragma warning restore CA1031
        {
            return Task.FromException<TResult>(ex);
        }

        return _capacity.Wait(0, CancellationToken.None) ? Add(entry, cancellationToken) : WaitForSlotThenAddAsync(entry, cancellationToken);
    }

    /// <summary>Faults queued items, lets a running batch finish and waits for the dispatcher to exit. Idempotent.</summary>
    public void Dispose()
    {
        Stop();
        Thread? thread = Volatile.Read(ref _thread);
        if (thread is not null && thread != Thread.CurrentThread)
        {
            thread.Join();
        }
    }

    /// <summary>As <see cref="Dispose"/>, waiting for the dispatcher asynchronously.</summary>
    public async ValueTask DisposeAsync()
    {
        Stop();
        Thread? thread = Volatile.Read(ref _thread);
        if (thread is not null && thread != Thread.CurrentThread)
        {
            await _stopped.Task.ConfigureAwait(false);
        }
    }

    private async Task<TResult> WaitForSlotThenAddAsync(Entry entry, CancellationToken cancellationToken)
    {
        using (var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _disposing.Token))
        {
            try
            {
                await _capacity.WaitAsync(linked.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                throw new ObjectDisposedException(GetType().FullName, "The micro-batcher was disposed while the item waited for a queue slot.");
            }
        }

        return await Add(entry, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Adds an entry that already holds a queue slot.</summary>
    private Task<TResult> Add(Entry entry, CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            if (_disposed)
            {
                _capacity.Release();
                return Task.FromException<TResult>(new ObjectDisposedException(GetType().FullName));
            }

            entry.EnqueuedAt = Stopwatch.GetTimestamp();
            entry.Node = _queue.AddLast(entry);
            if (_thread is null)
            {
                var thread = new Thread(DispatchLoop) { IsBackground = true, Name = "Starlings micro-batcher" };
                thread.Start();
                Volatile.Write(ref _thread, thread);
            }

            Monitor.Pulse(_gate);
        }

        if (!cancellationToken.CanBeCanceled)
        {
            return entry.Task;
        }

        var registration = cancellationToken.UnsafeRegister(static (state, token) =>
        {
            var (batcher, e) = ((MicroBatcher<TItem, TResult>, Entry))state!;
            batcher.CancelEntry(e, token);
        }, (this, entry));
        return AwaitAndUnregisterAsync(entry.Task, registration);
    }

    private static async Task<TResult> AwaitAndUnregisterAsync(Task<TResult> task, CancellationTokenRegistration registration)
    {
        try
        {
            return await task.ConfigureAwait(false);
        }
        finally
        {
            await registration.DisposeAsync().ConfigureAwait(false);
        }
    }

    private void CancelEntry(Entry entry, CancellationToken token)
    {
        bool releaseSlot = false;
        lock (_gate)
        {
            if (!entry.TrySetCanceled(token))
            {
                return;
            }

            if (entry.Node is not null)
            {
                _queue.Remove(entry.Node);
                entry.Node = null;
                releaseSlot = true;
            }
            else if (entry.Batch is { Done: false } batch && --batch.Live == 0)
            {
                // Every caller of the running batch has cancelled: nobody needs its results.
                batch.Cancellation.Cancel();
            }
        }

        if (releaseSlot)
        {
            _capacity.Release();
        }
    }

    /// <summary>Marks the batcher disposed and faults the queue.</summary>
    private void Stop()
    {
        List<Entry> pending;
        lock (_gate)
        {
            if (!_disposed)
            {
                _disposed = true;
                pending = [.. _queue];
                _queue.Clear();
                foreach (var entry in pending)
                {
                    entry.Node = null;
                }

                Monitor.PulseAll(_gate);
            }
            else
            {
                pending = [];
            }

        }

        _disposing.Cancel();
        foreach (var entry in pending)
        {
            entry.TrySetException(new ObjectDisposedException(GetType().FullName, "The micro-batcher was disposed before the item was dispatched."));
        }

        if (pending.Count > 0)
        {
            _capacity.Release(pending.Count);
        }
    }

    private void DispatchLoop()
    {
        var batch = new List<Entry>(_options.MaxRows);
        try
        {
            while (true)
            {
                Batch running;
                int width;
                lock (_gate)
                {
                    while (true)
                    {
                        if (_disposed)
                        {
                            return;
                        }

                        if (_queue.First is null)
                        {
                            Monitor.Wait(_gate);
                            continue;
                        }

                        long wait = Collect(batch, out width);
                        if (wait <= 0)
                        {
                            break;
                        }

                        batch.Clear();
                        // Monitor.Wait truncates to whole milliseconds; round up so a sub-millisecond remainder sleeps
                        // instead of spinning on the lock.
                        Monitor.Wait(_gate, (int)Math.Max(1, Math.Ceiling(Stopwatch.GetElapsedTime(0, wait).TotalMilliseconds)));
                    }

                    long now = Stopwatch.GetTimestamp();
                    running = new Batch(batch.Count);
                    foreach (var entry in batch)
                    {
                        _queue.Remove(entry.Node!);
                        entry.Node = null;
                        entry.Batch = running;
                        _queueTicks += now - entry.EnqueuedAt;
                        _tokens += entry.Tokens;
                    }

                    _batches++;
                    _items += batch.Count;
                    _tokenCells += (long)width * batch.Count;
                }

                _capacity.Release(batch.Count);
                Run(batch, running);
                batch.Clear();
            }
        }
        finally
        {
            _stopped.TrySetResult();
        }
    }

    /// <summary>
    /// Group-key equality that never throws on the dispatcher thread: a key whose <see cref="object.Equals(object)"/>
    /// throws is treated as unequal, so its item runs in a batch it leads (alone).
    /// </summary>
    private static bool SameKey(object? a, object? b)
    {
        try
        {
            return Equals(a, b);
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Fills <paramref name="batch"/> from the queue (under the lock) and returns 0 when it should be dispatched now, or
    /// the Stopwatch ticks until the oldest item's wait expires.
    /// </summary>
    private long Collect(List<Entry> batch, out int width)
    {
        var first = _queue.First!.Value;
        object? key = first.Key;
        int maxRows = _options.MaxRows;
        long maxCells = _options.MaxTokens ?? long.MaxValue;
        bool full = false;
        width = 0;
        for (var node = _queue.First; node is not null; node = node.Next)
        {
            var entry = node.Value;
            if (batch.Count > 0 && !SameKey(entry.Key, key))
            {
                continue;
            }

            int newWidth = Math.Max(width, entry.Tokens);
            if (batch.Count > 0 && (long)(batch.Count + 1) * newWidth > maxCells)
            {
                full = true;
                break;
            }

            batch.Add(entry);
            width = newWidth;
            if (batch.Count == maxRows)
            {
                full = true;
                break;
            }
        }

        if (full)
        {
            return 0;
        }

        long remaining = first.EnqueuedAt + _maxWaitTicks - Stopwatch.GetTimestamp();
        return Math.Max(remaining, 0);
    }

    private void Run(List<Entry> batch, Batch running)
    {
        var items = new TItem[batch.Count];
        for (int i = 0; i < items.Length; i++)
        {
            items[i] = batch[i].Item;
        }

        IReadOnlyList<TResult>? results = null;
        Exception? error = null;
        long started = Stopwatch.GetTimestamp();
        try
        {
            results = _runBatch(items, running.Cancellation.Token);
            if (results is null || results.Count != items.Length)
            {
                error = new InvalidOperationException($"The batch function returned {results?.Count.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "null"} results for {items.Length} items.");
            }
        }
#pragma warning disable CA1031 // A failed batch faults its callers; the dispatcher keeps serving the queue.
        catch (Exception ex)
#pragma warning restore CA1031
        {
            error = ex;
        }

        long elapsed = Stopwatch.GetTimestamp() - started;
        lock (_gate)
        {
            running.Done = true;
            _runTicks += elapsed;
        }

        running.Cancellation.Dispose();
        for (int i = 0; i < batch.Count; i++)
        {
            var entry = batch[i];
            if (error is OperationCanceledException && running.Cancellation.IsCancellationRequested)
            {
                entry.TrySetCanceled();
            }
            else if (error is not null)
            {
                entry.TrySetException(error);
            }
            else
            {
                entry.TrySetResult(results![i]);
            }
        }
    }

    private sealed class Entry(TItem item, int tokens, object? key) : TaskCompletionSource<TResult>(TaskCreationOptions.RunContinuationsAsynchronously)
    {
        public TItem Item { get; } = item;

        public int Tokens { get; } = tokens;

        public object? Key { get; } = key;

        public long EnqueuedAt { get; set; }

        /// <summary>Non-null while queued (guarded by the batcher's lock).</summary>
        public LinkedListNode<Entry>? Node { get; set; }

        /// <summary>Set at dispatch (guarded by the batcher's lock).</summary>
        public Batch? Batch { get; set; }
    }

    private sealed class Batch(int live)
    {
        public CancellationTokenSource Cancellation { get; } = new();

        /// <summary>Callers of the batch that have not cancelled (guarded by the batcher's lock).</summary>
        public int Live { get; set; } = live;

        /// <summary>The batch function has returned (guarded by the batcher's lock).</summary>
        public bool Done { get; set; }
    }
}
