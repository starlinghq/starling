namespace Starlings.TestSupport;

/// <summary>
/// A <see cref="TaskScheduler"/> for the execution-scheduler tests: runs each task on a dedicated thread and records
/// how many tasks it ran and on which managed threads.
/// </summary>
internal sealed class CountingTaskScheduler : TaskScheduler
{
    private int _count;

    /// <summary>Tasks run so far.</summary>
    public int Count => Volatile.Read(ref _count);

    /// <summary>Managed thread ids the tasks ran on.</summary>
    public System.Collections.Concurrent.ConcurrentBag<int> Threads { get; } = [];

    protected override IEnumerable<Task> GetScheduledTasks() => [];

    protected override void QueueTask(Task task)
    {
        var thread = new Thread(() =>
        {
            Interlocked.Increment(ref _count);
            Threads.Add(Environment.CurrentManagedThreadId);
            TryExecuteTask(task);
        })
        { IsBackground = true };
        thread.Start();
    }

    protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued) => false;
}
