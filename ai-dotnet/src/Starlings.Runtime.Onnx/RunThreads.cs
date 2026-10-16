using System.Runtime.ExceptionServices;

namespace Starlings.Runtime.Onnx;

/// <summary>
/// A small process-wide set of long-lived threads that execute ONNX Runtime runs for sessions created with
/// <see cref="OnnxSessionOptions.DedicatedRunThreads"/>. The caller blocks until its run completes on one of them.
/// </summary>
/// <remarks>
/// <para>
/// Why: with ONNX Runtime's CPU arena off, a run allocates and frees its buffers with <c>malloc</c> on the thread that
/// calls it. glibc serves each thread from its own malloc arena, and a non-main arena keeps the free memory at its
/// top resident after the buffers are freed: <c>malloc_trim</c> does not return it, and <c>free</c> returns it only
/// above the dynamic trim threshold (up to 64 MiB). When runs come from a rotating set of thread-pool threads, each
/// new caller thread can land on a fresh arena and leave another 20 to 60 MiB resident, so resident memory steps up
/// over an hour (docs/evidence/eg2-soak-reliability/). Runs executed on these threads reuse the same few arenas, so
/// that memory is reused by the next run instead of accumulating.
/// </para>
/// <para>
/// Threads are created on demand, only when every existing one is busy, up to <see cref="Environment.ProcessorCount"/>
/// (more concurrent CPU runs than processors gain nothing; extra runs wait for a free thread). They are background
/// threads and live for the process. A run started from one of these threads executes inline.
/// </para>
/// </remarks>
internal static class RunThreads
{
    private static readonly object s_gate = new();
    private static readonly Queue<WorkItem> s_queue = new();
    private static int s_threads;

    // Threads waiting for an item that no enqueuer has claimed yet.
    private static int s_free;

    // Threads claimed (woken or just started) that have not yet taken the gate again. Each will take a queued item.
    private static int s_waking;

    [ThreadStatic]
    private static bool t_isRunThread;

    /// <summary>Upper bound on the number of threads.</summary>
    internal static int MaxThreads { get; } = Math.Max(1, Environment.ProcessorCount);

    /// <summary>Threads created so far.</summary>
    internal static int ThreadCount => Volatile.Read(ref s_threads);

    /// <summary>Whether the current thread is one of these threads.</summary>
    internal static bool IsRunThread => t_isRunThread;

    /// <summary>Runs <paramref name="action"/> on one of the threads and waits for it; rethrows its exception.</summary>
    internal static void Invoke(Action action)
    {
        if (t_isRunThread)
        {
            action();
            return;
        }

        // Not disposed: the worker may still be returning from Set when the caller wakes, and an event whose wait
        // handle is never requested holds no kernel object.
        var item = new WorkItem(action);
        lock (s_gate)
        {
            s_queue.Enqueue(item);
            if (s_free > 0)
            {
                // Claim one waiting thread.
                s_free--;
                s_waking++;
                Monitor.Pulse(s_gate);
            }
            else if (s_queue.Count > s_waking && s_threads < MaxThreads)
            {
                // Start a thread only when the threads already on their way cannot take every queued item: a thread
                // that was woken for an item another thread took first is still counted in s_waking.
                s_threads++;
                s_waking++;
                // UnsafeStart: a thread that lives for the process must not capture the first caller's ExecutionContext.
                new Thread(Loop) { IsBackground = true, Name = "starlings-onnx-run" }.UnsafeStart();
            }
        }

        item.Done.Wait();
        item.Error?.Throw();
    }

    private static void Loop()
    {
        t_isRunThread = true;
        bool starting = true;
        WorkItem? finished = null;
        while (true)
        {
            WorkItem? item;
            lock (s_gate)
            {
                // Completing the previous item under the lock means this thread already counts as free (or has taken
                // the next item) when its caller queues its next run, so a caller that runs repeatedly keeps reusing
                // the same threads instead of starting new ones.
                finished?.Done.Set();
                if (starting)
                {
                    // Started by an enqueuer, which counted this thread as waking.
                    s_waking--;
                    starting = false;
                }

                while (!s_queue.TryDequeue(out item))
                {
                    s_free++;
                    Monitor.Wait(s_gate);
                    s_waking--;
                }
            }

            item.Execute();
            finished = item;
        }
    }

    private sealed class WorkItem(Action action)
    {
        public ManualResetEventSlim Done { get; } = new(false, spinCount: 0);

        public ExceptionDispatchInfo? Error { get; private set; }

        public void Execute()
        {
            try
            {
                action();
            }
            catch (Exception ex)
            {
                Error = ExceptionDispatchInfo.Capture(ex);
            }
        }
    }
}
