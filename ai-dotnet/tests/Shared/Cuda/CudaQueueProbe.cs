using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Starlings.TestSupport;

/// <summary>
/// Regression probe for <c>docs/evidence/gliner2-cuda-wrong-output/</c>: ONNX Runtime 1.30 returned from a CUDA run
/// terminated at a node boundary while device work it had queued was still pending, on buffers it had already
/// released, and the next run computed on memory that work still wrote (a GLiNER2 soak returned empty entities). The
/// probe cancels a call at deadlines spread over its duration and, as soon as each cancelled call returns, times
/// <c>cudaDeviceSynchronize</c>: any work still queued by the call shows as drain time. Before the fix, about one
/// cancelled GLiNER2 call in 20 left up to 33 ms of work queued; after it, none.
/// </summary>
internal static class CudaQueueProbe
{
    /// <summary>Drain time above which work counts as still queued (a drained device answers in microseconds).</summary>
    public const double PendingThresholdMs = 2.0;

    // The CUDA 13 runtime the ONNX Runtime 1.30 CUDA provider already loaded (a blittable signature; no marshalling).
#pragma warning disable SYSLIB1054 // LibraryImport would need unsafe code in these test projects.
    [DllImport("libcudart.so.13")]
    private static extern int cudaDeviceSynchronize();
#pragma warning restore SYSLIB1054

    /// <summary>Milliseconds until the device finished the work this process had queued.</summary>
    public static double DrainMilliseconds()
    {
        long start = Stopwatch.GetTimestamp();
        int status = cudaDeviceSynchronize();
        double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        if (status != 0)
        {
            throw new InvalidOperationException($"cudaDeviceSynchronize returned CUDA error {status}.");
        }

        return ms;
    }

    /// <summary>
    /// Runs <paramref name="call"/> until <paramref name="cancellations"/> calls were cancelled (at most four times as
    /// many attempts), with deadlines uniform over one uncancelled call's duration, and returns the drain time after
    /// each cancelled call.
    /// </summary>
    public static ProbeResult CancelledCalls(Action<CancellationToken> call, int cancellations, int seed = 1)
    {
        call(CancellationToken.None);
        DrainMilliseconds();
        long t = Stopwatch.GetTimestamp();
        call(CancellationToken.None);
        double callMs = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
        int maxDeadline = Math.Max(2, (int)callMs);
        var random = new Random(seed);
        var drains = new List<double>();
        int attempts = 0;
        while (drains.Count < cancellations && attempts++ < 4 * cancellations)
        {
            using var cts = new CancellationTokenSource(random.Next(1, maxDeadline + 1));
            try
            {
                call(cts.Token);
                DrainMilliseconds();
            }
            catch (OperationCanceledException)
            {
                drains.Add(DrainMilliseconds());
            }
        }

        return new ProbeResult(callMs, attempts, drains);
    }

    internal sealed record ProbeResult(double CallMs, int Attempts, List<double> Drains)
    {
        public int Pending => Drains.Count(d => d > PendingThresholdMs);

        public override string ToString() => Drains.Count == 0
            ? $"no call was cancelled in {Attempts} attempts (uncancelled call {CallMs:F1} ms)"
            : $"uncancelled call {CallMs:F1} ms; {Drains.Count} of {Attempts} calls cancelled; drain after a cancelled call: " +
              $"median {Drains.Order().ElementAt(Drains.Count / 2):F3} ms, max {Drains.Max():F3} ms, above {PendingThresholdMs} ms: {Pending}";
    }
}
