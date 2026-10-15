// Local-only stress reproducer for the GLiNER2 CUDA wrong output of the 2026-10-15 G4 soak (cycle 388, request 14: empty
// entities); evidence in docs/evidence/gliner2-cuda-wrong-output/. Not in AI.slnx; references the GPU natives. Run it
// through run.sh. Modes:
//
//   (default)  the soak's pattern: the 12 soak inputs and labels, --concurrency callers, every --deadline-every-th
//              request a 16-text batch cancelled after a deadline in [--deadline-min, --deadline-max] ms; each other
//              request checked against the first output; on a mismatch, raw count and span logits and the request
//              history. Options --cycles, --requests (per cycle), --keep-loaded (else a new extractor per cycle), --seed.
//   --pairs n  one thread runs n batches of --long-count texts (--batch per graph call) cancelled after a deadline;
//              --waiters threads run short requests meanwhile and check them.
//   --probe n  n batches cancelled after a deadline; after each call returns, times cudaDeviceSynchronize: the device
//              work the call left queued.
//   --noise n  in any mode, n other extractors run uncancelled batches on the same GPU (contention).
//   --cpu      CPU provider instead of CUDA.
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

string snapshot = args[0];
int cycles = 50, requests = 100, concurrency = 2, deadlineEvery = 10, dMin = 5, dMax = 150, seed = 1;
bool cpu = false, keep = false;
int pairs = 0, batch = 8, longCount = 16, waiters = 1, noise = 0, probe = 0;
for (int i = 1; i < args.Length; i++)
{
    int Next() => int.Parse(args[++i], CultureInfo.InvariantCulture);
    switch (args[i])
    {
        case "--cycles": cycles = Next(); break;
        case "--requests": requests = Next(); break;
        case "--concurrency": concurrency = Next(); break;
        case "--deadline-every": deadlineEvery = Next(); break;
        case "--deadline-min": dMin = Next(); break;
        case "--deadline-max": dMax = Next(); break;
        case "--seed": seed = Next(); break;
        case "--cpu": cpu = true; break;
        case "--keep-loaded": keep = true; break;
        case "--pairs": pairs = Next(); break;
        case "--batch": batch = Next(); break;
        case "--waiters": waiters = Next(); break;
        case "--noise": noise = Next(); break;
        case "--probe": probe = Next(); break;
        case "--long-count": longCount = Next(); break;
        default: throw new ArgumentException(args[i]);
    }
}

string[] texts =
[
    "Ada Lovelace met Charles Babbage in London in June 1833.",
    "The committee in Geneva approved the budget on 4 March 2025 after a long debate.",
    "Marie Curie worked at the University of Paris and won two Nobel Prizes.",
    "Der Zug nach München fährt um 14 Uhr vom Hauptbahnhof Berlin ab.",
    "El museo del Prado en Madrid abre a las nueve de la mañana.",
    "東京タワーは1958年に完成しました。",
    "Satya Nadella spoke at the Microsoft Build conference in Seattle.",
    "A short one.",
    "Researchers at CERN in Switzerland announced new results from the Large Hadron Collider on Tuesday.",
    "Jean Dupont, directeur de la société Lumière, a visité Lyon et Marseille en avril.",
    "The river Thames flows through Oxford, Reading and London before reaching the North Sea.",
    "Квантовые компьютеры обещают ускорить некоторые вычисления.",
];
string longText = string.Join(' ', Enumerable.Repeat(
    "In October the board of Northwind Traders met in Copenhagen with representatives of the city council, " +
    "the harbour authority and two universities to discuss a new logistics centre near the old shipyard. " +
    "Anna Larsen presented the plan, and Peter Holm asked about the cost and the timetable.", 4));
string[] longBatch = Enumerable.Repeat(longText, 16).ToArray();
string[] labels = ["person", "organization", "location", "date"];

var files = Gliner2ModelAssets.OpenDirectory(snapshot, Starlings.Acquisition.CacheVerification.Size);
var options = new Gliner2ExtractorOptions
{
    Session = new OnnxSessionOptions { Provider = cpu ? OnnxExecutionProvider.Cpu : OnnxExecutionProvider.Cuda },
};

static string Spans(Gliner2ExtractionResult r) => string.Join(";", r.Entities.Select(e => $"{e.Label}:{e.Start}-{e.End}"));

if (probe > 0)
{
    // Device work still queued when a call returns: time cudaDeviceSynchronize right after each call.
    using var ex = Gliner2EntityExtractor.Create(files.ModelPath, files.TokenizerPath, options with { MaxBatchSize = batch });
    var batchInputs = Enumerable.Repeat(longText, longCount).Select(t => ex.Prepare(t, labels)).ToArray();
    ex.RunModel(batchInputs);
    var rnd = new Random(seed);
    var cancelledDrain = new List<double>();
    var completedDrain = new List<double>();
    var callMs = new List<double>();
    for (int p = 0; p < probe; p++)
    {
        using var cts = new CancellationTokenSource(rnd.Next(dMin, dMax + 1));
        bool wasCancelled = false;
        long t0 = Stopwatch.GetTimestamp();
        try
        {
            ex.RunModel(batchInputs, cts.Token);
        }
        catch (OperationCanceledException)
        {
            wasCancelled = true;
        }

        callMs.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        long t = Stopwatch.GetTimestamp();
        int rc = Cuda.cudaDeviceSynchronize();
        double ms = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
        if (rc != 0)
        {
            throw new InvalidOperationException("cudaDeviceSynchronize " + rc);
        }

        (wasCancelled ? cancelledDrain : completedDrain).Add(ms);
    }

    static string Stats(List<double> v) => v.Count == 0 ? "n=0" : string.Create(CultureInfo.InvariantCulture,
        $"n={v.Count} p50={v.Order().ElementAt(v.Count / 2):F3} p95={v.Order().ElementAt((int)(v.Count * 0.95)):F3} max={v.Max():F3} ms; >1 ms: {v.Count(x => x > 1)}");
    Console.WriteLine("PROBE call returned -> device drained, cancelled calls: " + Stats(cancelledDrain));
    Console.WriteLine("PROBE call returned -> device drained, completed calls: " + Stats(completedDrain));
    Console.WriteLine("PROBE call duration: " + Stats(callMs));
    return 0;
}

// GPU contention: `noise` other sessions run uncancelled long batches on the same device for the whole run.
bool noiseDone = false;
var noiseWorkers = Enumerable.Range(0, noise).Select(_ => Task.Factory.StartNew(() =>
{
    using var other = Gliner2EntityExtractor.Create(files.ModelPath, files.TokenizerPath, options);
    var batchTexts = Enumerable.Repeat(longText, 16).ToArray();
    while (!Volatile.Read(ref noiseDone))
    {
        other.ExtractBatch(batchTexts, labels);
    }
}, TaskCreationOptions.LongRunning)).ToArray();
if (pairs > 0)
{
    // One thread runs long batches cancelled after a deadline; `waiters` threads run short requests meanwhile, so one
    // is waiting on the extractor's run gate whenever a cancelled run returns.
    using var ex = Gliner2EntityExtractor.Create(files.ModelPath, files.TokenizerPath, options with { MaxBatchSize = batch });
    var refs = texts.Select(t => Spans(ex.Extract(t, labels))).ToArray();
    var batchInputs = Enumerable.Repeat(longText, longCount).Select(t => ex.Prepare(t, labels)).ToArray();
    var shortInputs = texts.Select(t => ex.Prepare(t, labels)).ToArray();
    var rnd = new Random(seed);
    int bad = 0, cancels = 0, checkedShort = 0;
    bool done = false;
    var sw = Stopwatch.StartNew();
    var shortWorkers = Enumerable.Range(0, waiters).Select(w => Task.Factory.StartNew(() =>
    {
        for (int i = w; !Volatile.Read(ref done); i++)
        {
            int input = i % texts.Length;
            var o = ex.RunModel([shortInputs[input]])[0];
            Interlocked.Increment(ref checkedShort);
            if (Spans(ex.Decode(shortInputs[input], o)) != refs[input])
            {
                Interlocked.Increment(ref bad);
            }
        }
    }, TaskCreationOptions.LongRunning)).ToArray();
    for (int p = 0; p < pairs; p++)
    {
        using var cts = new CancellationTokenSource(rnd.Next(dMin, dMax + 1));
        try
        {
            ex.RunModel(batchInputs, cts.Token);
        }
        catch (OperationCanceledException)
        {
            cancels++;
        }
    }

    Volatile.Write(ref done, true);
    Task.WaitAll(shortWorkers);
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"PAIRS long {pairs} cancelled {cancels}; short checked {checkedShort} wrong {bad}; {sw.Elapsed.TotalSeconds:F0}s (waiters {waiters}, batch {batch}, long {longCount}, deadline {dMin}-{dMax} ms)"));
    Volatile.Write(ref noiseDone, true);
    Task.WaitAll(noiseWorkers);
    return bad == 0 ? 0 : 1;
}

string[]? reference = null;
Gliner2ModelOutput[]? referenceOutputs = null;
long total = 0, mismatches = 0, cancelled = 0, completedLong = 0;
var clock = Stopwatch.StartNew();
Gliner2EntityExtractor? kept = null;
for (int cycle = 0; cycle < cycles; cycle++)
{
    var extractor = kept ?? Gliner2EntityExtractor.Create(files.ModelPath, files.TokenizerPath, options);
    if (keep)
    {
        kept = extractor;
    }

    if (reference is null)
    {
        var inputs = texts.Select(t => extractor.Prepare(t, labels)).ToArray();
        referenceOutputs = inputs.Select(i => extractor.RunModel([i])[0]).ToArray();
        reference = inputs.Select((inp, i) => Spans(extractor.Decode(inp, referenceOutputs[i]))).ToArray();
        Console.WriteLine($"serialized={extractor.SerializesNativeCalls} providers={string.Join(",", extractor.Session.Diagnostics.SessionProviders)}");
    }

    var history = new ConcurrentQueue<string>();
    void Log(string s) => history.Enqueue(string.Create(CultureInfo.InvariantCulture, $"{clock.Elapsed.TotalMilliseconds:F3} [{Environment.CurrentManagedThreadId}] {s}"));
    int next = -1;
    var workers = Enumerable.Range(0, concurrency).Select(worker => Task.Factory.StartNew(() =>
    {
        var random = new Random(HashCode.Combine(seed, cycle, worker));
        int k;
        while ((k = Interlocked.Increment(ref next)) < requests)
        {
            if (deadlineEvery > 0 && k % deadlineEvery == deadlineEvery - 1)
            {
                int ms = random.Next(dMin, dMax + 1);
                using var cts = new CancellationTokenSource();
                cts.CancelAfter(ms);
                Log($"k={k} long start deadline {ms} ms");
                try
                {
                    extractor.ExtractBatch(longBatch, labels, cancellationToken: cts.Token);
                    Interlocked.Increment(ref completedLong);
                    Log($"k={k} long completed");
                }
                catch (OperationCanceledException)
                {
                    Interlocked.Increment(ref cancelled);
                    Log($"k={k} long cancelled");
                }

                continue;
            }

            int input = k % texts.Length;
            Log($"k={k} input {input} start");
            var prepared = extractor.Prepare(texts[input], labels);
            var output = extractor.RunModel([prepared])[0];
            string got = Spans(extractor.Decode(prepared, output));
            Interlocked.Increment(ref total);
            Log($"k={k} input {input} done '{got}'");
            if (got != reference[input])
            {
                Interlocked.Increment(ref mismatches);
                var sb = new StringBuilder();
                sb.AppendLine(CultureInfo.InvariantCulture, $"MISMATCH cycle {cycle} request {k} input {input}: '{got}' vs '{reference[input]}'");
                var refOut = referenceOutputs![input];
                sb.AppendLine("count logits: " + string.Join(", ", output.CountLogits.ToArray().Select(v => v.ToString("G6", CultureInfo.InvariantCulture))));
                sb.AppendLine("ref count:    " + string.Join(", ", refOut.CountLogits.ToArray().Select(v => v.ToString("G6", CultureInfo.InvariantCulture))));
                var s = output.SpanLogits.ToArray();
                var rs = refOut.SpanLogits.ToArray();
                int nan = s.Count(float.IsNaN);
                int zero = s.Count(v => v == 0f);
                double maxDiff = s.Zip(rs, (a, b) => Math.Abs((double)a - b)).Max();
                sb.AppendLine(CultureInfo.InvariantCulture, $"span logits: n={s.Length} nan={nan} zero={zero} min={s.Min()} max={s.Max()} maxAbsDiffFromRef={maxDiff}");
                sb.AppendLine("span first 16: " + string.Join(", ", s.Take(16).Select(v => v.ToString("G6", CultureInfo.InvariantCulture))));
                sb.AppendLine("ref  first 16: " + string.Join(", ", rs.Take(16).Select(v => v.ToString("G6", CultureInfo.InvariantCulture))));
                sb.AppendLine("history (last 40):");
                foreach (var h in history.TakeLast(40))
                {
                    sb.AppendLine("  " + h);
                }

                lock (texts)
                {
                    Console.WriteLine(sb.ToString());
                }
            }
        }
    }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
    Task.WaitAll(workers);
    if (!keep)
    {
        extractor.Dispose();
    }

    if (cycle % 10 == 9 || cycle == cycles - 1)
    {
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"cycle {cycle + 1}/{cycles} t={clock.Elapsed.TotalSeconds:F0}s checked {total} mismatches {mismatches} long cancelled {cancelled} completed {completedLong}"));
    }
}

kept?.Dispose();
Console.WriteLine($"RESULT checked {total} mismatches {mismatches} long cancelled {cancelled} completed {completedLong}");
Volatile.Write(ref noiseDone, true);
Task.WaitAll(noiseWorkers);
return mismatches == 0 ? 0 : 1;

internal static class Cuda
{
#pragma warning disable SYSLIB1054, CA5392
    [System.Runtime.InteropServices.DllImport("libcudart.so.13")]
    internal static extern int cudaDeviceSynchronize();
#pragma warning restore SYSLIB1054, CA5392
}
