// Explicit model acquisition from the command line (P-009).
//
//   dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- <command> <lock> [options]
//
// Commands:
//   download <lock>   Download missing or invalid files of the lock into the cache and verify them.
//   verify <lock>     Offline: check that the cache holds every file with the right size and SHA-256.
//   status <lock>     Offline: like verify, but lists missing and failing files instead of stopping at an error (exit 1 unless complete).
//   clean <lock>      Remove the lock's partial downloads and stale lock files (files another download holds are skipped).
//   lock <lock>       Print the lock as JSON.
//   path <lock>       Print the snapshot directory.
// <lock> is a known name (multilingual-e5-small, gliner2-base-v1, embeddinggemma-2, embeddinggemma-2-image) or the path of a JSON lock file. Files with an explicit
// source (the GLiNER2 and EmbeddingGemma 2 ONNX graphs) download from that address, without the Hub token.
// Options:
//   --cache <dir>     Cache root (default: $STARLING_MODELS, else the user cache directory).
//   --size-only       Check cached files by size only.
//   --parallel <n>    Download up to n files of the lock at once (default 1).
//   --mirror <url>    A Hub-layout mirror tried after the endpoint when it fails (repeatable; never receives the token unless on huggingface.co).
//   --from <dir>      A directory laid out like the cache to copy files from before downloading (repeatable).
//   --offline         Never use the network: files come only from the cache and --from directories.
//   --lock-timeout <s> Give up after s seconds when another download holds a file's lock (default: wait).
// The access token for gated repositories is read from $HF_TOKEN and never printed.
using System.Diagnostics;
using System.Globalization;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.Gliner2;

return await ToolMain.RunAsync(ToolBody);

async Task<int> ToolBody()
{
    var known = new Dictionary<string, ModelAssetLock>(StringComparer.OrdinalIgnoreCase)
    {
        [E5ModelAssets.Lock.Name] = E5ModelAssets.Lock,
        [Gliner2ModelAssets.Lock.Name] = Gliner2ModelAssets.Lock,
        [EmbeddingGemma2ModelAssets.Lock.Name] = EmbeddingGemma2ModelAssets.Lock,
        [EmbeddingGemma2ModelAssets.ImageLock.Name] = EmbeddingGemma2ModelAssets.ImageLock,
    };

    if (args.Length < 2 || args[0] is "-h" or "--help")
    {
        return Usage();
    }

    string command = args[0];
    string? cache = null;
    var verification = CacheVerification.Sha256;
    int parallel = 1;
    bool offline = false;
    TimeSpan? lockTimeout = null;
    var mirrors = new List<Uri>();
    var sources = new List<string>();
    for (int i = 2; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--cache" when i + 1 < args.Length:
                cache = args[++i];
                break;
            case "--size-only":
                verification = CacheVerification.Size;
                break;
            case "--parallel" when i + 1 < args.Length && int.TryParse(args[i + 1], NumberStyles.None, CultureInfo.InvariantCulture, out int n) && n >= 1:
                parallel = n;
                i++;
                break;
            case "--mirror" when i + 1 < args.Length && Uri.TryCreate(args[i + 1], UriKind.Absolute, out var mirror):
                mirrors.Add(mirror);
                i++;
                break;
            case "--from" when i + 1 < args.Length:
                sources.Add(args[++i]);
                break;
            case "--offline":
                offline = true;
                break;
            case "--lock-timeout" when i + 1 < args.Length && double.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds >= 0:
                lockTimeout = TimeSpan.FromSeconds(seconds);
                i++;
                break;
            default:
                return Usage($"unknown option {args[i]}");
        }
    }

    ModelAssetLock assetLock;
    try
    {
        assetLock = known.TryGetValue(args[1], out var named) ? named : ModelAssetLock.Load(args[1]);
    }
    catch (Exception ex) when (ex is IOException or FormatException or UnauthorizedAccessException)
    {
        return Usage($"'{args[1]}' is neither a known lock ({string.Join(", ", known.Keys)}) nor a readable lock file: {ex.Message}");
    }

    var options = new ModelAcquisitionOptions
    {
        CacheDirectory = cache,
        Verification = verification,
        MaxConcurrentFiles = parallel,
        FallbackEndpoints = mirrors,
        LocalSources = sources,
        AllowNetwork = !offline,
        LockWaitTimeout = lockTimeout,
    };
    using var store = new ModelAssetStore(options);
    using var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        cts.Cancel();
    };

    try
    {
        switch (command)
        {
            case "download":
            {
                Console.Error.WriteLine($"acquiring {assetLock} ({Mb(assetLock.TotalSize)}) into {store.GetSnapshotDirectory(assetLock)}");
                var watch = Stopwatch.StartNew();
                var snapshot = await store.DownloadAsync(assetLock, new ConsoleProgress(), cts.Token);
                Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"done in {watch.Elapsed.TotalSeconds:F1} s"));
                PrintFiles(snapshot);
                return 0;
            }

            case "verify":
            {
                var snapshot = store.OpenLocal(assetLock);
                PrintFiles(snapshot);
                return 0;
            }

            case "status":
            {
                var status = store.Check(assetLock);
                Console.WriteLine($"{status.State}  {status.SnapshotDirectory}  {Mb(status.MissingBytes)} to acquire");
                foreach (string path in status.MissingPaths)
                {
                    Console.WriteLine($"missing  {path}");
                }

                foreach (var problem in status.Problems)
                {
                    Console.WriteLine($"{problem.Kind,-8} {problem.Path}: {problem.Message}");
                }

                return status.IsComplete ? 0 : 1;
            }

            case "clean":
            {
                var result = store.RemovePartialDownloads(assetLock);
                foreach (var entry in result.Removed)
                {
                    Console.WriteLine($"removed  {entry.Kind,-15} {entry.Path}  {entry.Bytes} bytes");
                }

                foreach (var entry in result.Skipped)
                {
                    Console.WriteLine($"in use   {entry.Kind,-15} {entry.Path}");
                }

                return 0;
            }

            case "lock":
                Console.Write(assetLock.ToJson());
                return 0;

            case "path":
                Console.WriteLine(store.GetSnapshotDirectory(assetLock));
                return 0;

            default:
                return Usage($"unknown command {command}");
        }
    }
    catch (ModelAcquisitionException ex)
    {
        Console.Error.WriteLine($"error ({ex.GetType().Name}): {ex.Message}");
        return 1;
    }
    catch (OperationCanceledException)
    {
        Console.Error.WriteLine("cancelled; partial downloads are kept and resumed by the next download.");
        return 130;
    }

    static void PrintFiles(ModelSnapshot snapshot)
    {
        foreach (var file in snapshot.Lock.Files)
        {
            Console.WriteLine($"ok   {snapshot.GetFilePath(file.Path)}  {file.Size} bytes  sha256 {(snapshot.Verification == CacheVerification.Sha256 ? file.Sha256 : "(size checked only)")}");
        }
    }

    static string Mb(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e6:F1} MB");

    static int Usage(string? error = null)
    {
        if (error is not null)
        {
            Console.Error.WriteLine("error: " + error);
        }

        Console.Error.WriteLine(
            "usage: <download|verify|status|clean|lock|path> <multilingual-e5-small | gliner2-base-v1 | embeddinggemma-2 | embeddinggemma-2-image | lock.json> " +
            "[--cache <dir>] [--size-only] [--parallel <n>] [--mirror <url>]... [--from <dir>]... [--offline] [--lock-timeout <seconds>]");
        return 2;
    }
}

internal sealed class ConsoleProgress : IProgress<ModelAssetProgress>
{
    public void Report(ModelAssetProgress value) =>
        Console.Error.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"  [{value.FileIndex + 1}/{value.FileCount}] {value.Stage,-11} {value.Path}  {value.BytesProcessed / 1e6,8:F1} / {value.TotalBytes / 1e6:F1} MB" +
            $"  (lock {value.LockBytesProcessed / 1e6:F1} / {value.LockTotalBytes / 1e6:F1} MB)"));
}
