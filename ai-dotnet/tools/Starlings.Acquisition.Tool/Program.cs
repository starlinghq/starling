// Explicit model acquisition from the command line (P-009).
//
//   dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- <command> <lock> [options]
//
// Commands:
//   download <lock>   Download missing or invalid files of the lock into the cache and verify them.
//   verify <lock>     Offline: check that the cache holds every file with the right size and SHA-256.
//   lock <lock>       Print the lock as JSON.
//   path <lock>       Print the snapshot directory.
// <lock> is a known name (multilingual-e5-small, gliner2-base-v1, embeddinggemma-2, embeddinggemma-2-image) or the path of a JSON lock file. Files with an explicit
// source (the GLiNER2 and EmbeddingGemma 2 ONNX graphs) download from that address, without the Hub token.
// Options:
//   --cache <dir>     Cache root (default: $STARLING_MODELS, else the user cache directory).
//   --size-only       Check cached files by size only.
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

    var options = new ModelAcquisitionOptions { CacheDirectory = cache, Verification = verification };
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

        Console.Error.WriteLine("usage: <download|verify|lock|path> <multilingual-e5-small | gliner2-base-v1 | embeddinggemma-2 | embeddinggemma-2-image | lock.json> [--cache <dir>] [--size-only]");
        return 2;
    }
}

internal sealed class ConsoleProgress : IProgress<ModelAssetProgress>
{
    public void Report(ModelAssetProgress value) =>
        Console.Error.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"  {value.Stage,-11} {value.Path}  {value.BytesProcessed / 1e6,8:F1} / {value.TotalBytes / 1e6:F1} MB"));
}
