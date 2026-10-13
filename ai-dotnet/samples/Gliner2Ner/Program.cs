// Entity extraction with fastino/gliner2-base-v1 from the Starlings packages.
//
//   dotnet run -- download [--cache <dir>]                                explicit acquisition (network)
//   dotnet run -- extract  [--cache <dir> | --snapshot <dir>] [--threshold <t>] [text] [--labels a,b,c]
//                                                                         offline: loads the verified files and extracts
//
// Without --cache the files go to $STARLING_MODELS or the user cache directory. --snapshot verifies an existing directory
// holding tokenizer.json and onnx/gliner2-entities-fp32.onnx instead. Inference never downloads anything.
using System.Globalization;
using System.Reflection;
using Starlings.Acquisition;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

if (args.Length == 0 || args[0] is not ("download" or "extract"))
{
    Console.Error.WriteLine("usage: Gliner2Ner download [--cache <dir>] | extract [--cache <dir> | --snapshot <dir>] [--threshold <t>] [--labels a,b] [text]");
    return 2;
}

string? cache = null;
string? snapshot = null;
float threshold = 0.5f;
string[] labels = ["person", "organization", "location", "date", "product"];
var rest = new List<string>();
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--cache" when i + 1 < args.Length:
            cache = args[++i];
            break;
        case "--snapshot" when i + 1 < args.Length:
            snapshot = args[++i];
            break;
        case "--threshold" when i + 1 < args.Length:
            threshold = float.Parse(args[++i], CultureInfo.InvariantCulture);
            break;
        case "--labels" when i + 1 < args.Length:
            labels = args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            break;
        default:
            rest.Add(args[i]);
            break;
    }
}

var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };

if (args[0] == "download")
{
    // 1. Explicit acquisition of the pinned lock. tokenizer.json comes from the Hub repository at the pinned commit; the
    //    ONNX graph is Starling's conversion, downloaded from its explicit source. Cached files are verified, missing
    //    ones downloaded and verified. This is the only step that uses the network.
    Console.WriteLine($"lock:  {Gliner2ModelAssets.Lock}");
    Console.WriteLine($"cache: {acquisition.ResolveCacheDirectory()}");
    if (Gliner2ModelAssets.ModelSourceIsPlaceholder)
    {
        Console.WriteLine($"note:  the graph's source is not published yet ({Gliner2ModelAssets.ModelSource}); place it in the snapshot directory first.");
    }

    var progress = new FileProgress();
    Gliner2ModelFiles downloaded = await Gliner2ModelAssets.DownloadAsync(acquisition, progress);
    foreach (ModelAssetFile file in Gliner2ModelAssets.Lock.Files)
    {
        string how = progress.Downloaded.Contains(file.Path) ? "downloaded" : "already cached";
        Console.WriteLine($"{file.Path}: {how}, {file.Size} bytes, sha256 {file.Sha256} verified");
    }

    Console.WriteLine($"snapshot: {downloaded.Snapshot.Directory}");
    return 0;
}

// 2. Offline load: verifies the files (size and SHA-256) and never downloads.
Gliner2ModelFiles files = snapshot is null ? Gliner2ModelAssets.OpenLocal(acquisition) : Gliner2ModelAssets.OpenDirectory(snapshot);
using var extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions { Session = new OnnxSessionOptions() });

Console.WriteLine($"packages: {Version<Gliner2EntityExtractor>()}, {Version<Starlings.Tokenizers.Tokenizer>()}");
Console.WriteLine($"onnxruntime: native {OnnxRuntimeInfo.NativeVersion}, session providers {string.Join(",", extractor.Session.Diagnostics.SessionProviders)}");

string[] texts = rest.Count > 0
    ? [.. rest]
    :
    [
        "Apple CEO Tim Cook announced the iPhone 15 in Cupertino on September 12, 2023.",
        "Marie Curie won the Nobel Prize in Physics in 1903 together with Pierre Curie",
    ];

// 3. One label set per call; texts are batched. Offsets index the original text (Text == text[Start..End]).
var results = extractor.ExtractBatch(texts, labels, threshold);
for (int i = 0; i < texts.Length; i++)
{
    Console.WriteLine($"text: {texts[i]}");
    foreach (var e in results[i].Entities)
    {
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"  {e.Label}: {e.Text} [{e.Start}, {e.End}) {e.Confidence:F4}"));
    }
}

// 4. Inputs above 512 tokens are rejected, not truncated.
try
{
    extractor.Extract(string.Join(' ', Enumerable.Repeat("word", 600)), labels);
}
catch (Gliner2InputTooLongException ex)
{
    Console.WriteLine($"long input rejected: {ex.TokenCount} tokens > {ex.MaxTokens}");
}

return 0;

static string Version<T>() =>
    typeof(T).Assembly.GetName().Name + " " +
    (typeof(T).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "?");

internal sealed class FileProgress : IProgress<ModelAssetProgress>
{
    public HashSet<string> Downloaded { get; } = [];

    public void Report(ModelAssetProgress value)
    {
        if (value.Stage == ModelAssetStage.Downloading)
        {
            Downloaded.Add(value.Path);
        }
    }
}
