using Starlings.BrowserSpike;

// Usage: BrowserSpike.Native <repo root> <STARLING_MODELS> [encode seconds] [warm loads]
// Prints the same JSON document as RunTokenizerSuite in the browser.
if (args.Length < 2)
{
    Console.Error.WriteLine("usage: BrowserSpike.Native <repo root> <models root> [encode seconds] [warm loads]");
    return 2;
}

string fixtures = Path.Combine(args[0], "model-pipeline", "protected");
string models = args[1];
double seconds = args.Length > 2 ? double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 3;
int warmLoads = args.Length > 3 ? int.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 3;
string json = await TokenizerSuite.RunAsync(new FileSource(fixtures, models), "native (CoreCLR)", seconds, warmLoads);
Console.WriteLine(json);
return 0;

internal sealed class FileSource(string fixtures, string models) : ISpikeSource
{
    public Task<byte[]> ReadAssetAsync(string path) => File.ReadAllBytesAsync(Path.Combine(models, path));

    public Task<byte[]> ReadFixtureAsync(string path) => File.ReadAllBytesAsync(Path.Combine(fixtures, path));
}
