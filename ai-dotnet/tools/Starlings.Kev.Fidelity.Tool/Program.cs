using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.Kev;
using Starlings.Runtime.Onnx;

return ToolMain.Run(ToolBody);

// The dotnet stage of the kev-onnx-cpu-v1 onboarding route: reads the reference stage's reference.json, runs every
// case through Starlings.Models.Kev from the reference ids (cached and uncached) and from the record text, and writes
// <out>/dotnet.json for the decision evaluator.
int ToolBody()
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || args[i] is not ("--profile" or "--snapshot" or "--reference" or "--out" or "--threads" or "--device")
            || !options.TryAdd(args[i], args[i + 1]))
        {
            Console.Error.WriteLine("usage: --profile <candidate.json> --snapshot <bundle dir> --reference <reference.json> --out <dir> [--threads 2] [--device cpu|cuda[:N]]");
            return 2;
        }
    }

    if (ToolSettings.RequiredOptions.Any(key => !options.ContainsKey(key)))
    {
        Console.Error.WriteLine("missing --profile, --snapshot, --reference, or --out");
        return 2;
    }

    int threads = options.TryGetValue("--threads", out var rawThreads) ? int.Parse(rawThreads, CultureInfo.InvariantCulture) : 2;
    string candidatePath = options["--profile"];
    using var candidate = JsonDocument.Parse(File.ReadAllText(candidatePath));
    string repository = candidate.RootElement.GetProperty("origin").GetProperty("repo").GetString()!;
    string revision = candidate.RootElement.GetProperty("origin").GetProperty("revision").GetString()!;
    var profile = KevModelProfile.ForRepository(repository)
        ?? throw new NotSupportedException($"No Kev contract for {repository}.");
    if (candidate.RootElement.GetProperty("id").GetString() != profile.Id || revision != profile.Revision)
    {
        throw new NotSupportedException($"Candidate {candidatePath} is not the {profile.Id} contract at {profile.Revision}.");
    }

    string candidateHash = Sha256(candidatePath);
    string referencePath = options["--reference"];
    using var reference = JsonDocument.Parse(File.ReadAllText(referencePath));
    var root = reference.RootElement;
    if (root.GetProperty("profile").GetString() != profile.Id || root.GetProperty("candidate_sha256").GetString() != candidateHash)
    {
        throw new InvalidDataException("Reference inputs do not belong to the selected candidate contract.");
    }

    var cases = root.GetProperty("cases").EnumerateArray().ToArray();
    if (cases.Length == 0 || cases.Select(c => c.GetProperty("id").GetString()).Distinct(StringComparer.Ordinal).Count() != cases.Length)
    {
        throw new InvalidDataException("Reference inputs require uniquely identified cases.");
    }

    string snapshot = Path.GetFullPath(options["--snapshot"]);
    string Bundle(string file) => Path.Combine(snapshot, file);
    // --device cuda[:N] runs the same bundle as the profile's onnx-fp32-cuda sibling (use_tf32=0, forced by Runtime.Onnx);
    // only the -p:UseCuda=true build has the CUDA provider, and session creation says so otherwise.
    string device = options.TryGetValue("--device", out var rawDevice) ? rawDevice : "cpu";
    var sessionOptions = new OnnxSessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = 1 };
    if (device == "cuda" || device.StartsWith("cuda:", StringComparison.Ordinal))
    {
        int deviceId = device == "cuda" ? 0 : int.Parse(device[5..], CultureInfo.InvariantCulture);
        sessionOptions = sessionOptions with { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = deviceId };
    }
    else if (device != "cpu")
    {
        Console.Error.WriteLine($"unknown --device {device}: cpu, cuda or cuda:N");
        return 2;
    }
    using var model = KevDecisionModel.Create(
        profile, Bundle(KevModelAssets.ModelFile), Bundle(KevModelAssets.HeadFile), Bundle(KevModelAssets.HeadMetadataFile),
        Bundle(KevModelAssets.TokenizerFile), new KevOptions { Session = sessionOptions });

    var rows = new List<Dictionary<string, object>>();
    foreach (var c in cases)
    {
        var record = Record(c.GetProperty("record"));
        var referenceEncoding = new KevEncoding(
            [.. c.GetProperty("state_ids").EnumerateArray().Select(e => e.GetInt32())],
            [.. c.GetProperty("rows").EnumerateArray().Select(r => new KevRow(
                [.. r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())],
                [.. r.GetProperty("pos").EnumerateArray().Select(e => e.GetInt32())],
                r.GetProperty("decide").GetInt32(),
                [.. r.GetProperty("opts").EnumerateArray().Select(e => e.GetInt32())]))]);
        var encoding = model.Encode(record);
        rows.Add(new Dictionary<string, object>
        {
            ["id"] = c.GetProperty("id").GetString()!,
            ["state_ids"] = encoding.StateIds.ToArray(),
            ["rows"] = encoding.Rows.Select(r => new { ids = r.Ids.ToArray(), pos = r.Positions.ToArray(), decide = r.Decide, opts = r.Options.ToArray() }).ToArray(),
            ["output"] = model.Run(referenceEncoding),
            ["uncached_output"] = model.RunUncached(referenceEncoding),
            ["text_output"] = model.Run(encoding),
        });
        Console.WriteLine($"{rows.Count}/{cases.Length} {rows[^1]["id"]}");
    }

    bool cancellationVerified = false;
    try
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        model.Score(Record(cases[0].GetProperty("record")), cancellation.Token);
    }
    catch (OperationCanceledException)
    {
        cancellationVerified = true;
    }

    var diagnostics = model.Session.Diagnostics;
    var result = new Dictionary<string, object>
    {
        ["format"] = "starlings.kev-candidate-results", ["format_version"] = 1,
        ["profile"] = profile.Id, ["candidate_sha256"] = candidateHash, ["stage"] = "dotnet",
        ["reference_sha256"] = Sha256(referencePath),
        ["environment"] = new
        {
            machine = Environment.MachineName, dotnet = RuntimeInformation.FrameworkDescription, onnxruntime = diagnostics.OrtVersion,
            provider = diagnostics.SessionProviders[0], profile_run = model.ProfileId, threads, inter_op_threads = 1,
            provider_options = diagnostics.ProviderOptions,
            cuda_visible_devices = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"),
            model_sha256 = new[] { KevModelAssets.ModelFile, KevModelAssets.ModelDataFile, KevModelAssets.HeadFile, KevModelAssets.HeadMetadataFile, KevModelAssets.TokenizerFile }
                .ToDictionary(f => f, f => Sha256(Bundle(f))),
        },
        ["temperature"] = model.Temperature,
        ["cases"] = rows,
        ["cancellation_verified"] = cancellationVerified,
    };
    string output = Path.GetFullPath(options["--out"]);
    Directory.CreateDirectory(output);
    File.WriteAllText(Path.Combine(output, "dotnet.json"), JsonSerializer.Serialize(result, ToolSettings.Json) + "\n");
    Console.WriteLine($"wrote {rows.Count} cases to {output}");
    return 0;

    static KevRecord Record(JsonElement r) => new(
        r.GetProperty("state").GetString()!,
        [.. r.GetProperty("questions").EnumerateArray().Select((q, i) => new KevRecordQuestion(
            q.TryGetProperty("id", out var id) ? id.GetString()! : i.ToString(CultureInfo.InvariantCulture),
            q.GetProperty("instr").GetString()!,
            [.. q.GetProperty("options").EnumerateArray().Select(o => o.GetString()!)]))]);

    static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}

internal static class ToolSettings
{
    public static readonly string[] RequiredOptions = ["--profile", "--snapshot", "--reference", "--out"];
    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
}
