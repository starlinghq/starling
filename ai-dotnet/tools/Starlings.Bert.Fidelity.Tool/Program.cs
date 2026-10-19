using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.Bert;
using Starlings.Runtime.Onnx;
using Microsoft.ML.OnnxRuntime;

return ToolMain.Run(ToolBody);

int ToolBody()
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || args[i] is not ("--profile" or "--snapshot" or "--reference" or "--out" or "--threads")
            || !options.TryAdd(args[i], args[i + 1]))
        {
            Console.Error.WriteLine("usage: --profile <candidate.json> --snapshot <dir> --reference <reference.json> --out <dir> [--threads 2]");
            return 2;
        }
    }

    if (ToolSettings.RequiredOptions.Any(key => !options.ContainsKey(key)))
    {
        Console.Error.WriteLine("missing --profile, --snapshot, --reference, or --out");
        return 2;
    }

    int threads = options.TryGetValue("--threads", out var rawThreads) ? int.Parse(rawThreads, System.Globalization.CultureInfo.InvariantCulture) : 2;
    var profile = BertModelProfile.Load(options["--profile"]);
    string profileHash = Sha256(options["--profile"]);
    using var reference = JsonDocument.Parse(File.ReadAllText(options["--reference"]));
    var root = reference.RootElement;
    if (root.GetProperty("profile").GetString() != profile.Id || root.GetProperty("candidate_sha256").GetString() != profileHash)
    {
        throw new InvalidDataException("Reference inputs do not belong to the selected candidate contract.");
    }

    string output = Path.GetFullPath(options["--out"]);
    Directory.CreateDirectory(output);
    var sourceRows = root.GetProperty("cases").EnumerateArray().ToArray();
    if (sourceRows.Length < 2 || sourceRows.Select(row => row.GetProperty("id").GetString()).Distinct(StringComparer.Ordinal).Count() != sourceRows.Length)
    {
        throw new InvalidDataException("Reference inputs require at least two uniquely identified cases.");
    }

    using var embedder = profile.Task == BertTask.Embeddings ? BertTextEmbedder.Create(profile, options["--snapshot"], threads, ToolSettings.MaxBatchSize) : null;
    using var reranker = profile.Task == BertTask.Reranking ? BertReranker.Create(profile, options["--snapshot"], threads, ToolSettings.MaxBatchSize) : null;
    OnnxModelSession session = embedder?.Session ?? reranker!.Session;
    var rows = new List<Dictionary<string, object>>();
    var supplied = sourceRows.Select(Input).ToArray();
    var tokenized = new BertInput[sourceRows.Length];
    var hiddenLayouts = new List<Dictionary<string, object>>();
    using (var hiddenFile = profile.Task == BertTask.Embeddings ? File.Create(Path.Combine(output, "dotnet-hidden.f32")) : null)
    {
        for (int i = 0; i < sourceRows.Length; i++)
        {
            var source = sourceRows[i];
            string id = source.GetProperty("id").GetString()!;
            string text = source.GetProperty("text").GetString()!;
            string prompt = source.GetProperty("prompt").GetString()!;
            string? pair = source.TryGetProperty("pair", out var pairValue) && pairValue.ValueKind != JsonValueKind.Null ? pairValue.GetString() : null;
            tokenized[i] = embedder is not null ? embedder.Tokenize(text, prompt) : reranker!.Tokenize(new BertPair(text, pair!));
            float[] fromIds = embedder is not null ? embedder.EmbedTokens([supplied[i]])[0] : [reranker!.ScoreTokens([supplied[i]])[0]];
            float[] fromText = embedder is not null ? embedder.Embed(text, prompt) : [reranker!.Score(new BertPair(text, pair!))];
            var row = new Dictionary<string, object>
            {
                ["id"] = id, ["input_ids"] = tokenized[i].InputIds.ToArray(),
                ["attention_mask"] = tokenized[i].AttentionMask.ToArray(), ["token_type_ids"] = tokenized[i].TokenTypeIds.ToArray(),
                ["output"] = fromIds, ["text_output"] = fromText,
            };
            if (reranker is not null)
            {
                row["logits"] = fromIds;
                row["score"] = fromIds[0];
                row["text_score"] = fromText[0];
            }
            if (hiddenFile is not null)
            {
                float[] hidden = Hidden(session, supplied[i], profile);
                hiddenLayouts.Add(new Dictionary<string, object>
                {
                    ["id"] = id, ["offset"] = hiddenFile.Position / sizeof(float), ["shape"] = new[] { supplied[i].Length, profile.HiddenSize },
                });
                if (!BitConverter.IsLittleEndian)
                {
                    throw new PlatformNotSupportedException("Fidelity tensor sidecars require little-endian float32.");
                }

                hiddenFile.Write(MemoryMarshal.AsBytes(hidden.AsSpan()));
            }

            rows.Add(row);
        }
    }

    float[][] idsBatch = embedder is not null ? embedder.EmbedTokens(supplied) : reranker!.ScoreTokens(supplied).Select(score => new[] { score }).ToArray();
    float[][] textBatch;
    var textChunks = new List<object>();
    if (embedder is not null)
    {
        textBatch = new float[sourceRows.Length][];
        foreach (var group in Enumerable.Range(0, sourceRows.Length).GroupBy(index => sourceRows[index].GetProperty("prompt").GetString()!))
        {
            var indices = group.ToArray();
            var values = embedder.EmbedBatch(indices.Select(index => sourceRows[index].GetProperty("text").GetString()!).ToArray(), group.Key);
            textChunks.AddRange(Chunks(indices, tokenized, sourceRows));
            for (int i = 0; i < indices.Length; i++)
            {
                textBatch[indices[i]] = values[i];
            }
        }
    }
    else
    {
        textBatch = reranker!.ScoreBatch(sourceRows.Select(row => new BertPair(row.GetProperty("text").GetString()!, row.GetProperty("pair").GetString()!)).ToArray())
            .Select(score => new[] { score }).ToArray();
        textChunks.AddRange(Chunks(Enumerable.Range(0, sourceRows.Length).ToArray(), tokenized, sourceRows));
    }

    bool cancellationVerified = false;
    try
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        if (embedder is not null)
        {
            embedder.EmbedTokens(supplied, cancellation.Token);
        }
        else
        {
            reranker!.ScoreTokens(supplied, cancellation.Token);
        }
    }
    catch (OperationCanceledException)
    {
        cancellationVerified = true;
    }

    var diagnostics = session.Diagnostics;
    var result = new Dictionary<string, object>
    {
        ["format"] = "starlings.bert-candidate-results", ["format_version"] = 1,
        ["profile"] = profile.Id, ["task"] = root.GetProperty("task").GetString()!, ["candidate_sha256"] = profileHash,
        ["stage"] = "dotnet", ["reference_sha256"] = Sha256(options["--reference"]),
        ["assets"] = profile.RuntimeLock.Files.Select(file => new { path = file.Path, sha256 = file.Sha256 }).ToArray(),
        ["environment"] = new { machine = Environment.MachineName, dotnet = RuntimeInformation.FrameworkDescription,
            onnxruntime = diagnostics.OrtVersion, provider = "CPUExecutionProvider", threads, inter_op_threads = 1 },
        ["cases"] = rows,
        ["batches"] = new[]
        {
            new { path = "ids", case_ids = rows.Select(row => row["id"]).ToArray(), input_lengths = supplied.Select(input => input.Length).ToArray(), output = idsBatch,
                chunks = Chunks(Enumerable.Range(0, sourceRows.Length).ToArray(), supplied, sourceRows) },
            new { path = "text", case_ids = rows.Select(row => row["id"]).ToArray(), input_lengths = tokenized.Select(input => input.Length).ToArray(), output = textBatch,
                chunks = textChunks.ToArray() },
        },
        ["checks"] = new { pre_cancelled_request = cancellationVerified },
    };
    if (profile.Task == BertTask.Embeddings)
    {
        result["hidden"] = new { file = "dotnet-hidden.f32", sha256 = Sha256(Path.Combine(output, "dotnet-hidden.f32")), cases = hiddenLayouts };
    }

    File.WriteAllText(Path.Combine(output, "dotnet.json"), JsonSerializer.Serialize(result, ToolSettings.Json) + "\n");
    Console.WriteLine($"wrote {rows.Count} {profile.Task} cases and ids/text batches to {output}");
    return 0;

    static BertInput Input(JsonElement row) => new(
        row.GetProperty("input_ids").EnumerateArray().Select(value => value.GetInt64()).ToArray(),
        row.GetProperty("attention_mask").EnumerateArray().Select(value => value.GetInt64()).ToArray(),
        row.GetProperty("token_type_ids").EnumerateArray().Select(value => value.GetInt64()).ToArray());

    static object[] Chunks(int[] indices, BertInput[] inputs, JsonElement[] rows) => indices.Chunk(ToolSettings.MaxBatchSize).Select(chunk => (object)new
    {
        case_ids = chunk.Select(index => rows[index].GetProperty("id").GetString()!).ToArray(),
        input_lengths = chunk.Select(index => inputs[index].Length).ToArray(),
        width = chunk.Max(index => inputs[index].Length),
    }).ToArray();

    static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    static float[] Hidden(OnnxModelSession session, BertInput input, BertModelProfile profile)
    {
        long[] shape = [1, input.Length];
        using var ids = OrtValue.CreateTensorValueFromMemory(input.InputIds.ToArray(), shape);
        using var mask = OrtValue.CreateTensorValueFromMemory(input.AttentionMask.ToArray(), shape);
        using var types = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds.ToArray(), shape);
        using var outputs = profile.TokenTypeIds
            ? session.Run(["input_ids", "attention_mask", "token_type_ids"], [ids, mask, types], [profile.OutputName])
            : session.Run(["input_ids", "attention_mask"], [ids, mask], [profile.OutputName]);
        return outputs[0].GetTensorDataAsSpan<float>().ToArray();
    }
}

internal static class ToolSettings
{
    // Single source for the session batch size and the recorded chunk layout: the session splits text
    // batches into runs of this size, and Chunks() must describe exactly those runs.
    public const int MaxBatchSize = 32;

    public static readonly string[] RequiredOptions = ["--profile", "--snapshot", "--reference", "--out"];
    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
}
