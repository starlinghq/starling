using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace GenCheck;

/// <summary>One prompt from prompts-v1.jsonl plus the reference rendering and ids for it.</summary>
public sealed record PromptCase(
    string Id,
    JsonArray Messages,
    JsonArray? Tools,
    JsonObject? TemplateKwargs,
    int MaxNewTokens,
    string RefRendered,
    int[] RefPromptIds,
    int[] RefOutputIds);

public static class Shared
{
    public static readonly JsonSerializerOptions Compact = new() { WriteIndented = false, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    /// <summary>Reads JSON Lines, splitting on '\n' only (prompt text may contain U+2028 and similar).</summary>
    public static IEnumerable<JsonObject> ReadJsonl(string path)
    {
        foreach (var line in File.ReadAllText(path, Encoding.UTF8).Split('\n'))
        {
            if (line.Trim().Length > 0)
            {
                yield return JsonNode.Parse(line)!.AsObject();
            }
        }
    }

    /// <summary>Joins the prompt set with the reference results (rendered text and prompt ids).</summary>
    public static List<PromptCase> LoadCases(string promptsPath, string referenceDir)
    {
        var refs = ReadJsonl(Path.Combine(referenceDir, "results.jsonl")).ToDictionary(r => (string)r["id"]!);
        var cases = new List<PromptCase>();
        foreach (var p in ReadJsonl(promptsPath))
        {
            var id = (string)p["id"]!;
            var r = refs[id];
            cases.Add(new PromptCase(
                id,
                p["messages"]!.AsArray(),
                p["tools"]?.AsArray(),
                p["chat_template_kwargs"]?.AsObject(),
                (int)p["max_new_tokens"]!,
                (string)r["rendered"]!,
                r["prompt_ids"]!.AsArray().Select(x => (int)x!).ToArray(),
                r["output_ids"]!.AsArray().Select(x => (int)x!).ToArray()));
        }
        return cases;
    }

    /// <summary>Log-softmax with double accumulation, returned as float32.</summary>
    public static float[] LogSoftmax(ReadOnlySpan<float> logits)
    {
        double max = double.NegativeInfinity;
        foreach (var v in logits)
        {
            if (v > max)
            {
                max = v;
            }
        }
        double sum = 0;
        foreach (var v in logits)
        {
            sum += Math.Exp(v - max);
        }
        var lse = max + Math.Log(sum);
        var outp = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            outp[i] = (float)(logits[i] - lse);
        }
        return outp;
    }

    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        var best = 0;
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }
        return best;
    }

    public static void WriteStep0(string outDir, string id, float[] logprobs)
    {
        var dir = Path.Combine(outDir, "step0");
        Directory.CreateDirectory(dir);
        var bytes = new byte[logprobs.Length * 4];
        Buffer.BlockCopy(logprobs, 0, bytes, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("big-endian host");
        }
        File.WriteAllBytes(Path.Combine(dir, id + ".f32"), bytes);
    }

    public static JsonArray ToJson(IEnumerable<int> ids) => new(ids.Select(i => (JsonNode?)JsonValue.Create(i)).ToArray());

    public static JsonArray ToJson(IEnumerable<double> xs) => new(xs.Select(i => (JsonNode?)JsonValue.Create(i)).ToArray());

    /// <summary>Summary statistics for exploratory timings.</summary>
    public static JsonObject Stats(IReadOnlyList<double> xs)
    {
        var s = xs.OrderBy(x => x).ToArray();
        double Pct(double p) => s.Length == 0 ? double.NaN : s[Math.Min(s.Length - 1, (int)Math.Ceiling(p * s.Length) - 1)];
        return new JsonObject
        {
            ["n"] = s.Length,
            ["mean"] = s.Length == 0 ? double.NaN : s.Average(),
            ["median"] = Pct(0.5),
            ["min"] = s.Length == 0 ? double.NaN : s[0],
            ["max"] = s.Length == 0 ? double.NaN : s[^1],
        };
    }

    public static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    public static JsonObject EnvironmentInfo() => new()
    {
        ["dotnet"] = Environment.Version.ToString(),
        ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
        ["rid"] = System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier,
        ["processors"] = Environment.ProcessorCount,
        ["loadavg"] = File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : null,
        ["utc"] = DateTime.UtcNow.ToString("O"),
    };
}
