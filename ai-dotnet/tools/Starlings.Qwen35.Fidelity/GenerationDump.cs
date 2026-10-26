// Candidate outputs for the generation task-quality evaluator (G3): `model-pipeline evaluate --g3 generation`.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -- generation \
//     --bundle <snapshot dir> --prompts <prompts.jsonl> --out <dir> [--variant fp32x|fp16x] [--device cpu|cuda] [--threads N]
//
// Runs the public chat path once per prompt row: the row's messages through the owned chat template and tokenizer
// (Qwen35ChatModel.Generate, chat_template_kwargs.enable_thinking honoured), greedy decoding, the row's
// max_new_tokens. Writes <out>/generations.jsonl, one JSON line per row in file order:
// {"id", "output_text", "output_ids", "finish_reason", "prompt_tokens", "completion_tokens"}, where output_text is the
// decoded completion without the stop token (the text the G3 metric scores) and output_ids include the stop token
// when generation stopped on one (the format of results.jsonl in --modes parity). Also <out>/environment.json.
// Rows with tool definitions are refused (the owned renderer has no tools input in this tool). A row the package rejects
// with a typed error (GenerationContextLengthException, Qwen35TemplateException) is written as {"id", "backend",
// "rejected": "<type>: <message>"} without output_text; the run continues and exits 1 (the evaluator refuses such a
// file, so a rejected row is never scored). Exit codes: 0 every row generated; 1 rows rejected; 2 usage or input error.
// The tool applies no budgets and reads nothing under model-pipeline/protected/.
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;

internal static class GenerationDump
{
    public static int Run(string[] args)
    {
        string? bundle = null, prompts = null, output = null;
        string device = "cpu";
        var variant = Qwen35Variant.Fp32x;
        int threads = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--bundle" when i + 1 < args.Length:
                    bundle = args[++i];
                    break;
                case "--prompts" when i + 1 < args.Length:
                    prompts = args[++i];
                    break;
                case "--out" when i + 1 < args.Length:
                    output = args[++i];
                    break;
                case "--device" when i + 1 < args.Length && args[i + 1] is "cpu" or "cuda":
                    device = args[++i];
                    break;
                case "--variant" when i + 1 < args.Length && args[i + 1] is "fp32x" or "fp16x":
                    variant = args[++i] == "fp16x" ? Qwen35Variant.Fp16x : Qwen35Variant.Fp32x;
                    break;
                case "--threads" when i + 1 < args.Length && int.TryParse(args[i + 1], CultureInfo.InvariantCulture, out int n) && n > 0:
                    threads = n;
                    i++;
                    break;
                default:
                    return Usage($"unknown or invalid option {args[i]}");
            }
        }

        if (bundle is null || prompts is null || output is null)
        {
            return Usage("--bundle, --prompts and --out are required");
        }

        List<(string Id, List<Qwen35Message> Messages, bool Thinking, int MaxNewTokens)> rows;
        try
        {
            rows = ReadPrompts(prompts);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException or System.Text.Json.JsonException)
        {
            return Usage($"{prompts}: {e.Message}");
        }

        Directory.CreateDirectory(output);
        var clock = Stopwatch.StartNew();
        Qwen35ChatModel model;
        try
        {
            var files = Qwen35ModelAssets.OpenDirectory(bundle, variant, CacheVerification.Size);
            model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions
            {
                Provider = device == "cuda" ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu,
                IntraOpThreads = threads > 0 ? threads : null,
            });
        }
        catch (ModelAcquisitionException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        var rejected = new List<string>();
        using (model)
        {
            double loadMs = clock.Elapsed.TotalMilliseconds;
            string backend = $"df-qwen35-{device}-{Qwen35ModelAssets.VariantDirectory(variant)}";
            string path = Path.Combine(output, "generations.jsonl");
            using (var writer = new StreamWriter(path, append: false, new UTF8Encoding(false)) { NewLine = "\n" })
            {
                foreach (var (id, messages, thinking, maxNewTokens) in rows)
                {
                    GenerationResult result;
                    try
                    {
                        result = model.Generate(messages, new Qwen35ChatOptions { MaxNewTokens = maxNewTokens, EnableThinking = thinking });
                    }
                    catch (Exception e) when (e is GenerationContextLengthException or Qwen35TemplateException)
                    {
                        // A typed rejection of this prompt is evidence, not a crash: the row records it without
                        // output_text, so `model-pipeline evaluate --g3 generation` refuses the file (it never scores it).
                        string reason = $"{e.GetType().Name}: {e.Message}";
                        rejected.Add(id);
                        writer.WriteLine(new JsonObject { ["id"] = id, ["backend"] = backend, ["rejected"] = reason }.ToJsonString(Json.Compact));
                        writer.Flush();
                        Console.Error.WriteLine($"{id}: rejected: {reason}");
                        continue;
                    }

                    int[] outputIds = result.StopTokenId is int stop ? [.. result.TokenIds, stop] : [.. result.TokenIds];
                    var row = new JsonObject
                    {
                        ["id"] = id,
                        ["backend"] = backend,
                        ["output_text"] = result.Text,
                        ["output_ids"] = new JsonArray(outputIds.Select(t => (JsonNode?)t).ToArray()),
                        ["finish_reason"] = result.FinishReason == GenerationFinishReason.Stop ? "eos" : "length",
                        ["prompt_tokens"] = result.Usage.PromptTokens,
                        ["completion_tokens"] = result.Usage.CompletionTokens,
                        ["prefill_ms"] = result.Timings.Prefill.TotalMilliseconds,
                        ["decode_ms"] = result.Timings.Decode.TotalMilliseconds,
                    };
                    writer.WriteLine(row.ToJsonString(Json.Compact));
                    writer.Flush();
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"{id,-24} prompt {result.Usage.PromptTokens,5} gen {outputIds.Length,4} {row["finish_reason"],-6} " +
                        $"{result.Timings.Prefill.TotalSeconds:F2}s+{result.Timings.Decode.TotalSeconds:F2}s"));
                }
            }

            var env = new JsonObject
            {
                ["runner"] = "ai-dotnet/tools/Starlings.Qwen35.Fidelity generation",
                ["backend"] = backend,
                ["profile"] = model.Info.Profile,
                ["checkpoint"] = model.Info.Checkpoint,
                ["genai_version"] = model.Info.GenAIVersion,
                ["tokenizer"] = model.Info.Tokenizer,
                ["chat_template"] = model.Info.ChatTemplate,
                ["decoding"] = "greedy",
                ["threads"] = threads,
                ["model_load_ms"] = loadMs,
                ["prompts_sha256"] = Sha256(prompts),
                ["rows"] = rows.Count,
                ["dotnet"] = Environment.Version.ToString(),
                ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
                ["cpu_count"] = Environment.ProcessorCount,
                ["source_revision"] = Environment.GetEnvironmentVariable("DF_SOURCE_REVISION"),
            };
            File.WriteAllText(Path.Combine(output, "environment.json"), env.ToJsonString(Json.Indented) + "\n");
            Console.WriteLine($"wrote {path}: {rows.Count} rows");
        }

        if (rejected.Count > 0)
        {
            Console.Error.WriteLine($"error: {rejected.Count} of {rows.Count} rows rejected (recorded with their reason): {string.Join(", ", rejected)}");
            return 1;
        }

        return 0;
    }

    private static List<(string, List<Qwen35Message>, bool, int)> ReadPrompts(string path)
    {
        var rows = new List<(string, List<Qwen35Message>, bool, int)>();
        foreach (string line in File.ReadLines(path, Encoding.UTF8))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            var p = JsonNode.Parse(line)!.AsObject();
            string id = (string?)p["id"] ?? throw new InvalidDataException("a row has no id");
            if (p["tools"] is not null)
            {
                throw new NotSupportedException($"{id}: tool definitions are not supported by the generation command");
            }

            var messages = (p["messages"]?.AsArray() ?? throw new InvalidDataException($"{id}: no messages")).Select(m =>
            {
                string role = (string)m!["role"]!;
                var parsed = role switch
                {
                    "system" => Qwen35Role.System,
                    "user" => Qwen35Role.User,
                    "assistant" => Qwen35Role.Assistant,
                    _ => throw new NotSupportedException($"{id}: role {role}"),
                };
                return new Qwen35Message(parsed, (string)m["content"]!, (string?)m["reasoning_content"]);
            }).ToList();
            bool thinking = p["chat_template_kwargs"]?["enable_thinking"] is JsonNode t && (bool)t;
            int maxNewTokens = (int?)p["max_new_tokens"] ?? throw new InvalidDataException($"{id}: no max_new_tokens");
            rows.Add((id, messages, thinking, maxNewTokens));
        }

        return rows;
    }

    private static string Sha256(string path)
    {
        using var s = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(s));
    }

    private static int Usage(string error)
    {
        Console.Error.WriteLine($"error: {error}");
        Console.Error.WriteLine("usage: Starlings.Qwen35.Fidelity generation --bundle <dir> --prompts <jsonl> --out <dir> [--variant fp32x|fp16x] [--device cpu|cuda] [--threads N]");
        return 2;
    }
}
