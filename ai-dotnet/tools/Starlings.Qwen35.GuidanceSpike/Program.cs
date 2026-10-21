// Spike S5 (Corvid C-003): does an llguidance grammar remove invalid tool names from Qwen3.5-0.8B tool calls, and
// what does it cost in latency?
//
//   Starlings.Qwen35.GuidanceSpike --model-dir DIR --out runs.jsonl [--cuda] [--cases N] [--start N] [--smoke]
//
// DIR is an ortgenai-fp32x directory that also holds tokenizer.json (ONNX Runtime GenAI's guidance reads the
// tokenizer from the model directory; the published Starling bundle keeps it in transformers-5.18.0/ instead).
// Each case runs four interleaved conditions, greedy, thinking off:
//   free-all       all 12 tools offered, no grammar
//   guided-all     all 12 tools, Qwen35ToolGrammar (Required); one grammar for every case
//   free-subset    the expected tool plus 4 others chosen by case id, no grammar
//   guided-subset  the same 5 tools with their grammar; the grammar differs per case (no compile-cache reuse)
// --aa replaces each guided condition with a repeat of its free one (an A/A control for run-to-run noise).
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;

return ToolMain.Run(() => Spike.Run(args));

internal static class Spike
{
    private const int MaxNewTokens = 256;

    public static int Run(string[] args)
    {
        string? modelDir = null, outPath = null;
        bool cuda = false, smoke = false, aa = false;
        int count = int.MaxValue, start = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model-dir": modelDir = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--cuda": cuda = true; break;
                case "--smoke": smoke = true; break;
                case "--aa": aa = true; break;
                case "--cases": count = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--start": start = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
            }
        }

        if (modelDir is null || (outPath is null && !smoke))
        {
            Console.Error.WriteLine("usage: --model-dir DIR --out FILE [--cuda] [--cases N] [--start N] [--smoke]");
            return 2;
        }

        string fixtures = Path.Combine(AppContext.BaseDirectory, "fixtures");
        using var toolsDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtures, "tools.json")));
        Qwen35Tool[] allTools = [.. toolsDoc.RootElement.EnumerateArray().Select(Qwen35Tool.FromJson)];
        var cases = File.ReadLines(Path.Combine(fixtures, "cases.jsonl"))
            .Where(l => l.Length > 0).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();

        Qwen35ModelFiles files = Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x) with { ModelDirectory = Path.GetFullPath(modelDir) };
        var load = Stopwatch.StartNew();
        using var model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions
        {
            Provider = cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu,
        });
        Console.Error.WriteLine($"loaded {model.Info.Profile} on {model.Info.Provider} ({model.Info.GenAIVersion}, {model.Info.GenAINativePackage}) in {load.ElapsedMilliseconds} ms");

        if (smoke)
        {
            return Smoke(model, allTools);
        }

        // Warm-up (excluded): one free and one guided generation on a prompt outside the set.
        var warm = new[] { Qwen35Message.User("Remind me to stretch at 4pm.") };
        Generate(model, warm, allTools, null);
        Generate(model, warm, allTools, Qwen35ToolGrammar.Build(allTools));

        using var output = new StreamWriter(outPath!, append: start > 0);
        foreach (var c in cases.Skip(start).Take(count))
        {
            int id = c["id"]!.GetValue<int>();
            string expected = c["tool"]!.GetValue<string>();
            var messages = new[] { Qwen35Message.User(c["prompt"]!.GetValue<string>()) };
            var others = allTools.Where(t => t.Name != expected).OrderBy(t => Hash(id, t.Name)).Take(4);
            Qwen35Tool[] subset = [.. allTools.Where(t => t.Name == expected || others.Contains(t))];

            var conditions = aa
                ? new[] { ("free-all", allTools, false), ("free-all-repeat", allTools, false), ("free-subset", subset, false), ("free-subset-repeat", subset, false) }
                : new[] { ("free-all", allTools, false), ("guided-all", allTools, true), ("free-subset", subset, false), ("guided-subset", subset, true) };
            foreach (var (condition, tools, guided) in conditions)
            {
                var compile = Stopwatch.StartNew();
                GenerationGuidance? guidance = guided ? Qwen35ToolGrammar.Build(tools) : null;
                double grammarBuildMs = compile.Elapsed.TotalMilliseconds;
                var run = Generate(model, messages, tools, guidance);
                var record = Score(c, tools, run);
                record["condition"] = condition;
                record["grammar_build_ms"] = grammarBuildMs;
                output.WriteLine(record.ToJsonString());
                output.Flush();
                Console.Error.WriteLine($"{id,3} {condition,-18} {record["outcome"],-12} {record["called"],-16} ttft={run.TtftMs,7:F1} total={run.TotalMs,7:F1} tok={run.Tokens}");
            }
        }

        return 0;
    }

    private static int Smoke(Qwen35ChatModel model, Qwen35Tool[] tools)
    {
        var messages = new[] { Qwen35Message.User("What's the weather in Porto?") };
        var forced = Generate(model, messages, tools, GenerationGuidance.Regex("FORCED-[0-9]{3}"));
        Console.WriteLine($"regex  -> {JsonSerializer.Serialize(forced.Text)} finish={forced.Finish}");
        Console.WriteLine(Qwen35ToolGrammar.BuildLark(tools));
        var guided = Generate(model, messages, tools, Qwen35ToolGrammar.Build(tools));
        Console.WriteLine($"lark   -> {JsonSerializer.Serialize(guided.Text)} finish={guided.Finish}");
        var parsed = Qwen35ToolCallParser.Parse(guided.Text, tools, enableThinking: false);
        Console.WriteLine($"parsed -> {string.Join(", ", parsed.ToolCalls.Select(t => t.Name + " " + t.Arguments.GetRawText()))}");
        var free = Generate(model, messages, tools, null);
        Console.WriteLine($"free   -> {JsonSerializer.Serialize(free.Text)} finish={free.Finish}");
        return 0;
    }

    private sealed record GenRun(string Text, int Tokens, double TtftMs, double TotalMs, double PrefillMs, string Finish, int PromptTokens);

    private static GenRun Generate(Qwen35ChatModel model, Qwen35Message[] messages, Qwen35Tool[] tools, GenerationGuidance? guidance)
    {
        var options = new Qwen35ChatOptions { MaxNewTokens = MaxNewTokens, Tools = tools, Guidance = guidance };
        var prompt = model.CreatePrompt(messages, tools);
        var clock = Stopwatch.StartNew();
        double ttft = -1;
        GenerationResult? result = null;
        foreach (var update in model.GenerateStreaming(prompt, options))
        {
            if (ttft < 0 && update.TokenIds.Length > 0)
            {
                ttft = clock.Elapsed.TotalMilliseconds;
            }

            result = update.Result ?? result;
        }

        double total = clock.Elapsed.TotalMilliseconds;
        return new GenRun(result!.Text, result.Usage.CompletionTokens, ttft < 0 ? total : ttft, total,
            result.Timings.Prefill.TotalMilliseconds, result.FinishReason.ToString(), result.Usage.PromptTokens);
    }

    private static JsonObject Score(JsonObject c, Qwen35Tool[] tools, GenRun run)
    {
        string expected = c["tool"]!.GetValue<string>();
        var parsed = Qwen35ToolCallParser.Parse(run.Text, tools, enableThinking: false);
        var offered = tools.ToDictionary(t => t.Name, StringComparer.Ordinal);
        string outcome;
        string? called = null;
        bool? argsValid = null, argsMatch = null;
        if (parsed.ToolCalls.Count == 0)
        {
            // A Qwen-style block the parser could not read, or an invented name in a well-formed block?
            var m = System.Text.RegularExpressions.Regex.Match(run.Text, "<function=([^>\n]*)>");
            if (m.Success)
            {
                called = m.Groups[1].Value;
                outcome = offered.ContainsKey(called) ? "malformed" : "invalid_name";
            }
            else
            {
                outcome = "no_call";
            }
        }
        else
        {
            var call = parsed.ToolCalls[0];
            called = call.Name;
            if (!offered.TryGetValue(call.Name, out var tool))
            {
                outcome = "invalid_name";
            }
            else
            {
                outcome = call.Name == expected ? "correct" : "wrong_tool";
                argsValid = ArgumentsValid(tool, call.Arguments);
                if (call.Name == expected)
                {
                    argsMatch = ArgumentsMatch(c["args"] as JsonObject, call.Arguments);
                }
            }
        }

        return new JsonObject
        {
            ["id"] = c["id"]!.GetValue<int>(),
            ["tempting"] = c["tempting"]?.GetValue<bool>() ?? false,
            ["expected"] = expected,
            ["offered"] = tools.Length,
            ["outcome"] = outcome,
            ["called"] = called,
            ["calls"] = parsed.ToolCalls.Count,
            ["args_valid"] = argsValid,
            ["args_match"] = argsMatch,
            ["ttft_ms"] = run.TtftMs,
            ["total_ms"] = run.TotalMs,
            ["prefill_ms"] = run.PrefillMs,
            ["completion_tokens"] = run.Tokens,
            ["prompt_tokens"] = run.PromptTokens,
            ["finish"] = run.Finish,
            ["text"] = run.Text,
        };
    }

    /// <summary>Required parameters present, no undeclared parameters, values of the declared type and enum.</summary>
    private static bool ArgumentsValid(Qwen35Tool tool, JsonElement arguments)
    {
        if (arguments.ValueKind != JsonValueKind.Object || tool.Parameters.ValueKind != JsonValueKind.Object)
        {
            return arguments.ValueKind is JsonValueKind.Object or JsonValueKind.Undefined;
        }

        var properties = tool.Parameters.GetProperty("properties");
        if (tool.Parameters.TryGetProperty("required", out var required))
        {
            foreach (var r in required.EnumerateArray())
            {
                if (!arguments.TryGetProperty(r.GetString()!, out _))
                {
                    return false;
                }
            }
        }

        foreach (var argument in arguments.EnumerateObject())
        {
            if (!properties.TryGetProperty(argument.Name, out var schema))
            {
                return false;
            }

            if (schema.TryGetProperty("enum", out var values) &&
                !values.EnumerateArray().Any(v => JsonElement.DeepEquals(v, argument.Value)))
            {
                return false;
            }

            string type = schema.TryGetProperty("type", out var t) ? t.GetString()! : "string";
            bool ok = type switch
            {
                "integer" => argument.Value.ValueKind == JsonValueKind.Number && argument.Value.TryGetInt64(out _),
                "number" => argument.Value.ValueKind == JsonValueKind.Number,
                "boolean" => argument.Value.ValueKind is JsonValueKind.True or JsonValueKind.False,
                "string" => argument.Value.ValueKind == JsonValueKind.String,
                _ => true,
            };
            if (!ok)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Every labelled argument equal (strings ignoring case and surrounding whitespace).</summary>
    private static bool ArgumentsMatch(JsonObject? labels, JsonElement arguments)
    {
        if (labels is null)
        {
            return true;
        }

        foreach (var (key, value) in labels)
        {
            if (arguments.ValueKind != JsonValueKind.Object || !arguments.TryGetProperty(key, out var actual))
            {
                return false;
            }

            bool equal = value!.GetValueKind() == JsonValueKind.String
                ? actual.ValueKind == JsonValueKind.String &&
                  string.Equals(actual.GetString()!.Trim(), value.GetValue<string>(), StringComparison.OrdinalIgnoreCase)
                : actual.ValueKind == JsonValueKind.Number && actual.GetRawText() == value.ToJsonString();
            if (!equal)
            {
                return false;
            }
        }

        return true;
    }

    private static int Hash(int id, string name)
    {
        unchecked
        {
            int h = id * 7919;
            foreach (char ch in name)
            {
                h = h * 31 + ch;
            }

            return h & 0x7fffffff;
        }
    }
}
