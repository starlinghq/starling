// Candidate outputs for a tokenizer fixture file (G1 on a tokenizer holdout or release set):
// `model-pipeline evaluate --tokenizer-fixtures`.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- tokenizer --model e5|embeddinggemma2|qwen35|gliner2
//       --tokenizer <tokenizer.json> --fixtures <tokenizer fixture .jsonl> --out <dir>
//
// Reads only each fixture row's "id" and "text" (gliner2: also the labels in its recorded schema) from the fixture file,
// named explicitly; no default path, so a release set is addressed only by its path.
// e5, embeddinggemma2, qwen35: the owned tokenizer (Starlings.Tokenizers, the pinned tokenizer.json as the file
// configures it, special tokens added) per text: {"id", "encoding": {"ids", "tokens", "type_ids", "attention_mask",
// "special_tokens_mask", "offsets": {"unicode-scalars", "utf16-code-units", "utf8-bytes"}}, "decode":
// {"skip_special_tokens", "keep_special_tokens"}}, the fields of the reference fixture format.
// gliner2: Gliner2Preprocessor.Build(text, labels) per row: {"id", "whitespace_splitter": {"normalized_text", "words":
// [{"word", "offsets": {"unicode-scalars"}}], "schema_tokens", "input_ids", "text_word_first_positions",
// "schema_special_positions"}}.
// Writes tokenizer.candidate.jsonl and environment.json (tokenizer digest, commit) into --out. Compares nothing and
// writes nothing under model-pipeline/protected/.
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.Gliner2;
using Starlings.Tokenizers;

internal static class TokenizerDump
{
    private static readonly JsonSerializerOptions Compact = new() { WriteIndented = false };

    public static int Run(string[] args)
    {
        string? model = null, tokenizerPath = null, fixtures = null, outDir = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" when i + 1 < args.Length:
                    model = args[++i];
                    break;
                case "--tokenizer" when i + 1 < args.Length:
                    tokenizerPath = args[++i];
                    break;
                case "--fixtures" when i + 1 < args.Length:
                    fixtures = args[++i];
                    break;
                case "--out" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
                default:
                    return Usage($"unknown option {args[i]}");
            }
        }

        if (model is null || tokenizerPath is null || fixtures is null || outDir is null)
        {
            return Usage("--model, --tokenizer, --fixtures and --out are required");
        }

        // The digest the model's library pins (Qwen3.5: its manifest's processing.tokenizer asset; the evaluator
        // checks the recorded digest against the manifest for every model).
        string? pinned = model switch
        {
            "e5" => E5ModelAssets.TokenizerSha256,
            "embeddinggemma2" => EmbeddingGemma2ModelAssets.TokenizerSha256,
            "gliner2" => Gliner2ModelAssets.TokenizerSha256,
            "qwen35" => null,
            _ => "unknown",
        };
        if (pinned == "unknown")
        {
            return Usage("--model must be e5, embeddinggemma2, qwen35 or gliner2");
        }

        if (!File.Exists(fixtures) || !File.Exists(tokenizerPath))
        {
            return Usage($"not found: {(File.Exists(fixtures) ? tokenizerPath : fixtures)}");
        }

        string sha256 = ToolSupport.Sha256(tokenizerPath);
        if (pinned is not null && !string.Equals(sha256, pinned, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine($"error: {tokenizerPath}: SHA-256 {sha256} is not the pinned {pinned}");
            return 2;
        }

        var sb = new StringBuilder();
        int rows = 0;
        if (model == "gliner2")
        {
            var preprocessor = Gliner2Preprocessor.FromFile(tokenizerPath, Gliner2ModelAssets.TokenizerSha256);
            foreach (var row in ReadRows(fixtures))
            {
                sb.Append(Gliner2Row(preprocessor, row).ToJsonString(Compact)).Append('\n');
                rows++;
            }
        }
        else
        {
            var tokenizer = Tokenizer.FromFile(tokenizerPath);
            foreach (var row in ReadRows(fixtures))
            {
                sb.Append(EncodingRow(tokenizer, row).ToJsonString(Compact)).Append('\n');
                rows++;
            }
        }

        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "tokenizer.candidate.jsonl");
        File.WriteAllText(outPath, sb.ToString());
        string? repoRoot = ToolSupport.FindRepositoryRoot();
        var environment = new JsonObject
        {
            ["format"] = "starlings.tokenizer-candidate",
            ["format_version"] = 1,
            ["model"] = model,
            ["producer"] = new JsonObject
            {
                ["name"] = "Starlings.Fidelity.Tool tokenizer",
                ["repository_commit"] = ToolSupport.GitHead(repoRoot),
                ["tokenizer_library"] = typeof(Tokenizer).Assembly.GetName().Version?.ToString(),
                ["tokenizer_library_informational_version"] = typeof(Tokenizer).Assembly
                    .GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                    .OfType<System.Reflection.AssemblyInformationalVersionAttribute>().FirstOrDefault()?.InformationalVersion,
                ["dotnet"] = System.Environment.Version.ToString(),
            },
            ["tokenizer"] = new JsonObject { ["path"] = Path.GetFileName(tokenizerPath), ["sha256"] = sha256 },
            ["fixtures"] = new JsonObject { ["path"] = fixtures, ["sha256"] = ToolSupport.Sha256(fixtures) },
            ["rows"] = rows,
        };
        File.WriteAllText(Path.Combine(outDir, "environment.json"), environment.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        Console.WriteLine($"wrote {outPath}: {rows} rows");
        return 0;
    }

    private static IEnumerable<JsonNode> ReadRows(string path)
    {
        foreach (string line in File.ReadLines(path))
        {
            if (line.Trim().Length > 0)
            {
                yield return JsonNode.Parse(line)!;
            }
        }
    }

    private static JsonObject EncodingRow(Tokenizer tokenizer, JsonNode row)
    {
        string text = (string)row["text"]!;
        var encoding = tokenizer.Encode(text, addSpecialTokens: true);
        int n = encoding.Count;
        var ids = new int[n];
        var scalars = new JsonArray();
        var utf16 = new JsonArray();
        var utf8 = new JsonArray();
        var utf8Prefix = Utf8Prefix(text);
        for (int i = 0; i < n; i++)
        {
            ids[i] = encoding.Ids[i];
            var s = encoding.ScalarOffsets[i];
            var u = encoding.Offsets[i];
            scalars.Add(new JsonArray(s.Start, s.End));
            utf16.Add(new JsonArray(u.Start, u.End));
            utf8.Add(new JsonArray(utf8Prefix[u.Start], utf8Prefix[u.End]));
        }

        return new JsonObject
        {
            ["id"] = (string)row["id"]!,
            ["encoding"] = new JsonObject
            {
                ["ids"] = Array(ids.Select(v => (long)v)),
                ["tokens"] = new JsonArray(Enumerable.Range(0, n).Select(i => (JsonNode?)encoding.Tokens[i]).ToArray()),
                ["type_ids"] = Array(Enumerable.Range(0, n).Select(i => (long)encoding.TypeIds[i])),
                ["attention_mask"] = Array(Enumerable.Range(0, n).Select(i => (long)encoding.AttentionMask[i])),
                ["special_tokens_mask"] = Array(Enumerable.Range(0, n).Select(i => (long)encoding.SpecialTokensMask[i])),
                ["offsets"] = new JsonObject { ["unicode-scalars"] = scalars, ["utf16-code-units"] = utf16, ["utf8-bytes"] = utf8 },
            },
            ["decode"] = new JsonObject
            {
                ["skip_special_tokens"] = tokenizer.Decode(ids, skipSpecialTokens: true),
                ["keep_special_tokens"] = tokenizer.Decode(ids, skipSpecialTokens: false),
            },
        };
    }

    /// <summary>UTF-8 byte offset of every UTF-16 position of the text (length + 1 entries).</summary>
    private static int[] Utf8Prefix(string text)
    {
        var prefix = new int[text.Length + 1];
        int bytes = 0;
        for (int i = 0; i < text.Length; i++)
        {
            prefix[i] = bytes;
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                bytes += 4;
                prefix[++i] = bytes - 2; // inside a pair: not a valid boundary, never an offset
            }
            else
            {
                bytes += text[i] < 0x80 ? 1 : text[i] < 0x800 ? 2 : 3;
            }
        }

        prefix[text.Length] = bytes;
        return prefix;
    }

    private static JsonObject Gliner2Row(Gliner2Preprocessor preprocessor, JsonNode row)
    {
        // Labels are the tokens after each [E] marker of the recorded schema (the fixture's request).
        var schema = row["whitespace_splitter"]!["schema_tokens"]![0]!.AsArray().Select(t => (string)t!).ToList();
        var labels = Enumerable.Range(0, schema.Count).Where(i => schema[i] == "[E]").Select(i => schema[i + 1]).ToList();
        var input = preprocessor.Build((string)row["text"]!, Gliner2Preprocessor.NormalizeLabels(labels, "labels"), null);
        var words = new JsonArray();
        for (int w = 0; w < input.WordCount; w++)
        {
            words.Add(new JsonObject
            {
                ["word"] = input.Words[w],
                ["offsets"] = new JsonObject { ["unicode-scalars"] = new JsonArray(input.WordStarts[w], input.WordEnds[w]) },
            });
        }

        return new JsonObject
        {
            ["id"] = (string)row["id"]!,
            ["whitespace_splitter"] = new JsonObject
            {
                ["normalized_text"] = input.NormalizedText,
                ["words"] = words,
                ["schema_tokens"] = new JsonArray(new JsonArray(input.SchemaTokens.Select(t => (JsonNode?)t).ToArray())),
                ["input_ids"] = Array(input.InputIds.ToArray()),
                ["text_word_first_positions"] = Array(input.WordIndex.ToArray()),
                ["schema_special_positions"] = new JsonArray(Array(input.SchemaIndex.ToArray())),
            },
        };
    }

    private static JsonArray Array(IEnumerable<long> values) => new(values.Select(v => (JsonNode?)v).ToArray());

    private static int Usage(string error)
    {
        Console.Error.WriteLine($"error: {error}");
        Console.Error.WriteLine("usage: Starlings.Fidelity.Tool tokenizer --model e5|embeddinggemma2|qwen35|gliner2 --tokenizer <tokenizer.json> --fixtures <fixture.jsonl> --out <dir>");
        return 2;
    }
}
