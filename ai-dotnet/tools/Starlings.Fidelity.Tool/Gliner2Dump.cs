// GLiNER2 stage results (P-015) for `model-pipeline evaluate`, task "entity-extraction": stage results format
// version 2 (model-pipeline/schemas/stage-results.schema.json; outputs per stage in
// model_pipeline.results.ENTITY_EXTRACTION_OUTPUTS). Runs the C# GLiNER2 path on every case of the protected
// extraction.jsonl and writes:
//
//   tokenizer.results.json   stage "tokenizer" (stage 1, the owned C# processor): normalized_text, words,
//                            start_map, end_map, input_ids, attention_mask, word_index, schema_index.
//   dotnet.results.json      stage "dotnet": from the fixture inputs (stage 3) input_ids, word_index, schema_index,
//                            count_logits, span_logits (valid spans only, label/start word/width order, the layout
//                            of span_logits_valid) and entities; from text (stage 4) text_entities; padded batches
//                            of span_logits (fixture inputs) and text_entities (text path), grouped by label count.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- gliner2 --out <dir>
//       [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]
//
// Entity offsets are Unicode scalar values of the normalized text, exclusive end (the fixtures' unit). From the
// fixture inputs the entities are GLiNER2's spans as decoded (normalized-text offsets); text_entities are the
// package's public results (Gliner2Entity.ScalarStart/ScalarEnd index the original text, a prefix of the normalized
// text, so the unit is the same; they differ from GLiNER2's only for spans over the appended "." , none of which the
// fixtures contain).
//
// Cases outside the C# contract still get outputs, through internal paths of the package, so the report covers every
// protected case: GLiNER2's max_len word cap (unsupported in the public API) and inputs over 512 tokens (rejected by
// the public API; the graph runs them). They are listed in environment.cases_outside_contract. The tool applies no
// bounds and writes nothing under model-pipeline/protected/.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

internal static class Gliner2Dump
{
    private const string Task = "entity-extraction";

    public static int Run(string[] args)
    {
        string? outDir = null;
        string? snapshot = Environment.GetEnvironmentVariable("STARLING_GLINER2_SNAPSHOT");
        string? fixturesDir = null;
        string profile = "gliner2-base-v1/onnx-fp32-cpu";
        int? threads = null;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--out" when i + 1 < args.Length:
                    outDir = args[++i];
                    break;
                case "--snapshot" when i + 1 < args.Length:
                    snapshot = args[++i];
                    break;
                case "--fixtures" when i + 1 < args.Length:
                    fixturesDir = args[++i];
                    break;
                case "--profile" when i + 1 < args.Length:
                    profile = args[++i];
                    break;
                case "--threads" when i + 1 < args.Length && int.TryParse(args[i + 1], out int n) && n > 0:
                    threads = n;
                    i++;
                    break;
                default:
                    return Usage($"unknown option {args[i]}");
            }
        }

        if (outDir is null)
        {
            return Usage("--out is required");
        }

        string? repoRoot = ToolSupport.FindRepositoryRoot();
        fixturesDir ??= repoRoot is null ? null : Path.Combine(repoRoot, "model-pipeline", "protected", "gliner2-base-v1", "onnx-fp32-cpu");
        string? fixturePath = fixturesDir is null ? null : Path.Combine(fixturesDir, "extraction.jsonl");
        if (fixturePath is null || !File.Exists(fixturePath))
        {
            return Usage($"extraction fixtures not found ({fixturePath ?? "no repository root"}); pass --fixtures");
        }

        Gliner2ModelFiles files;
        try
        {
            files = string.IsNullOrEmpty(snapshot) ? Gliner2ModelAssets.OpenLocal() : Gliner2ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
        }
        catch (ModelAcquisitionException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        Directory.CreateDirectory(outDir);
        string? commit = ToolSupport.GitHead(repoRoot);
        using var extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions
        {
            Session = new OnnxSessionOptions { IntraOpNumThreads = threads },
        });
        var d = extractor.Session.Diagnostics;
        var tokenizerAsset = new Asset(Gliner2ModelAssets.TokenizerFile, Gliner2ModelAssets.TokenizerSha256);
        var modelAsset = new Asset(Gliner2ModelAssets.ModelFile, d.ModelSha256 ?? ToolSupport.Sha256(files.ModelPath));

        var cases = ReadCases(fixturePath);
        var outside = new Dictionary<string, string>(StringComparer.Ordinal);
        var textInputs = new Dictionary<string, Gliner2ModelInput>(StringComparer.Ordinal);
        foreach (var c in cases)
        {
            var input = extractor.PrepareOutsideContract(c.Text, c.LabelsRequested, c.Descriptions, c.MaxLen);
            textInputs[c.Id] = input;
            if (c.MaxLen is not null)
            {
                outside[c.Id] = $"max_len word cap {c.MaxLen} (unsupported in the public API; internal path)";
            }
            else if (input.TokenCount > extractor.Options.MaxSequenceLength)
            {
                outside[c.Id] = $"{input.TokenCount} tokens > {extractor.Options.MaxSequenceLength} (the public API rejects it; internal path without the limit)";
            }
        }

        var environment = StageEnvironment(d, outside);

        // Stage 1: the owned processor.
        var tokenizerStage = new StageWriter(profile, "tokenizer", "Starlings.Models.Gliner2 processor (Python-compatible normalization, splitter and lower-casing; schema prefix through the owned tokenizer, profile gliner2-base-v1-unigram)", commit, environment, [tokenizerAsset], Task);
        foreach (var c in cases)
        {
            var input = textInputs[c.Id];
            tokenizerStage.Case(c.Id)
                .Add("normalized_text", input.NormalizedText)
                .Add("words", input.Words.ToArray())
                .Add("start_map", Longs(input.WordStarts))
                .Add("end_map", Longs(input.WordEnds))
                .Add("input_ids", input.InputIds.ToArray())
                .Add("attention_mask", Enumerable.Repeat(1L, input.TokenCount).ToArray())
                .Add("word_index", input.WordIndex.ToArray())
                .Add("schema_index", input.SchemaIndex.ToArray());
        }

        // Stage 3: the C# runtime and decoder on the fixture inputs.
        var dotnetStage = new StageWriter(profile, "dotnet", $"C# Gliner2EntityExtractor: RunModel + decoder on the fixture inputs, and Prepare/RunModel/Decode from text; ONNX Runtime {d.OrtVersion} CPU (managed binding {d.ManagedBindingVersion})", commit, environment, [modelAsset, tokenizerAsset], Task);
        var fixtureInputs = cases.ToDictionary(c => c.Id, c => c.FixtureInput(), StringComparer.Ordinal);
        foreach (var c in cases)
        {
            var input = fixtureInputs[c.Id];
            var o = extractor.RunModelOutsideContract([input])[0];
            dotnetStage.Case(c.Id)
                .Add("input_ids", input.InputIds.ToArray())
                .Add("word_index", input.WordIndex.ToArray())
                .Add("schema_index", input.SchemaIndex.ToArray())
                .Add("count_logits", o.CountLogits.ToArray())
                .Add("span_logits", o.GetValidSpanLogits())
                .Add("entities", Upstream(input, o, c.Threshold));
        }

        // Stage 4: from text.
        foreach (var c in cases)
        {
            var input = textInputs[c.Id];
            var o = extractor.RunModelOutsideContract([input])[0];
            dotnetStage.Case(c.Id).Add("text_entities", Public(extractor.Decode(input, o, c.Threshold)));
        }

        // Padded batches: one label count per batch (the graph's input contract), up to 8 rows, corpus order; only
        // chunks with more than one row (rows of different lengths pad).
        foreach (var (name, chunk) in Chunks(cases, c => fixtureInputs[c.Id].Labels.Count))
        {
            var outputs = extractor.RunModelOutsideContract(chunk.Select(c => fixtureInputs[c.Id]).ToList());
            dotnetStage.Batch($"span-logits-{name}", "span_logits", chunk.Select(c => c.Id).ToList(), outputs.Select(o => o.GetValidSpanLogits()).ToList());
        }

        foreach (var (name, chunk) in Chunks(cases.Where(c => !outside.ContainsKey(c.Id)).ToList(), c => textInputs[c.Id].Labels.Count))
        {
            var inputs = chunk.Select(c => textInputs[c.Id]).ToList();
            var outputs = extractor.RunModel(inputs);
            var rows = chunk.Select((c, i) => Public(extractor.Decode(inputs[i], outputs[i], c.Threshold))).ToList();
            dotnetStage.Batch($"text-entities-{name}", "text_entities", chunk.Select(c => c.Id).ToList(), rows);
        }

        tokenizerStage.Write(Path.Combine(outDir, "tokenizer.results.json"));
        dotnetStage.Write(Path.Combine(outDir, "dotnet.results.json"));
        Console.WriteLine($"wrote {Path.Combine(outDir, "tokenizer.results.json")} and dotnet.results.json: {cases.Count} cases ({outside.Count} outside the contract, internal paths)");
        return 0;
    }

    private static IEnumerable<(string Name, List<FixtureCaseG2> Chunk)> Chunks(List<FixtureCaseG2> cases, Func<FixtureCaseG2, int> labelCount)
    {
        foreach (var group in cases.GroupBy(labelCount).OrderBy(g => g.Key))
        {
            var list = group.ToList();
            for (int start = 0, part = 0; start < list.Count; start += 8, part++)
            {
                var chunk = list.Skip(start).Take(8).ToList();
                if (chunk.Count > 1)
                {
                    yield return ($"labels{group.Key}-part{part}", chunk);
                }
            }
        }
    }

    private static EntityResult Upstream(Gliner2ModelInput input, Gliner2ModelOutput output, float threshold)
    {
        var spans = Gliner2Decoder.DecodeUpstream(input, output, threshold);
        var items = spans is null
            ? []
            : spans.SelectMany((list, label) => list.Select(s => new EntityItem(input.Labels[label], s.Text, s.Start, s.End, s.Confidence))).ToList();
        return new EntityResult(spans is not null, items);
    }

    private static EntityResult Public(Gliner2ExtractionResult result) =>
        new(result.GateOpen, result.Entities.Select(e => new EntityItem(e.Label, e.Text, e.ScalarStart, e.ScalarEnd, e.Confidence)).ToList());

    private static long[] Longs(ReadOnlySpan<int> values)
    {
        var result = new long[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = values[i];
        }

        return result;
    }

    private static Dictionary<string, object?> StageEnvironment(OnnxSessionDiagnostics d, Dictionary<string, string> outside) => new()
    {
        ["machine"] = System.Environment.MachineName,
        ["provider"] = "cpu",
        ["provider_options"] = d.ProviderOptions.ToDictionary(kv => kv.Key, kv => (object?)kv.Value),
        ["dotnet"] = RuntimeInformation.FrameworkDescription,
        ["os"] = RuntimeInformation.OSDescription,
        ["process_architecture"] = RuntimeInformation.ProcessArchitecture.ToString(),
        ["cpu_count"] = System.Environment.ProcessorCount,
        ["cpu"] = ToolSupport.CpuModel(),
        ["onnxruntime"] = d.OrtVersion,
        ["onnxruntime_managed"] = d.ManagedBindingVersion,
        ["session_providers"] = d.SessionProviders,
        ["intra_op_threads"] = d.IntraOpNumThreads,
        ["graph_optimization_level"] = d.GraphOptimizationLevel,
        ["package"] = $"Starlings.Models.Gliner2 {typeof(Gliner2EntityExtractor).Assembly.GetName().Version}",
        ["cases_outside_contract"] = outside,
    };

    private static List<FixtureCaseG2> ReadCases(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n')
            .Where(l => l.Trim().Length > 0)
            .Select(l =>
            {
                using var doc = JsonDocument.Parse(l);
                return FixtureCaseG2.Parse(doc.RootElement);
            })
            .ToList();

    private static int Usage(string error)
    {
        Console.Error.WriteLine($"error: {error}");
        Console.Error.WriteLine("usage: Starlings.Fidelity.Tool gliner2 --out <dir> [--snapshot <dir>] [--fixtures <dir>] [--profile <id>] [--threads <n>]");
        return 2;
    }
}

/// <summary>One case of extraction.jsonl: the request, and the fixture's own model inputs for the ids path.</summary>
internal sealed record FixtureCaseG2(
    string Id,
    string Text,
    List<string> LabelsRequested,
    Dictionary<string, string>? Descriptions,
    float Threshold,
    int? MaxLen,
    string NormalizedText,
    List<string> Words,
    int[] StartMap,
    int[] EndMap,
    List<string> SchemaTokens,
    List<string> Labels,
    long[] InputIds,
    long[] WordIndex,
    long[] SchemaIndex)
{
    public static FixtureCaseG2 Parse(JsonElement r)
    {
        Dictionary<string, string>? descriptions = r.GetProperty("descriptions").ValueKind == JsonValueKind.Object
            ? r.GetProperty("descriptions").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal)
            : null;
        return new FixtureCaseG2(
            r.GetProperty("id").GetString()!,
            r.GetProperty("text").GetString()!,
            Strings(r.GetProperty("labels_requested")),
            descriptions,
            (float)r.GetProperty("threshold").GetDouble(),
            r.GetProperty("max_len").ValueKind == JsonValueKind.Number ? r.GetProperty("max_len").GetInt32() : null,
            r.GetProperty("normalized_text").GetString()!,
            Strings(r.GetProperty("words")),
            r.GetProperty("start_map").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            r.GetProperty("end_map").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Strings(r.GetProperty("schema_tokens")),
            Strings(r.GetProperty("labels")),
            r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
            r.GetProperty("word_index").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
            r.GetProperty("schema_index").EnumerateArray().Select(e => e.GetInt64()).ToArray());
    }

    /// <summary>The fixture's model inputs as a <see cref="Gliner2ModelInput"/> (no C# preprocessing involved).</summary>
    public Gliner2ModelInput FixtureInput()
    {
        int[] normalized = NormalizedText.EnumerateRunes().Select(r => r.Value).ToArray();
        int original = Text.EnumerateRunes().Count();
        return new Gliner2ModelInput(Text, NormalizedText, normalized, original, Labels, SchemaTokens, Words, StartMap, EndMap, InputIds, WordIndex, SchemaIndex);
    }

    private static List<string> Strings(JsonElement a) => a.EnumerateArray().Select(e => e.GetString()!).ToList();
}
