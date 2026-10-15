// Candidate outputs for the task-quality evaluator (G3, workstream R5): `model-pipeline evaluate --g3`.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- retrieval --model e5|embeddinggemma2
//       --dataset <dir with queries.jsonl, corpus.jsonl> --out <embeddings.jsonl> [--snapshot <dir>] [--threads <n>]
//       [--device cpu|cuda[:N]]
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- retrieval
//       --model embeddinggemma2-image|embeddinggemma2-video --dataset <dir> --media <dir> --out <embeddings.jsonl>
//       [--snapshot <dir>] [--threads <n>] [--device cpu|cuda[:N]] [--ffmpeg <FFmpeg 8 lib dir>]
//   dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- ner
//       --dataset <gold.jsonl> --out <predictions.jsonl> [--snapshot <dir>] [--threads <n>] [--device cpu|cuda[:N]]
//
// --device cuda needs the CUDA build (ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda; use_tf32=0, see ToolDevice.cs).
//
// retrieval: one JSON line per query and document, {"kind": "query"|"document", "id", "prompt", "embedding"}, through
// the public text embedders: e5 with the "query: " and "passage: " prefixes (E5TextEmbedder), EmbeddingGemma 2 with
// its query and document prompts (EmbeddingGemma2TextEmbedder). Only each row's "text" is embedded (no title), so
// every producer embeds the same string.
// Image and video retrieval (embeddinggemma2-image, embeddinggemma2-video): text queries against a corpus of media files,
// {"id", "file", "sha256"} rows of corpus.jsonl, read from --media. Every file's SHA-256 is checked before any embedding
// (a mismatch ends the run, exit 2). Everything goes through EmbeddingGemma2MultimodalEmbedder at the package defaults:
// queries with the query prompt (EmbeddingGemma2MultimodalInput.FromText, EmbeddingGemma2Prompt.Query), images
// (SkiaSharp decoder) and videos (FFmpeg 8 adapter, --ffmpeg or the default library search) one at a time without a
// prompt, as upstream's model card says for media. A document the package rejects with a typed per-input error is
// written as {"kind": "document", "id", "rejected": "<type>: <message>"} without an embedding, so the evaluator refuses
// the file; the run continues and exits 1.
// ner: one JSON line per gold row, {"id", "entities": [{"start", "end", "label", "text", "confidence"}]}, from
// Gliner2EntityExtractor.Extract(text, labels) at the extractor's default operating point; offsets are Unicode
// scalar values of the row text (Gliner2Entity.ScalarStart/ScalarEnd), the gold format's unit.
// A row Extract rejects (Gliner2InputTooLongException) is written as {"id", "rejected": "<type>: <message>"} without
// entities, so the evaluator refuses the file instead of scoring the row as empty; the run continues and exits 1.
// The tool applies no budgets and reads nothing under model-pipeline/protected/.
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

internal static class TaskQualityDump
{
    private static readonly JsonSerializerOptions Compact = new() { WriteIndented = false };

    public static int Run(string command, string[] args)
    {
        string? model = null, dataset = null, outPath = null, snapshot = null, media = null, ffmpeg = null;
        int? threads = null;
        var device = ToolDevice.Cpu;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" when i + 1 < args.Length:
                    model = args[++i];
                    break;
                case "--dataset" when i + 1 < args.Length:
                    dataset = args[++i];
                    break;
                case "--out" when i + 1 < args.Length:
                    outPath = args[++i];
                    break;
                case "--snapshot" when i + 1 < args.Length:
                    snapshot = args[++i];
                    break;
                case "--media" when i + 1 < args.Length:
                    media = args[++i];
                    break;
                case "--ffmpeg" when i + 1 < args.Length:
                    ffmpeg = args[++i];
                    break;
                case "--threads" when i + 1 < args.Length && int.TryParse(args[i + 1], out int n) && n > 0:
                    threads = n;
                    i++;
                    break;
                case "--device" when i + 1 < args.Length && ToolDevice.TryParse(args[i + 1], out var parsed):
                    device = parsed;
                    i++;
                    break;
                default:
                    return Usage($"unknown option {args[i]}");
            }
        }

        if (dataset is null || outPath is null)
        {
            return Usage("--dataset and --out are required");
        }

        if (device.Unavailable() is { } unavailable)
        {
            return Usage(unavailable);
        }

        Console.WriteLine($"device: {device.Label}{(device.IsCuda ? $" (cuda device {device.CudaDeviceId}, CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")})" : string.Empty)}");

        try
        {
            return command switch
            {
                "retrieval" when model == "e5" => RetrievalE5(dataset, outPath, snapshot, threads, device),
                "retrieval" when model == "embeddinggemma2" => RetrievalGemma(dataset, outPath, snapshot, threads, device),
                "retrieval" when model is "embeddinggemma2-image" or "embeddinggemma2-video" && media is null => Usage("--media is required for image and video retrieval"),
                "retrieval" when model is "embeddinggemma2-image" or "embeddinggemma2-video" => RetrievalGemmaMedia(model == "embeddinggemma2-video", dataset, media!, outPath, snapshot, threads, device, ffmpeg),
                "retrieval" => Usage("--model must be e5, embeddinggemma2, embeddinggemma2-image or embeddinggemma2-video"),
                _ => Ner(dataset, outPath, snapshot, threads, device),
            };
        }
        catch (Exception ex) when (ex is ModelAcquisitionException or EmbeddingGemma2FFmpegException or InvalidDataException)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }
    }

    private static List<(string Id, string Text)> ReadTexts(string path)
    {
        var rows = new List<(string, string)>();
        foreach (string line in File.ReadLines(path))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            var node = JsonNode.Parse(line)!;
            rows.Add(((string)node["id"]!, (string)node["text"]!));
        }

        return rows;
    }

    private static int WriteEmbeddings(string outPath, string dataset, Func<IReadOnlyList<string>, bool, float[][]> embed, string queryPrompt, string documentPrompt)
    {
        var queries = ReadTexts(Path.Combine(dataset, "queries.jsonl"));
        var corpus = ReadTexts(Path.Combine(dataset, "corpus.jsonl"));
        var sb = new StringBuilder();
        foreach (var (rows, isQuery) in new[] { (queries, true), (corpus, false) })
        {
            for (int start = 0; start < rows.Count; start += 16)
            {
                var chunk = rows.Skip(start).Take(16).ToList();
                var vectors = embed(chunk.Select(r => r.Text).ToList(), isQuery);
                for (int k = 0; k < chunk.Count; k++)
                {
                    AppendRow(sb, isQuery ? "query" : "document", chunk[k].Id, isQuery ? queryPrompt : documentPrompt, vectors[k]);
                }
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, sb.ToString());
        Console.WriteLine($"wrote {outPath}: {queries.Count} queries, {corpus.Count} documents");
        return 0;
    }

    private static int RetrievalE5(string dataset, string outPath, string? snapshot, int? threads, ToolDevice device)
    {
        snapshot ??= Environment.GetEnvironmentVariable("STARLING_E5_SNAPSHOT");
        var files = string.IsNullOrEmpty(snapshot) ? E5ModelAssets.OpenLocal() : E5ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
        using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
        {
            Session = device.Apply(new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256, IntraOpNumThreads = threads }),
        });
        var embedder = new E5TextEmbedder(tokenizer, encoder);
        return WriteEmbeddings(outPath, dataset, (texts, isQuery) =>
            embedder.EmbedBatch(texts.Select(t => (isQuery ? E5TextEmbedder.QueryPrefix : E5TextEmbedder.PassagePrefix) + t).ToList()),
            E5TextEmbedder.QueryPrefix, E5TextEmbedder.PassagePrefix);
    }

    private static int RetrievalGemma(string dataset, string outPath, string? snapshot, int? threads, ToolDevice device)
    {
        snapshot ??= Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        var files = string.IsNullOrEmpty(snapshot) ? EmbeddingGemma2ModelAssets.OpenLocal() : EmbeddingGemma2ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath);
        using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
        {
            Session = device.Apply(new OnnxSessionOptions { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256, IntraOpNumThreads = threads }),
        });
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        return WriteEmbeddings(outPath, dataset, (texts, isQuery) =>
            embedder.EmbedBatch(texts, isQuery ? EmbeddingGemma2Prompt.Query : EmbeddingGemma2Prompt.Document),
            EmbeddingGemma2Prompt.Query.Name, EmbeddingGemma2Prompt.Document.Name);
    }

    /// <summary>The documents of a media corpus: {"id", "file", "sha256"} rows of corpus.jsonl.</summary>
    internal static List<(string Id, string File, string Sha256)> ReadMediaCorpus(string path)
    {
        var rows = new List<(string, string, string)>();
        foreach (string line in File.ReadLines(path))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            var node = JsonNode.Parse(line)!;
            string id = (string?)node["id"] ?? throw new InvalidDataException($"{path}: a row has no id");
            string file = (string?)node["file"] ?? throw new InvalidDataException($"{path}: {id} has no file (not a media corpus)");
            string sha256 = (string?)node["sha256"] ?? throw new InvalidDataException($"{path}: {id} has no sha256");
            if (Path.GetFileName(file) != file || file.StartsWith('.'))
            {
                throw new InvalidDataException($"{path}: {id}: {file} is not a plain file name");
            }

            rows.Add((id, file, sha256));
        }

        return rows;
    }

    /// <summary>The media files that are missing or whose SHA-256 differs from the corpus, as messages.</summary>
    internal static List<string> VerifyMedia(string mediaDir, IEnumerable<(string Id, string File, string Sha256)> corpus)
    {
        var problems = new List<string>();
        foreach (var (id, file, sha256) in corpus)
        {
            string path = Path.Combine(mediaDir, file);
            if (!File.Exists(path))
            {
                problems.Add($"{id}: {path} not found");
                continue;
            }

            using var stream = File.OpenRead(path);
            string actual = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(stream));
            if (!string.Equals(actual, sha256, StringComparison.Ordinal))
            {
                problems.Add($"{id}: {file} has SHA-256 {actual}, corpus.jsonl says {sha256}");
            }
        }

        return problems;
    }

    private static int RetrievalGemmaMedia(bool video, string dataset, string mediaDir, string outPath, string? snapshot, int? threads, ToolDevice device, string? ffmpegDir)
    {
        var queries = ReadTexts(Path.Combine(dataset, "queries.jsonl"));
        var corpus = ReadMediaCorpus(Path.Combine(dataset, "corpus.jsonl"));
        var problems = VerifyMedia(mediaDir, corpus);
        if (problems.Count > 0)
        {
            foreach (string problem in problems)
            {
                Console.Error.WriteLine($"error: {problem}");
            }

            return 2;
        }

        snapshot ??= Environment.GetEnvironmentVariable("STARLING_EMBEDDINGGEMMA2_SNAPSHOT");
        var files = string.IsNullOrEmpty(snapshot) ? EmbeddingGemma2ModelAssets.OpenLocalImageAssets() : EmbeddingGemma2ModelAssets.OpenImageDirectory(snapshot);
        var decoder = video ? new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = ffmpegDir }) : null;
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions
        {
            Session = device.Apply(new OnnxSessionOptions { IntraOpNumThreads = threads }),
        });
        var sb = new StringBuilder();
        for (int start = 0; start < queries.Count; start += 16)
        {
            var chunk = queries.Skip(start).Take(16).ToList();
            var vectors = embedder.EmbedBatch(chunk.Select(q => EmbeddingGemma2MultimodalInput.FromText(q.Text)).ToList(), EmbeddingGemma2Prompt.Query);
            for (int k = 0; k < chunk.Count; k++)
            {
                AppendRow(sb, "query", chunk[k].Id, EmbeddingGemma2Prompt.Query.Name, vectors[k]);
            }
        }

        int rejected = 0;
        var watch = System.Diagnostics.Stopwatch.StartNew();
        foreach (var (id, file, _) in corpus)
        {
            string path = Path.Combine(mediaDir, file);
            try
            {
                float[] vector = video ? embedder.Embed(decoder!.DecodeFile(path).Video) : embedder.Embed(EmbeddingGemma2ImageDecoder.DecodeFile(path));
                AppendRow(sb, "document", id, EmbeddingGemma2Prompt.None.Name, vector);
                Console.WriteLine($"{id}: embedded ({watch.Elapsed.TotalSeconds:F0} s)");
            }
            catch (Exception ex) when (EmbeddingGemma2VideoDump.IsRowRejection(ex) || EmbeddingGemma2ImageDump.IsRowRejection(ex))
            {
                sb.Append(new JsonObject { ["kind"] = "document", ["id"] = id, ["rejected"] = StageWriter.RejectionReason(ex) }.ToJsonString(Compact)).Append('\n');
                Console.Error.WriteLine($"{id}: rejected: {StageWriter.RejectionReason(ex)}");
                rejected++;
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, sb.ToString());
        Console.WriteLine($"wrote {outPath}: {queries.Count} queries, {corpus.Count} {(video ? "videos" : "images")}");
        if (rejected > 0)
        {
            Console.Error.WriteLine($"error: {rejected} of {corpus.Count} documents rejected (recorded with their reason, without an embedding)");
            return 1;
        }

        return 0;
    }

    private static void AppendRow(StringBuilder sb, string kind, string id, string prompt, float[] vector)
    {
        var row = new JsonObject
        {
            ["kind"] = kind,
            ["id"] = id,
            ["prompt"] = prompt,
            ["embedding"] = new JsonArray(vector.Select(v => (JsonNode?)v).ToArray()),
        };
        sb.Append(row.ToJsonString(Compact)).Append('\n');
    }

    private static int Ner(string dataset, string outPath, string? snapshot, int? threads, ToolDevice device)
    {
        snapshot ??= Environment.GetEnvironmentVariable("STARLING_GLINER2_SNAPSHOT");
        var files = string.IsNullOrEmpty(snapshot) ? Gliner2ModelAssets.OpenLocal() : Gliner2ModelAssets.OpenDirectory(snapshot, CacheVerification.Sha256);
        using var extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions
        {
            Session = device.Apply(new OnnxSessionOptions { IntraOpNumThreads = threads }),
        });
        var sb = new StringBuilder();
        int rows = 0, rejected = 0;
        foreach (string line in File.ReadLines(dataset))
        {
            if (line.Trim().Length == 0)
            {
                continue;
            }

            var node = JsonNode.Parse(line)!;
            string text = (string)node["text"]!;
            var labels = node["labels"]!.AsArray().Select(l => (string)l!).ToList();
            Gliner2ExtractionResult result;
            try
            {
                result = extractor.Extract(text, labels);
            }
            catch (Gliner2InputTooLongException ex)
            {
                // A typed rejection is recorded without entities (never as an empty prediction): the evaluator refuses
                // the file rather than scoring the row as finding nothing.
                string id = (string)node["id"]!;
                sb.Append(new JsonObject { ["id"] = id, ["rejected"] = StageWriter.RejectionReason(ex) }.ToJsonString(Compact)).Append('\n');
                Console.Error.WriteLine($"{id}: rejected: {StageWriter.RejectionReason(ex)}");
                rows++;
                rejected++;
                continue;
            }

            var entities = new JsonArray(result.Entities.Select(e => (JsonNode?)new JsonObject
            {
                ["start"] = e.ScalarStart,
                ["end"] = e.ScalarEnd,
                ["label"] = e.Label,
                ["text"] = e.Text,
                ["confidence"] = e.Confidence,
            }).ToArray());
            sb.Append(new JsonObject { ["id"] = (string)node["id"]!, ["entities"] = entities }.ToJsonString(Compact)).Append('\n');
            rows++;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllText(outPath, sb.ToString());
        Console.WriteLine($"wrote {outPath}: {rows} rows");
        if (rejected > 0)
        {
            Console.Error.WriteLine($"error: {rejected} of {rows} rows rejected (recorded with their reason, without entities)");
            return 1;
        }

        return 0;
    }

    private static int Usage(string error)
    {
        Console.Error.WriteLine($"error: {error}");
        Console.Error.WriteLine("usage: Starlings.Fidelity.Tool retrieval --model e5|embeddinggemma2 --dataset <dir> --out <embeddings.jsonl> [--snapshot <dir>] [--threads <n>] [--device cpu|cuda[:N]]");
        Console.Error.WriteLine("       Starlings.Fidelity.Tool retrieval --model embeddinggemma2-image|embeddinggemma2-video --dataset <dir> --media <dir> --out <embeddings.jsonl> [--snapshot <dir>] [--threads <n>] [--device cpu|cuda[:N]] [--ffmpeg <FFmpeg 8 lib dir>]");
        Console.Error.WriteLine("       Starlings.Fidelity.Tool ner --dataset <gold.jsonl> --out <predictions.jsonl> [--snapshot <dir>] [--threads <n>] [--device cpu|cuda[:N]]");
        return 2;
    }
}
