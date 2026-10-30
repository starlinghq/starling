// PP-OCRv5 development-parity evidence for model-pipeline's ppocr-onnx-cpu-v1 route. For each corpus image:
//   dotnet-image:       the PNG decoded (SkiaSharp, through the EmbeddingGemma 2 decoder), the detector input, and the
//                       public PaddleOcrEngine.Recognize lines (boxes, box scores, text, confidence);
//   dotnet-tensors:     the reference detector input and each reference recognizer input through the .NET sessions
//                       (probability map, recognizer probabilities), and the CTC text of those probabilities;
//   dotnet-postprocess: the owned DB post-processing on the reference probability map (boxes and box scores);
//   dotnet-crop:        the recognizer input cropped by the .NET code from each reference box.
// Writes dotnet.json plus a float32 sidecar in the layout run_ppocr.py uses; the evaluator compares them.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Models.PaddleOcr;
using Starlings.Runtime.Onnx;

return ToolMain.Run(ToolBody);

int ToolBody()
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || args[i] is not ("--profile" or "--snapshot" or "--corpus" or "--reference" or "--out" or "--threads")
            || !options.TryAdd(args[i], args[i + 1]))
        {
            Console.Error.WriteLine("usage: --profile <candidate.json> --snapshot <dir with det/ and rec/> --corpus <ocr-corpus.jsonl> --reference <reference.json> --out <dir> [--threads 2]");
            return 2;
        }
    }

    if (Settings.RequiredOptions.Any(key => !options.ContainsKey(key)))
    {
        Console.Error.WriteLine("missing --profile, --snapshot, --corpus, --reference, or --out");
        return 2;
    }

    if (!BitConverter.IsLittleEndian)
    {
        throw new PlatformNotSupportedException("Fidelity tensor sidecars require a little-endian host.");
    }

    int threads = options.TryGetValue("--threads", out var rawThreads) ? int.Parse(rawThreads, System.Globalization.CultureInfo.InvariantCulture) : 2;
    string profilePath = options["--profile"];
    using var profileDocument = JsonDocument.Parse(File.ReadAllText(profilePath));
    var profile = profileDocument.RootElement;
    Contract.Check(profile);
    string profileId = profile.GetProperty("id").GetString()!;
    string profileHash = Sha256(profilePath);

    string referencePath = options["--reference"];
    using var referenceDocument = JsonDocument.Parse(File.ReadAllText(referencePath));
    var reference = referenceDocument.RootElement;
    if (reference.GetProperty("profile").GetString() != profileId || reference.GetProperty("candidate_sha256").GetString() != profileHash)
    {
        throw new InvalidDataException("Reference inputs do not belong to the selected candidate contract.");
    }

    var referenceTensors = ReferenceTensors.Read(referencePath, reference);
    var corpus = ReadCorpus(options["--corpus"]);
    string snapshot = options["--snapshot"];
    var files = PaddleOcrModelAssets.OpenDirectories(Path.Combine(snapshot, "det"), Path.Combine(snapshot, "rec"));
    string output = Path.GetFullPath(options["--out"]);
    Directory.CreateDirectory(output);
    var session = new OnnxSessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = 1 };
    using var engine = PaddleOcrEngine.Create(files, new PaddleOcrOptions { DetectorSession = session, RecognizerSession = session });

    var rows = new List<Dictionary<string, object>>();
    var layout = new List<Dictionary<string, object>>();
    string tensorPath = Path.Combine(output, "dotnet-tensors.f32");
    PaddleOcrImage? firstImage = null;
    using (var tensors = File.Create(tensorPath))
    {
        foreach (var source in reference.GetProperty("cases").EnumerateArray())
        {
            string id = source.GetProperty("id").GetString()!;
            if (!corpus.TryGetValue(id, out var entry))
            {
                throw new InvalidDataException($"{id}: the reference names an image the corpus does not hold.");
            }

            var decoded = EmbeddingGemma2ImageDecoder.DecodeFile(entry.Path);
            var image = PaddleOcrImage.FromRgb24(decoded.Width, decoded.Height, decoded.Pixels.Span);
            firstImage ??= image;

            // dotnet-image: the public path, plus its own detector input.
            var input = PaddleOcrProcessing.Detector(image);
            Write(tensors, layout, id, "det_input", null, input.Tensor, [1, 3, input.Height, input.Width]);
            var lines = engine.Recognize(image);

            // dotnet-tensors: the reference detector input through the .NET detector session.
            var (referenceInput, inputShape) = referenceTensors.Get(id, "det_input", null);
            int height = inputShape[2], width = inputShape[3];
            var probability = engine.RunDetector(referenceInput, height, width, CancellationToken.None);
            Write(tensors, layout, id, "prob_map", null, probability, [1, 1, height, width]);

            // dotnet-postprocess: the owned DB post-processing on the reference probability map.
            var (referenceProbability, probabilityShape) = referenceTensors.Get(id, "prob_map", null);
            if (probabilityShape[2] != height || probabilityShape[3] != width)
            {
                throw new InvalidDataException($"{id}: reference probability map and detector input sizes differ.");
            }

            var postprocessed = PaddleOcrProcessing.Sort(PaddleOcrProcessing.PostProcess(referenceProbability, height, width, image.Height, image.Width));

            // dotnet-crop and dotnet-tensors per reference line.
            var forced = new List<Dictionary<string, object>>();
            var referenceLines = source.GetProperty("lines").EnumerateArray().ToArray();
            for (int index = 0; index < referenceLines.Length; index++)
            {
                var box = referenceLines[index].GetProperty("box").EnumerateArray().SelectMany(point => point.EnumerateArray().Select(value => value.GetInt32())).ToArray();
                var crop = PaddleOcrProcessing.Crop(image, box)
                    ?? throw new InvalidDataException($"{id}/{index}: the reference box gives an empty crop.");
                var (cropTensor, cropWidth) = PaddleOcrProcessing.Recognizer(crop.Pixels, crop.Height, crop.Width);
                Write(tensors, layout, id, "rec_input", index, cropTensor, [1, 3, 48, cropWidth]);

                var (recognizerInput, recognizerShape) = referenceTensors.Get(id, "rec_input", index);
                var (probabilities, steps) = engine.RunRecognizer(recognizerInput, recognizerShape[3], CancellationToken.None);
                Write(tensors, layout, id, "rec_probs", index, probabilities, [1, steps, engine.Configuration.Characters.Count]);
                var (text, confidence) = PaddleOcrProcessing.Decode(probabilities, steps, engine.Configuration.Characters);
                forced.Add(new Dictionary<string, object> { ["text"] = text, ["score"] = (double)confidence });
            }

            rows.Add(new Dictionary<string, object>
            {
                ["id"] = id,
                ["image_sha256"] = entry.Sha256,
                ["lines"] = lines.Select(line => new Dictionary<string, object>
                {
                    ["box"] = Points(line.Box.ToArray()),
                    ["box_score"] = (double)line.DetectionScore,
                    ["text"] = line.Text,
                    ["score"] = (double)line.Confidence,
                }).ToArray(),
                ["postprocess_lines"] = postprocessed.Select(box => new Dictionary<string, object>
                {
                    ["box"] = Points(box.Points),
                    ["box_score"] = box.Score,
                    ["text"] = string.Empty,
                    ["score"] = 0.0,
                }).ToArray(),
                ["forced"] = forced,
            });
        }
    }

    // A request whose token is already cancelled must not run either graph.
    bool cancellationVerified = false;
    using (var cancelled = new CancellationTokenSource())
    {
        cancelled.Cancel();
        try
        {
            engine.Recognize(firstImage ?? throw new InvalidDataException("No images."), cancelled.Token);
        }
        catch (OperationCanceledException)
        {
            cancellationVerified = true;
        }
    }

    var result = new Dictionary<string, object>
    {
        ["format"] = "starlings.ppocr-candidate-results",
        ["format_version"] = 1,
        ["profile"] = profileId,
        ["task"] = profile.GetProperty("task").GetString()!,
        ["candidate_sha256"] = profileHash,
        ["stage"] = "dotnet",
        ["reference_status"] = "proposed",
        ["reference_sha256"] = Sha256(referencePath),
        ["assets"] = new[]
        {
            new { path = "det/" + PaddleOcrModelAssets.GraphFile, sha256 = PaddleOcrModelAssets.DetectorSha256 },
            new { path = "det/" + PaddleOcrModelAssets.ConfigFile, sha256 = PaddleOcrModelAssets.DetectorConfigSha256 },
            new { path = "rec/" + PaddleOcrModelAssets.GraphFile, sha256 = PaddleOcrModelAssets.RecognizerSha256 },
            new { path = "rec/" + PaddleOcrModelAssets.ConfigFile, sha256 = PaddleOcrModelAssets.RecognizerConfigSha256 },
        },
        ["environment"] = new
        {
            machine = Environment.MachineName,
            dotnet = RuntimeInformation.FrameworkDescription,
            onnxruntime = OnnxRuntimeInfo.NativeVersion,
            provider = "CPUExecutionProvider",
            threads,
            inter_op_threads = 1,
            image_decoder = "Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp",
        },
        ["cases"] = rows,
        ["tensors"] = new { file = Path.GetFileName(tensorPath), sha256 = Sha256(tensorPath), cases = layout },
        ["checks"] = new { pre_cancelled_request = cancellationVerified },
    };
    File.WriteAllText(Path.Combine(output, "dotnet.json"), JsonSerializer.Serialize(result, Settings.Json) + "\n");
    Console.WriteLine($"wrote {rows.Count} PP-OCRv5 images (image, tensor, post-processing and crop paths) to {output}");
    return 0;

    static int[][] Points(int[] xy) => [[xy[0], xy[1]], [xy[2], xy[3]], [xy[4], xy[5]], [xy[6], xy[7]]];

    static void Write(FileStream tensors, List<Dictionary<string, object>> layout, string id, string name, int? line, float[] values, int[] shape)
    {
        if (values.Length != shape.Aggregate(1, (product, size) => product * size))
        {
            throw new InvalidDataException($"{id}/{name}: {values.Length} values do not fill shape [{string.Join(", ", shape)}].");
        }

        var row = new Dictionary<string, object> { ["id"] = id, ["name"] = name, ["offset"] = tensors.Position / sizeof(float), ["shape"] = shape };
        if (line is int value)
        {
            row["line"] = value;
        }

        layout.Add(row);
        tensors.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    }

    static Dictionary<string, CorpusImage> ReadCorpus(string corpusPath)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(corpusPath))!;
        var images = new Dictionary<string, CorpusImage>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(corpusPath).Where(line => line.Length > 0))
        {
            using var row = JsonDocument.Parse(line);
            string id = row.RootElement.GetProperty("id").GetString()!;
            string path = Path.GetFullPath(Path.Combine(directory, row.RootElement.GetProperty("image").GetString()!));
            if (!path.StartsWith(directory + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{id}: image path escapes the corpus directory.");
            }

            string sha256 = row.RootElement.GetProperty("sha256").GetString()!;
            if (Sha256(path) != sha256)
            {
                throw new InvalidDataException($"{id}: image differs from its pinned SHA-256.");
            }

            images.Add(id, new CorpusImage(path, sha256));
        }

        return images;
    }
}

static string Sha256(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}

internal sealed record CorpusImage(string Path, string Sha256);

/// <summary>The reference stage's float32 sidecar, verified against its recorded SHA-256.</summary>
internal sealed class ReferenceTensors
{
    private readonly byte[] _blob;
    private readonly Dictionary<(string, string, int?), (long Offset, int[] Shape)> _layout = [];

    private ReferenceTensors(byte[] blob) => _blob = blob;

    public static ReferenceTensors Read(string referencePath, JsonElement reference)
    {
        var tensors = reference.GetProperty("tensors");
        string file = tensors.GetProperty("file").GetString()!;
        if (Path.GetFileName(file) != file)
        {
            throw new InvalidDataException("Unsafe reference tensor sidecar path.");
        }

        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(referencePath))!, file);
        byte[] blob = File.ReadAllBytes(path);
        if (Convert.ToHexStringLower(SHA256.HashData(blob)) != tensors.GetProperty("sha256").GetString())
        {
            throw new InvalidDataException("Reference tensor sidecar differs from its recorded SHA-256.");
        }

        var result = new ReferenceTensors(blob);
        foreach (var entry in tensors.GetProperty("cases").EnumerateArray())
        {
            int? line = entry.TryGetProperty("line", out var value) ? value.GetInt32() : null;
            var shape = entry.GetProperty("shape").EnumerateArray().Select(size => size.GetInt32()).ToArray();
            result._layout.Add((entry.GetProperty("id").GetString()!, entry.GetProperty("name").GetString()!, line), (entry.GetProperty("offset").GetInt64(), shape));
        }

        return result;
    }

    public (float[] Values, int[] Shape) Get(string id, string name, int? line)
    {
        if (!_layout.TryGetValue((id, name, line), out var entry))
        {
            throw new InvalidDataException($"Reference tensor {id}/{name}/{line} is missing.");
        }

        long count = entry.Shape.Aggregate(1L, (product, size) => product * size);
        if ((entry.Offset + count) * sizeof(float) > _blob.Length)
        {
            throw new InvalidDataException($"Reference tensor {id}/{name}/{line} lies outside the sidecar.");
        }

        return (MemoryMarshal.Cast<byte, float>(_blob.AsSpan((int)(entry.Offset * sizeof(float)), (int)(count * sizeof(float)))).ToArray(), entry.Shape);
    }
}

internal static class Settings
{
    public static readonly string[] RequiredOptions = ["--profile", "--snapshot", "--corpus", "--reference", "--out"];
    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
}

/// <summary>The candidate contract against the library's fixed constants: Python and C# must agree exactly.</summary>
internal static class Contract
{
    public static void Check(JsonElement profile)
    {
        var origin = profile.GetProperty("origin");
        Same("task", profile.GetProperty("task").GetString(), "ocr");
        Same("origin.repo", origin.GetProperty("repo").GetString(), PaddleOcrModelAssets.DetectorRepository);
        Same("origin.revision", origin.GetProperty("revision").GetString(), PaddleOcrModelAssets.DetectorRevision);
        var recognizer = origin.GetProperty("components").EnumerateArray().Single(entry => entry.GetProperty("role").GetString() == "rec");
        Same("origin.components[rec].repo", recognizer.GetProperty("repo").GetString(), PaddleOcrModelAssets.RecognizerRepository);
        Same("origin.components[rec].revision", recognizer.GetProperty("revision").GetString(), PaddleOcrModelAssets.RecognizerRevision);
        Pins("origin.files", origin.GetProperty("files"), PaddleOcrModelAssets.DetectorLock);
        Pins("origin.components[rec].files", recognizer.GetProperty("files"), PaddleOcrModelAssets.RecognizerLock);
        var runtime = profile.GetProperty("runtime");
        Same("runtime.version", runtime.GetProperty("version").GetString(), OnnxRuntimeInfo.NativeVersion);
        Same("runtime.precision", runtime.GetProperty("precision").GetString(), "fp32");
        var behavior = profile.GetProperty("behavior");
        Same("behavior.color_order", behavior.GetProperty("color_order").GetString(), "BGR");
        var resize = behavior.GetProperty("detector").GetProperty("resize");
        Same("detector.resize.limit_side_len", resize.GetProperty("limit_side_len").GetInt32(), PaddleOcrContract.LimitSideLength);
        Same("detector.resize.limit_type", resize.GetProperty("limit_type").GetString(), "max");
        Same("detector.resize.max_side_limit", resize.GetProperty("max_side_limit").GetInt32(), PaddleOcrContract.MaxSideLimit);
        Same("detector.resize.multiple", resize.GetProperty("multiple").GetInt32(), PaddleOcrContract.SideMultiple);
        Same("detector.resize.interpolation", resize.GetProperty("interpolation").GetString(), "opencv-inter-linear-u8");
        var normalize = behavior.GetProperty("detector").GetProperty("normalize");
        Same("detector.normalize.mean", string.Join(",", normalize.GetProperty("mean").EnumerateArray().Select(v => v.GetSingle())), string.Join(",", PaddleOcrContract.Mean));
        Same("detector.normalize.std", string.Join(",", normalize.GetProperty("std").EnumerateArray().Select(v => v.GetSingle())), string.Join(",", PaddleOcrContract.Std));
        var db = behavior.GetProperty("db");
        Same("db.thresh", db.GetProperty("thresh").GetSingle(), PaddleOcrContract.Threshold);
        Same("db.box_thresh", db.GetProperty("box_thresh").GetDouble(), PaddleOcrContract.BoxThreshold);
        Same("db.max_candidates", db.GetProperty("max_candidates").GetInt32(), PaddleOcrContract.MaxCandidates);
        Same("db.unclip_ratio", db.GetProperty("unclip_ratio").GetDouble(), PaddleOcrContract.UnclipRatio);
        Same("db.min_size", db.GetProperty("min_size").GetInt32(), PaddleOcrContract.MinSize);
        Same("db.score_mode", db.GetProperty("score_mode").GetString(), "fast");
        Same("db.box_type", db.GetProperty("box_type").GetString(), "quad");
        Same("db.use_dilation", db.GetProperty("use_dilation").GetBoolean(), false);
        var crop = behavior.GetProperty("crop");
        Same("crop.tall_ratio", crop.GetProperty("tall_ratio").GetDouble(), PaddleOcrContract.TallRatio);
        Same("crop.interpolation", crop.GetProperty("interpolation").GetString(), "opencv-inter-cubic-u8");
        Same("crop.border", crop.GetProperty("border").GetString(), "replicate");
        var recognizerBehavior = behavior.GetProperty("recognizer");
        Same("recognizer.image_shape", string.Join(",", recognizerBehavior.GetProperty("image_shape").EnumerateArray().Select(v => v.GetInt32())), $"3,{PaddleOcrContract.RecognizerHeight},{PaddleOcrContract.RecognizerWidth}");
        Same("recognizer.max_width", recognizerBehavior.GetProperty("max_width").GetInt32(), PaddleOcrContract.RecognizerMaxWidth);
        Same("recognizer.batch_size", recognizerBehavior.GetProperty("batch_size").GetInt32(), 1);
        var ctc = behavior.GetProperty("ctc");
        Same("ctc.dictionary_size", ctc.GetProperty("dictionary_size").GetInt32(), PaddleOcrContract.DictionarySize);
        Same("ctc.dictionary_sha256", ctc.GetProperty("dictionary_sha256").GetString(), PaddleOcrContract.DictionarySha256);
        Same("ctc.classes", ctc.GetProperty("classes").GetInt32(), PaddleOcrContract.Classes);
        Same("ctc.use_space_char", ctc.GetProperty("use_space_char").GetBoolean(), true);
        Same("ctc.decoding", ctc.GetProperty("decoding").GetString(), "greedy");
        var pipeline = behavior.GetProperty("pipeline");
        Same("pipeline.sort_row_tolerance", pipeline.GetProperty("sort_row_tolerance").GetInt32(), PaddleOcrContract.SortRowTolerance);
        Same("pipeline.textline_orientation", pipeline.GetProperty("textline_orientation").GetBoolean(), false);
        Same("pipeline.document_preprocessing", pipeline.GetProperty("document_preprocessing").GetBoolean(), false);
    }

    private static void Pins(string field, JsonElement files, Starlings.Acquisition.ModelAssetLock assetLock)
    {
        var pinned = files.EnumerateArray().ToDictionary(file => file.GetProperty("path").GetString()!, file => (file.GetProperty("size").GetInt64(), file.GetProperty("sha256").GetString()!), StringComparer.Ordinal);
        var locked = assetLock.Files.ToDictionary(file => file.Path, file => (file.Size, file.Sha256), StringComparer.Ordinal);
        if (pinned.Count != locked.Count || pinned.Any(pair => !locked.TryGetValue(pair.Key, out var value) || value != pair.Value))
        {
            throw new InvalidDataException($"{field} differs from the .NET asset lock.");
        }
    }

    private static void Same<T>(string field, T actual, T expected)
    {
        if (!EqualityComparer<T>.Default.Equals(actual, expected))
        {
            throw new InvalidDataException($"Candidate contract field {field} is {actual}, the .NET library implements {expected}.");
        }
    }
}
