// Clean CUDA consumer for the Starlings release-candidate drill (G6 of the CUDA profiles).
// tools/release-drill-ai.sh --cuda copies this file into a temporary directory outside the repository and builds it
// against the packed packages from a local folder feed plus Microsoft.ML.OnnxRuntime.Gpu.Linux from nuget.org. It must
// compile and run with nothing but those packages: no project references, no repository paths, no test framework.
//
//   dotnet CudaConsumer.dll <profile> <cache> <fixture dir> <tolerances.yaml> [<image dir>]
//   dotnet CudaConsumer.dll eg2-video <cache> <fixture dir> <tolerances.yaml> <video dir> <FFmpeg lib dir> 8|9
//
// Offline only. Resolves the profile's lock from the cache (size and SHA-256), opens CUDA sessions on device 0 (the
// drill sets CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to one GPU), prints the version diagnostics and
// compares the outputs with the protected fixtures within the CUDA profile's protected bounds, read from <tolerances.yaml>.
//
//   e5           multilingual-e5-small/onnx-fp32-cuda: embeddings.jsonl, tokenizer.jsonl (onnx-fp32-cpu fixtures)
//   gliner2      gliner2-base-v1/onnx-fp32-cuda: extraction.jsonl (onnx-fp32-cpu fixtures)
//   eg2-text     embeddinggemma-2/onnx-fp32-cuda: embeddings.jsonl (onnx-fp32-cpu fixtures)
//   eg2-image    embeddinggemma-2/image-fp32-cuda: image-embeddings.jsonl (image-fp32-cpu fixtures); <image dir> holds
//                the make_eg2_images.py set with its images.json
//   eg2-video    embeddinggemma-2/video-fp32-cuda: video-embeddings.jsonl (video-fp32-cpu fixtures); <video dir> holds the
//                22 make_eg2_videos.py fixture videos (SHA-256 checked against the protected videos.json). Decoded with
//                the FFmpeg adapter package of the given major (Starlings.Models.EmbeddingGemma2.Video.FFmpeg or .FFmpeg9)
//                on the user's own LGPL FFmpeg shared libraries (no opt-in to other builds): sampled indices and the
//                SHA-256 of every sampled frame exact (decoder bound), embeddings within the CUDA profile's bounds
//
// Exit code 0 only if every check passes.
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Extensions.AI;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using FF8 = Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
using FF9 = Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;
using Starlings.Models.Gliner2;
using Starlings.Runtime.Onnx;

// An exception ends with "error: " and the full exception on stderr and exit code 70 (EX_SOFTWARE), not an
// unhandled-exception abort and core dump (as the other release-drill consumers do).
try
{
    return await ToolBody();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex}");
    return 70;
}

async Task<int> ToolBody()
{
    if (args.Length < 4 || args[0] is not ("e5" or "gliner2" or "eg2-text" or "eg2-image" or "eg2-video") || (args[0] == "eg2-image" && args.Length < 5)
        || (args[0] == "eg2-video" && (args.Length < 7 || args[6] is not ("8" or "9"))))
    {
        Console.Error.WriteLine("usage: e5|gliner2|eg2-text|eg2-image <cache> <fixture dir> <tolerances.yaml> [<image dir>]");
        Console.Error.WriteLine("       eg2-video <cache> <fixture dir> <tolerances.yaml> <video dir> <FFmpeg lib dir> 8|9");
        return 2;
    }

    string profile = args[0];
    string cache = args[1];
    string fixtureDir = args[2];
    Dictionary<string, double> bounds = ReadFloatBounds(args[3]);
    const string ExpectedOrt = "1.30.0";

    int failures = 0;
    void Check(bool ok, string what)
    {
        Console.WriteLine($"{(ok ? "ok:  " : "FAIL:")} {what}");
        if (!ok)
        {
            failures++;
        }
    }

    double Bound(string key) => bounds.TryGetValue(key, out double v) ? v : throw new InvalidOperationException($"{args[3]}: no float.{key}");

    // Where the code comes from: package versions and the files actually loaded.
    foreach (Type t in new[] { typeof(ModelAssetStore), typeof(OnnxModelSession), typeof(Starlings.Tokenizers.Tokenizer), typeof(E5Encoder), typeof(E5EmbeddingGenerator), typeof(Gliner2EntityExtractor), typeof(EmbeddingGemma2Encoder), typeof(EmbeddingGemma2EmbeddingGenerator), typeof(EmbeddingGemma2ImageDecoder), typeof(FF8.EmbeddingGemma2VideoDecoder), typeof(FF9.EmbeddingGemma2VideoDecoder) })
    {
        Assembly a = t.Assembly;
        Console.WriteLine($"assembly: {a.GetName().Name} {a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion} ({a.Location})");
    }

    Console.WriteLine($"runtime:  {RuntimeInformation.FrameworkDescription} {RuntimeInformation.RuntimeIdentifier}");
    Console.WriteLine($"gpu:      CUDA_DEVICE_ORDER={Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER")} CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
    Console.WriteLine($"bounds:   {args[3]}: {string.Join(", ", bounds.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key} {kv.Value.ToString("G", CultureInfo.InvariantCulture)}"))}");

    // Native dependency resolution (G6): the natives of Microsoft.ML.OnnxRuntime.Gpu.Linux load, match the managed binding
    // and offer the CUDA execution provider.
    Check(OnnxRuntimeInfo.ManagedVersion == ExpectedOrt && OnnxRuntimeInfo.NativeVersion == ExpectedOrt,
        $"ONNX Runtime native {OnnxRuntimeInfo.NativeVersion}, managed {OnnxRuntimeInfo.ManagedVersion} (expected {ExpectedOrt})");
    Check(OnnxRuntimeInfo.AvailableProviders.Contains(OnnxRuntimeInfo.CudaProviderName),
        $"providers: {string.Join(", ", OnnxRuntimeInfo.AvailableProviders)}");

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false };
    var cuda = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };

    switch (profile)
    {
        case "e5":
            await RunE5();
            break;
        case "gliner2":
            RunGliner2();
            break;
        case "eg2-text":
            await RunEg2Text();
            break;
        case "eg2-image":
            await RunEg2Image(args[4]);
            break;
        case "eg2-video":
            await RunEg2Video(args[4], args[5], args[6] == "9" ? 9 : 8);
            break;
    }

    // The CUDA provider library the process actually mapped, and no other tokenizer implementation.
    if (File.Exists("/proc/self/maps"))
    {
        string[] mapped = [.. File.ReadAllLines("/proc/self/maps").Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).LastOrDefault() ?? "")
            .Where(p => p.StartsWith('/')).Distinct()];
        string[] ort = [.. mapped.Where(p => Path.GetFileName(p).StartsWith("libonnxruntime", StringComparison.Ordinal))];
        // The machine's CUDA runtime, cuBLAS and cuDNN (not in any package; the profile's documented prerequisites).
        string[] system = [.. mapped.Where(p => Path.GetFileName(p) is var n && (n.StartsWith("libcudart", StringComparison.Ordinal) || n.StartsWith("libcublas", StringComparison.Ordinal) || n.StartsWith("libcudnn", StringComparison.Ordinal)))];
        foreach (string p in ort.Concat(system))
        {
            Console.WriteLine($"mapped:   {p}");
        }

        string appDir = AppContext.BaseDirectory;
        Check(ort.Any(p => p.EndsWith("/libonnxruntime_providers_cuda.so", StringComparison.Ordinal)) && ort.All(p => p.StartsWith(appDir, StringComparison.Ordinal)),
            $"ONNX Runtime and its CUDA provider are mapped from the consumer's own build output ({appDir})");
        string[] foreignNative = [.. mapped.Where(p => p.Contains("tokenizers", StringComparison.OrdinalIgnoreCase) && !p.EndsWith("Starlings.Tokenizers.dll", StringComparison.Ordinal))];
        Check(foreignNative.Length == 0, $"no native tokenizers library mapped ({string.Join(", ", foreignNative)})");
    }

    string[] foreignManaged = [.. AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetName().Name ?? "")
        .Where(n => n.Contains("Tokenizer", StringComparison.OrdinalIgnoreCase) && n != "Starlings.Tokenizers")];
    Check(foreignManaged.Length == 0, $"no other managed tokenizer assembly loaded ({string.Join(", ", foreignManaged)})");

    Console.WriteLine(failures == 0 ? "RESULT: PASS" : $"RESULT: FAIL ({failures})");
    return failures == 0 ? 0 : 1;

    void SessionChecks(string what, OnnxModelSession session, string sha256)
    {
        OnnxSessionDiagnostics d = session.Diagnostics;
        Console.WriteLine($"session:  {what}: ORT {d.OrtVersion} (managed {d.ManagedBindingVersion}); providers {string.Join(",", d.SessionProviders)}; " +
            $"options {string.Join(", ", d.ProviderOptions.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}; model sha256 {d.ModelSha256}");
        Check(d.RequestedProvider == OnnxExecutionProvider.Cuda && d.SessionProviders.Count > 0 && d.SessionProviders[0] == OnnxRuntimeInfo.CudaProviderName,
            $"{what}: session runs on {OnnxRuntimeInfo.CudaProviderName}");
        Check(d.ProviderOptions.TryGetValue("use_tf32", out string? tf32) && tf32 == "0", $"{what}: use_tf32=0");
        Check(d.ProviderOptions.TryGetValue("device_id", out string? device) && device == "0", $"{what}: device_id=0");
        Check(d.ModelSha256 == sha256, $"{what}: model sha256 is the lock's {sha256}");
    }

    async Task RunE5()
    {
        E5ModelFiles files = E5ModelAssets.OpenLocal(acquisition);
        var texts = new Dictionary<string, string>();
        foreach (JsonElement row in ReadJsonl(Path.Combine(fixtureDir, "tokenizer.jsonl")))
        {
            texts[row.GetProperty("id").GetString()!] = row.GetProperty("text").GetString()!;
        }

        var cases = ReadJsonl(Path.Combine(fixtureDir, "embeddings.jsonl"))
            .Select(r => (Id: r.GetProperty("id").GetString()!, Ids: r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(), Embedding: Floats(r.GetProperty("embedding"))))
            .ToList();
        Check(cases.Count == 46, $"e5 fixture cases: {cases.Count} (expected 46)");

        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
        using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions { Session = cuda with { ExpectedModelSha256 = E5ModelAssets.ModelSha256 } });
        SessionChecks("e5", encoder.Session, E5ModelAssets.ModelSha256);
        var embedder = new E5TextEmbedder(tokenizer, encoder);
        int idsExact = cases.Count(c => tokenizer.Tokenize(texts[c.Id]).InputIds.Span.SequenceEqual(c.Ids));
        Check(idsExact == cases.Count, $"owned tokenizer ids exact: {idsExact}/{cases.Count}");

        float[][] expected = [.. cases.Select(c => c.Embedding)];
        float[][] single = [.. cases.Select(c => embedder.Embed(texts[c.Id]))];
        float[][] batch = embedder.EmbedBatch([.. cases.Select(c => texts[c.Id])]);
        Compare("e5 typed API on CUDA, one text per call, vs protected", single, expected, Bound("embedding_max_abs"));
        Compare("e5 typed API on CUDA, one batch of 46, vs protected", batch, expected, Bound("embedding_max_abs"));
        Compare("e5 typed API on CUDA, batch vs own single rows", batch, single, Bound("padded_batch_vs_single_embedding_max_abs"));

        using var generator = new E5EmbeddingGenerator(embedder);
        GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync(cases.Select(c => texts[c.Id]));
        Compare("e5 Extensions.AI generator over the CUDA session vs protected", [.. generated.Select(e => e.Vector.ToArray())], expected, Bound("embedding_max_abs"));
    }

    void RunGliner2()
    {
        Gliner2ModelFiles files = Gliner2ModelAssets.OpenLocal(acquisition);
        using var extractor = Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions { Session = cuda });
        SessionChecks("gliner2", extractor.Session, Gliner2ModelAssets.ModelSha256);
        double confidenceBound = Bound("confidence_max_abs");
        int inContract = 0, exact = 0;
        double maxConfidence = 0;
        foreach (JsonElement row in ReadJsonl(Path.Combine(fixtureDir, "extraction.jsonl")))
        {
            string id = row.GetProperty("id").GetString()!;
            string text = row.GetProperty("text").GetString()!;
            string[] labels = [.. row.GetProperty("labels_requested").EnumerateArray().Select(e => e.GetString()!)];
            float threshold = row.GetProperty("threshold").GetSingle();
            int tokens = row.GetProperty("input_ids").GetArrayLength();
            if (row.GetProperty("max_len").ValueKind == JsonValueKind.Number || tokens > Gliner2ExtractorOptions.ProfileMaxSequenceLength)
            {
                // Outside the contract: the max_len word cap is unsupported and inputs above 512 tokens are rejected.
                if (tokens > Gliner2ExtractorOptions.ProfileMaxSequenceLength && row.GetProperty("max_len").ValueKind != JsonValueKind.Number)
                {
                    bool rejected = false;
                    try
                    {
                        extractor.Extract(text, labels, threshold);
                    }
                    catch (Gliner2InputTooLongException)
                    {
                        rejected = true;
                    }

                    Check(rejected, $"gliner2 {id}: {tokens} tokens > 512 rejected");
                }

                continue;
            }

            Dictionary<string, string>? descriptions = row.GetProperty("descriptions").ValueKind == JsonValueKind.Object
                ? row.GetProperty("descriptions").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal)
                : null;
            Gliner2ExtractionResult result = extractor.ExtractBatch([text], labels, descriptions, threshold)[0];
            inContract++;
            bool same = true;
            foreach (JsonProperty label in row.GetProperty("output").GetProperty("entities").EnumerateObject())
            {
                var want = label.Value.EnumerateArray()
                    .Select(e => (Text: e.GetProperty("text").GetString()!, Confidence: e.GetProperty("confidence").GetDouble()))
                    .OrderBy(e => e.Text, StringComparer.Ordinal).ToList();
                var got = result.GetEntities(label.Name).Select(e => (e.Text, Confidence: (double)e.Confidence))
                    .OrderBy(e => e.Text, StringComparer.Ordinal).ToList();
                if (!want.Select(w => w.Text).SequenceEqual(got.Select(g => g.Text), StringComparer.Ordinal))
                {
                    same = false;
                    Console.WriteLine($"      {id}/{label.Name}: expected [{string.Join(" | ", want.Select(w => w.Text))}], got [{string.Join(" | ", got.Select(g => g.Text))}]");
                    continue;
                }

                for (int i = 0; i < want.Count; i++)
                {
                    maxConfidence = Math.Max(maxConfidence, Math.Abs(want[i].Confidence - got[i].Confidence));
                }
            }

            exact += same ? 1 : 0;
        }

        Check(inContract == 54 && exact == inContract, $"gliner2 entities equal the reference (label and text) on {exact}/{inContract} in-contract cases (expected 54)");
        Check(maxConfidence <= confidenceBound, $"gliner2 confidence max abs {maxConfidence:E3} (bound confidence_max_abs {confidenceBound:G})");
    }

    async Task RunEg2Text()
    {
        EmbeddingGemma2ModelFiles files = EmbeddingGemma2ModelAssets.OpenLocal(acquisition);
        using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions { Session = cuda with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 } });
        SessionChecks("eg2-text", encoder.Session, EmbeddingGemma2ModelAssets.ModelSha256);
        var embedder = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath), encoder);
        var rows = ReadJsonl(Path.Combine(fixtureDir, "embeddings.jsonl"))
            .Select(r => (Id: r.GetProperty("id").GetString()!, Prompt: EmbeddingGemma2Prompt.FromName(r.GetProperty("prompt").GetString()!), Text: r.GetProperty("text").GetString()!, Embedding: Floats(r.GetProperty("embedding"))))
            .ToList();
        Check(rows.Count == 138, $"eg2 text fixture rows: {rows.Count} (expected 138)");
        float[][] expected = [.. rows.Select(r => r.Embedding)];
        float[][] single = [.. rows.Select(r => embedder.Embed(r.Text, r.Prompt))];
        Compare("eg2 typed API on CUDA, one text per call, vs protected", single, expected, Bound("embedding_max_abs"), Bound("min_cosine"));
        // Padded batches of up to 8 rows with the same prompt, against the same rows alone.
        var batched = new float[rows.Count][];
        foreach (var group in rows.Select((r, i) => (r, i)).GroupBy(x => x.r.Prompt.Name))
        {
            foreach (var chunk in group.Chunk(8))
            {
                float[][] vectors = embedder.EmbedBatch([.. chunk.Select(x => x.r.Text)], chunk[0].r.Prompt);
                for (int k = 0; k < chunk.Length; k++)
                {
                    batched[chunk[k].i] = vectors[k];
                }
            }
        }

        Compare("eg2 typed API on CUDA, padded batches of 8, vs own single rows", batched, single, Bound("padded_batch_vs_single_embedding_max_abs"));
        var byPrompt = rows.Select((r, i) => (r, i)).Where(x => x.r.Prompt.Name == "query").ToList();
        using var generator = new EmbeddingGemma2EmbeddingGenerator(embedder, new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query });
        GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync(byPrompt.Select(x => x.r.Text));
        Compare($"eg2 Extensions.AI generator over the CUDA session, {byPrompt.Count} query rows, vs protected",
            [.. generated.Select(e => e.Vector.ToArray())], [.. byPrompt.Select(x => x.r.Embedding)], Bound("embedding_max_abs"));
    }

    async Task RunEg2Image(string imageDir)
    {
        EmbeddingGemma2ImageModelFiles files = EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition);
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = cuda });
        SessionChecks("eg2-image vision", embedder.VisionSession, EmbeddingGemma2ModelAssets.VisionModelSha256);
        SessionChecks("eg2-image multimodal text", embedder.MultimodalSession, EmbeddingGemma2ModelAssets.MultimodalModelSha256);
        var fileById = JsonDocument.Parse(File.ReadAllText(Path.Combine(imageDir, "images.json"))).RootElement.GetProperty("images").EnumerateArray()
            .ToDictionary(e => e.GetProperty("id").GetString()!, e => Path.Combine(imageDir, e.GetProperty("file").GetString()!), StringComparer.Ordinal);
        var used = new List<(string Id, string? ImageId, EmbeddingGemma2MultimodalInput Input, EmbeddingGemma2Prompt Prompt, float[] Embedding)>();
        int skipped = 0;
        foreach (JsonElement row in ReadJsonl(Path.Combine(fixtureDir, "image-embeddings.jsonl")))
        {
            string id = row.GetProperty("id").GetString()!;
            string kind = row.GetProperty("kind").GetString()!;
            var prompt = EmbeddingGemma2Prompt.FromName(row.GetProperty("prompt").GetString()!);
            if (kind == "text")
            {
                used.Add((id, null, EmbeddingGemma2MultimodalInput.FromText(row.GetProperty("text").GetString()!), prompt, Floats(row.GetProperty("embedding"))));
                continue;
            }

            if (kind != "image")
            {
                skipped++;   // interleaved rows: covered by the package tests (docs/evidence/embeddinggemma2-images/)
                continue;
            }

            string imageId = row.GetProperty("images")[0].GetProperty("image").GetString()!;
            EmbeddingGemma2Image image;
            try
            {
                image = EmbeddingGemma2ImageDecoder.DecodeFile(fileById[imageId]);
            }
            catch (EmbeddingGemma2UnsupportedImageException ex)
            {
                // The 16-bit fixture image is rejected by contract (capability image_16_bit unsupported).
                Check(imageId == "mode-gray16", $"eg2-image {id}: rejected ({ex.Reason}); only mode-gray16 may be rejected");
                continue;
            }

            used.Add((id, imageId, EmbeddingGemma2MultimodalInput.Create(image), prompt, Floats(row.GetProperty("embedding"))));
        }

        int images = used.Count(u => u.ImageId is not null);
        Console.WriteLine($"eg2-image rows: {images} image rows decoded with the SkiaSharp adapter, {used.Count - images} text rows, {skipped} interleaved rows not run here");
        Check(images >= 30, $"eg2-image: {images} image rows compared (expected at least 30)");
        float[][] expected = [.. used.Select(u => u.Embedding)];
        float[][] single = [.. used.Select(u => embedder.Embed(u.Input, u.Prompt))];
        Compare("eg2-image typed API on CUDA, one input per call, vs protected", single, expected, Bound("embedding_max_abs"), Bound("min_cosine"));

        // Microsoft.Extensions.AI: the multimodal generator over the same CUDA sessions, image bytes decoded by the adapter.
        var plain = used.Where(u => u.ImageId is not null && u.Prompt.Name == EmbeddingGemma2Prompt.NoneName).ToList();
        using var generator = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
        {
            ImageDecoder = (bytes, _) => EmbeddingGemma2ImageDecoder.Decode(bytes.Span),
        });
        var contents = plain.Select(u => new DataContent(File.ReadAllBytes(fileById[u.ImageId!]), fileById[u.ImageId!].EndsWith(".png", StringComparison.Ordinal) ? "image/png" : "image/jpeg")).ToList();
        IEmbeddingGenerator<DataContent, Embedding<float>> imageGenerator = generator;
        GeneratedEmbeddings<Embedding<float>> generated = await imageGenerator.GenerateAsync(contents);
        Compare($"eg2-image Extensions.AI generator over the CUDA sessions, {plain.Count} image rows, vs protected",
            [.. generated.Select(e => e.Vector.ToArray())], [.. plain.Select(u => u.Embedding)], Bound("embedding_max_abs"));
    }

    async Task RunEg2Video(string videoDir, string ffmpegDir, int major)
    {
        // The adapter package of the requested major on the user's own FFmpeg; a GPL or nonfree build is refused (no opt-in).
        Func<string, (EmbeddingGemma2Video Video, IReadOnlyList<int> Indices)> decodeFile;
        Func<ReadOnlyMemory<byte>, EmbeddingGemma2Video> decodeBytes;
        string ffVersion, ffDir, ffLicenses, adapter;
        bool lgpl;
        if (major == 9)
        {
            var d = new FF9.EmbeddingGemma2VideoDecoder(new FF9.EmbeddingGemma2FFmpegOptions { LibraryDirectory = ffmpegDir });
            decodeFile = path => { var v = d.DecodeFile(path); return (v.Video, v.FrameIndices); };
            decodeBytes = bytes => d.Decode(bytes).Video;
            (ffVersion, ffDir, lgpl, adapter) = (d.Libraries.Version, d.Libraries.Directory ?? "", d.Libraries.IsLgpl, Describe(typeof(FF9.EmbeddingGemma2VideoDecoder)));
            ffLicenses = string.Join(", ", d.Libraries.Licenses.Select(kv => $"{kv.Key} {kv.Value}"));
            Check(FF9.EmbeddingGemma2FFmpeg.SupportedFFmpegMajor == 9, "eg2-video: Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 binds FFmpeg 9");
        }
        else
        {
            var d = new FF8.EmbeddingGemma2VideoDecoder(new FF8.EmbeddingGemma2FFmpegOptions { LibraryDirectory = ffmpegDir });
            decodeFile = path => { var v = d.DecodeFile(path); return (v.Video, v.FrameIndices); };
            decodeBytes = bytes => d.Decode(bytes).Video;
            (ffVersion, ffDir, lgpl, adapter) = (d.Libraries.Version, d.Libraries.Directory ?? "", d.Libraries.IsLgpl, Describe(typeof(FF8.EmbeddingGemma2VideoDecoder)));
            ffLicenses = string.Join(", ", d.Libraries.Licenses.Select(kv => $"{kv.Key} {kv.Value}"));
            Check(FF8.EmbeddingGemma2FFmpeg.SupportedFFmpegMajor == 8, "eg2-video: Starlings.Models.EmbeddingGemma2.Video.FFmpeg binds FFmpeg 8");
        }

        Console.WriteLine($"ffmpeg:   {ffVersion} from {ffDir}; {adapter}; licenses: {ffLicenses}");
        Check(lgpl, $"eg2-video: the user's FFmpeg {ffVersion} is an LGPL build");
        Check(ffVersion.StartsWith($"n{major}.", StringComparison.Ordinal), $"eg2-video: FFmpeg {ffVersion} is major {major}");

        // The fixture videos are the protected ones, byte for byte.
        var fileById = new Dictionary<string, string>(StringComparer.Ordinal);
        int filesOk = 0;
        foreach (JsonElement v in JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDir, "videos.json"))).RootElement.GetProperty("videos").EnumerateArray())
        {
            string path = Path.Combine(videoDir, v.GetProperty("file").GetString()!);
            fileById[v.GetProperty("id").GetString()!] = path;
            filesOk += File.Exists(path) && Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))) == v.GetProperty("sha256").GetString() ? 1 : 0;
        }

        Check(fileById.Count == 22 && filesOk == fileById.Count, $"eg2-video: {filesOk}/{fileById.Count} fixture videos present with the protected SHA-256 (expected 22)");

        EmbeddingGemma2ImageModelFiles files = EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition);
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = cuda });
        SessionChecks("eg2-video vision", embedder.VisionSession, EmbeddingGemma2ModelAssets.VisionModelSha256);
        SessionChecks("eg2-video multimodal text", embedder.MultimodalSession, EmbeddingGemma2ModelAssets.MultimodalModelSha256);

        // Video rows read from a file (any prompt) and the text queries; interleaved and array rows are covered by the
        // package tests (docs/evidence/embeddinggemma2-video/).
        var used = new List<(string Id, string? Video, EmbeddingGemma2MultimodalInput Input, EmbeddingGemma2Prompt Prompt, float[] Embedding)>();
        int indicesExact = 0, framesExact = 0, framesTotal = 0, skipped = 0;
        foreach (JsonElement row in ReadJsonl(Path.Combine(fixtureDir, "video-embeddings.jsonl")))
        {
            string id = row.GetProperty("id").GetString()!;
            string kind = row.GetProperty("kind").GetString()!;
            var prompt = EmbeddingGemma2Prompt.FromName(row.GetProperty("prompt").GetString()!);
            if (kind == "text")
            {
                used.Add((id, null, EmbeddingGemma2MultimodalInput.FromText(row.GetProperty("text").GetString()!), prompt, Floats(row.GetProperty("embedding"))));
                continue;
            }

            if (kind != "video" || row.GetProperty("input_form").GetString() != "path")
            {
                skipped++;
                continue;
            }

            JsonElement spec = row.GetProperty("videos")[0];
            string videoId = spec.GetProperty("video").GetString()!;
            var (video, indices) = decodeFile(fileById[videoId]);
            int[] wantIndices = [.. spec.GetProperty("sampled_indices").EnumerateArray().Select(e => e.GetInt32())];
            string[] wantFrames = [.. spec.GetProperty("frame_sha256").EnumerateArray().Select(e => e.GetString()!)];
            indicesExact += indices.SequenceEqual(wantIndices) ? 1 : 0;
            for (int k = 0; k < wantFrames.Length; k++)
            {
                framesTotal++;
                framesExact += k < video.Frames.Count && Convert.ToHexStringLower(SHA256.HashData(video.Frames[k].Pixels.Span)) == wantFrames[k] ? 1 : 0;
            }

            used.Add((id, videoId, EmbeddingGemma2MultimodalInput.FromVideo(video), prompt, Floats(row.GetProperty("embedding"))));
        }

        int videoRows = used.Count(u => u.Video is not null);
        Console.WriteLine($"eg2-video rows: {videoRows} video rows decoded with the FFmpeg {major} adapter, {used.Count - videoRows} text rows, {skipped} interleaved or array rows not run here");
        Check(videoRows >= 23, $"eg2-video: {videoRows} video rows compared (expected at least 23)");
        Check(indicesExact == videoRows, $"eg2-video: sampled frame indices equal the reference on {indicesExact}/{videoRows} videos (exact)");
        Check(framesTotal > 0 && framesExact == framesTotal, $"eg2-video: decoded RGB of {framesExact}/{framesTotal} sampled frames equal torchcodec's (SHA-256; decoder bound exact)");
        float[][] expected = [.. used.Select(u => u.Embedding)];
        float[][] single = [.. used.Select(u => embedder.Embed(u.Input, u.Prompt))];
        Compare("eg2-video typed API on CUDA, one input per call, vs protected", single, expected, Bound("embedding_max_abs"), Bound("min_cosine"));

        // Microsoft.Extensions.AI: the multimodal generator over the same CUDA sessions, the six smallest video files from
        // their bytes through the adapter.
        var plain = used.Where(u => u.Video is not null && u.Prompt.Name == EmbeddingGemma2Prompt.NoneName)
            .OrderBy(u => new FileInfo(fileById[u.Video!]).Length).Take(6).ToList();
        using var generator = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
        {
            VideoDecoder = (bytes, _) => decodeBytes(bytes),
        });
        var contents = plain.Select(u => new DataContent(File.ReadAllBytes(fileById[u.Video!]), fileById[u.Video!].EndsWith(".webm", StringComparison.Ordinal) ? "video/webm" : "video/mp4")).ToList();
        IEmbeddingGenerator<DataContent, Embedding<float>> videoGenerator = generator;
        GeneratedEmbeddings<Embedding<float>> generated = await videoGenerator.GenerateAsync(contents);
        Compare($"eg2-video Extensions.AI generator over the CUDA sessions, {plain.Count} videos from bytes, vs protected",
            [.. generated.Select(e => e.Vector.ToArray())], [.. plain.Select(u => u.Embedding)], Bound("embedding_max_abs"));
    }

    static string Describe(Type t) =>
        $"{t.Assembly.GetName().Name} {t.Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion?.Split('+')[0]}";

    void Compare(string what, float[][] actual, float[][] expected, double bound, double minCosineBound = 0)
    {
        double max = 0, minCos = 1;
        bool finite = true;
        bool shapes = actual.Length == expected.Length;
        for (int r = 0; shapes && r < actual.Length; r++)
        {
            shapes &= actual[r].Length == expected[r].Length;
            if (!shapes)
            {
                break;
            }

            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < actual[r].Length; i++)
            {
                finite &= float.IsFinite(actual[r][i]);
                max = Math.Max(max, Math.Abs((double)actual[r][i] - expected[r][i]));
                dot += (double)actual[r][i] * expected[r][i];
                na += (double)actual[r][i] * actual[r][i];
                nb += (double)expected[r][i] * expected[r][i];
            }

            minCos = Math.Min(minCos, dot / Math.Sqrt(na * nb));
        }

        Check(shapes && finite && max <= bound && minCos >= minCosineBound,
            string.Create(CultureInfo.InvariantCulture, $"{what}: {actual.Length} rows, max abs {max:E3} (bound {bound:G}), min cosine {minCos:F9}{(minCosineBound > 0 ? $" (bound {minCosineBound:G})" : "")}{(finite ? "" : ", NaN or Inf")}"));
    }

    static float[] Floats(JsonElement array) => [.. array.EnumerateArray().Select(e => e.GetSingle())];

    // Split on '\n' only: fixture text contains U+0085 and U+2028.
    static List<JsonElement> ReadJsonl(string path) =>
        [.. File.ReadAllText(path).Split('\n').Where(l => l.Length > 0).Select(l => JsonDocument.Parse(l).RootElement)];

    // The float section of a protected tolerances.yaml: "  key: number" lines under "float:".
    static Dictionary<string, double> ReadFloatBounds(string path)
    {
        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        string section = "";
        foreach (string raw in File.ReadAllLines(path))
        {
            string line = raw.Split('#')[0].TrimEnd();
            if (line.Length == 0)
            {
                continue;
            }

            if (!char.IsWhiteSpace(line[0]))
            {
                section = line.Split(':')[0];
                continue;
            }

            string[] kv = line.Trim().Split(':', 2);
            if (section == "float" && kv.Length == 2 && double.TryParse(kv[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
            {
                result[kv[0]] = value;
            }
        }

        return result;
    }
}
