// Semantic search with google/embeddinggemma-2 from the Starlings packages: text, text to image, text to video.
//
//   dotnet run -- download        [--cache <dir>]                           explicit acquisition (network)
//   dotnet run -- search          [--cache <dir> | --snapshot <dir>] [query ...]   offline: embeds, ranks by cosine
//   dotnet run -- download-images [--cache <dir>]                           explicit acquisition of the image graphs
//   dotnet run -- images          [--cache <dir> | --snapshot <dir>] --images <dir> [query ...]
//                                 offline: decodes the PNG and JPEG files of <dir>, ranks them for text queries
//   dotnet run -- videos          [--cache <dir> | --snapshot <dir>] --videos <dir> [--ffmpeg <dir>] [query ...]
//                                 offline: decodes the videos of <dir> with the FFmpeg adapter on your own FFmpeg 8.x
//                                 shared libraries (--ffmpeg, else $STARLING_FFMPEG_DIR, else the system
//                                 library path; none are shipped) and ranks them for text queries (image lock files)
//
// Without --cache the files go to $STARLING_MODELS or the user cache directory. --snapshot verifies an existing directory
// holding tokenizer.json, config_sentence_transformers.json and onnx/embeddinggemma2-text-fp32.onnx (images: also
// onnx/embeddinggemma2-vision-fp32.onnx and onnx/embeddinggemma2-text-mm-fp32.onnx) instead. Inference
// never downloads anything. Deployments must adhere to the Gemma Prohibited Use Policy
// (https://ai.google.dev/gemma/prohibited_use_policy).
using System.Reflection;
using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Extensions.AI;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
using Starlings.Runtime.Onnx;
using Microsoft.Extensions.AI;

// A failure ends with a one-line message and exit code 1, not an unhandled-exception abort.
try
{
    return await ToolBody();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> ToolBody()
{
    if (args.Length == 0 || args[0] is not ("download" or "search" or "download-images" or "images" or "videos"))
    {
        Console.Error.WriteLine("usage: EmbeddingGemma2Search download [--cache <dir>] | search [--cache <dir> | --snapshot <dir>] [query ...]");
        Console.Error.WriteLine("       EmbeddingGemma2Search download-images [--cache <dir>] | images [--cache <dir> | --snapshot <dir>] --images <dir> [query ...]");
        Console.Error.WriteLine("       EmbeddingGemma2Search videos [--cache <dir> | --snapshot <dir>] --videos <dir> [--ffmpeg <FFmpeg 8 lib dir>] [query ...]");
        return 2;
    }

    string? cache = null;
    string? snapshot = null;
    string? imageDirectory = null;
    string? videoDirectory = null;
    string? ffmpegDirectory = null;
    var rest = new List<string>();
    for (int i = 1; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--cache" when i + 1 < args.Length:
                cache = args[++i];
                break;
            case "--snapshot" when i + 1 < args.Length:
                snapshot = args[++i];
                break;
            case "--images" when i + 1 < args.Length:
                imageDirectory = args[++i];
                break;
            case "--videos" when i + 1 < args.Length:
                videoDirectory = args[++i];
                break;
            case "--ffmpeg" when i + 1 < args.Length:
                ffmpegDirectory = args[++i];
                break;
            default:
                rest.Add(args[i]);
                break;
        }
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };

    if (args[0] == "download")
    {
        // 1. Explicit acquisition of the pinned lock: tokenizer.json and config_sentence_transformers.json from the Hub
        //    repository at the pinned commit, the text-path graph from its GitHub release. Cached files are verified.
        Console.WriteLine($"lock:  {EmbeddingGemma2ModelAssets.Lock}");
        Console.WriteLine($"cache: {acquisition.ResolveCacheDirectory()}");
        var progress = new FileProgress();
        EmbeddingGemma2ModelFiles downloaded = await EmbeddingGemma2ModelAssets.DownloadAsync(acquisition, progress);
        foreach (ModelAssetFile file in EmbeddingGemma2ModelAssets.Lock.Files)
        {
            string how = progress.Downloaded.Contains(file.Path) ? "downloaded" : "already cached";
            Console.WriteLine($"{file.Path}: {how}, {file.Size} bytes, sha256 {file.Sha256} verified");
        }

        Console.WriteLine($"snapshot: {downloaded.Snapshot.Directory}");
        return 0;
    }

    if (args[0] == "download-images")
    {
        // Image embeddings need two more graphs (vision, multimodal text; release embeddinggemma-2-vision-onnx-r1) next to
        // the text files. Same snapshot directory, so a text download is reused.
        var imageProgress = new FileProgress();
        EmbeddingGemma2ImageModelFiles downloadedImages = await EmbeddingGemma2ModelAssets.DownloadImageAssetsAsync(acquisition, imageProgress);
        foreach (ModelAssetFile file in EmbeddingGemma2ModelAssets.ImageLock.Files)
        {
            string how = imageProgress.Downloaded.Contains(file.Path) ? "downloaded" : "already cached";
            Console.WriteLine($"{file.Path}: {how}, {file.Size} bytes, sha256 {file.Sha256} verified");
        }

        Console.WriteLine($"snapshot: {downloadedImages.Text.Snapshot.Directory}");
        return 0;
    }

    if (args[0] == "images")
    {
        return await SearchImagesAsync(acquisition, snapshot, imageDirectory, rest);
    }

    if (args[0] == "videos")
    {
        return await SearchVideosAsync(acquisition, snapshot, videoDirectory, ffmpegDirectory, rest);
    }

    // 2. Offline load: verifies the files (size and SHA-256) and never downloads.
    EmbeddingGemma2ModelFiles files = string.IsNullOrEmpty(snapshot)
        ? EmbeddingGemma2ModelAssets.OpenLocal(acquisition)
        : EmbeddingGemma2ModelAssets.OpenDirectory(snapshot);
    using var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, new EmbeddingGemma2EncoderOptions
    {
        Session = new OnnxSessionOptions { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 },
    });
    var embedder = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath), encoder);

    Console.WriteLine($"packages: {Version<EmbeddingGemma2Encoder>()}, {Version<Starlings.Tokenizers.Tokenizer>()}, {Version<EmbeddingGemma2EmbeddingGenerator>()}");
    Console.WriteLine($"onnxruntime: native {OnnxRuntimeInfo.NativeVersion}, managed {OnnxRuntimeInfo.ManagedVersion}, session providers {string.Join(",", encoder.Session.Diagnostics.SessionProviders)}");

    (string Id, string Text)[] documents =
    [
        ("protein", "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day."),
        ("summit", "The summit of Mount Everest is 8,849 metres above sea level, the highest point on Earth."),
        ("rust", "Eisen rostet, wenn es Sauerstoff und Wasser ausgesetzt ist; dabei entsteht Eisenoxid."),
        ("tokyo", "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。"),
        ("bread", "Para hacer pan se necesita harina, agua, sal y levadura; la masa reposa antes de hornearse."),
    ];
    string[] queries = rest.Count > 0
        ? [.. rest]
        : ["how much protein should a female eat", "Warum rostet Eisen?", "日本の首都はどこですか", "highest mountain in the world"];

    // 3. Typed API. The model card's prompts are explicit: "title: none | text: " for documents and
    //    "task: search result | query: " for queries. Nothing adds them implicitly.
    float[][] documentVectors = embedder.EmbedBatch([.. documents.Select(d => d.Text)], EmbeddingGemma2Prompt.Document);
    float[][] queryVectors = embedder.EmbedBatch(queries, EmbeddingGemma2Prompt.Query);
    for (int q = 0; q < queries.Length; q++)
    {
        var ranked = Rank(queryVectors[q], documentVectors);
        Console.WriteLine($"query: {queries[q]}");
        Console.WriteLine($"  top: {documents[ranked[0].Index].Id} ({ranked[0].Score:F4}); next: {documents[ranked[1].Index].Id} ({ranked[1].Score:F4})");
    }

    // 4. Microsoft.Extensions.AI, with a Matryoshka size: Dimensions = 256 keeps the leading 256 values, renormalized.
    using IEmbeddingGenerator<string, Embedding<float>> generator =
        new EmbeddingGemma2EmbeddingGenerator(embedder, new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query });
    GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync(queries);
    double maxDiff = 0;
    for (int q = 0; q < queries.Length; q++)
    {
        maxDiff = Math.Max(maxDiff, MaxAbsDiff(generated[q].Vector.Span, queryVectors[q]));
    }

    var small = new EmbeddingGenerationOptions { Dimensions = 256 };
    GeneratedEmbeddings<Embedding<float>> smallQueries = await generator.GenerateAsync(queries, small);
    small.AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = "document" };
    GeneratedEmbeddings<Embedding<float>> smallDocuments = await generator.GenerateAsync(documents.Select(d => d.Text), small);
    float[][] smallDocumentVectors = [.. smallDocuments.Select(e => e.Vector.ToArray())];
    int sameTop = 0;
    for (int q = 0; q < queries.Length; q++)
    {
        sameTop += Rank(smallQueries[q].Vector.ToArray(), smallDocumentVectors)[0].Index == Rank(queryVectors[q], documentVectors)[0].Index ? 1 : 0;
    }

    EmbeddingGeneratorMetadata? metadata = generator.GetService<EmbeddingGeneratorMetadata>();
    Console.WriteLine($"extensions.ai: {metadata?.ProviderName} {metadata?.DefaultModelId} dims {metadata?.DefaultModelDimensions}; " +
        $"{generated.Count} embeddings, {generated.Usage?.InputTokenCount} tokens, max abs difference from the typed API {maxDiff:G3}");
    Console.WriteLine($"matryoshka 256: {smallQueries[0].Dimensions} dimensions; same top document for {sameTop} of {queries.Length} queries");

    // 5. Inputs above 8,192 tokens are rejected, never truncated.
    try
    {
        embedder.Embed(string.Join(' ', Enumerable.Repeat("long", 9000)));
        Console.WriteLine("long input: not rejected (unexpected)");
        return 1;
    }
    catch (EmbeddingGemma2InputTooLongException ex)
    {
        Console.WriteLine($"long input rejected: {ex.TokenCount} tokens > {ex.MaximumTokens}");
    }

    return 0;

    // Text to image: images decoded with the SkiaSharp adapter (PIL's RGB), embedded alone and without a prompt (the model
    // card: "Pass images, video, and audio without any prefix"); queries with the query prompt on the released text graph.
    static async Task<int> SearchImagesAsync(ModelAcquisitionOptions acquisition, string? snapshot, string? imageDirectory, List<string> rest)
    {
        if (imageDirectory is null)
        {
            Console.Error.WriteLine("images: --images <dir> is required (PNG and JPEG files)");
            return 2;
        }

        EmbeddingGemma2ImageModelFiles files = string.IsNullOrEmpty(snapshot)
            ? EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition)
            : EmbeddingGemma2ModelAssets.OpenImageDirectory(snapshot);
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files);
        string[] paths = [.. Directory.EnumerateFiles(imageDirectory)
            .Where(p => p.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
            .Order(StringComparer.Ordinal)];
        var images = new List<(string Name, string Path, EmbeddingGemma2Image Image)>();
        foreach (string path in paths)
        {
            try
            {
                images.Add((Path.GetFileName(path), path, EmbeddingGemma2ImageDecoder.DecodeFile(path)));
            }
            catch (EmbeddingGemma2UnsupportedImageException ex)
            {
                Console.WriteLine($"skipped {Path.GetFileName(path)}: {ex.Reason}");
            }
        }

        if (images.Count < 2)
        {
            Console.Error.WriteLine($"images: {imageDirectory} holds fewer than two decodable PNG or JPEG files");
            return 2;
        }

        Console.WriteLine($"images: {images.Count} decoded from {imageDirectory}; vision session providers {string.Join(",", embedder.VisionSession.Diagnostics.SessionProviders)}");
        float[][] imageVectors = embedder.EmbedBatch([.. images.Select(i => i.Image)]);
        string[] queries = rest.Count > 0
            ? [.. rest]
            : ["a photo of a cat", "a cup of coffee", "a rocket launch", "portrait of an astronaut", "deep space full of galaxies"];
        float[][] queryVectors = embedder.EmbedBatch([.. queries.Select(q => EmbeddingGemma2MultimodalInput.FromText(q))], EmbeddingGemma2Prompt.Query);
        for (int q = 0; q < queries.Length; q++)
        {
            var ranked = Rank(queryVectors[q], imageVectors);
            Console.WriteLine($"image query: {queries[q]}");
            Console.WriteLine($"  top: {images[ranked[0].Index].Name} ({ranked[0].Score:F4}); next: {images[ranked[1].Index].Name} ({ranked[1].Score:F4})");
        }

        // Text and image in one input: the caption precedes the image tokens.
        float[] captioned = embedder.Embed(EmbeddingGemma2MultimodalInput.Create("A photo: ", images[0].Image));
        Console.WriteLine($"interleaved: {captioned.Length} dimensions, cosine to the image alone {Dot(captioned, imageVectors[0]):F4}");

        // Microsoft.Extensions.AI: IEmbeddingGenerator<DataContent, Embedding<float>> with the SkiaSharp decoder plugged in.
        using var generator = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
        {
            ImageDecoder = (bytes, _) => EmbeddingGemma2ImageDecoder.Decode(bytes.Span),
        });
        var contents = images
            .Select(i => new DataContent(File.ReadAllBytes(i.Path), i.Path.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? "image/png" : "image/jpeg"))
            .ToList();
        IEmbeddingGenerator<DataContent, Embedding<float>> imageGenerator = generator;
        GeneratedEmbeddings<Embedding<float>> generated = await imageGenerator.GenerateAsync(contents);
        double maxDiff = 0;
        for (int i = 0; i < images.Count; i++)
        {
            maxDiff = Math.Max(maxDiff, MaxAbsDiff(generated[i].Vector.Span, imageVectors[i]));
        }

        Console.WriteLine($"extensions.ai images: {generated.Count} embeddings, {generated.Usage?.InputTokenCount} tokens, max abs difference from the typed API {maxDiff:G3}");
        return 0;
    }

    // Text to video: videos decoded with the FFmpeg adapter on the user's own FFmpeg 8.x (torchcodec's metadata, sampling at
    // 1 frame per second up to 32 frames, torchcodec's RGB), embedded alone and without a prompt; queries with the query
    // prompt on the released text graph.
    static async Task<int> SearchVideosAsync(ModelAcquisitionOptions acquisition, string? snapshot, string? videoDirectory, string? ffmpegDirectory, List<string> rest)
    {
        if (videoDirectory is null)
        {
            Console.Error.WriteLine("videos: --videos <dir> is required (MP4, WebM, MKV or MOV files)");
            return 2;
        }

        EmbeddingGemma2VideoDecoder decoder;
        try
        {
            decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = ffmpegDirectory });
        }
        catch (EmbeddingGemma2FFmpegException ex)
        {
            Console.Error.WriteLine($"videos: {ex.Message}");
            return 3;
        }

        Console.WriteLine($"ffmpeg: {decoder.Libraries.Version} ({(decoder.Libraries.IsLgpl ? "LGPL" : "not LGPL")}) from {decoder.Libraries.Directory ?? "the system library path"}; {Version<EmbeddingGemma2VideoDecoder>()}");
        EmbeddingGemma2ImageModelFiles files = string.IsNullOrEmpty(snapshot)
            ? EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition)
            : EmbeddingGemma2ModelAssets.OpenImageDirectory(snapshot);

        // CPU: frame batches equal single frames bit for bit, so larger vision calls only trade memory for speed.
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 8, MaxVisionBatchPatches = 8 * 1_260 });
        string[] paths = [.. Directory.EnumerateFiles(videoDirectory)
            .Where(p => Path.GetExtension(p).ToLowerInvariant() is ".mp4" or ".webm" or ".mkv" or ".mov")
            .Order(StringComparer.Ordinal)];
        var videos = new List<(string Name, string Path, EmbeddingGemma2DecodedVideo Decoded)>();
        foreach (string path in paths)
        {
            var decoded = decoder.DecodeFile(path);
            videos.Add((Path.GetFileName(path), path, decoded));
            var m = decoded.Metadata;
            Console.WriteLine($"decoded {Path.GetFileName(path)}: {m.Codec} {m.Width}x{m.Height}, {m.FrameCount} frames, {m.DurationSeconds:F2} s -> frames [{string.Join(", ", decoded.FrameIndices)}]");
        }

        if (videos.Count < 2)
        {
            Console.Error.WriteLine($"videos: {videoDirectory} holds fewer than two videos");
            return 2;
        }

        float[][] videoVectors = embedder.EmbedBatch([.. videos.Select(v => EmbeddingGemma2MultimodalInput.FromVideo(v.Decoded.Video))]);
        string[] queries = rest.Count > 0
            ? [.. rest]
            : ["colour bars test pattern with a moving white square", "colourful circles and rectangles moving over a dark background", "white words scrolling across a colour gradient"];
        float[][] queryVectors = embedder.EmbedBatch([.. queries.Select(q => EmbeddingGemma2MultimodalInput.FromText(q))], EmbeddingGemma2Prompt.Query);
        for (int q = 0; q < queries.Length; q++)
        {
            var ranked = Rank(queryVectors[q], videoVectors);
            Console.WriteLine($"video query: {queries[q]}");
            Console.WriteLine($"  top: {videos[ranked[0].Index].Name} ({ranked[0].Score:F4}); next: {videos[ranked[1].Index].Name} ({ranked[1].Score:F4})");
        }

        // Microsoft.Extensions.AI: a video DataContent through the FFmpeg decoder, equal to the typed API (shortest video).
        int shortest = Enumerable.Range(0, videos.Count).MinBy(i => videos[i].Decoded.FrameIndices.Count);
        using var generator = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
        {
            VideoDecoder = (bytes, _) => decoder.Decode(bytes).Video,
        });
        IEmbeddingGenerator<DataContent, Embedding<float>> videoGenerator = generator;
        var generated = await videoGenerator.GenerateAsync([new DataContent(File.ReadAllBytes(videos[shortest].Path), "video/mp4")]);
        Console.WriteLine($"extensions.ai videos: {videos[shortest].Name}, {generated.Usage?.InputTokenCount} tokens, max abs difference from the typed API {MaxAbsDiff(generated[0].Vector.Span, videoVectors[shortest]):G3}");

        // Decoded frames without timing are rejected by default (upstream would keep every frame and thin to 32).
        try
        {
            embedder.Embed(EmbeddingGemma2Video.FromFramesWithoutTiming(videos[shortest].Decoded.Video.Frames));
            Console.WriteLine("frames without timing: not rejected (unexpected)");
            return 1;
        }
        catch (EmbeddingGemma2VideoTimingRequiredException)
        {
            Console.WriteLine("frames without timing rejected: EmbeddingGemma2VideoTimingRequiredException");
        }

        return 0;
    }

    static (int Index, float Score)[] Rank(float[] query, float[][] documents) =>
        [.. documents.Select((d, i) => (Index: i, Score: Dot(query, d))).OrderByDescending(r => r.Score)];

    static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    static double MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        }

        return max;
    }

    static string Version<T>()
    {
        AssemblyName name = typeof(T).Assembly.GetName();
        string? informational = typeof(T).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        return $"{name.Name} {informational?.Split('+')[0]}";
    }
}

/// <summary>Records which files were actually transferred. Called synchronously by the store.</summary>
internal sealed class FileProgress : IProgress<ModelAssetProgress>
{
    public HashSet<string> Downloaded { get; } = [];

    public void Report(ModelAssetProgress value)
    {
        if (value.Stage == ModelAssetStage.Downloading)
        {
            Downloaded.Add(value.Path);
        }
    }
}
