// Native AOT consumer for the Starlings packages (plan 10 G6; production-readiness R9).
// tools/aot-check-ai.sh copies this file with AotConsumer.csproj.in into a temporary directory outside the repository,
// once per target, restores the packed packages from a local folder feed, and builds it twice: a normal JIT build and
// `dotnet publish -p:PublishAot=true`. Both binaries run the same inference on the cached model files and write their
// results to a file; the script compares the two files (discrete values exactly, floats within a bound, diagnostics
// that legitimately differ under AOT reported only).
//
//   AotConsumer <cache> <results file> [<empty download dir>]
//
// The target is fixed at build time (AotTarget in the project file, one preprocessor symbol per target), so each
// published binary references only the packages of its target and its trim/AOT warnings belong to those packages.
// The download dir is used only by the acquisition target (a real Hub download of e5's tokenizer.json, no token).
// Exit code 0 only if the inference ran; equality is the script's job.
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Starlings.Acquisition;
#if AOT_RUNTIME || AOT_E5 || AOT_E5_MEAI || AOT_E5_MLNET || AOT_GLINER2 || AOT_EG2 || AOT_EG2_MEAI || AOT_EG2_MLNET
using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;
#endif
#if AOT_E5 || AOT_E5_MEAI || AOT_E5_MLNET
using Starlings.Models.E5;
#endif
#if AOT_E5_MEAI
using Starlings.Models.E5.Extensions.AI;
#endif
#if AOT_E5_MLNET
using Starlings.Models.E5.MLNet;
#endif
#if AOT_GLINER2
using Starlings.Models.Gliner2;
#endif
#if AOT_EG2 || AOT_EG2_MEAI || AOT_EG2_MLNET
using Starlings.Models.EmbeddingGemma2;
#endif
#if AOT_EG2_MEAI
using Starlings.Models.EmbeddingGemma2.Extensions.AI;
#endif
#if AOT_EG2_MLNET
using Starlings.Models.EmbeddingGemma2.MLNet;
#endif
#if AOT_E5_MEAI || AOT_EG2_MEAI || AOT_QWEN35_MEAI
using Microsoft.Extensions.AI;
#endif
#if AOT_E5_MLNET || AOT_EG2_MLNET
using Microsoft.ML;
using Microsoft.ML.Data;
#endif
#if AOT_QWEN35 || AOT_QWEN35_MEAI
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
#endif
#if AOT_QWEN35_MEAI
using Starlings.Models.Qwen35.Extensions.AI;
#endif

// An exception ends with "error: " and the full exception on stderr and exit code 70 (EX_SOFTWARE), not an
// unhandled-exception abort and core dump (as ai-dotnet/tools/Shared/ToolMain.cs does for the in-repository tools).
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
    if (args.Length is < 2 or > 3)
    {
        Console.Error.WriteLine("usage: AotConsumer <cache> <results file> [<empty download dir>]");
        return 2;
    }

    string cache = args[0];
    var results = new StringBuilder();
    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache, UseEnvironmentToken = false };
    string target = typeof(Marker).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>().First(a => a.Key == "AotTarget").Value!;
    // Native AOT never compiles code at run time; a JIT run does.
    bool aot = !System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeCompiled;
    Console.WriteLine($"target:  {target}");
    Console.WriteLine($"mode:    {(aot ? "Native AOT" : "JIT")} ({RuntimeInformation.FrameworkDescription}, {RuntimeInformation.RuntimeIdentifier}, dynamic code {(System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported ? "supported" : "not supported")})");
    Put("target", target);
#if AOT_CUDA
    const string Provider = "cuda";
#else
    const string Provider = "cpu";
#endif
    Put("provider", Provider);

    string[] queries = ["query: how much protein should a female eat", "query: 南瓜的家常做法", "query: Warum rostet Eisen?"];
    string[] passages =
    [
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "passage: Iron rusts when it reacts with oxygen and water to form iron oxide.",
        "passage: 清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精",
    ];

#if AOT_ACQUISITION
    // Acquisition only: lock JSON round trip, offline verification of the cached e5 files, and (with a download dir) a
    // real download of tokenizer.json from the public Hub repository without a token.
    var e5Lock = new ModelAssetLock("multilingual-e5-small-tokenizer", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        [new ModelAssetFile("tokenizer.json", 17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39")]);
    string json = e5Lock.ToJson();
    ModelAssetLock parsed = ModelAssetLock.Parse(json);
    Put("lock.json", json);
    Put("lock.parsed", $"{parsed.Name} {parsed.Repository}@{parsed.Revision} {string.Join(",", parsed.Files.Select(f => $"{f.Path}:{f.Size}:{f.Sha256}"))}");
    using (var store = new ModelAssetStore(acquisition))
    {
        ModelSnapshot local = store.OpenLocal(e5Lock);
        Put("offline.verified", Path.GetFileName(local.GetFilePath("tokenizer.json")) + " " + new FileInfo(local.GetFilePath("tokenizer.json")).Length);
    }

    if (args.Length == 3)
    {
        using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = args[2], UseEnvironmentToken = false });
        ModelSnapshot downloaded = await store.DownloadAsync(e5Lock);
        Put("download.verified", Path.GetFileName(downloaded.GetFilePath("tokenizer.json")) + " " + new FileInfo(downloaded.GetFilePath("tokenizer.json")).Length);
    }
#endif

#if AOT_RUNTIME
    // Runtime.Onnx only: native resolution and one raw session run over fixed ids (e5's graph, no tokenizer).
    Put("ort", $"native {OnnxRuntimeInfo.NativeVersion}, managed {OnnxRuntimeInfo.ManagedVersion}, providers {string.Join(",", OnnxRuntimeInfo.AvailableProviders)}");
    string e5Graph = Path.Combine(cache, "intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx/model.onnx");
    using (var session = OnnxModelSession.Create(e5Graph, Sessions()))
    {
        Put("session.inputs", string.Join(",", session.Inputs.Select(i => i.Name)));
        Put("session.outputs", string.Join(",", session.Outputs.Select(o => o.Name)));
        long[] ids = [0, 41, 1222, 83, 10, 3034, 5, 2];
        long[] shape = [1, ids.Length];
        var names = new List<string>();
        var values = new List<OrtValue>();
        foreach (var input in session.Inputs)
        {
            long[] data = input.Name switch
            {
                "input_ids" => ids,
                "attention_mask" => [.. ids.Select(_ => 1L)],
                _ => new long[ids.Length],
            };
            names.Add(input.Name);
            values.Add(OrtValue.CreateTensorValueFromMemory(data, shape));
        }

        using var outputs = session.Run(names, values, [session.Outputs[0].Name]);
        Put("session.output.shape", string.Join("x", outputs.First().GetTensorTypeAndShape().Shape));
        PutVector("session.output", outputs.First().GetTensorDataAsSpan<float>().ToArray());
        foreach (OrtValue v in values)
        {
            v.Dispose();
        }
    }
#endif

#if AOT_E5
    E5ModelFiles e5Files = E5ModelAssets.OpenLocal(acquisition);
    var e5Tokenizer = E5Tokenizer.FromFile(e5Files.TokenizerPath, E5ModelAssets.TokenizerSha256);
    using (var encoder = E5Encoder.Create(e5Files.ModelPath, new E5EncoderOptions { Session = Sessions() with { ExpectedModelSha256 = E5ModelAssets.ModelSha256 } }))
    {
        Put("session.provider", encoder.Session.Diagnostics.SessionProviders[0]);
        var embedder = new E5TextEmbedder(e5Tokenizer, encoder);
        string[] texts = [.. queries, .. passages];
        for (int i = 0; i < texts.Length; i++)
        {
            Put($"ids[{i}]", string.Join(",", e5Tokenizer.Tokenize(texts[i]).InputIds.ToArray()));
            PutVector($"embed[{i}]", embedder.Embed(texts[i]));
        }

        float[][] batch = embedder.EmbedBatch(texts);
        for (int i = 0; i < batch.Length; i++)
        {
            PutVector($"batch[{i}]", batch[i]);
        }
    }
#endif

#if AOT_E5_MEAI
    using (var generator = E5EmbeddingGenerator.Create(E5ModelAssets.OpenLocal(acquisition), new E5EmbeddingGeneratorOptions { Encoder = new E5EncoderOptions { Session = Sessions() } }))
    {
        Put("session.provider", generator.Embedder.Encoder.Session.Diagnostics.SessionProviders[0]);
        GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync([.. queries, .. passages]);
        for (int i = 0; i < generated.Count; i++)
        {
            PutVector($"generated[{i}]", generated[i].Vector.ToArray());
        }

        Put("usage.input", generated.Usage?.InputTokenCount?.ToString(CultureInfo.InvariantCulture) ?? "none");
        Put("metadata", generator.GetService<EmbeddingGeneratorMetadata>()?.DefaultModelId ?? "none");
    }
#endif

#if AOT_E5_MLNET
    E5ModelFiles mlFiles = E5ModelAssets.OpenLocal(acquisition);
    using (var encoder = E5Encoder.Create(mlFiles.ModelPath, new E5EncoderOptions { Session = Sessions() }))
    {
        var embedder = new E5TextEmbedder(E5Tokenizer.FromFile(mlFiles.TokenizerPath, E5ModelAssets.TokenizerSha256), encoder);
        var ml = new MLContext(seed: 1);
        IDataView view = ml.Data.LoadFromEnumerable([.. queries.Concat(passages).Select(t => new TextRow { Text = t })]);
        using var transformer = ml.Transforms.ApplyE5Embedding(embedder, "Embedding", "Text", new E5EmbeddingOptions { BatchSize = 4 }).Fit(view);
        float[][] rows = [.. transformer.Transform(view).GetColumn<float[]>("Embedding")];
        for (int i = 0; i < rows.Length; i++)
        {
            PutVector($"mlnet[{i}]", rows[i]);
        }
    }
#endif

#if AOT_GLINER2
    using (var extractor = Gliner2EntityExtractor.Create(Gliner2ModelAssets.OpenLocal(acquisition), new Gliner2ExtractorOptions { Session = Sessions() }))
    {
        string[] texts =
        [
            "Apple CEO Tim Cook announced the iPhone 15 in Cupertino on September 12, 2023.",
            "Marie Curie won the Nobel Prize in Physics in 1903 together with Pierre Curie",
        ];
        Put("session.provider", extractor.Session.Diagnostics.SessionProviders[0]);
        var extracted = extractor.ExtractBatch(texts, ["person", "organization", "location", "date", "product"], 0.5f);
        for (int i = 0; i < texts.Length; i++)
        {
            Put($"entities[{i}]", string.Join("; ", extracted[i].Entities.Select(e => $"{e.Label}:{e.Text}[{e.Start},{e.End})")));
            PutVector($"confidence[{i}]", [.. extracted[i].Entities.Select(e => e.Confidence)]);
        }
    }
#endif

#if AOT_EG2
    EmbeddingGemma2ModelFiles eg2Files = EmbeddingGemma2ModelAssets.OpenLocal(acquisition);
    using (var encoder = EmbeddingGemma2Encoder.Create(eg2Files.ModelPath, new EmbeddingGemma2EncoderOptions { Session = Sessions() with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 } }))
    {
        Put("session.provider", encoder.Session.Diagnostics.SessionProviders[0]);
        var embedder = new EmbeddingGemma2TextEmbedder(EmbeddingGemma2Tokenizer.FromFile(eg2Files.TokenizerPath), encoder);
        float[][] q = embedder.EmbedBatch([.. queries.Select(StripPrefix)], EmbeddingGemma2Prompt.Query);
        float[][] d = embedder.EmbedBatch([.. passages.Select(StripPrefix)], EmbeddingGemma2Prompt.Document);
        for (int i = 0; i < q.Length; i++)
        {
            PutVector($"query[{i}]", q[i]);
            PutVector($"document[{i}]", d[i]);
        }
    }

    // Image path: a synthetic RGB gradient (no decoder package), vision graph plus multimodal text graph.
    using (var multimodal = EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition), new EmbeddingGemma2MultimodalOptions { Session = Sessions() }))
    {
        const int W = 96, H = 64;
        byte[] rgb = new byte[W * H * 3];
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                int o = ((y * W) + x) * 3;
                rgb[o] = (byte)(x * 255 / (W - 1));
                rgb[o + 1] = (byte)(y * 255 / (H - 1));
                rgb[o + 2] = (byte)((x + y) % 256);
            }
        }

        Put("vision.provider", multimodal.VisionSession.Diagnostics.SessionProviders[0]);
        Put("multimodal.provider", multimodal.MultimodalSession.Diagnostics.SessionProviders[0]);
        var image = EmbeddingGemma2Image.FromRgb24(W, H, rgb);
        PutVector("image", multimodal.Embed(image));
        PutVector("interleaved", multimodal.Embed(EmbeddingGemma2MultimodalInput.Create("A gradient: ", image)));

        // Video path from already-decoded frames (no FFmpeg): three frames, 1 fps, 3 seconds; the frames shift the gradient.
        var frames = new List<EmbeddingGemma2Image>();
        for (int f = 0; f < 3; f++)
        {
            byte[] frame = new byte[rgb.Length];
            for (int i = 0; i < rgb.Length; i++)
            {
                frame[i] = (byte)(rgb[i] + (f * 40));
            }

            frames.Add(EmbeddingGemma2Image.FromRgb24(W, H, frame));
        }

        var video = EmbeddingGemma2Video.FromDecodedFrames(frames, framesPerSecond: 1.0, durationSeconds: 3.0);
        Put("video.sampled", string.Join(",", video.GetSampledFrameIndices()));
        PutVector("video", multimodal.Embed(video));
    }
#endif

#if AOT_EG2_MEAI
    using (var generator = EmbeddingGemma2EmbeddingGenerator.Create(EmbeddingGemma2ModelAssets.OpenLocal(acquisition), new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query, Encoder = new EmbeddingGemma2EncoderOptions { Session = Sessions() } }))
    {
        Put("session.provider", generator.Embedder.Encoder.Session.Diagnostics.SessionProviders[0]);
        GeneratedEmbeddings<Embedding<float>> generated = await generator.GenerateAsync([.. queries.Select(StripPrefix)]);
        for (int i = 0; i < generated.Count; i++)
        {
            PutVector($"generated[{i}]", generated[i].Vector.ToArray());
        }

        GeneratedEmbeddings<Embedding<float>> small = await generator.GenerateAsync([.. queries.Select(StripPrefix)], new EmbeddingGenerationOptions { Dimensions = 256 });
        PutVector("generated.256[0]", small[0].Vector.ToArray());
        Put("usage.input", generated.Usage?.InputTokenCount?.ToString(CultureInfo.InvariantCulture) ?? "none");
    }
#endif

#if AOT_EG2_MLNET
    EmbeddingGemma2ModelFiles eg2MlFiles = EmbeddingGemma2ModelAssets.OpenLocal(acquisition);
    {
        var ml = new MLContext(seed: 1);
        IDataView view = ml.Data.LoadFromEnumerable([.. passages.Select(t => new TextRow { Text = StripPrefix(t) })]);
        using var transformer = ml.Transforms.ApplyEmbeddingGemma2Embedding(eg2MlFiles, "Embedding", "Text", new EmbeddingGemma2EmbeddingOptions { Prompt = EmbeddingGemma2Prompt.Document, BatchSize = 3, Encoder = new EmbeddingGemma2EncoderOptions { Session = Sessions() } }).Fit(view);
        float[][] rows = [.. transformer.Transform(view).GetColumn<float[]>("Embedding")];
        for (int i = 0; i < rows.Length; i++)
        {
            PutVector($"mlnet[{i}]", rows[i]);
        }
    }
#endif

#if AOT_QWEN35 || AOT_QWEN35_MEAI
#if AOT_CUDA
    // On CUDA both variants: fp32x and fp16x (fp16x is built for CUDA only).
    Qwen35Variant[] variants = [Qwen35Variant.Fp32x, Qwen35Variant.Fp16x];
#else
    Qwen35Variant[] variants = [Qwen35Variant.Fp32x];
#endif
    foreach (var variant in variants)
    using (var model = Qwen35ChatModel.Load(Qwen35ModelAssets.OpenLocal(variant, acquisition), new Qwen35ModelOptions { Provider = Provider == "cuda" ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu }))
    {
        string v = variant == Qwen35Variant.Fp32x ? string.Empty : $"{variant}.";
        Put(v + "model", $"{model.Info.Profile} {model.Info.Provider}");
        // Reported, not compared: GenAI's version comes from the .deps.json, which a Native AOT binary does not have.
        Info(v + "genai.version", model.Info.GenAIVersion);
        const string Question = "What is the capital of France? Answer in one sentence.";
#if AOT_QWEN35
        GenerationResult result = model.Generate([Qwen35Message.User(Question)], new Qwen35ChatOptions { MaxNewTokens = 32 });
        Put(v + "text", result.Text);
        Put(v + "token_ids", string.Join(",", result.TokenIds));
        Put(v + "finish", $"{result.FinishReason} {result.Usage.PromptTokens}+{result.Usage.CompletionTokens}");
#else
        using IChatClient client = new Qwen35ChatClient(model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { MaxNewTokens = 32 } });
        ChatResponse response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, Question)]);
        Put(v + "text", response.Text);
        Put(v + "finish", $"{response.FinishReason} {response.Usage?.InputTokenCount}+{response.Usage?.OutputTokenCount}");
        var streamed = new StringBuilder();
        await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, Question)]))
        {
            streamed.Append(update.Text);
        }

        Put(v + "streamed", streamed.ToString());
#endif
    }
#endif

    File.WriteAllText(args[1], results.ToString());
    Console.Write(results.ToString().Length > 4000 ? results.ToString()[..4000] + "...\n" : results.ToString());
    Console.WriteLine("RESULT: RAN");
    return 0;

    void Put(string key, string value) => Row("str", key, value);

    // A diagnostic that may legitimately differ between JIT and AOT: the script reports it but does not compare it.
    void Info(string key, string value) => Row("inf", key, value);

    void Row(string kind, string key, string value) =>
        results.Append(kind).Append('\t').Append(key).Append('\t').Append(value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\n", "\\n", StringComparison.Ordinal)).Append('\n');

    void PutVector(string key, float[] values) =>
        results.Append("vec\t").Append(key).Append('\t').AppendJoin(',', values.Select(v => v.ToString("R", CultureInfo.InvariantCulture))).Append('\n');

#if AOT_RUNTIME || AOT_E5 || AOT_E5_MEAI || AOT_E5_MLNET || AOT_GLINER2 || AOT_EG2 || AOT_EG2_MEAI || AOT_EG2_MLNET
    // Session settings of this build's provider: CUDA device 0 (TF32 off, forced by Starlings.Runtime.Onnx) or the CPU.
    static OnnxSessionOptions Sessions() =>
#if AOT_CUDA
        new() { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 };
#else
        new();
#endif
#endif

#if AOT_EG2 || AOT_EG2_MEAI || AOT_EG2_MLNET
    static string StripPrefix(string text) => text[(text.IndexOf(':', StringComparison.Ordinal) + 2)..];
#endif
}

/// <summary>Anchor for the assembly metadata.</summary>
internal sealed class Marker;

#if AOT_E5_MLNET || AOT_EG2_MLNET
/// <summary>Input row of the ML.NET pipeline.</summary>
internal sealed class TextRow
{
    public string Text { get; set; } = string.Empty;
}
#endif
