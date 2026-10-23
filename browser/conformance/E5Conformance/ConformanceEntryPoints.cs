using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.JSInterop;
using Starlings.Models.E5;
using Starlings.Models.E5.Browser;
using Starlings.Runtime.OnnxWeb;
using Starlings.Tokenizers;

namespace Starlings.E5Conformance;

/// <summary>
/// Entry points <c>run_conformance.py</c> calls with <c>DotNet.invokeMethodAsync('E5Conformance', name, ...)</c>.
/// The server mounts the protected fixtures at <c>/data/fixtures/</c>, the pinned tokenizer.json under
/// <c>/data/tok/</c> and the graphs under <c>/data/models/</c>.
/// </summary>
public static class ConformanceEntryPoints
{
    private const string Fixtures = "data/fixtures/multilingual-e5-small/onnx-fp32-cpu/";
    private static HttpClient? _http;

    /// <summary>Called once from Program.</summary>
    public static void Initialize(Uri baseAddress) => _http = new HttpClient { BaseAddress = baseAddress };

    private static HttpClient Http => _http ?? throw new InvalidOperationException("Not initialized.");

    /// <summary>
    /// Loads one variant on one provider through <see cref="E5BrowserEmbedder.LoadAsync"/> and runs every check whose
    /// result <c>evaluate_e5_browser.py</c> compares with the committed tolerances. Returns one JSON document.
    /// </summary>
    [JSInvokable]
    public static async Task<string> Run(string variant, string modelUrl, string tokenizerUrl, string provider, int threads, bool proxy, int iterations)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            // The build mode (AOT or interpreter) is the harness label (--label); Mono on WebAssembly has no API for it.
            json.WriteString("runtime", RuntimeInformation.FrameworkDescription);
            json.WriteString("variant", variant);
            json.WriteString("provider", provider);
            try
            {
                await RunCoreAsync(json, variant, modelUrl, tokenizerUrl, provider, threads, proxy, iterations);
            }
#pragma warning disable CA1031 // The harness records any failure in the result document.
            catch (Exception ex)
#pragma warning restore CA1031
            {
                json.WriteString("error", ex.ToString());
            }

            json.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static async Task RunCoreAsync(Utf8JsonWriter json, string variant, string modelUrl, string tokenizerUrl, string provider, int threads, bool proxy, int iterations)
    {
        var options = new E5BrowserLoadOptions
        {
            Variant = Enum.Parse<E5BrowserVariant>(variant, ignoreCase: true),
            ModelUrl = modelUrl,
            TokenizerUrl = tokenizerUrl,
            Provider = provider == "webgpu" ? OnnxWebExecutionProvider.WebGpu : OnnxWebExecutionProvider.Wasm,
            Runtime = new OnnxWebRuntimeOptions
            {
                Bundle = provider == "webgpu" ? Starlings.Runtime.OnnxWeb.WebGpu.OnnxWebGpuBundle.WebGpu : OnnxWebBundle.Wasm,
                NumThreads = threads > 0 ? threads : null,
                Proxy = proxy,
            },
            // Evidence runs measure the network path; the cache is exercised by a separate reload check.
            Fetch = new OnnxWebFetchOptions { CacheName = null },
        };

        var stages = new List<(string Stage, double Ms)>();
        var stopwatch = Stopwatch.StartNew();
        var last = E5BrowserLoadStage.Runtime;
        var progress = new SyncProgress<E5BrowserLoadProgress>(p =>
        {
            if (p.Stage != last)
            {
                stages.Add((last.ToString(), stopwatch.Elapsed.TotalMilliseconds));
                last = p.Stage;
            }
        });
        using var embedder = await E5BrowserEmbedder.LoadAsync(options, progress);
        stages.Add(("Done", stopwatch.Elapsed.TotalMilliseconds));
        json.WriteStartObject("load");
        foreach (var (stage, ms) in stages)
        {
            json.WriteNumber(stage + "_until_ms", ms);
        }

        json.WriteNumber("session_create_ms", embedder.Encoder.Session!.CreateTime.TotalMilliseconds);
        json.WriteNumber("model_bytes", embedder.Encoder.Session.ModelBytes);
        json.WriteString("model_sha256", embedder.Encoder.Session.ModelSha256);
        json.WriteEndObject();
        json.WriteString("profile", embedder.Profile);
        json.WriteNumber("dimension", embedder.Encoder.Dimension);
        WriteRuntimeInfo(json, OnnxWebRuntime.Info!);

        List<JsonElement> embeddingRows = ReadJsonLines(await Http.GetByteArrayAsync(Fixtures + "embeddings.jsonl"));
        List<JsonElement> tokenizerRows = ReadJsonLines(await Http.GetByteArrayAsync(Fixtures + "tokenizer.jsonl"));
        var texts = tokenizerRows.ToDictionary(r => r.GetProperty("id").GetString()!, r => r.GetProperty("text").GetString()!);

        Console.WriteLine("conformance stage: tokenizer");
        // Tokenization, two ways: the e5 pipeline tokenizer (special tokens, truncation to 512) against the ids the
        // protected embeddings were computed from, and the owned tokenizer itself against the full tokenizer fixtures.
        json.WriteStartArray("e5_tokenizer");
        foreach (var row in embeddingRows)
        {
            string id = row.GetProperty("id").GetString()!;
            var input = embedder.Tokenizer.Tokenize(texts[id]);
            json.WriteStartObject();
            json.WriteString("id", id);
            WriteLongs(json, "input_ids", input.InputIds.Span);
            WriteLongs(json, "attention_mask", input.AttentionMask.Span);
            json.WriteEndObject();
        }

        json.WriteEndArray();

        var raw = Tokenizer.FromUtf8Json(await Http.GetByteArrayAsync(tokenizerUrl));
        json.WriteStartArray("tokenizer");
        foreach (var row in tokenizerRows)
        {
            if (row.TryGetProperty("pair", out var pair) && pair.ValueKind == JsonValueKind.String)
            {
                continue; // Pair encoding is not an embedding input (manifest behavior.unsupported).
            }

            TokenEncoding encoding = raw.Encode(row.GetProperty("text").GetString()!);
            json.WriteStartObject();
            json.WriteString("id", row.GetProperty("id").GetString());
            WriteInts(json, "ids", encoding.Ids);
            WriteInts(json, "attention_mask", encoding.AttentionMask);
            json.WriteStartArray("offsets_utf16");
            foreach (var offset in encoding.Offsets)
            {
                json.WriteStartArray();
                json.WriteNumberValue(offset.Start);
                json.WriteNumberValue(offset.End);
                json.WriteEndArray();
            }

            json.WriteEndArray();
            json.WriteEndObject();
        }

        json.WriteEndArray();

        Console.WriteLine("conformance stage: embeddings");
        // Embeddings from the fixture ids, one row per call (the reference is batch 1).
        var fixtureInputs = embeddingRows.ToDictionary(
            r => r.GetProperty("id").GetString()!,
            r => new EncoderInput(r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray()));
        stopwatch.Restart();
        _ = await embedder.Encoder.EmbedAsync(fixtureInputs.Values.First());
        json.WriteNumber("first_run_ms", stopwatch.Elapsed.TotalMilliseconds);
        json.WriteStartObject("embeddings");
        foreach (var (id, input) in fixtureInputs)
        {
            WriteFloats(json, id, await embedder.Encoder.EmbedAsync(input));
        }

        json.WriteEndObject();

        Console.WriteLine("conformance stage: batch8");
        // Padded batch of the first 8 short cases (shorter rows padded to the longest), same encoder.
        var shortIds = fixtureInputs.Where(kv => kv.Value.Length < 512).Take(8).ToList();
        float[][] batch = await embedder.Encoder.EmbedBatchAsync([.. shortIds.Select(kv => kv.Value)]);
        json.WriteStartObject("batch8");
        for (int i = 0; i < shortIds.Count; i++)
        {
            WriteFloats(json, shortIds[i].Key, batch[i]);
        }

        json.WriteEndObject();

        Console.WriteLine("conformance stage: generator");
        // Text in, embedding out, through the IEmbeddingGenerator (owned tokenizer, no prefix), one call for all texts.
        using var generator = new E5BrowserEmbeddingGenerator(embedder);
        var ids = embeddingRows.Select(r => r.GetProperty("id").GetString()!).ToList();
        var generated = await generator.GenerateAsync(ids.Select(id => texts[id]));
        json.WriteNumber("generator_input_tokens", generated.Usage?.InputTokenCount ?? -1);
        json.WriteStartObject("generator_embeddings");
        for (int i = 0; i < ids.Count; i++)
        {
            WriteFloats(json, ids[i], generated[i].Vector.ToArray());
        }

        json.WriteEndObject();

        Console.WriteLine("conformance stage: latency");
        // Warm latency: one 34-token passage, the short batch of 8, one 512-token input.
        string passageId = fixtureInputs.OrderBy(kv => Math.Abs(kv.Value.Length - 34)).First().Key;
        var passage = fixtureInputs[passageId];
        var long512 = fixtureInputs.Values.First(i => i.Length == 512);
        json.WriteStartObject("latency_ms");
        await Time(json, $"passage{passage.Length}_b1", iterations, () => embedder.Encoder.EmbedAsync(passage));
        await Time(json, "short_b8", iterations, () => embedder.Encoder.EmbedBatchAsync([.. shortIds.Select(kv => kv.Value)]));
        await Time(json, "long512_b1", Math.Max(3, iterations / 2), () => embedder.Encoder.EmbedAsync(long512));
        await Time(json, $"generator_text_{passageId}_b1", iterations, () => generator.GenerateAsync([texts[passageId]]));
        json.WriteEndObject();
    }

    /// <summary>
    /// Reload check: loads the variant twice with Cache Storage on; the second load must read both files from the
    /// cache and give the same embedding bit for bit. Returns JSON.
    /// </summary>
    [JSInvokable]
    public static async Task<string> RunCacheCheck(string variant, string modelUrl, string tokenizerUrl)
    {
        var options = new E5BrowserLoadOptions
        {
            Variant = Enum.Parse<E5BrowserVariant>(variant, ignoreCase: true),
            ModelUrl = modelUrl,
            TokenizerUrl = tokenizerUrl,
            Fetch = new OnnxWebFetchOptions { CacheName = "starlings-conformance" },
        };
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            float[]? first = null;
            for (int round = 0; round < 2; round++)
            {
                var stopwatch = Stopwatch.StartNew();
                using var embedder = await E5BrowserEmbedder.LoadAsync(options);
                float[] v = await embedder.EmbedAsync("query: wie spät ist es?");
                json.WriteNumber($"round{round}_load_ms", stopwatch.Elapsed.TotalMilliseconds);
                if (first is null)
                {
                    first = v;
                }
                else
                {
                    json.WriteBoolean("same_embedding", first.AsSpan().SequenceEqual(v));
                }
            }

            json.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static async Task Time(Utf8JsonWriter json, string name, int iterations, Func<Task> action)
    {
        for (int i = 0; i < 3; i++)
        {
            await action();
        }

        var samples = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            await action();
            samples[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }

        Array.Sort(samples);
        json.WriteStartObject(name);
        json.WriteNumber("median", samples[samples.Length / 2]);
        json.WriteStartArray("samples");
        foreach (double s in samples)
        {
            json.WriteNumberValue(Math.Round(s, 3));
        }

        json.WriteEndArray();
        json.WriteEndObject();
    }

    private static void WriteRuntimeInfo(Utf8JsonWriter json, OnnxWebRuntimeInfo info)
    {
        json.WriteStartObject("onnxruntime_web");
        json.WriteString("version", info.OrtVersion);
        json.WriteString("bundle", info.Bundle);
        json.WriteBoolean("cross_origin_isolated", info.CrossOriginIsolated);
        json.WriteNumber("hardware_concurrency", info.HardwareConcurrency);
        json.WriteNumber("num_threads", info.NumThreads);
        json.WriteBoolean("proxy", info.Proxy);
        json.WriteBoolean("webgpu_exposed", info.WebGpuExposed);
        json.WriteEndObject();
    }

    private static List<JsonElement> ReadJsonLines(byte[] bytes)
    {
        var rows = new List<JsonElement>();
        foreach (string line in Encoding.UTF8.GetString(bytes).Split('\n'))
        {
            if (line.Length > 0)
            {
                using var document = JsonDocument.Parse(line);
                rows.Add(document.RootElement.Clone());
            }
        }

        return rows;
    }

    private static void WriteFloats(Utf8JsonWriter json, string name, ReadOnlySpan<float> values)
    {
        json.WriteStartArray(name);
        foreach (float v in values)
        {
            json.WriteNumberValue(v);
        }

        json.WriteEndArray();
    }

    private static void WriteLongs(Utf8JsonWriter json, string name, ReadOnlySpan<long> values)
    {
        json.WriteStartArray(name);
        foreach (long v in values)
        {
            json.WriteNumberValue(v);
        }

        json.WriteEndArray();
    }

    private static void WriteInts(Utf8JsonWriter json, string name, IReadOnlyList<int> values)
    {
        json.WriteStartArray(name);
        foreach (int v in values)
        {
            json.WriteNumberValue(v);
        }

        json.WriteEndArray();
    }

    private sealed class SyncProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}
