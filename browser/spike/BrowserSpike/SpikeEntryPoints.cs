using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Starlings.Tokenizers;
using Microsoft.JSInterop;

namespace Starlings.BrowserSpike;

/// <summary>
/// Entry points the Playwright harness calls with <c>DotNet.invokeMethodAsync('BrowserSpike', name, ...)</c>.
/// Data is served by the harness: <c>/data/tok/</c> (pinned assets, <c>$STARLING_MODELS</c> layout),
/// <c>/data/fixtures/</c> (<c>model-pipeline/protected/</c>), <c>/data/models/</c> (graphs), <c>/ort/</c>
/// (onnxruntime-web).
/// </summary>
public static class SpikeEntryPoints
{
    private const int Dimension = 384;
    private static HttpClient? _http;
    private static IJSRuntime? _js;

    /// <summary>Called once from Program.</summary>
    public static void Initialize(Uri baseAddress, IJSRuntime js)
    {
        _http = new HttpClient { BaseAddress = baseAddress };
        _js = js;
    }

    private static HttpClient Http => _http ?? throw new InvalidOperationException("Not initialized.");

    private static IJSRuntime Js => _js ?? throw new InvalidOperationException("Not initialized.");

    /// <summary>Tokenizer conformance and timing for every profile (the same code as the native run).</summary>
    [JSInvokable]
    public static Task<string> RunTokenizerSuite(double encodeSeconds, int warmLoads) =>
        TokenizerSuite.RunAsync(new HttpSource(Http), RuntimeLabel(), encodeSeconds, warmLoads);

    /// <summary>
    /// e5-small through onnxruntime-web: loads one graph on one backend, embeds the 46 protected cases one by one
    /// (fixture input ids, C# mean pooling and normalization as in Starlings.Models.E5), embeds a padded
    /// batch of 8, then times warm requests in JS (session.run only) and end to end from C# (tokenize, interop, run,
    /// pool).
    /// </summary>
    [JSInvokable]
    public static async Task<string> RunE5(string modelUrl, string backend, int threads, int iterations)
    {
        await using IJSObjectReference ort = await Js.InvokeAsync<IJSObjectReference>("import", "./spike-ort.js");
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = false }))
        {
            json.WriteStartObject();
            json.WriteString("runtime", RuntimeLabel());
            json.WriteString("model_url", modelUrl);
            json.WriteString("backend", backend);
            json.WriteNumber("threads_requested", threads);

            string load = await ort.InvokeAsync<string>("create", modelUrl, backend, threads);
            json.WritePropertyName("load");
            json.WriteRawValue(load);
            using (JsonDocument loadDoc = JsonDocument.Parse(load))
            {
                if (loadDoc.RootElement.TryGetProperty("error", out _))
                {
                    json.WriteEndObject();
                    await json.FlushAsync();
                    return Encoding.UTF8.GetString(buffer.ToArray());
                }
            }

            byte[] embeddingsJsonl = await Http.GetByteArrayAsync("data/fixtures/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl");
            List<JsonElement> rows = TokenizerSuite.ReadJsonLines(embeddingsJsonl);

            // Single-row embeddings of every case.
            var firstRun = Stopwatch.StartNew();
            long[] firstIds = [.. rows[0].GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64())];
            _ = await RunAsync(ort, [firstIds]);
            json.WriteNumber("first_run_ms", firstRun.Elapsed.TotalMilliseconds);

            json.WriteStartArray("embeddings");
            firstRun.Restart();
            foreach (JsonElement row in rows)
            {
                long[] ids = [.. row.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64())];
                float[] embedding = (await RunAsync(ort, [ids]))[0];
                json.WriteStartObject();
                json.WriteString("id", row.GetProperty("id").GetString());
                WriteFloats(json, "embedding", embedding);
                json.WriteEndObject();
            }

            json.WriteEndArray();
            json.WriteNumber("all_cases_ms", firstRun.Elapsed.TotalMilliseconds);

            // Padded batch of the first 8 short cases (the bench_e5_ort.py short-b8 workload).
            long[][] b8 = ShortBatch(rows, 8);
            float[][] batchEmbeddings = await RunAsync(ort, b8);
            json.WriteStartArray("batch8");
            List<JsonElement> shortRows = [.. rows.Where(r => r.GetProperty("input_ids").GetArrayLength() < 512)];
            for (int i = 0; i < b8.Length; i++)
            {
                json.WriteStartObject();
                json.WriteString("id", shortRows[i].GetProperty("id").GetString());
                WriteFloats(json, "embedding", batchEmbeddings[i]);
                json.WriteEndObject();
            }

            json.WriteEndArray();

            // Warm latency in JS (session.run only, inputs prebuilt).
            long[] passage = [.. rows.First(r => r.GetProperty("id").GetString() == "e5-passage-prefix").GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64())];
            long[] long512 = [.. rows.First(r => r.GetProperty("id").GetString() == "long-english").GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64())];
            json.WriteStartObject("latency_js");
            foreach ((string name, long[][] batch, int n) in new[]
            {
                ("passage34-b1", new[] { passage }, iterations),
                ("short-b8", b8, iterations),
                ("long512-b1", new[] { long512 }, Math.Max(3, iterations / 4)),
            })
            {
                (byte[] idBytes, byte[] maskBytes, int seq) = Pack(batch);
                string timing = await ort.InvokeAsync<string>("bench", idBytes, maskBytes, batch.Length, seq, 3, n);
                json.WritePropertyName(name);
                json.WriteRawValue(timing);
            }

            json.WriteEndObject();

            // End to end from C#: owned tokenizer, interop, run, pooling, for one passage text and a batch of 8 texts.
            byte[] tokJson = await Http.GetByteArrayAsync("data/tok/" + TokenizerProfile.All[0].AssetPath);
            var lw = Stopwatch.StartNew();
            Tokenizer tokenizer = Tokenizer.FromUtf8Json(tokJson).WithTruncation(new TruncationOptions(512));
            json.WriteNumber("e2e_tokenizer_load_ms", lw.Elapsed.TotalMilliseconds);
            List<JsonElement> textRows = TokenizerSuite.ReadJsonLines(await Http.GetByteArrayAsync("data/fixtures/multilingual-e5-small/onnx-fp32-cpu/tokenizer.jsonl"));
            string passageText = textRows.First(r => r.GetProperty("id").GetString() == "e5-passage-prefix").GetProperty("text").GetString()!;
            string[] batchTexts = [.. shortRows.Take(8).Select(r => textRows.First(t => t.GetProperty("id").GetString() == r.GetProperty("id").GetString()).GetProperty("text").GetString()!)];
            json.WriteStartObject("latency_e2e");
            foreach ((string name, string[] texts) in new[] { ("passage34-b1", new[] { passageText }), ("short-b8", batchTexts) })
            {
                for (int w = 0; w < 3; w++)
                {
                    await EmbedTextsAsync(ort, tokenizer, texts);
                }

                var samples = new List<double>();
                for (int i = 0; i < iterations; i++)
                {
                    var sw = Stopwatch.StartNew();
                    await EmbedTextsAsync(ort, tokenizer, texts);
                    samples.Add(sw.Elapsed.TotalMilliseconds);
                }

                WriteSamples(json, name, samples);
            }

            json.WriteEndObject();

            string memory = await ort.InvokeAsync<string>("memory");
            json.WritePropertyName("memory");
            json.WriteRawValue(memory);
            await ort.InvokeVoidAsync("release");
            json.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static async Task EmbedTextsAsync(IJSObjectReference ort, Tokenizer tokenizer, string[] texts)
    {
        long[][] ids = new long[texts.Length][];
        for (int i = 0; i < texts.Length; i++)
        {
            ImmutableIdsToLongs(tokenizer.Encode(texts[i]), out ids[i]);
        }

        _ = await RunAsync(ort, ids);
    }

    private static void ImmutableIdsToLongs(TokenEncoding encoding, out long[] ids)
    {
        ids = new long[encoding.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = encoding.Ids[i];
        }
    }

    /// <summary>Runs rows right-padded with id 1 (mask 0) and returns the pooled, normalized embeddings.</summary>
    private static async Task<float[][]> RunAsync(IJSObjectReference ort, long[][] rows)
    {
        (byte[] idBytes, byte[] maskBytes, int seq) = Pack(rows);
        byte[] hiddenBytes = await ort.InvokeAsync<byte[]>("run", idBytes, maskBytes, rows.Length, seq);
        ReadOnlySpan<float> hidden = MemoryMarshal.Cast<byte, float>(hiddenBytes);
        ReadOnlySpan<long> mask = MemoryMarshal.Cast<byte, long>(maskBytes);
        var result = new float[rows.Length][];
        for (int r = 0; r < rows.Length; r++)
        {
            result[r] = new float[Dimension];
            MeanPoolAndNormalize(hidden.Slice(r * seq * Dimension, seq * Dimension), mask.Slice(r * seq, seq), result[r]);
        }

        return result;
    }

    private static (byte[] Ids, byte[] Mask, int Seq) Pack(long[][] rows)
    {
        int seq = rows.Max(r => r.Length);
        var ids = new long[rows.Length * seq];
        var mask = new long[rows.Length * seq];
        for (int r = 0; r < rows.Length; r++)
        {
            for (int t = 0; t < seq; t++)
            {
                bool real = t < rows[r].Length;
                ids[(r * seq) + t] = real ? rows[r][t] : 1;
                mask[(r * seq) + t] = real ? 1 : 0;
            }
        }

        return (MemoryMarshal.AsBytes(ids.AsSpan()).ToArray(), MemoryMarshal.AsBytes(mask.AsSpan()).ToArray(), seq);
    }

    private static long[][] ShortBatch(List<JsonElement> rows, int size) =>
        [.. rows.Where(r => r.GetProperty("input_ids").GetArrayLength() < 512).Take(size)
            .Select(r => r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray())];

    /// <summary>The library's algorithm (E5Encoder.MeanPoolAndNormalize): double accumulation, one rounding.</summary>
    private static void MeanPoolAndNormalize(ReadOnlySpan<float> hidden, ReadOnlySpan<long> mask, Span<float> destination)
    {
        Span<double> sum = stackalloc double[Dimension];
        sum.Clear();
        double count = 0;
        for (int t = 0; t < mask.Length; t++)
        {
            if (mask[t] == 0)
            {
                continue;
            }

            count++;
            ReadOnlySpan<float> row = hidden.Slice(t * Dimension, Dimension);
            for (int d = 0; d < Dimension; d++)
            {
                sum[d] += row[d];
            }
        }

        count = Math.Max(count, 1e-9);
        double norm = 0;
        for (int d = 0; d < Dimension; d++)
        {
            sum[d] /= count;
            norm += sum[d] * sum[d];
        }

        norm = Math.Max(Math.Sqrt(norm), 1e-12);
        for (int d = 0; d < Dimension; d++)
        {
            destination[d] = (float)(sum[d] / norm);
        }
    }

    private static void WriteFloats(Utf8JsonWriter json, string name, float[] values)
    {
        json.WriteStartArray(name);
        foreach (float v in values)
        {
            json.WriteNumberValue(v);
        }

        json.WriteEndArray();
    }

    private static void WriteSamples(Utf8JsonWriter json, string name, List<double> samples)
    {
        json.WriteStartArray(name);
        foreach (double v in samples)
        {
            json.WriteNumberValue(Math.Round(v, 3));
        }

        json.WriteEndArray();
    }

    private static string RuntimeLabel() => "browser-wasm (" + RuntimeInformation.FrameworkDescription + ")";

    private sealed class HttpSource(HttpClient http) : ISpikeSource
    {
        public Task<byte[]> ReadAssetAsync(string path) => http.GetByteArrayAsync("data/tok/" + path);

        public Task<byte[]> ReadFixtureAsync(string path) => http.GetByteArrayAsync("data/fixtures/" + path);
    }
}
