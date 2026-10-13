using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Tokenizers;

namespace Starlings.BrowserSpike;

/// <summary>A tokenizer profile of the spike: the pinned asset and the protected fixtures it must match exactly.</summary>
/// <param name="Name">Profile name (profiles/ in starlinghq/tokenizers).</param>
/// <param name="AssetPath">Path under the model directory (<c>$STARLING_MODELS</c> layout).</param>
/// <param name="Sha256">Pinned SHA-256 of the asset.</param>
/// <param name="FixturePath">Path of <c>tokenizer.jsonl</c> under <c>model-pipeline/protected/</c>.</param>
/// <param name="BatchPath">Path of <c>tokenizer-batch.json</c> under <c>model-pipeline/protected/</c>, if any.</param>
/// <param name="EmbeddingInputsPath">Path of <c>embeddings.jsonl</c> whose <c>input_ids</c> the 512-truncated encode must equal, if any.</param>
public sealed record TokenizerProfile(
    string Name,
    string AssetPath,
    string Sha256,
    string FixturePath,
    string? BatchPath,
    string? EmbeddingInputsPath)
{
    /// <summary>The three profiles of the spike: Unigram (e5), byte-level BPE (Qwen chat), Gemma BPE with byte fallback.</summary>
    public static readonly TokenizerProfile[] All =
    [
        new(
            "xlmr-unigram-e5",
            "intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json",
            "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
            "multilingual-e5-small/onnx-fp32-cpu/tokenizer.jsonl",
            "multilingual-e5-small/onnx-fp32-cpu/tokenizer-batch.json",
            "multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl"),
        new(
            "qwen35-bpe-chat",
            "Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/transformers-5.18.0/tokenizer.json",
            "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
            "qwen-bpe/chat/tokenizer/tokenizer.jsonl",
            null,
            null),
        new(
            "gemma-bpe-embeddinggemma2",
            "google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json",
            "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
            "embeddinggemma-2/onnx-fp32-cpu/tokenizer.jsonl",
            "embeddinggemma-2/onnx-fp32-cpu/tokenizer-batch.json",
            null),
    ];
}

/// <summary>Where the suite reads bytes from: files natively, HTTP in the browser.</summary>
public interface ISpikeSource
{
    /// <summary>Reads a pinned model asset by its <c>$STARLING_MODELS</c>-relative path.</summary>
    Task<byte[]> ReadAssetAsync(string path);

    /// <summary>Reads a protected fixture by its <c>model-pipeline/protected</c>-relative path.</summary>
    Task<byte[]> ReadFixtureAsync(string path);
}

/// <summary>
/// The same conformance and timing code natively and in the browser: exact comparison of every recorded field of the
/// protected fixtures (ids, tokens, type ids, masks, sequence ids, offsets in UTF-16, scalar and UTF-8 units, decode
/// with and without special tokens, truncation and padded batch where recorded), then load and encode timings.
/// Results are written as JSON with <see cref="Utf8JsonWriter"/> (no reflection serialization, trimming-safe).
/// </summary>
public static class TokenizerSuite
{
    /// <summary>Runs every profile and returns one JSON document.</summary>
    public static async Task<string> RunAsync(ISpikeSource source, string runtime, double encodeSeconds, int warmLoads)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            json.WriteStartObject();
            json.WriteString("runtime", runtime);
            json.WriteString("framework", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
            json.WriteString("os", System.Runtime.InteropServices.RuntimeInformation.OSDescription);
            json.WriteString("architecture", System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString());
            json.WriteStartArray("profiles");
            foreach (TokenizerProfile profile in TokenizerProfile.All)
            {
                await RunProfileAsync(source, profile, json, encodeSeconds, warmLoads).ConfigureAwait(false);
                await json.FlushAsync().ConfigureAwait(false);
            }

            json.WriteEndArray();
            json.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static async Task RunProfileAsync(ISpikeSource source, TokenizerProfile profile, Utf8JsonWriter json, double encodeSeconds, int warmLoads)
    {
        json.WriteStartObject();
        json.WriteString("profile", profile.Name);

        var sw = Stopwatch.StartNew();
        byte[] asset = await source.ReadAssetAsync(profile.AssetPath).ConfigureAwait(false);
        json.WriteNumber("asset_read_ms", sw.Elapsed.TotalMilliseconds);
        json.WriteNumber("asset_bytes", asset.Length);
        string sha = Convert.ToHexStringLower(SHA256.HashData(asset));
        json.WriteBoolean("asset_sha256_ok", sha == profile.Sha256);

        sw.Restart();
        Tokenizer tokenizer = Tokenizer.FromUtf8Json(asset);
        json.WriteNumber("first_load_ms", sw.Elapsed.TotalMilliseconds);
        var warm = new List<double>();
        for (int i = 0; i < warmLoads; i++)
        {
            sw.Restart();
            _ = Tokenizer.FromUtf8Json(asset);
            warm.Add(sw.Elapsed.TotalMilliseconds);
        }

        WriteNumbers(json, "warm_load_ms", warm);

        List<JsonElement> rows = ReadJsonLines(await source.ReadFixtureAsync(profile.FixturePath).ConfigureAwait(false));
        var mismatches = new List<string>();
        int fieldChecks = 0;
        foreach (JsonElement row in rows)
        {
            fieldChecks += CheckCase(tokenizer, row, mismatches);
        }

        if (profile.BatchPath is not null)
        {
            fieldChecks += CheckBatch(tokenizer, rows, await source.ReadFixtureAsync(profile.BatchPath).ConfigureAwait(false), mismatches);
        }

        int embeddingInputCases = 0;
        if (profile.EmbeddingInputsPath is not null)
        {
            Tokenizer truncating = tokenizer.WithTruncation(new TruncationOptions(512));
            Dictionary<string, string> texts = rows.ToDictionary(r => r.GetProperty("id").GetString()!, r => r.GetProperty("text").GetString()!);
            foreach (JsonElement e in ReadJsonLines(await source.ReadFixtureAsync(profile.EmbeddingInputsPath).ConfigureAwait(false)))
            {
                string id = e.GetProperty("id").GetString()!;
                embeddingInputCases++;
                fieldChecks++;
                if (!Ints(e.GetProperty("input_ids")).SequenceEqual(truncating.Encode(texts[id]).Ids))
                {
                    mismatches.Add(id + ": embedding input_ids");
                }
            }
        }

        json.WriteNumber("cases", rows.Count);
        json.WriteNumber("embedding_input_cases", embeddingInputCases);
        json.WriteNumber("field_checks", fieldChecks);
        json.WriteNumber("mismatches", mismatches.Count);
        json.WriteStartArray("mismatch_details");
        foreach (string m in mismatches.Take(20))
        {
            json.WriteStringValue(m);
        }

        json.WriteEndArray();

        // Encode throughput over the fixture texts (single-sequence cases), after a warm-up pass.
        string[] texts1 = [.. rows.Where(r => !r.TryGetProperty("pair", out _)).Select(r => r.GetProperty("text").GetString()!)];
        long chars = texts1.Sum(t => (long)t.Length);
        foreach (string t in texts1)
        {
            _ = tokenizer.Encode(t);
        }

        long encodes = 0;
        long tokens = 0;
        sw.Restart();
        while (sw.Elapsed.TotalSeconds < encodeSeconds)
        {
            foreach (string t in texts1)
            {
                tokens += tokenizer.Encode(t).Count;
            }

            encodes += texts1.Length;
        }

        double seconds = sw.Elapsed.TotalSeconds;
        json.WriteStartObject("encode");
        json.WriteNumber("texts_per_pass", texts1.Length);
        json.WriteNumber("chars_per_pass", chars);
        json.WriteNumber("passes", encodes / texts1.Length);
        json.WriteNumber("seconds", seconds);
        json.WriteNumber("texts_per_s", encodes / seconds);
        json.WriteNumber("tokens_per_s", tokens / seconds);
        json.WriteNumber("chars_per_s", encodes / texts1.Length * chars / seconds);
        json.WriteNumber("us_per_text", seconds * 1e6 / encodes);
        json.WriteEndObject();

        json.WriteEndObject();
    }

    private static int CheckCase(Tokenizer tokenizer, JsonElement row, List<string> mismatches)
    {
        string id = row.GetProperty("id").GetString()!;
        string text = row.GetProperty("text").GetString()!;
        string? pair = row.TryGetProperty("pair", out JsonElement p) ? p.GetString() : null;
        TokenEncoding actual = pair is null ? tokenizer.Encode(text) : tokenizer.Encode(text, pair);
        int checks = CheckEncoding(row.GetProperty("encoding"), actual, text, pair, id, mismatches);

        JsonElement decode = row.GetProperty("decode");
        int[] ids = Ints(row.GetProperty("encoding").GetProperty("ids"));
        Check(decode.GetProperty("skip_special_tokens").GetString() == tokenizer.Decode(ids, skipSpecialTokens: true), id, "decode(skip)", mismatches);
        Check(decode.GetProperty("keep_special_tokens").GetString() == tokenizer.Decode(ids, skipSpecialTokens: false), id, "decode(keep)", mismatches);
        checks += 2;

        if (row.TryGetProperty("truncated", out JsonElement truncated))
        {
            Tokenizer t = tokenizer.WithTruncation(new TruncationOptions(truncated.GetProperty("max_length").GetInt32()));
            TokenEncoding te = pair is null ? t.Encode(text) : t.Encode(text, pair);
            checks += CheckEncoding(truncated.GetProperty("encoding"), te, text, pair, id + "/truncated", mismatches);
        }

        return checks;
    }

    private static int CheckEncoding(JsonElement expected, TokenEncoding actual, string text, string? pair, string id, List<string> mismatches)
    {
        int checks = 0;
        void C(bool ok, string field)
        {
            checks++;
            Check(ok, id, field, mismatches);
        }

        C(Ints(expected.GetProperty("ids")).SequenceEqual(actual.Ids), "ids");
        C(expected.GetProperty("tokens").EnumerateArray().Select(e => e.GetString()!).SequenceEqual(actual.Tokens), "tokens");
        C(Ints(expected.GetProperty("type_ids")).SequenceEqual(actual.TypeIds), "type_ids");
        C(Ints(expected.GetProperty("attention_mask")).SequenceEqual(actual.AttentionMask), "attention_mask");
        C(Ints(expected.GetProperty("special_tokens_mask")).SequenceEqual(actual.SpecialTokensMask), "special_tokens_mask");
        JsonElement offsets = expected.GetProperty("offsets");
        C(Offsets(offsets.GetProperty("utf16-code-units")).SequenceEqual(actual.Offsets), "offsets(utf16)");
        C(Offsets(offsets.GetProperty("unicode-scalars")).SequenceEqual(actual.ScalarOffsets), "offsets(scalars)");
        if (offsets.TryGetProperty("utf8-bytes", out JsonElement utf8))
        {
            TokenOffset[] expectedBytes = Offsets(utf8);
            bool ok = expectedBytes.Length == actual.Count;
            for (int i = 0; ok && i < actual.Count; i++)
            {
                string src = actual.SequenceIds[i] == 1 ? pair! : text;
                ok = expectedBytes[i] == ScalarToUtf8(src, actual.ScalarOffsets[i]);
            }

            C(ok, "offsets(utf8)");
        }

        if (expected.TryGetProperty("sequence_ids", out JsonElement seq))
        {
            int?[] expectedSeq = [.. seq.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? (int?)null : e.GetInt32())];
            C(expectedSeq.SequenceEqual(actual.SequenceIds), "sequence_ids");
        }

        return checks;
    }

    private static int CheckBatch(Tokenizer tokenizer, List<JsonElement> rows, byte[] batchJson, List<string> mismatches)
    {
        using JsonDocument doc = JsonDocument.Parse(batchJson);
        JsonElement padding = doc.RootElement.GetProperty("padding");
        int padId = padding.GetProperty("pad_id").GetInt32();
        Tokenizer padded = tokenizer.WithPadding(new PaddingOptions(padId, tokenizer.IdToToken(padId)!));
        Dictionary<string, JsonElement> byId = rows.ToDictionary(r => r.GetProperty("id").GetString()!);
        JsonElement[] batchRows = [.. doc.RootElement.GetProperty("rows").EnumerateArray()];
        string[] texts = [.. batchRows.Select(r => byId[r.GetProperty("id").GetString()!].GetProperty("text").GetString()!)];
        IReadOnlyList<TokenEncoding> batch = padded.EncodeBatch(texts);
        int checks = 0;
        for (int i = 0; i < batchRows.Length; i++)
        {
            string id = "batch/" + batchRows[i].GetProperty("id").GetString();
            Check(Ints(batchRows[i].GetProperty("ids")).SequenceEqual(batch[i].Ids), id, "ids", mismatches);
            Check(Ints(batchRows[i].GetProperty("attention_mask")).SequenceEqual(batch[i].AttentionMask), id, "attention_mask", mismatches);
            checks += 2;
        }

        return checks;
    }

    private static void Check(bool ok, string id, string field, List<string> mismatches)
    {
        if (!ok)
        {
            mismatches.Add(id + ": " + field);
        }
    }

    private static TokenOffset ScalarToUtf8(string text, TokenOffset scalar)
    {
        int ToUtf8(int s)
        {
            int bytes = 0;
            int count = 0;
            foreach (Rune r in text.EnumerateRunes())
            {
                if (count == s)
                {
                    break;
                }

                bytes += r.Utf8SequenceLength;
                count++;
            }

            return bytes;
        }

        return new TokenOffset(ToUtf8(scalar.Start), ToUtf8(scalar.End));
    }

    /// <summary>Reads JSON Lines, splitting on LF only (corpus text contains U+0085 and U+2028).</summary>
    public static List<JsonElement> ReadJsonLines(byte[] utf8)
    {
        var rows = new List<JsonElement>();
        foreach (string line in Encoding.UTF8.GetString(utf8).Split('\n'))
        {
            if (line.Length > 0)
            {
                using JsonDocument doc = JsonDocument.Parse(line);
                rows.Add(doc.RootElement.Clone());
            }
        }

        return rows;
    }

    private static int[] Ints(JsonElement array) => [.. array.EnumerateArray().Select(e => e.GetInt32())];

    private static TokenOffset[] Offsets(JsonElement array) =>
        [.. array.EnumerateArray().Select(p => new TokenOffset(p[0].GetInt32(), p[1].GetInt32()))];

    private static void WriteNumbers(Utf8JsonWriter json, string name, IEnumerable<double> values)
    {
        json.WriteStartArray(name);
        foreach (double v in values)
        {
            json.WriteNumberValue(Math.Round(v, 3));
        }

        json.WriteEndArray();
    }

    /// <summary>Formats a number invariantly (for log lines).</summary>
    public static string F(double value) => value.ToString("0.###", CultureInfo.InvariantCulture);
}
