using System.Runtime.InteropServices;
using System.Runtime.InteropServices.JavaScript;
using System.Text.Json;
using Starlings.Models.E5;
using Starlings.Models.E5.Browser;
using Starlings.Runtime.OnnxWeb;
using Starlings.Runtime.OnnxWeb.WebGpu;

namespace E5Wasm;

/// <summary>
/// The JavaScript API of the bundle. dotnet.js exposes these as
/// <c>(await runtime.getAssemblyExports("E5Wasm.dll")).E5Wasm.E5Exports.*</c>; ../web/public/e5/e5-worker.js wraps them
/// in a promise API. Vectors cross to JavaScript as <c>Float32Array</c>s: a [JSExport] cannot return an array from an
/// async method, so the result is copied into a JavaScript-owned array through a memory view (<see cref="Float32"/>).
/// </summary>
public static partial class E5Exports
{
    private static E5BrowserEmbedder? _embedder;

    /// <summary>
    /// Loads the model once: <paramref name="variant"/> <c>fp32</c> (from the Hugging Face Hub when
    /// <paramref name="modelUrl"/> is empty) or <c>int8embed</c> (self-hosted at <paramref name="modelUrl"/>), on
    /// <paramref name="provider"/> <c>wasm</c> or <c>webgpu</c>. Reports progress as JSON
    /// (<c>{"stage","received","total"}</c>) and returns the runtime as JSON.
    /// </summary>
    [JSExport]
    public static async Task<string> Load(
        string variant,
        string? modelUrl,
        string provider,
        [JSMarshalAs<JSType.Function<JSType.String>>] Action<string> onProgress)
    {
        if (_embedder is not null)
        {
            return Describe();
        }

        var graph = variant switch
        {
            "fp32" => E5BrowserVariant.Fp32,
            "int8embed" => E5BrowserVariant.Int8Embed,
            _ => throw new ArgumentException($"variant '{variant}' is not 'fp32' or 'int8embed'.", nameof(variant)),
        };
        bool webGpu = provider switch
        {
            "wasm" => false,
            "webgpu" => true,
            _ => throw new ArgumentException($"provider '{provider}' is not 'wasm' or 'webgpu'.", nameof(provider)),
        };
        _embedder = await E5BrowserEmbedder.LoadAsync(
            new E5BrowserLoadOptions
            {
                Variant = graph,
                ModelUrl = string.IsNullOrEmpty(modelUrl) ? null : modelUrl,
                Provider = webGpu ? OnnxWebExecutionProvider.WebGpu : OnnxWebExecutionProvider.Wasm,
                // One onnxruntime-web bundle per worker; the WASM-only one is the smaller download.
                Runtime = OnnxWebRuntime.Info is not null
                    ? null
                    : new OnnxWebRuntimeOptions { Bundle = webGpu ? OnnxWebGpuBundle.WebGpu : OnnxWebBundle.Wasm },
            },
            new Progress(onProgress));
        return Describe();
    }

    /// <summary>Embeds one text with the e5 prefix of <paramref name="inputType"/> (<c>query</c>, <c>passage</c> or <c>none</c>).</summary>
    [JSExport]
    public static async Task<JSObject> Embed(string text, string inputType)
    {
        float[] vector = await Embedder.EmbedAsync(Prefix(inputType) + text);
        return Float32(vector);
    }

    /// <summary>
    /// Embeds texts in one call (padded batches of up to 32). Returns one <c>Float32Array</c> of
    /// <c>texts.length x dimension</c> values, row-major.
    /// </summary>
    [JSExport]
    public static async Task<JSObject> EmbedBatch([JSMarshalAs<JSType.Array<JSType.String>>] string[] texts, string inputType)
    {
        string prefix = Prefix(inputType);
        float[][] vectors = await Embedder.EmbedBatchAsync([.. texts.Select(t => prefix + t)]);
        return Float32([.. vectors.SelectMany(v => v)]);
    }

    /// <summary>Releases the model (the next <see cref="Load"/> loads it again).</summary>
    [JSExport]
    public static void Unload()
    {
        _embedder?.Dispose();
        _embedder = null;
    }

    private static E5BrowserEmbedder Embedder => _embedder ?? throw new InvalidOperationException("Call load() first.");

    private static string Prefix(string inputType) => inputType switch
    {
        "query" => E5TextEmbedder.QueryPrefix,
        "passage" => E5TextEmbedder.PassagePrefix,
        "none" => string.Empty,
        // A misspelt input type would silently drop e5's prefix and degrade retrieval: reject it.
        _ => throw new ArgumentException($"inputType '{inputType}' is not 'query', 'passage' or 'none'.", nameof(inputType)),
    };

    private static string Describe()
    {
        var info = OnnxWebRuntime.Info!;
        return JsonSerializer.Serialize(
            new RuntimeDescription(_embedder!.Profile, _embedder.Encoder.Dimension, info.OrtVersion, info.Bundle, info.NumThreads, info.HardwareConcurrency, info.CrossOriginIsolated),
            E5Json.Default.RuntimeDescription);
    }

    private static JSObject Float32(float[] values) => MakeFloat32(MemoryMarshal.AsBytes(values.AsSpan()));

    /// <summary>Defined by the worker (<c>globalThis.e5interop.makeFloat32</c>): copies the bytes into a new Float32Array.</summary>
    [JSImport("globalThis.e5interop.makeFloat32")]
    private static partial JSObject MakeFloat32([JSMarshalAs<JSType.MemoryView>] Span<byte> bytes);

    private sealed class Progress(Action<string> report) : IProgress<E5BrowserLoadProgress>
    {
        public void Report(E5BrowserLoadProgress value) =>
            report(JsonSerializer.Serialize(new ProgressReport(value.Stage.ToString(), value.BytesReceived, value.TotalBytes), E5Json.Default.ProgressReport));
    }
}

/// <summary>What <see cref="E5Exports.Load"/> returns.</summary>
public sealed record RuntimeDescription(string? Profile, int Dimension, string? OrtVersion, string Bundle, int Threads, int Cores, bool CrossOriginIsolated);

/// <summary>One load progress report.</summary>
public sealed record ProgressReport(string Stage, long Received, long? Total);

[System.Text.Json.Serialization.JsonSourceGenerationOptions(PropertyNamingPolicy = System.Text.Json.Serialization.JsonKnownNamingPolicy.CamelCase)]
[System.Text.Json.Serialization.JsonSerializable(typeof(RuntimeDescription))]
[System.Text.Json.Serialization.JsonSerializable(typeof(ProgressReport))]
internal sealed partial class E5Json : System.Text.Json.Serialization.JsonSerializerContext;
