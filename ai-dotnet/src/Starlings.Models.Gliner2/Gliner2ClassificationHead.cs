using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;

namespace Starlings.Models.Gliner2;

/// <summary>
/// GLiNER2's classification head (<c>classifier</c>: <c>Linear(768, 1536)</c>, ReLU, <c>Linear(1536, 1)</c>), read from
/// the checkpoint's <c>model.safetensors</c> (<see cref="Gliner2ModelAssets.ClassificationWeightsFile"/>). Experimental
/// (<see cref="Gliner2Capabilities"/>, <c>classification</c>).
/// </summary>
/// <remarks>
/// The locked ONNX graph does not contain this head; it exposes the backbone's <c>last_hidden_state</c>, and the head
/// maps the hidden state of each <c>[L]</c> label marker to one logit, as GLiNER2 2.0.0 does
/// (<c>inference/runtime.py</c>, <c>_extract_classification_result</c>). Only the four <c>classifier.*</c> tensors are
/// read (about 4.7 MB of the 834 MB file); their bytes are checked against <see cref="HeadSha256"/>. Immutable and safe
/// to use from several threads.
/// </remarks>
public sealed class Gliner2ClassificationHead
{
    /// <summary>Hidden size of the backbone.</summary>
    public const int HiddenSize = 768;

    /// <summary>Width of the head's intermediate layer.</summary>
    public const int IntermediateSize = 2 * HiddenSize;

    /// <summary>
    /// SHA-256 (lower-case hex) of the raw little-endian float32 bytes of <c>classifier.0.weight</c>,
    /// <c>classifier.0.bias</c>, <c>classifier.2.weight</c> and <c>classifier.2.bias</c>, concatenated in that order, in
    /// <c>fastino/gliner2-base-v1</c> at <see cref="Gliner2ModelAssets.Revision"/>.
    /// </summary>
    public const string HeadSha256 = "9a7a68ecacee7d7888050499542e2586588387745cc41f943e72a0ad3b3a0194";

    internal static readonly (string Name, int[] Shape)[] Tensors =
    [
        ("classifier.0.weight", [IntermediateSize, HiddenSize]),
        ("classifier.0.bias", [IntermediateSize]),
        ("classifier.2.weight", [1, IntermediateSize]),
        ("classifier.2.bias", [1]),
    ];

    private readonly float[] _w0;
    private readonly float[] _b0;
    private readonly float[] _w2;
    private readonly float _b2;

    private Gliner2ClassificationHead(float[] w0, float[] b0, float[] w2, float b2)
    {
        _w0 = w0;
        _b0 = b0;
        _w2 = w2;
        _b2 = b2;
    }

    /// <summary>Reads the head from <c>model.safetensors</c> and checks its bytes against <see cref="HeadSha256"/>.</summary>
    /// <exception cref="InvalidDataException">Not a safetensors file (including a malformed header), a tensor is missing or has another dtype or shape, or the bytes differ from the pinned head.</exception>
    public static Gliner2ClassificationHead FromSafetensors(string path) => FromSafetensors(path, HeadSha256);

    /// <summary>As <see cref="FromSafetensors(string)"/>; <paramref name="expectedSha256"/> null skips the digest (tests only).</summary>
    internal static Gliner2ClassificationHead FromSafetensors(string path, string? expectedSha256)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        long fileLength = RandomAccess.GetLength(handle);
        Span<byte> prefix = stackalloc byte[8];
        if (fileLength < 8 || RandomAccess.Read(handle, prefix, 0) != 8)
        {
            throw new InvalidDataException($"{path}: too short for a safetensors file.");
        }

        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerLength == 0 || headerLength > 100_000_000 || 8 + (long)headerLength > fileLength)
        {
            throw new InvalidDataException($"{path}: safetensors header length {headerLength} is invalid.");
        }

        var header = new byte[headerLength];
        ReadExactly(handle, header, 8, path);
        long dataStart = 8 + (long)headerLength;
        using var json = ParseHeader(header, path);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var arrays = new float[Tensors.Length][];
        for (int t = 0; t < Tensors.Length; t++)
        {
            var (name, shape) = Tensors[t];
            if (!json.RootElement.TryGetProperty(name, out var entry))
            {
                throw new InvalidDataException($"{path}: tensor '{name}' is missing; this is not a GLiNER2 checkpoint.");
            }

            var (dtype, actualShape, begin, end) = ReadEntry(entry, name, path);
            if (dtype != "F32" || !actualShape.AsSpan().SequenceEqual(shape))
            {
                throw new InvalidDataException(
                    $"{path}: tensor '{name}' is {dtype} [{string.Join(", ", actualShape)}]; expected F32 [{string.Join(", ", shape)}].");
            }

            int count = shape.Aggregate(1, (a, b) => a * b);
            if (end - begin != 4L * count || dataStart + end > fileLength || begin < 0)
            {
                throw new InvalidDataException($"{path}: tensor '{name}' has invalid data offsets [{begin}, {end}].");
            }

            var bytes = new byte[end - begin];
            ReadExactly(handle, bytes, dataStart + begin, path);
            hash.AppendData(bytes);
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(4 * i, 4));
            }

            arrays[t] = values;
        }

        string digest = Convert.ToHexStringLower(hash.GetHashAndReset());
        if (expectedSha256 is not null && !string.Equals(digest, expectedSha256, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{path}: classification head SHA-256 {digest} does not match the pinned {expectedSha256}.");
        }

        return new Gliner2ClassificationHead(arrays[0], arrays[1], arrays[2], arrays[3][0]);
    }

    /// <summary>One logit per row of <paramref name="markers"/> (row-major [rows, <see cref="HiddenSize"/>]).</summary>
    public float[] Logits(ReadOnlySpan<float> markers)
    {
        if (markers.Length % HiddenSize != 0)
        {
            throw new ArgumentException($"Length {markers.Length} is not a multiple of {HiddenSize}.", nameof(markers));
        }

        int rows = markers.Length / HiddenSize;
        var logits = new float[rows];
        var hidden = new float[IntermediateSize];
        for (int r = 0; r < rows; r++)
        {
            var x = markers.Slice(r * HiddenSize, HiddenSize);
            for (int j = 0; j < IntermediateSize; j++)
            {
                var w = _w0.AsSpan(j * HiddenSize, HiddenSize);
                double sum = 0;
                for (int k = 0; k < HiddenSize; k++)
                {
                    sum += (double)w[k] * x[k];
                }

                float h = (float)sum + _b0[j];
                hidden[j] = h > 0 ? h : 0;
            }

            double logit = 0;
            for (int j = 0; j < IntermediateSize; j++)
            {
                logit += (double)_w2[j] * hidden[j];
            }

            logits[r] = (float)logit + _b2;
        }

        return logits;
    }

    private static JsonDocument ParseHeader(byte[] header, string path)
    {
        JsonDocument json;
        try
        {
            json = JsonDocument.Parse(header);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"{path}: safetensors header is not valid JSON.", ex);
        }

        if (json.RootElement.ValueKind != JsonValueKind.Object)
        {
            json.Dispose();
            throw new InvalidDataException($"{path}: safetensors header is not a JSON object.");
        }

        return json;
    }

    /// <summary>Reads one header entry; any malformed field becomes <see cref="InvalidDataException"/>.</summary>
    private static (string? Dtype, int[] Shape, long Begin, long End) ReadEntry(JsonElement entry, string name, string path)
    {
        try
        {
            string? dtype = entry.GetProperty("dtype").GetString();
            int[] shape = entry.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var offsets = entry.GetProperty("data_offsets");
            if (offsets.GetArrayLength() != 2)
            {
                throw new InvalidDataException($"{path}: tensor '{name}' does not have two data offsets.");
            }

            return (dtype, shape, offsets[0].GetInt64(), offsets[1].GetInt64());
        }
        catch (Exception ex) when (ex is InvalidOperationException or KeyNotFoundException or FormatException or IndexOutOfRangeException)
        {
            throw new InvalidDataException($"{path}: tensor '{name}' has a malformed header entry.", ex);
        }
    }

    private static void ReadExactly(Microsoft.Win32.SafeHandles.SafeFileHandle handle, byte[] buffer, long offset, string path)
    {
        int done = 0;
        while (done < buffer.Length)
        {
            int n = RandomAccess.Read(handle, buffer.AsSpan(done), offset + done);
            if (n <= 0)
            {
                throw new InvalidDataException($"{path}: unexpected end of file.");
            }

            done += n;
        }
    }
}
