using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace Starlings.Models.Kev;

/// <summary>
/// Kev's pointer head (<c>kev/model.py</c> <c>PointerHead</c>): <c>q, k = Linear(d, 256)</c> with bias; option logits
/// <c>(k(h_opts) @ q(h_decide)) / sqrt(256) / T</c>; softmax over the question's options. Weights from
/// <c>head.safetensors</c>, temperature and dimensions from <c>head.json</c> (both written by
/// <c>model-pipeline/reference/kev/kev_merge.py</c>), checked against the profile. Arithmetic is in double.
/// </summary>
internal sealed class KevPointerHead
{
    private readonly float[] _qWeight;
    private readonly float[] _qBias;
    private readonly float[] _kWeight;
    private readonly float[] _kBias;
    private readonly int _hidden;
    private readonly double _scale;

    private KevPointerHead(float[] qWeight, float[] qBias, float[] kWeight, float[] kBias, int hidden, double temperature)
    {
        (_qWeight, _qBias, _kWeight, _kBias, _hidden) = (qWeight, qBias, kWeight, kBias, hidden);
        Temperature = temperature;
        _scale = 1.0 / Math.Sqrt(KevModelProfile.HeadDim) / temperature;
    }

    /// <summary>The calibration temperature in effect.</summary>
    public double Temperature { get; }

    public static KevPointerHead Load(KevModelProfile profile, string headPath, string metadataPath, double? temperatureOverride = null)
    {
        using (var meta = JsonDocument.Parse(File.ReadAllText(metadataPath)))
        {
            var root = meta.RootElement;
            Check(root.GetProperty("hidden_size").GetInt32() == profile.HiddenSize, "hidden_size");
            Check(root.GetProperty("head_dim").GetInt32() == KevModelProfile.HeadDim, "head_dim");
            Check(root.GetProperty("temperature").GetDouble() == profile.Temperature, "temperature");
            Check(root.GetProperty("head_pt_sha256").GetString() == profile.HeadPtSha256, "head_pt_sha256");
            Check(root.GetProperty("adapter_sha256").GetString() == profile.AdapterSha256, "adapter_sha256");
            Check(root.GetProperty("base_revision").GetString() == profile.BaseRevision, "base_revision");
        }

        var tensors = ReadSafetensors(headPath);
        float[] Tensor(string name, params int[] shape)
        {
            if (!tensors.TryGetValue(name, out var t) || !t.Shape.SequenceEqual(shape))
            {
                throw new InvalidDataException($"head.safetensors: {name} is missing or not [{string.Join(", ", shape)}].");
            }

            return t.Data;
        }

        double temperature = temperatureOverride ?? profile.Temperature;
        if (!(temperature > 0) || double.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperatureOverride), temperature, "The temperature must be positive and finite.");
        }

        int d = profile.HiddenSize, p = KevModelProfile.HeadDim;
        return new KevPointerHead(Tensor("q.weight", p, d), Tensor("q.bias", p), Tensor("k.weight", p, d), Tensor("k.bias", p), d, temperature);

        void Check(bool ok, string field)
        {
            if (!ok)
            {
                throw new InvalidDataException($"head.json: {field} does not match the {profile.Id} contract.");
            }
        }
    }

    /// <summary>Calibrated option probabilities from the <c>&lt;decide&gt;</c> hidden state and each option's <c>&lt;/opt&gt;</c> hidden state.</summary>
    public float[] Probabilities(ReadOnlySpan<float> decide, IReadOnlyList<float[]> options)
    {
        double[] q = Project(_qWeight, _qBias, decide);
        var logits = new double[options.Count];
        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            double[] k = Project(_kWeight, _kBias, options[i]);
            double s = 0;
            for (int j = 0; j < k.Length; j++)
            {
                s += k[j] * q[j];
            }

            logits[i] = s * _scale;
            max = Math.Max(max, logits[i]);
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = Math.Exp(logits[i] - max);
            sum += logits[i];
        }

        var probs = new float[logits.Length];
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] = (float)(logits[i] / sum);
        }

        return probs;
    }

    private double[] Project(float[] w, float[] b, ReadOnlySpan<float> x)
    {
        if (x.Length != _hidden)
        {
            throw new ArgumentException($"Hidden state has {x.Length} values; expected {_hidden}.", nameof(x));
        }

        var y = new double[b.Length];
        for (int i = 0; i < y.Length; i++)
        {
            var row = w.AsSpan(i * _hidden, _hidden);
            double s = b[i];
            for (int j = 0; j < row.Length; j++)
            {
                s += (double)row[j] * x[j];
            }

            y[i] = s;
        }

        return y;
    }

    /// <summary>Minimal safetensors reader: 8-byte little-endian header length, JSON header, raw little-endian F32 tensors.</summary>
    internal static Dictionary<string, (int[] Shape, float[] Data)> ReadSafetensors(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length < 8)
        {
            throw new InvalidDataException("head.safetensors is truncated.");
        }

        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(bytes);
        if (headerLength <= 0 || 8 + headerLength > bytes.Length)
        {
            throw new InvalidDataException("head.safetensors has an invalid header length.");
        }

        int dataStart = checked(8 + (int)headerLength);
        using var header = JsonDocument.Parse(Encoding.UTF8.GetString(bytes, 8, (int)headerLength));
        var result = new Dictionary<string, (int[], float[])>(StringComparer.Ordinal);
        foreach (var entry in header.RootElement.EnumerateObject())
        {
            if (entry.Name == "__metadata__")
            {
                continue;
            }

            if (entry.Value.GetProperty("dtype").GetString() != "F32")
            {
                throw new InvalidDataException($"head.safetensors: {entry.Name} is not F32.");
            }

            int[] shape = [.. entry.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
            var offsets = entry.Value.GetProperty("data_offsets");
            long start = offsets[0].GetInt64(), end = offsets[1].GetInt64();
            long count = shape.Aggregate(1L, (a, n) => a * n);
            if (start < 0 || end < start || end - start != count * sizeof(float) || dataStart + end > bytes.Length)
            {
                throw new InvalidDataException($"head.safetensors: {entry.Name} has invalid offsets.");
            }

            var data = new float[count];
            for (long i = 0; i < count; i++)
            {
                data[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan((int)(dataStart + start + i * 4), 4));
            }

            result[entry.Name] = (shape, data);
        }

        return result;
    }
}
