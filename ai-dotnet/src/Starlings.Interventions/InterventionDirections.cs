using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Starlings.Interventions;

/// <summary>
/// The directions <c>Q</c> of one site: <c>rank</c> rows of <c>hidden</c> float32 values, row-major.
/// </summary>
/// <remarks>
/// File format (V1, <c>float32</c>): raw IEEE 754 binary32 values, little-endian, row-major <c>[rank, hidden]</c>, no
/// header, exactly <c>rank * hidden * 4</c> bytes. In NumPy: <c>q.astype("&lt;f4").tobytes()</c>. Rows should be
/// orthonormal; that is the profile author's job and is not enforced (see <see cref="MaxOrthonormalityError"/>).
/// </remarks>
public sealed class InterventionDirections
{
    private readonly float[] _values;

    private InterventionDirections(int rank, int hiddenSize, float[] values)
    {
        Rank = rank;
        HiddenSize = hiddenSize;
        _values = values;
    }

    /// <summary>Number of rows.</summary>
    public int Rank { get; }

    /// <summary>Length of each row.</summary>
    public int HiddenSize { get; }

    /// <summary>All values, row-major <c>[rank, hidden]</c>.</summary>
    public ReadOnlyMemory<float> Values => _values;

    /// <summary>Row <paramref name="index"/>.</summary>
    public ReadOnlySpan<float> Row(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Rank);
        return _values.AsSpan(index * HiddenSize, HiddenSize);
    }

    /// <summary>Creates directions from row-major values, checking the shape and that every value is finite.</summary>
    /// <exception cref="InterventionContractException">Wrong length or a non-finite value.</exception>
    public static InterventionDirections Create(int rank, int hiddenSize, ReadOnlySpan<float> values)
    {
        CheckShape(rank, hiddenSize);
        if (values.Length != (long)rank * hiddenSize)
        {
            throw new InterventionContractException($"{values.Length} values for directions of shape [{rank}, {hiddenSize}].");
        }

        float[] copy = values.ToArray();
        CheckFinite(copy, hiddenSize);
        return new InterventionDirections(rank, hiddenSize, copy);
    }

    /// <summary>Decodes a directions file's bytes for <paramref name="site"/> (size and finite values; no hash check).</summary>
    /// <exception cref="InterventionContractException">Wrong size or a non-finite value.</exception>
    public static InterventionDirections Decode(ReadOnlySpan<byte> bytes, InterventionSite site)
    {
        ArgumentNullException.ThrowIfNull(site);
        return Decode(bytes, site.Rank, site.HiddenSize);
    }

    /// <summary>Decodes raw little-endian float32 bytes of shape <c>[rank, hidden]</c>.</summary>
    /// <exception cref="InterventionContractException">Wrong size or a non-finite value.</exception>
    public static InterventionDirections Decode(ReadOnlySpan<byte> bytes, int rank, int hiddenSize)
    {
        CheckShape(rank, hiddenSize);
        long expected = (long)rank * hiddenSize * sizeof(float);
        if (bytes.Length != expected)
        {
            throw new InterventionContractException(
                $"directions file is {bytes.Length} bytes; shape [{rank}, {hiddenSize}] float32 needs {expected}.");
        }

        float[] values = new float[rank * hiddenSize];
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(values);
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(i * sizeof(float), sizeof(float)));
            }
        }

        CheckFinite(values, hiddenSize);
        return new InterventionDirections(rank, hiddenSize, values);
    }

    /// <summary>
    /// Reads <paramref name="path"/> as <paramref name="site"/>'s directions file, checking its size, its SHA-256
    /// against <see cref="InterventionSite.DirectionsSha256"/>, and that every value is finite.
    /// </summary>
    /// <exception cref="InterventionMismatchException">The SHA-256 differs.</exception>
    /// <exception cref="InterventionContractException">Wrong size or a non-finite value.</exception>
    public static InterventionDirections Load(string path, InterventionSite site)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(site);
        FileInfo info = new(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Directions file not found: {path}", path);
        }

        if (info.Length != site.DirectionsByteLength)
        {
            throw new InterventionContractException(
                $"{path} is {info.Length} bytes; shape [{site.Rank}, {site.HiddenSize}] float32 needs {site.DirectionsByteLength}.");
        }

        byte[] bytes = File.ReadAllBytes(path);
        string sha = InterventionHashes.ComputeSha256(bytes);
        if (!string.Equals(sha, site.DirectionsSha256, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"{path}: SHA-256 {sha}, the profile records {site.DirectionsSha256}.");
        }

        return Decode(bytes, site);
    }

    /// <summary>The file bytes of these directions (raw little-endian float32, row-major, no header).</summary>
    public byte[] Encode()
    {
        byte[] bytes = new byte[_values.Length * sizeof(float)];
        for (int i = 0; i < _values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), _values[i]);
        }

        return bytes;
    }

    /// <summary>
    /// Diagnostic: the largest absolute deviation of the Gram matrix <c>Q Q^T</c> from the identity (accumulated in
    /// double). Near zero for orthonormal rows. Informational only; nothing in the contract enforces it.
    /// </summary>
    public double MaxOrthonormalityError()
    {
        double worst = 0;
        for (int i = 0; i < Rank; i++)
        {
            ReadOnlySpan<float> a = Row(i);
            for (int j = i; j < Rank; j++)
            {
                ReadOnlySpan<float> b = Row(j);
                double dot = 0;
                for (int k = 0; k < HiddenSize; k++)
                {
                    dot += (double)a[k] * b[k];
                }

                worst = Math.Max(worst, Math.Abs(dot - (i == j ? 1.0 : 0.0)));
            }
        }

        return worst;
    }

    private static void CheckShape(int rank, int hiddenSize)
    {
        if (hiddenSize is < 1 or > InterventionSite.MaxHiddenSize || rank < 1 || rank > hiddenSize)
        {
            throw new InterventionContractException($"invalid directions shape [{rank}, {hiddenSize}].");
        }
    }

    private static void CheckFinite(float[] values, int hiddenSize)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                throw new InterventionContractException($"directions value [{i / hiddenSize}, {i % hiddenSize}] is not finite ({values[i]}).");
            }
        }
    }
}
