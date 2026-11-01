using System.Security.Cryptography;

namespace Starlings.Interventions;

/// <summary>
/// SHA-256 identities as the intervention contract spells them: 64 lower-case hexadecimal characters over the exact
/// bytes of a file.
/// </summary>
public static class InterventionHashes
{
    /// <summary>Whether <paramref name="value"/> is a SHA-256 in the contract's spelling (64 lower-case hex characters).</summary>
    public static bool IsSha256(string? value)
    {
        if (value is null || value.Length != 64)
        {
            return false;
        }

        foreach (char c in value)
        {
            if (!char.IsAsciiDigit(c) && c is not (>= 'a' and <= 'f'))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>The SHA-256 of <paramref name="data"/>, lower-case hex.</summary>
    public static string ComputeSha256(ReadOnlySpan<byte> data) => Convert.ToHexStringLower(SHA256.HashData(data));

    /// <summary>The SHA-256 of the file at <paramref name="path"/>, lower-case hex, read as a stream.</summary>
    public static string ComputeFileSha256(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20, FileOptions.SequentialScan);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    /// <summary>
    /// Checks that the file at <paramref name="path"/> has <paramref name="expectedSha256"/> (and, when given, the
    /// expected size, which is checked first because it is cheap).
    /// </summary>
    /// <exception cref="InterventionMismatchException">The size or the SHA-256 differs.</exception>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public static void VerifyFile(string path, string expectedSha256, long? expectedSize = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        RequireSha256(expectedSha256, nameof(expectedSha256));
        FileInfo info = new(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Intervention artifact not found: {path}", path);
        }

        if (expectedSize is long size && info.Length != size)
        {
            throw new InterventionMismatchException($"{path}: size {info.Length} bytes, expected {size}.");
        }

        string actual = ComputeFileSha256(path);
        if (!string.Equals(actual, expectedSha256, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"{path}: SHA-256 {actual}, expected {expectedSha256}.");
        }
    }

    internal static void RequireSha256(string value, string paramName)
    {
        ArgumentNullException.ThrowIfNull(value, paramName);
        if (!IsSha256(value))
        {
            throw new ArgumentException("Expected a SHA-256 as 64 lower-case hexadecimal characters.", paramName);
        }
    }
}
