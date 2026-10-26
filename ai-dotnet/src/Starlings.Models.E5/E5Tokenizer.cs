using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.E5;

/// <summary>
/// The owned tokenizer (<c>Starlings.Tokenizers</c>, profile <c>xlmr-unigram-e5</c>) as an
/// <see cref="IEncoderTokenizer"/>: the pinned <c>tokenizer.json</c>, special tokens added, right truncation to
/// <see cref="MaxLength"/> (default <see cref="E5Encoder.MaxSupportedSequenceLength"/>, 512), no padding. Text that
/// tokenizes to more than <see cref="MaxLength"/> tokens is cut on the right to <see cref="MaxLength"/> tokens, special
/// tokens included (<c>&lt;s&gt;</c> first, <c>&lt;/s&gt;</c> last), as the reference's <c>truncate-right</c> does; it is
/// not rejected. Give the encoder the same length so it accepts every output.
/// </summary>
/// <remarks>
/// Instances are immutable after construction. Concurrent <see cref="Tokenize"/> calls rely on the owned tokenizer's
/// immutability; thread safety is untested in the tokenizer library itself.
/// </remarks>
public sealed class E5Tokenizer : IEncoderTokenizer
{
    private readonly Tokenizer _tokenizer;

    private E5Tokenizer(Tokenizer tokenizer, int maxLength)
    {
        _tokenizer = tokenizer.WithPadding(null).WithTruncation(new TruncationOptions(maxLength));
        MaxLength = maxLength;
    }

    /// <summary>Maximum sequence length including <c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c>; longer text is truncated on the right to it.</summary>
    public int MaxLength { get; }

    /// <summary>
    /// Loads <c>tokenizer.json</c> from <paramref name="path"/>. When <paramref name="expectedSha256"/> is given, the
    /// file must match it (lower-case hex) or <see cref="InvalidDataException"/> is thrown. <paramref name="maxLength"/>
    /// is the right-truncation length (default <see cref="E5Encoder.MaxSupportedSequenceLength"/>).
    /// </summary>
    public static E5Tokenizer FromFile(string path, string? expectedSha256 = null, int maxLength = E5Encoder.MaxSupportedSequenceLength)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxLength, 2);
        return Load(File.ReadAllBytes(path), path, expectedSha256, maxLength);
    }

    /// <summary>
    /// Loads <c>tokenizer.json</c> from its UTF-8 bytes (for example fetched in a browser, where there is no file
    /// system). When <paramref name="expectedSha256"/> is given, the bytes must match it (lower-case hex) or
    /// <see cref="InvalidDataException"/> is thrown.
    /// </summary>
    public static E5Tokenizer FromUtf8Json(byte[] json, string? expectedSha256 = null, int maxLength = E5Encoder.MaxSupportedSequenceLength)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxLength, 2);
        return Load(json, "tokenizer.json", expectedSha256, maxLength);
    }

    private static E5Tokenizer Load(byte[] bytes, string source, string? expectedSha256, int maxLength)
    {
        if (expectedSha256 is not null)
        {
            string actual = Convert.ToHexStringLower(SHA256.HashData(bytes));
            if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"{source}: SHA-256 {actual} does not match the expected {expectedSha256}.");
            }
        }

        return new E5Tokenizer(Tokenizer.FromUtf8Json(bytes), maxLength);
    }

    /// <inheritdoc />
    public EncoderInput Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var encoding = _tokenizer.Encode(text);
        var ids = new long[encoding.Count];
        var mask = new long[encoding.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = encoding.Ids[i];
            mask[i] = encoding.AttentionMask[i];
        }

        return new EncoderInput(ids, mask);
    }
}
