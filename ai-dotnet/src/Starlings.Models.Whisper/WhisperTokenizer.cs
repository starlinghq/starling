using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.Whisper;

/// <summary>
/// The owned tokenizer (<c>Starlings.Tokenizers</c>, profile <c>whisper-bpe</c>) used to decode the generated token ids
/// back to transcript text. The model's special tokens are fixed in <see cref="WhisperModel"/>, so this type is only
/// needed for decode.
/// </summary>
/// <remarks>Instances are immutable after construction. Thread safety of concurrent <see cref="Decode"/> relies on the
/// owned tokenizer's immutability; it is untested in the tokenizer library itself.</remarks>
public sealed class WhisperTokenizer
{
    private readonly Tokenizer _tokenizer;

    private WhisperTokenizer(Tokenizer tokenizer) => _tokenizer = tokenizer;

    /// <summary>
    /// Loads <c>tokenizer.json</c> from <paramref name="path"/>. When <paramref name="expectedSha256"/> is given, the
    /// file must match it (lower-case hex) or <see cref="InvalidDataException"/> is thrown.
    /// </summary>
    public static WhisperTokenizer FromFile(string path, string? expectedSha256 = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        byte[] bytes = File.ReadAllBytes(path);
        if (expectedSha256 is not null)
        {
            string actual = Convert.ToHexStringLower(SHA256.HashData(bytes));
            if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"{path}: SHA-256 {actual} does not match the expected {expectedSha256}.");
            }
        }

        return new WhisperTokenizer(Tokenizer.FromUtf8Json(bytes));
    }

    /// <summary>Decodes token ids (for example <see cref="WhisperModel.Generate"/> output) to text, skipping special tokens.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var span = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            span[i] = ids[i];
        }

        return _tokenizer.Decode(span, skipSpecialTokens: true);
    }
}
