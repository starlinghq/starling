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

    /// <summary>
    /// Decodes token ids (for example <see cref="WhisperModel.Generate"/> output) to text, skipping special tokens and
    /// timestamp tokens (ids from <see cref="WhisperModel.TimestampBegin"/>). <c>tokenizer.json</c> does not mark the
    /// timestamp tokens special; transformers <c>WhisperTokenizer.decode</c> removes them from the text, and so does this.
    /// </summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var kept = new List<int>(ids.Count);
        foreach (int id in ids)
        {
            if (id < WhisperModel.TimestampBegin)
            {
                kept.Add(id);
            }
        }

        return _tokenizer.Decode(kept.ToArray(), skipSpecialTokens: true);
    }
}
