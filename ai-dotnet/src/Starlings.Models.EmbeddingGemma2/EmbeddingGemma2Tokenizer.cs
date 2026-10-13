using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// The owned tokenizer (<c>Starlings.Tokenizers</c>, profile <c>gemma-bpe-embeddinggemma2</c>) as an
/// <see cref="IEmbeddingGemma2Tokenizer"/>: the pinned <c>tokenizer.json</c>, <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>
/// added, no truncation, no padding (the encoder pads and rejects inputs above 8,192 tokens).
/// </summary>
/// <remarks>
/// <para>
/// The text is tokenized exactly as given; a prompt is part of the text (<see cref="EmbeddingGemma2TextEmbedder"/>
/// prepends it). Literal media placeholders in the text (<c>&lt;|image|&gt;</c>, <c>&lt;|audio|&gt;</c>,
/// <c>&lt;|video|&gt;</c> and the boundary tokens) tokenize to their special ids, which the encoder rejects: this
/// package embeds text only.
/// </para>
/// <para>Instances are immutable and safe for concurrent calls (the owned tokenizer is immutable).</para>
/// </remarks>
public sealed class EmbeddingGemma2Tokenizer : IEmbeddingGemma2Tokenizer
{
    private readonly Tokenizer _tokenizer;

    private EmbeddingGemma2Tokenizer(Tokenizer tokenizer)
    {
        _tokenizer = tokenizer.WithPadding(null).WithTruncation(null);
    }

    /// <summary>
    /// Loads <c>tokenizer.json</c> from <paramref name="path"/>. The file must match <paramref name="expectedSha256"/>
    /// (lower-case hex; default <see cref="EmbeddingGemma2ModelAssets.TokenizerSha256"/>, the pinned file) or
    /// <see cref="InvalidDataException"/> is thrown. Pass <see langword="null"/> to skip the check (another tokenizer
    /// file is not the qualified profile).
    /// </summary>
    public static EmbeddingGemma2Tokenizer FromFile(string path, string? expectedSha256 = EmbeddingGemma2ModelAssets.TokenizerSha256)
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

        return new EmbeddingGemma2Tokenizer(Tokenizer.FromUtf8Json(bytes));
    }

    /// <inheritdoc />
    public EmbeddingGemma2Input Tokenize(string text)
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

        return new EmbeddingGemma2Input(ids, mask);
    }
}
