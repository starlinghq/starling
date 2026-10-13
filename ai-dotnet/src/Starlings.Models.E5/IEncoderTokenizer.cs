namespace Starlings.Models.E5;

/// <summary>
/// Turns text into encoder input. This is where the owned tokenizer (<c>Starlings.Tokenizers</c>,
/// profile <c>xlmr-unigram-e5</c>) plugs in here through <see cref="E5Tokenizer"/>.
/// </summary>
/// <remarks>
/// An implementation must reproduce the pinned <c>tokenizer.json</c> exactly (ids and mask, G1): add
/// <c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c>, truncate on the right to at most 512 tokens including them, and return one
/// unpadded sequence. It must not add the e5 <c>"query: "</c> or <c>"passage: "</c> prefix; that is the caller's
/// decision (see <see cref="E5TextEmbedder"/>). Implementations must be safe for concurrent calls if the embedder is
/// shared across threads.
/// </remarks>
public interface IEncoderTokenizer
{
    /// <summary>Tokenizes one text. The returned memory must stay valid until the caller is done with it.</summary>
    EncoderInput Tokenize(string text);
}
