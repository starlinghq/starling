using System.Runtime.InteropServices;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// Turns a stream of generated token ids into text pieces with the owned tokenizer's decoder, never emitting half a
/// character: a byte-level token can carry part of a multi-byte UTF-8 sequence, and its text is held back until the
/// sequence completes. Concatenating every <see cref="Push"/> result and the final <see cref="Flush"/> equals
/// <see cref="Tokenizer.Decode"/> of all ids at once.
/// </summary>
/// <remarks>
/// Method (as in Hugging Face text-generation-inference): decode the ids since the last emitted piece together with the
/// piece before it, and emit the new suffix only when it is longer than before and does not end with U+FFFD (the
/// decoder's replacement for an incomplete UTF-8 sequence). A completion that really ends with invalid bytes is emitted
/// with its U+FFFD by <see cref="Flush"/>. Cost per step is proportional to the ids since the last emission. Not
/// thread-safe; one instance per request.
/// </remarks>
public sealed class IncrementalDetokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly bool _skipSpecialTokens;
    private readonly List<int> _ids = [];
    private int _prefixOffset;
    private int _readOffset;

    /// <summary>Creates a detokenizer.</summary>
    /// <param name="tokenizer">The owned tokenizer whose decoder produces the text.</param>
    /// <param name="skipSpecialTokens">Drop special added tokens, as <see cref="Tokenizer.Decode"/> does.</param>
    public IncrementalDetokenizer(Tokenizer tokenizer, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        _skipSpecialTokens = skipSpecialTokens;
    }

    /// <summary>Ids pushed so far.</summary>
    public int Count => _ids.Count;

    /// <summary>Adds one id and returns the text that became complete (possibly empty).</summary>
    public string Push(int tokenId)
    {
        _ids.Add(tokenId);
        ReadOnlySpan<int> ids = CollectionsMarshal.AsSpan(_ids);
        string prefixText = _prefixOffset == _readOffset ? string.Empty : _tokenizer.Decode(ids[_prefixOffset.._readOffset], _skipSpecialTokens);
        string newText = _tokenizer.Decode(ids[_prefixOffset..], _skipSpecialTokens);
        if (!newText.StartsWith(prefixText, StringComparison.Ordinal))
        {
            throw NotPrefixStable();
        }

        if (newText.Length > prefixText.Length && !newText.EndsWith('\uFFFD'))
        {
            _prefixOffset = _readOffset;
            _readOffset = _ids.Count;
            return newText[prefixText.Length..];
        }

        return string.Empty;
    }

    /// <summary>Returns any text still held back (an incomplete sequence decodes with U+FFFD) and resets the state.</summary>
    public string Flush()
    {
        ReadOnlySpan<int> ids = CollectionsMarshal.AsSpan(_ids);
        string prefixText = _prefixOffset == _readOffset ? string.Empty : _tokenizer.Decode(ids[_prefixOffset.._readOffset], _skipSpecialTokens);
        string text = _tokenizer.Decode(ids[_prefixOffset..], _skipSpecialTokens);
        if (!text.StartsWith(prefixText, StringComparison.Ordinal))
        {
            throw NotPrefixStable();
        }

        string rest = text[prefixText.Length..];
        _ids.Clear();
        _prefixOffset = 0;
        _readOffset = 0;
        return rest;
    }

    // Decoding more ids must extend the text of fewer ids. Byte-level decoding always does; a decoder that rewrites
    // earlier text (for example one that strips a leading space only at the start) is rejected rather than streamed
    // incorrectly.
    private static InvalidOperationException NotPrefixStable() =>
        new("The tokenizer's decoder is not prefix-stable; incremental detokenization is unsupported for it.");
}
