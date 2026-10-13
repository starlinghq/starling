namespace Starlings.Models.Gliner2;

/// <summary>
/// The model input GLiNER2 builds for one text and one entity schema (<c>SchemaTransformer</c>, inference mode), plus
/// what decoding needs: the schema prefix <c>( [P] entities ( [E] label ... ) ) [SEP_TEXT]</c> followed by the words,
/// each schema token and word tokenized alone with the owned tokenizer (no <c>[CLS]</c>/<c>[SEP]</c>).
/// </summary>
/// <remarks>
/// Created by <see cref="Gliner2EntityExtractor.Prepare(string, IReadOnlyList{string}, IReadOnlyDictionary{string, string}?)"/>.
/// Immutable. Names follow the reference fixtures (<c>extraction.jsonl</c> of the profile
/// <c>gliner2-base-v1/onnx-fp32-cpu</c>).
/// </remarks>
public sealed class Gliner2ModelInput
{
    private readonly int[] _wordStarts;
    private readonly int[] _wordEnds;
    private readonly long[] _inputIds;
    private readonly long[] _wordIndex;
    private readonly long[] _schemaIndex;

    internal Gliner2ModelInput(
        string text,
        string normalizedText,
        int[] normalizedScalars,
        int originalScalarCount,
        IReadOnlyList<string> labels,
        IReadOnlyList<string> schemaTokens,
        IReadOnlyList<string> words,
        int[] wordStarts,
        int[] wordEnds,
        long[] inputIds,
        long[] wordIndex,
        long[] schemaIndex)
    {
        Text = text;
        NormalizedText = normalizedText;
        NormalizedScalars = normalizedScalars;
        OriginalScalarCount = originalScalarCount;
        Labels = labels;
        SchemaTokens = schemaTokens;
        Words = words;
        _wordStarts = wordStarts;
        _wordEnds = wordEnds;
        _inputIds = inputIds;
        _wordIndex = wordIndex;
        _schemaIndex = schemaIndex;
    }

    /// <summary>The caller's text.</summary>
    public string Text { get; }

    /// <summary>GLiNER2's normalized text: <see cref="Text"/>, plus <c>"."</c> unless it ends in <c>.</c>, <c>!</c> or <c>?</c>.</summary>
    public string NormalizedText { get; }

    /// <summary>Labels in schema order: the caller's labels with duplicates removed (first occurrence kept), case kept.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>The schema tokens before <c>[SEP_TEXT]</c>, as GLiNER2 records them.</summary>
    public IReadOnlyList<string> SchemaTokens { get; }

    /// <summary>The lower-cased words of the normalized text.</summary>
    public IReadOnlyList<string> Words { get; }

    /// <summary>Start of each word, in Unicode scalar values of <see cref="NormalizedText"/> (reference <c>start_map</c>).</summary>
    public ReadOnlySpan<int> WordStarts => _wordStarts;

    /// <summary>Exclusive end of each word, in Unicode scalar values of <see cref="NormalizedText"/> (reference <c>end_map</c>).</summary>
    public ReadOnlySpan<int> WordEnds => _wordEnds;

    /// <summary>Token ids: schema prefix, <c>[SEP_TEXT]</c>, then the subword pieces of each word.</summary>
    public ReadOnlySpan<long> InputIds => _inputIds;

    /// <summary>Position of each word's first subword in <see cref="InputIds"/> (reference <c>word_index</c>).</summary>
    public ReadOnlySpan<long> WordIndex => _wordIndex;

    /// <summary>Positions of <c>[P]</c> and of each <c>[E]</c> in <see cref="InputIds"/> (reference <c>schema_index</c>).</summary>
    public ReadOnlySpan<long> SchemaIndex => _schemaIndex;

    /// <summary>Number of tokens (<see cref="InputIds"/> length); GLiNER2 adds no special tokens around them.</summary>
    public int TokenCount => _inputIds.Length;

    /// <summary>Number of words.</summary>
    public int WordCount => _wordStarts.Length;

    /// <summary>Unicode scalar values of <see cref="NormalizedText"/>.</summary>
    internal int[] NormalizedScalars { get; }

    /// <summary>Number of Unicode scalar values of <see cref="Text"/> (a prefix of the normalized scalars).</summary>
    internal int OriginalScalarCount { get; }
}
