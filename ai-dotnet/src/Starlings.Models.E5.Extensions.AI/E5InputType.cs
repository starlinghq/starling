namespace Starlings.Models.E5.Extensions.AI;

/// <summary>
/// Which e5 prefix the generator prepends to every input. e5 was trained with <c>"query: "</c> and
/// <c>"passage: "</c> prefixes (model card); omitting them degrades retrieval quality. The generator never detects
/// or removes an existing prefix: with <see cref="Query"/> the text <c>"query: x"</c> becomes
/// <c>"query: query: x"</c>.
/// </summary>
public enum E5InputType
{
    /// <summary>Default. Texts are tokenized exactly as given; the caller adds a prefix if wanted.</summary>
    None,

    /// <summary>Prepends <see cref="E5TextEmbedder.QueryPrefix"/> (<c>"query: "</c>), for search queries and symmetric tasks.</summary>
    Query,

    /// <summary>Prepends <see cref="E5TextEmbedder.PassagePrefix"/> (<c>"passage: "</c>), for documents being searched.</summary>
    Passage,
}
