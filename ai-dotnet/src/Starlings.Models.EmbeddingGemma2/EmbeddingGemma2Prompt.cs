using System.Diagnostics.CodeAnalysis;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// A named prompt of the pinned <c>config_sentence_transformers.json</c> (20 names, <c>default_prompt_name</c> null),
/// or <see cref="None"/>. The prompt text is prepended to the input text before tokenization and its tokens are pooled
/// with the text's (<c>include_prompt: true</c>). Nothing in this package adds a prompt implicitly: the default is
/// always <see cref="None"/>.
/// </summary>
/// <remarks>
/// The model card recommends <see cref="Query"/> (<c>"task: search result | query: "</c>) for search queries and
/// <see cref="Document"/> (<c>"title: none | text: "</c>) for documents. A document with a title is written by the
/// caller as <c>"title: {title} | text: {content}"</c> with <see cref="None"/>. Names are matched exactly
/// (ordinal, case-sensitive), as sentence-transformers does.
/// </remarks>
public sealed record EmbeddingGemma2Prompt
{
    private EmbeddingGemma2Prompt(string name, string text)
    {
        Name = name;
        Text = text;
    }

    /// <summary>Name of the "no prompt" value. Not a name in <c>config_sentence_transformers.json</c>.</summary>
    public const string NoneName = "none";

    /// <summary>Prompt text for search queries and the query-side task prompts.</summary>
    public const string SearchQueryText = "task: search result | query: ";

    /// <summary>Prompt text for documents.</summary>
    public const string DocumentText = "title: none | text: ";

    /// <summary>The prompt name as in <c>config_sentence_transformers.json</c>, or <see cref="NoneName"/>.</summary>
    public string Name { get; }

    /// <summary>The text prepended to the input before tokenization; empty for <see cref="None"/>.</summary>
    public string Text { get; }

    /// <summary>No prompt: the text is tokenized exactly as given. The default everywhere in this package.</summary>
    public static EmbeddingGemma2Prompt None { get; } = new(NoneName, string.Empty);

    /// <summary>The <c>query</c> prompt: <c>"task: search result | query: "</c>.</summary>
    public static EmbeddingGemma2Prompt Query { get; } = new("query", SearchQueryText);

    /// <summary>The <c>document</c> prompt: <c>"title: none | text: "</c>.</summary>
    public static EmbeddingGemma2Prompt Document { get; } = new("document", DocumentText);

    /// <summary>
    /// The 20 named prompts of <c>config_sentence_transformers.json</c> at the pinned revision, name to text, in the
    /// file's order. <see cref="None"/> is not among them.
    /// </summary>
    public static IReadOnlyDictionary<string, string> Named { get; } = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["BitextMining"] = SearchQueryText,
        ["Classification"] = "task: classification | query: ",
        ["Clustering"] = "task: clustering | query: ",
        ["CodeRetrieval"] = "task: code retrieval | query: ",
        ["Document"] = DocumentText,
        ["FactChecking"] = "task: fact checking | query: ",
        ["InstructionRetrieval"] = "task: code retrieval | query: ",
        ["MultilabelClassification"] = "task: classification | query: ",
        ["PairClassification"] = "task: sentence similarity | query: ",
        ["QuestionAnswering"] = "task: question answering | query: ",
        ["Reranking"] = SearchQueryText,
        ["Retrieval"] = SearchQueryText,
        ["Retrieval-document"] = DocumentText,
        ["Retrieval-query"] = SearchQueryText,
        ["STS"] = "task: sentence similarity | query: ",
        ["SearchQuery"] = SearchQueryText,
        ["SentenceSimilarity"] = "task: sentence similarity | query: ",
        ["Summarization"] = "task: sentence similarity | query: ",
        ["document"] = DocumentText,
        ["query"] = SearchQueryText,
    };

    /// <summary>
    /// The prompt with this exact name: one of <see cref="Named"/>, or <see cref="NoneName"/> for <see cref="None"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The name is not one of the 20 configured names or <c>"none"</c>.</exception>
    public static EmbeddingGemma2Prompt FromName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return TryFromName(name, out var prompt)
            ? prompt
            : throw new ArgumentException(
                $"'{name}' is not a prompt name of google/embeddinggemma-2. Expected '{NoneName}' or one of: {string.Join(", ", Named.Keys)} (case-sensitive).",
                nameof(name));
    }

    /// <summary>Looks up a prompt by exact name; see <see cref="FromName"/>.</summary>
    public static bool TryFromName(string name, [NotNullWhen(true)] out EmbeddingGemma2Prompt? prompt)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (string.Equals(name, NoneName, StringComparison.Ordinal))
        {
            prompt = None;
            return true;
        }

        if (string.Equals(name, Query.Name, StringComparison.Ordinal))
        {
            prompt = Query;
            return true;
        }

        if (string.Equals(name, Document.Name, StringComparison.Ordinal))
        {
            prompt = Document;
            return true;
        }

        if (Named.TryGetValue(name, out var text))
        {
            prompt = new EmbeddingGemma2Prompt(name, text);
            return true;
        }

        prompt = null;
        return false;
    }

    /// <summary>Applies the prompt: <see cref="Text"/> followed by <paramref name="text"/>.</summary>
    public string Apply(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Text.Length == 0 ? text : Text + text;
    }

    /// <inheritdoc/>
    public override string ToString() => Name;
}
