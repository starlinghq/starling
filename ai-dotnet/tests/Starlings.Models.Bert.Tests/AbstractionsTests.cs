namespace Starlings.Models.Bert.Tests;

/// <summary>The BERT route's embedder and reranker implement the shared interfaces. Weight-free.</summary>
public sealed class AbstractionsTests
{
    [Fact]
    public void TheEmbedderIsAnITextEmbedderAndTheRerankerAnIReranker()
    {
        Assert.True(typeof(ITextEmbedder).IsAssignableFrom(typeof(BertTextEmbedder)));
        Assert.True(typeof(IReranker).IsAssignableFrom(typeof(BertReranker)));
    }
}
