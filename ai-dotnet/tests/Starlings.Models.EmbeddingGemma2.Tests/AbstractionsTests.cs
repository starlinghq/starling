namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// The shared types of Starlings.Models.Abstractions as the EmbeddingGemma 2 package exposes them. Weight-free.
/// </summary>
public class AbstractionsTests
{
    [Fact]
    public void TheLengthExceptionIsTheSharedOneWithTheLimitAsMaxTokens()
    {
        var ex = new EmbeddingGemma2InputTooLongException(inputIndex: 3, tokenCount: 9000, maximumTokens: 8192);

        Assert.IsAssignableFrom<InputTooLongException>(ex);
        Assert.IsAssignableFrom<ArgumentException>(ex);
        Assert.Equal(3, ex.InputIndex);
        Assert.Equal(9000, ex.TokenCount);
        Assert.Equal(8192, ex.MaxTokens);
        Assert.Equal("inputs", ex.ParamName);
#pragma warning disable CS0618 // The former property stays for one minor version.
        Assert.Equal(8192, ex.MaximumTokens);
        Assert.Equal(100, new EmbeddingGemma2InputTooLongException("m") { MaximumTokens = 100 }.MaxTokens);
#pragma warning restore CS0618
    }

    [Fact]
    public void TheTextEmbedderIsAnITextEmbedder() =>
        Assert.True(typeof(ITextEmbedder).IsAssignableFrom(typeof(EmbeddingGemma2TextEmbedder)));

    [Fact]
    public void TheAssetsInstanceForwardsToTheStaticMembers() =>
        Assert.Same(EmbeddingGemma2ModelAssets.Lock, EmbeddingGemma2ModelAssets.Assets.Lock);

    [Fact]
    public void ProfileRecordsCarryTheContractFields()
    {
        var text = EmbeddingGemma2Capabilities.ProfileCapabilities[EmbeddingGemma2Capabilities.StableProfile];
        Assert.Equal(ProfileMaturity.Stable, text.Maturity);
        Assert.Equal(8192, text.MaxInputTokens);
        Assert.Equal(TruncationPolicy.Rejects, text.Truncation);
        Assert.Equal(768, text.Dimension);
        Assert.Equal(6, EmbeddingGemma2Capabilities.ProfileCapabilities.Count);
    }
}
