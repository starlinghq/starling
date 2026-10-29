namespace Starlings.Models.Gliner2.Tests;

/// <summary>The shared types of Starlings.Models.Abstractions as the GLiNER2 package exposes them. Weight-free.</summary>
public class AbstractionsTests
{
    [Fact]
    public void TheLengthExceptionIsTheSharedOne()
    {
        var ex = new Gliner2InputTooLongException("too long") { InputIndex = 2, TokenCount = 600, MaxTokens = 512 };

        Assert.IsAssignableFrom<InputTooLongException>(ex);
        Assert.IsAssignableFrom<ArgumentException>(ex);
        Assert.Equal(2, ex.InputIndex);
        Assert.Equal(600, ex.TokenCount);
        Assert.Equal(512, ex.MaxTokens);
    }

    [Fact]
    public void TheAssetsInstanceForwardsToTheStaticMembers() =>
        Assert.Same(Gliner2ModelAssets.Lock, Gliner2ModelAssets.Assets.Lock);

    [Fact]
    public void ProfileRecordsCarryTheContractFields()
    {
        var cpu = Gliner2Capabilities.ProfileCapabilities[Gliner2Capabilities.StableProfile];
        Assert.Equal(ProfileMaturity.Stable, cpu.Maturity);
        Assert.Equal(512, cpu.MaxInputTokens);
        Assert.Equal(TruncationPolicy.Rejects, cpu.Truncation);
        Assert.Null(cpu.Dimension);
    }
}
