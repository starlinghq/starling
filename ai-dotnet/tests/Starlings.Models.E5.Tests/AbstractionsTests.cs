using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The shared types of Starlings.Models.Abstractions as the e5 package exposes them: the typed length exception, the
/// <see cref="ITextEmbedder"/> shape and the <see cref="IModelAssets{TFiles}"/> instance. Weight-free.
/// </summary>
public class AbstractionsTests
{
    [Fact]
    public void AnInputAboveTheLimitThrowsTheTypedLengthException()
    {
        using var encoder = new E5Encoder(
            OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
            new E5EncoderOptions { MaxSequenceLength = 4 },
            ownsSession: true);
        var tooLong = new EncoderInput(new long[] { 0, 5, 5, 5, 2 });

        var ex = Assert.Throws<InputTooLongException>(() => encoder.EmbedBatch([new EncoderInput(new long[] { 0, 2 }), tooLong], TestContext.Current.CancellationToken));
        Assert.IsAssignableFrom<ArgumentException>(ex);
        Assert.Equal(1, ex.InputIndex);
        Assert.Equal(5, ex.TokenCount);
        Assert.Equal(4, ex.MaxTokens);
        Assert.Equal("Input 1 has 5 tokens; the maximum is 4. Truncate in the tokenizer. (Parameter 'inputs')", ex.Message);
        Assert.Equal(0, Assert.Throws<InputTooLongException>(() => encoder.Validate(tooLong)).InputIndex);
    }

    [Fact]
    public void TheTextEmbedderIsAnITextEmbedder() =>
        Assert.True(typeof(ITextEmbedder).IsAssignableFrom(typeof(E5TextEmbedder)));

    [Fact]
    public void TheAssetsInstanceForwardsToTheStaticMembers()
    {
        Assert.Same(E5ModelAssets.Lock, E5ModelAssets.Assets.Lock);
        string empty = Directory.CreateTempSubdirectory().FullName;
        try
        {
            Assert.ThrowsAny<Exception>(() => E5ModelAssets.Assets.OpenDirectory(empty));
        }
        finally
        {
            Directory.Delete(empty, recursive: true);
        }
    }

    [Fact]
    public void ProfileRecordsCarryTheContractFields()
    {
        var cpu = E5Capabilities.ProfileCapabilities[E5Capabilities.StableProfile];
        Assert.Equal(ProfileMaturity.Stable, cpu.Maturity);
        Assert.Equal(512, cpu.MaxInputTokens);
        Assert.Equal(TruncationPolicy.TruncatesRight, cpu.Truncation);
        Assert.Equal(384, cpu.Dimension);
        Assert.Equal(["cpu"], cpu.QualifiedProviders);
        Assert.Equal(["cuda"], E5Capabilities.ProfileCapabilities[E5Capabilities.StableCudaProfile].QualifiedProviders);
    }
}
