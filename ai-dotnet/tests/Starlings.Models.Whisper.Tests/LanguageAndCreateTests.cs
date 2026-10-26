namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// Language selection, the typed language error and the tokenizer digest check of <see cref="WhisperTranscriber.Create"/>.
/// Weight-free; runs in CI.
/// </summary>
public sealed class LanguageAndCreateTests
{
    [Fact]
    public void TheDefaultLanguageIsEnglish()
    {
        var options = new WhisperOptions();

        Assert.Equal(WhisperOptions.DefaultLanguage, options.Language);
        Assert.Equal(
            [WhisperModel.StartOfTranscript, WhisperModel.English, WhisperModel.Transcribe, WhisperModel.NoTimestamps],
            WhisperModel.Prompt(options.Language));
    }

    [Fact]
    public void NullLanguageKeepsTheAutoDetectPrompt()
    {
        Assert.Equal(WhisperModel.StartOfLanguageModel, WhisperModel.Prompt(null)[1]);
    }

    [Fact]
    public void LanguageCodesAreCaseInsensitive()
    {
        Assert.Equal(WhisperModel.LanguageToId["de"], WhisperModel.Prompt("DE")[1]);
    }

    [Theory]
    [InlineData("xx")]
    [InlineData("english")]
    [InlineData("")]
    public void AnUnknownLanguageThrowsTheTypedException(string language)
    {
        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(() => WhisperModel.Prompt(language));

        Assert.Equal(language, ex.Language);
        Assert.Equal("language", ex.ParamName);
        Assert.IsAssignableFrom<ArgumentException>(ex);
    }

    [Fact]
    public void CreateRejectsAnUnknownLanguageBeforeLoadingAnyFile()
    {
        var options = new WhisperOptions { Language = "xx" };

        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(
            () => WhisperTranscriber.Create("missing-encoder.onnx", "missing-decoder.onnx", "missing-tokenizer.json", options));

        Assert.Equal("xx", ex.Language);
        Assert.Equal(nameof(WhisperOptions.Language), ex.ParamName);
    }

    [Fact]
    public void CreateRejectsATokenizerThatIsNotThePinnedFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"whisper-tokenizer-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, "{}");
        try
        {
            // The graph paths do not exist: the digest check runs before the graphs load.
            var ex = Assert.Throws<InvalidDataException>(
                () => WhisperTranscriber.Create("missing-encoder.onnx", "missing-decoder.onnx", path));

            Assert.Contains(WhisperModelAssets.TokenizerSha256, ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
