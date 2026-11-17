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
            WhisperModel.Prompt(options.Language!));
    }

    [Fact]
    public void NullLanguageThrowsBecauseDetectionIsUnsupported()
    {
        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(() => WhisperModel.Prompt(null!));

        Assert.Null(ex.Language);
        Assert.Equal("language", ex.ParamName);
        Assert.Contains("language detection is unsupported", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LanguageDetectionIsUnsupported()
    {
        Assert.Equal(CapabilityStatus.Unsupported, WhisperCapabilities.All["language_detection"]);
    }

    [Fact]
    public void CreateRejectsANullLanguageBeforeLoadingAnyFile()
    {
        var options = new WhisperOptions { Language = null };

        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(
            () => WhisperTranscriber.Create("missing-encoder.onnx", "missing-decoder.onnx", "missing-tokenizer.json", options));

        Assert.Null(ex.Language);
        Assert.Equal(nameof(WhisperOptions.Language), ex.ParamName);
    }

    [Fact]
    public void ModelCreateRejectsANullLanguageBeforeLoadingAnyGraph()
    {
        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(
            () => WhisperModel.Create("missing-encoder.onnx", "missing-decoder.onnx", new WhisperOptions { Language = null }));

        Assert.Null(ex.Language);
        Assert.Equal(nameof(WhisperOptions.Language), ex.ParamName);
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
    public void ModelCreateRejectsAnUnknownLanguageBeforeLoadingAnyGraph()
    {
        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(
            () => WhisperModel.Create("missing-encoder.onnx", "missing-decoder.onnx", new WhisperOptions { Language = "xx" }));

        Assert.Equal("xx", ex.Language);
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
