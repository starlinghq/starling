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
    public void NullLanguageMeansDetectAndPromptStillRequiresOne()
    {
        // A null language is now detection (WhisperModel.DetectLanguage), so Prompt still needs a concrete code.
        var options = new WhisperOptions { Language = null };
        options.Validate(); // no throw: detection is allowed.

        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(() => WhisperModel.Prompt(null!));

        Assert.Null(ex.Language);
        Assert.Equal("language", ex.ParamName);
    }

    [Fact]
    public void LanguageDetectionIsExperimental()
    {
        Assert.Equal(CapabilityStatus.Experimental, WhisperCapabilities.All["language_detection"]);
    }

    [Fact]
    public void NullLanguageIsAllowedAndUnknownCodesStillThrow()
    {
        // Detection: a null language no longer rejects creation (the graphs still fail to load with missing paths).
        Assert.IsAssignableFrom<Exception>(
            Record.Exception(() => WhisperTranscriber.Create("missing-encoder.onnx", "missing-decoder.onnx", "missing-tokenizer.json", new WhisperOptions { Language = null })));

        // An unknown code is still rejected before any file loads.
        var ex = Assert.Throws<WhisperUnsupportedLanguageException>(
            () => WhisperTranscriber.Create("missing-encoder.onnx", "missing-decoder.onnx", "missing-tokenizer.json", new WhisperOptions { Language = "xx" }));
        Assert.Equal("xx", ex.Language);
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
