using Starlings.Acquisition;
using Starlings.Tokenizers;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// Pins the README's "unsupported" claim for the draft paraphrase-multilingual candidate: the pinned
/// Starlings.Tokenizers refuses its tokenizer.json (the <c>WhitespaceSplit</c> pre-tokenizer). Skips without the asset
/// (tools/fetch-tokenizer-assets.sh provides it). When a Tokenizers bump makes this fail, the model loads: drop the
/// README's unsupported note and onboard the .NET side.
/// </summary>
public sealed class UnsupportedTokenizerTests
{
    [Fact]
    public void ParaphraseMultilingualTokenizerIsRefusedByThePinnedTokenizers()
    {
        string root = RepositoryRoot();
        var profile = BertModelProfile.Load(Path.Combine(root, "model-pipeline", "candidates", "paraphrase-multilingual", "profile.json"));
        string models = Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(root, ".models");
        string path = Path.Combine(models, profile.SourceLock.Repository, profile.SourceLock.Revision, BertModelProfile.TokenizerFile);
        Assert.SkipUnless(File.Exists(path), $"{path} not found; run tools/fetch-tokenizer-assets.sh.");

        var error = Assert.Throws<UnsupportedTokenizerComponentException>(() => Tokenizer.FromFile(path));
        Assert.Contains("WhitespaceSplit", error.Message, StringComparison.Ordinal);
    }

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "candidates")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Repository root not found.");
    }
}
