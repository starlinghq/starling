using System.Globalization;
using Starlings.Acquisition;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The e5 asset lock against the registry manifests, and the documented flow: explicit download once, then offline
/// load into the encoder. The download step runs only with <c>STARLING_NETWORK_TESTS=1</c>.
/// </summary>
public class AcquisitionTests(ITestOutputHelper output)
{
    private const string NetworkVariable = "STARLING_NETWORK_TESTS";

    [Theory]
    [InlineData("onnx-fp32-cpu")]
    [InlineData("onnx-fp32-cuda")]
    public void LockMatchesTheRegistryManifest(string profile)
    {
        string manifest = File.ReadAllText(E5TestAssets.ManifestPath(profile));
        Assert.Contains($"url: https://huggingface.co/{E5ModelAssets.Repository}\n", manifest, StringComparison.Ordinal);
        Assert.Contains($"revision: \"{E5ModelAssets.Revision}\"", manifest, StringComparison.Ordinal);
        foreach (var file in E5ModelAssets.Lock.Files)
        {
            var (sha, size) = ReadManifestEntry(profile, file.Path);
            Assert.Equal(sha, file.Sha256);
            Assert.Equal(size, file.Size);
        }

        Assert.Equal(E5ModelAssets.Revision, E5ModelAssets.Lock.Revision);
        Assert.Equal(E5ModelAssets.ModelSha256, E5ModelAssets.Lock.GetFile(E5ModelAssets.ModelFile).Sha256);
        Assert.Equal(E5ModelAssets.TokenizerSha256, E5ModelAssets.Lock.GetFile(E5ModelAssets.TokenizerFile).Sha256);
    }

    [Fact]
    public void OfflineMissIsTypedAndSaysHowToDownload()
    {
        string empty = Path.Combine(Path.GetTempPath(), "starlings-e5-empty-" + Guid.NewGuid().ToString("N"));
        var ex = Assert.Throws<ModelAssetMissingException>(() => E5ModelAssets.OpenLocal(new ModelAcquisitionOptions { CacheDirectory = empty }));
        Assert.Equal([E5ModelAssets.ModelFile, E5ModelAssets.TokenizerFile], ex.MissingPaths);
        Assert.Contains("download multilingual-e5-small", ex.Message, StringComparison.Ordinal);
        Assert.False(Directory.Exists(empty), "Offline resolution must not create anything.");
    }

    [Fact]
    public async Task ExplicitDownloadThenOfflineLoadAndEmbed()
    {
        var options = new ModelAcquisitionOptions { CacheDirectory = E5TestAssets.ModelsRoot };
        if (Environment.GetEnvironmentVariable(NetworkVariable) == "1")
        {
            // Setup step: the only place the network is used. Already cached files are verified, not downloaded again.
            var downloaded = await E5ModelAssets.DownloadAsync(options, cancellationToken: TestContext.Current.CancellationToken);
            output.WriteLine($"acquired {downloaded.Snapshot.Directory}");
        }

        // Run time: offline, full SHA-256 verification, then the encoder from the verified path.
        E5ModelFiles files;
        try
        {
            files = E5ModelAssets.OpenLocal(options);
        }
        catch (ModelAssetMissingException ex)
        {
            Assert.Skip(ex.Message + $" (or set {NetworkVariable}=1 to download it in this test)");
            throw;
        }

        Assert.Equal(CacheVerification.Sha256, files.Snapshot.Verification);
        Assert.True(File.Exists(files.TokenizerPath));
        using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
        });
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
        var embedder = new E5TextEmbedder(tokenizer, encoder);
        var first = E5Fixtures.Cases[0];
        float[] embedding = embedder.Embed(E5Fixtures.Texts[first.Id], TestContext.Current.CancellationToken);
        var tolerances = E5Tolerances.Load(E5TestAssets.CpuFixtureDirectory);
        double maxAbs = embedding.Zip(first.Embedding, (a, b) => Math.Abs((double)a - b)).Max();
        output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{first.Id}: max abs {maxAbs:E3} (bound {tolerances.EmbeddingMaxAbs:E1})"));
        Assert.True(maxAbs <= tolerances.EmbeddingMaxAbs);
    }

    private static (string Sha256, long Size) ReadManifestEntry(string profile, string path)
    {
        var lines = File.ReadAllLines(E5TestAssets.ManifestPath(profile));
        for (int i = 0; i < lines.Length - 2; i++)
        {
            if (lines[i].Trim() == $"- path: {path}")
            {
                string sha = lines[i + 1].Trim();
                string size = lines[i + 2].Trim();
                Assert.StartsWith("sha256:", sha, StringComparison.Ordinal);
                Assert.StartsWith("size:", size, StringComparison.Ordinal);
                return (sha["sha256:".Length..].Trim().Trim('"'), long.Parse(size["size:".Length..].Trim(), CultureInfo.InvariantCulture));
            }
        }

        throw new InvalidOperationException($"{path} not in origin.files of {profile}.");
    }
}
