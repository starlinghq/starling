using Starlings.TestSupport;

namespace Starlings.Models.PaddleOcr.Tests;

/// <summary>
/// The package capability table (<see cref="PaddleOcrCapabilities.All"/>) equals the registry manifest of its profile
/// key for key (the manifest spells a package key's <c>.</c> as <c>_</c>), and the manifest's compatibility rows follow
/// the package statuses. Weight-free; runs in CI.
/// </summary>
public sealed class RegistryContractTests
{
    private static string RepositoryRoot { get; } = PaddleOcrTestAssets.RepositoryRoot;

    private static RegistryManifest Manifest { get; } = Assert.Single(RegistryManifest.Family(RepositoryRoot, "ppocrv5-mobile"));

    private static string Name(CapabilityStatus status) => status.ToString().ToLowerInvariant();

    [Fact]
    public void TheRegisteredProfileIsThePackageProfile()
    {
        Assert.Equal(PaddleOcrCapabilities.CpuProfile, Manifest.Id);
        Assert.Equal("experimental", Manifest.Maturity);
        Assert.Equal("ocr", Manifest.Behavior["tasks"]);
    }

    [Fact]
    public void CapabilitiesEqualTheManifestKeyForKey()
    {
        var package = PaddleOcrCapabilities.All.ToDictionary(
            entry => entry.Key.Replace('.', '_'),
            entry => Name(entry.Value),
            StringComparer.Ordinal);

        Assert.Equal(
            Manifest.Capabilities.OrderBy(entry => entry.Key, StringComparer.Ordinal),
            package.OrderBy(entry => entry.Key, StringComparer.Ordinal));
    }

    [Fact]
    public void CompatibilityRowsFollowThePackageStatuses()
    {
        Assert.Equal(Name(PaddleOcrCapabilities.All["ocr"]), Manifest.Compatibility["linux-x64 cpu"]);
        Assert.Equal(Name(PaddleOcrCapabilities.All["cuda"]), Manifest.Compatibility["linux-x64 cuda"]);
        Assert.Equal("untested", Manifest.Compatibility["windows-x64 cpu"]);
    }

    // The manifest's four origin files are the package's two asset locks, byte for byte (the recognizer's under base/,
    // the schema's second-repository spelling).
    [Fact]
    public void OriginFilesAreThePackageLocks()
    {
        string manifest = File.ReadAllText(Path.Combine(RepositoryRoot, "model-pipeline", "registry", "ppocrv5-mobile", "onnx-fp32-cpu", "manifest.yaml"))
            .Replace("\r\n", "\n", StringComparison.Ordinal);
        Assert.Contains($"url: https://huggingface.co/{PaddleOcrModelAssets.DetectorRepository}\n    revision: \"{PaddleOcrModelAssets.DetectorRevision}\"", manifest, StringComparison.Ordinal);
        Assert.Contains($"url: https://huggingface.co/{PaddleOcrModelAssets.RecognizerRepository}\n    revision: \"{PaddleOcrModelAssets.RecognizerRevision}\"", manifest, StringComparison.Ordinal);
        foreach (var (prefix, assetLock) in new[] { ("", PaddleOcrModelAssets.DetectorLock), ("base/", PaddleOcrModelAssets.RecognizerLock) })
        {
            foreach (var file in assetLock.Files)
            {
                Assert.Contains($"- path: {prefix}{file.Path}\n      sha256: \"{file.Sha256}\"\n      size: {file.Size}\n", manifest, StringComparison.Ordinal);
            }
        }
    }
}
