using Starlings.TestSupport;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The generated capability records (<see cref="BertCapabilities.ProfileCapabilities"/>) cover every registry profile of
/// the model families this package serves and equal each manifest field for field. Weight-free; runs in CI.
/// </summary>
public sealed class CapabilityRecordTests
{
    // The model families named in the header of the package's ProfileCapabilities.g.cs, one per built-in profile.
    private static readonly string[] Families =
    [
        "all-minilm-l6-v2", "bge-small-en-v1.5", "bge-base-en-v1.5", "bge-m3", "ms-marco-minilm-l6-v2", "bge-reranker-v2-m3",
        "nomic-embed-text-v1.5",
    ];

    [Fact]
    public void CapabilityRecordsEqualTheManifests()
    {
        string root = BuiltInProfileTests.RepositoryRoot();
        CapabilityRecordChecks.AssertRecordsEqualManifests(
            BertCapabilities.ProfileCapabilities,
            Families.SelectMany(family => RegistryManifest.Family(root, family)));
    }

    [Fact]
    public void EveryBuiltInProfileHasAFamily()
    {
        Assert.Equal(BertModelProfile.BuiltInIds.Count, Families.Length);
    }
}
