using Starlings.Models.QwenChat;
using Starlings.TestSupport;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>
/// The generated capability records (<see cref="QwenChatCapabilities.ProfileCapabilities"/>) cover every registry profile
/// this package serves and equal each manifest field for field. Weight-free; runs in CI.
/// </summary>
public sealed class CapabilityRecordTests
{
    [Fact]
    public void CapabilityRecordsEqualTheManifests()
    {
        string root = BundleLockTests.RepositoryRoot();
        CapabilityRecordChecks.AssertRecordsEqualManifests(
            QwenChatCapabilities.ProfileCapabilities,
            [.. RegistryManifest.Family(root, "qwen3-0.6b"), .. RegistryManifest.Family(root, "qwen2.5-0.5b-instruct")]);
    }

    [Fact]
    public void EveryPackageProfileHasARecord()
    {
        Assert.Contains(QwenChatProfile.Qwen3.Id, QwenChatCapabilities.ProfileCapabilities.Keys);
        Assert.Contains(QwenChatProfile.Qwen25.Id, QwenChatCapabilities.ProfileCapabilities.Keys);
    }
}
