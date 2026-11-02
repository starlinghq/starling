namespace Starlings.Models.QwenChat;

/// <summary>The capability records of the registry profiles this package serves.</summary>
public static class QwenChatCapabilities
{
    /// <summary>
    /// The capability record of every registry profile this package serves (<see cref="ModelProfileCapabilities"/>),
    /// generated from the profiles' registry manifests by <c>model-pipeline catalog</c>, keyed by profile id
    /// (<see cref="QwenChatProfile.Id"/>). Each record's <see cref="ModelProfileCapabilities.Capabilities"/> is the
    /// manifest's <c>capabilities</c> map key for key.
    /// </summary>
    public static IReadOnlyDictionary<string, ModelProfileCapabilities> ProfileCapabilities => GeneratedProfileCapabilities.Profiles;
}
