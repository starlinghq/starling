namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The test classes that load Kev bundles (a session of the smaller bundle holds about 3.5 GB) share one collection, so they run
/// one after another instead of loading several models at once.
/// </summary>
[CollectionDefinition(Name)]
public sealed class KevModelGroup
{
    /// <summary>Collection name.</summary>
    public const string Name = "Kev models";
}
