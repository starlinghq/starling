namespace Starlings.Models;

/// <summary>
/// Support status of one capability (plan section 9), shared by every model package. A missing entry never means
/// supported.
/// </summary>
/// <remarks>
/// The values and their order equal the per-package enums this type replaces (<c>Starlings.Models.E5.CapabilityStatus</c>,
/// <c>EmbeddingGemma2CapabilityStatus</c>, <c>Gliner2CapabilityStatus</c>), so a cast between them keeps the meaning.
/// The registry manifests spell the same statuses in lower case (<c>supported</c>, <c>unsupported</c>,
/// <c>experimental</c>, <c>untested</c>).
/// </remarks>
public enum CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected or the API does not offer it.</summary>
    Unsupported,

    /// <summary>Implemented and tested against the reference fixtures, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no sufficient evidence exists.</summary>
    Untested,
}
