using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>
/// A request's choice of intervention: which profile, at what strength <c>alpha</c>. Strength 0 leaves the model's
/// output unchanged; 1 applies the profile's operator in full. Any finite value is accepted; the profile author
/// documents the useful range. Immutable.
/// </summary>
public sealed record InterventionSelection
{
    /// <summary>Creates a selection.</summary>
    /// <param name="profileId">Id of the profile (<see cref="InterventionProfile.Id"/>).</param>
    /// <param name="strength">Request-level strength <c>alpha</c>; finite.</param>
    /// <exception cref="InterventionContractException">The id is malformed or the strength is not finite.</exception>
    public InterventionSelection(string profileId, float strength)
    {
        if (!StrictJson.IsProfileId(profileId))
        {
            throw new InterventionContractException($"'{profileId}' is not a valid profile id.");
        }

        if (!float.IsFinite(strength))
        {
            throw new InterventionContractException($"strength {strength} is not finite.");
        }

        ProfileId = profileId;
        Strength = strength;
    }

    /// <summary>Id of the selected profile.</summary>
    public string ProfileId { get; }

    /// <summary>Request-level strength <c>alpha</c>.</summary>
    public float Strength { get; }

    /// <summary>Checks that this selection names <paramref name="profile"/>.</summary>
    /// <exception cref="InterventionMismatchException">The selection names another profile.</exception>
    public void EnsureMatches(InterventionProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (!string.Equals(ProfileId, profile.Id, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"Selection names profile '{ProfileId}', the loaded graph carries '{profile.Id}'.");
        }
    }
}
