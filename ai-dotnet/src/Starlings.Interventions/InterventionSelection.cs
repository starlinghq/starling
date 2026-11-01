using System.Collections.ObjectModel;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>One entry of an <see cref="InterventionSelection"/>: a profile id and its request-level strength.</summary>
/// <param name="ProfileId">Id of the profile (<see cref="InterventionProfile.Id"/>).</param>
/// <param name="Strength">Request-level strength <c>alpha</c> of that profile; finite.</param>
public readonly record struct InterventionProfileStrength(string ProfileId, float Strength);

/// <summary>
/// A request's choice of intervention: a set of (profile id, strength <c>alpha</c>) pairs. Strength 0 leaves the
/// model's output unchanged; 1 applies the profile's operator in full. Any finite value is accepted; the profile author
/// documents the useful range. Immutable; two selections are equal when they hold the same pairs in the same order.
/// </summary>
/// <remarks>
/// A derived graph that carries several profiles takes one strength vector (<see cref="InterventionManifest.Profiles"/>
/// order); <see cref="Resolve"/> builds it outside the graph: each named profile gets its strength, every other profile
/// 0. So profiles combine freely and a profile left out (or given 0) is disabled. Where two profiles share a site, the
/// graph applies them in strength-vector order, not in the order of this selection.
/// </remarks>
public sealed record InterventionSelection
{
    private readonly InterventionProfileStrength[] _entries;

    /// <summary>Creates a selection of one profile.</summary>
    /// <param name="profileId">Id of the profile (<see cref="InterventionProfile.Id"/>).</param>
    /// <param name="strength">Request-level strength <c>alpha</c>; finite.</param>
    /// <exception cref="InterventionContractException">The id is malformed or the strength is not finite.</exception>
    public InterventionSelection(string profileId, float strength)
        : this([new InterventionProfileStrength(profileId, strength)])
    {
    }

    /// <summary>Creates a selection of one or more profiles, each with its strength.</summary>
    /// <param name="entries">At least one pair; profile ids unique and well formed; strengths finite.</param>
    /// <exception cref="InterventionContractException">The set is empty, an id is malformed or repeated, or a strength is not finite.</exception>
    public InterventionSelection(IEnumerable<InterventionProfileStrength> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        InterventionProfileStrength[] list = [.. entries];
        if (list.Length == 0)
        {
            throw new InterventionContractException("a selection names at least one profile; pass no selection for strength 0.");
        }

        HashSet<string> ids = new(StringComparer.Ordinal);
        foreach (InterventionProfileStrength entry in list)
        {
            if (!StrictJson.IsProfileId(entry.ProfileId))
            {
                throw new InterventionContractException($"'{entry.ProfileId}' is not a valid profile id.");
            }

            if (!float.IsFinite(entry.Strength))
            {
                throw new InterventionContractException($"strength {entry.Strength} of profile '{entry.ProfileId}' is not finite.");
            }

            if (!ids.Add(entry.ProfileId))
            {
                throw new InterventionContractException($"profile '{entry.ProfileId}' appears twice in the selection.");
            }
        }

        _entries = list;
        Entries = new ReadOnlyCollection<InterventionProfileStrength>(list);
    }

    /// <summary>The (profile id, strength) pairs, in the order given.</summary>
    public IReadOnlyList<InterventionProfileStrength> Entries { get; }

    /// <summary>Id of the selected profile, when the selection names exactly one.</summary>
    /// <exception cref="InvalidOperationException">The selection names several profiles; use <see cref="Entries"/>.</exception>
    public string ProfileId => Single.ProfileId;

    /// <summary>Request-level strength <c>alpha</c>, when the selection names exactly one profile.</summary>
    /// <exception cref="InvalidOperationException">The selection names several profiles; use <see cref="Entries"/>.</exception>
    public float Strength => Single.Strength;

    private InterventionProfileStrength Single => _entries.Length == 1
        ? _entries[0]
        : throw new InvalidOperationException($"The selection names {_entries.Length} profiles; use Entries.");

    /// <summary>The strength given to <paramref name="profileId"/>, or 0 when the selection does not name it.</summary>
    public float StrengthOf(string profileId)
    {
        foreach (InterventionProfileStrength entry in _entries)
        {
            if (string.Equals(entry.ProfileId, profileId, StringComparison.Ordinal))
            {
                return entry.Strength;
            }
        }

        return 0f;
    }

    /// <summary>
    /// The strength vector for a graph carrying <paramref name="profileIds"/> in strength-vector order: element i is the
    /// strength of profile i, 0 for a profile this selection does not name. A new array on every call.
    /// </summary>
    /// <exception cref="InterventionMismatchException">The selection names a profile that is not in <paramref name="profileIds"/>.</exception>
    public float[] Resolve(IReadOnlyList<string> profileIds)
    {
        ArgumentNullException.ThrowIfNull(profileIds);
        foreach (InterventionProfileStrength entry in _entries)
        {
            if (!profileIds.Contains(entry.ProfileId, StringComparer.Ordinal))
            {
                throw new InterventionMismatchException(
                    $"Selection names profile '{entry.ProfileId}', the loaded graph carries [{string.Join(", ", profileIds)}].");
            }
        }

        float[] vector = new float[profileIds.Count];
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = StrengthOf(profileIds[i]);
        }

        return vector;
    }

    /// <summary>Checks that this selection names <paramref name="profile"/> and nothing else.</summary>
    /// <exception cref="InterventionMismatchException">The selection names another profile, or several.</exception>
    public void EnsureMatches(InterventionProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        _ = Resolve([profile.Id]);
    }

    // The synthesized PrintMembers would read ProfileId and Strength, which throw for several profiles.
    private bool PrintMembers(System.Text.StringBuilder builder)
    {
        builder.Append("Entries = [");
        for (int i = 0; i < _entries.Length; i++)
        {
            builder.Append(i == 0 ? "" : ", ")
                .Append(_entries[i].ProfileId)
                .Append(": ")
                .Append(_entries[i].Strength.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        builder.Append(']');
        return true;
    }

    /// <inheritdoc/>
    public bool Equals(InterventionSelection? other) =>
        other is not null && _entries.AsSpan().SequenceEqual(other._entries);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        HashCode hash = default;
        foreach (InterventionProfileStrength entry in _entries)
        {
            hash.Add(entry);
        }

        return hash.ToHashCode();
    }
}
