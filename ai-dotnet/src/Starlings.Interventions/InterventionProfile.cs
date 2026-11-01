using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>
/// An intervention profile (<c>profile.json</c>, schema version 1 or 2): which base graph it applies to, the operator, and
/// the sites with their directions and scales. Immutable; construction and <see cref="Parse(ReadOnlyMemory{byte})"/>
/// validate every value. The format is specified in <c>docs/interventions/profile-format.md</c>.
/// </summary>
/// <remarks>
/// The profile's identity in a compiler manifest is the SHA-256 of the <c>profile.json</c> bytes as stored (see
/// <see cref="InterventionProfileFile"/>), not of a re-serialization. <see cref="Serialize"/> writes the canonical form
/// authors should store, so the stored bytes and the serialization agree.
/// The library ships the mechanism only. Deciding which callers may apply which profile is the host application's job.
/// </remarks>
public sealed class InterventionProfile
{
    /// <summary>Schema version 1: the version of a profile whose operator is <see cref="InterventionOperatorKind.DirectionalProjection"/>, unless it says otherwise.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>The newest schema version this library reads and writes: 2 adds <see cref="InterventionOperatorKind.DirectionalAddition"/>.</summary>
    public const int MaxSchemaVersion = 2;

    /// <summary>Largest accepted number of sites.</summary>
    public const int MaxSites = 1024;

    internal static readonly string[] PropertyNames = ["schema_version", "id", "description", "base_model_sha256", "operator", "sites"];

    /// <summary>Creates a profile. Sites are kept sorted by layer, then target.</summary>
    /// <param name="id">Profile id: lower-case ASCII letters, digits, '.', '_' and '-', starting with a letter or digit, at most 128 characters.</param>
    /// <param name="baseModelSha256">SHA-256 of the base <c>model.onnx</c> bytes (the base graph asset identity, as the package's asset lock records it).</param>
    /// <param name="operatorKind">The operator applied at every site.</param>
    /// <param name="sites">One or more sites; no two share a layer and target.</param>
    /// <param name="description">Optional free text for people; not interpreted.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionProfile(
        string id,
        string baseModelSha256,
        InterventionOperatorKind operatorKind,
        IEnumerable<InterventionSite> sites,
        string? description = null)
        : this(id, baseModelSha256, operatorKind, sites, description, operatorKind == InterventionOperatorKind.DirectionalProjection ? CurrentSchemaVersion : MaxSchemaVersion)
    {
    }

    /// <summary>Creates a profile with an explicit schema version (1 or 2; version 1 allows only <see cref="InterventionOperatorKind.DirectionalProjection"/>).</summary>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionProfile(
        string id,
        string baseModelSha256,
        InterventionOperatorKind operatorKind,
        IEnumerable<InterventionSite> sites,
        string? description,
        int schemaVersion)
    {
        if (schemaVersion is < CurrentSchemaVersion or > MaxSchemaVersion)
        {
            throw new InterventionContractException("schema_version", $"unsupported schema version {schemaVersion}; this library reads versions 1 and 2.");
        }

        ArgumentNullException.ThrowIfNull(sites);
        InterventionSite[] siteArray = [.. sites];
        if (!StrictJson.IsProfileId(id))
        {
            throw new InterventionContractException("id", $"'{id}' is not a valid profile id (lower-case [a-z0-9._-], starting with a letter or digit, at most 128).");
        }

        StrictJson.RequireSha256(baseModelSha256 ?? "", "base_model_sha256");
        if (!Enum.IsDefined(operatorKind))
        {
            throw new InterventionContractException("operator", $"unknown operator {operatorKind}.");
        }

        if (schemaVersion == CurrentSchemaVersion && operatorKind != InterventionOperatorKind.DirectionalProjection)
        {
            throw new InterventionContractException(
                "operator", $"operator '{InterventionNames.Of(operatorKind)}' needs schema version 2; version 1 knows only 'directional_projection'.");
        }

        if (siteArray.Length is 0 or > MaxSites)
        {
            throw new InterventionContractException("sites", $"{siteArray.Length} sites; expected 1..{MaxSites}.");
        }

        foreach (InterventionSite site in siteArray)
        {
            ArgumentNullException.ThrowIfNull(site, nameof(sites));
        }

        Array.Sort(siteArray, CompareSites);
        for (int i = 1; i < siteArray.Length; i++)
        {
            if (CompareSites(siteArray[i - 1], siteArray[i]) == 0)
            {
                throw new InterventionContractException(
                    "sites",
                    $"two sites at layer {siteArray[i].Layer}, target '{InterventionNames.Of(siteArray[i].Target)}'.");
            }
        }

        Id = id;
        Description = description;
        BaseModelSha256 = baseModelSha256!;
        Operator = operatorKind;
        Sites = new ReadOnlyCollection<InterventionSite>(siteArray);
        SchemaVersion = schemaVersion;
    }

    /// <summary>The schema version of the document (1 or 2); <see cref="Serialize"/> writes it unchanged.</summary>
    public int SchemaVersion { get; }

    /// <summary>Profile id.</summary>
    public string Id { get; }

    /// <summary>Optional free text for people.</summary>
    public string? Description { get; }

    /// <summary>SHA-256 of the base <c>model.onnx</c> bytes this profile applies to.</summary>
    public string BaseModelSha256 { get; }

    /// <summary>The operator applied at every site.</summary>
    public InterventionOperatorKind Operator { get; }

    /// <summary>The sites, sorted by layer, then target.</summary>
    public IReadOnlyList<InterventionSite> Sites { get; }

    /// <summary>Parses and validates a <c>profile.json</c> document (UTF-8, no byte order mark).</summary>
    /// <exception cref="InterventionContractException">The document violates the contract.</exception>
    public static InterventionProfile Parse(ReadOnlyMemory<byte> utf8Json)
    {
        using JsonDocument document = StrictJson.Parse(utf8Json, "profile");
        JsonElement root = document.RootElement;

        // The schema version is checked first, so a future document fails on its version, not on a field it added.
        if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("schema_version", out JsonElement version))
        {
            if (version.ValueKind != JsonValueKind.Number || !version.TryGetInt32(out int v) || v is < CurrentSchemaVersion or > MaxSchemaVersion)
            {
                throw new InterventionContractException(
                    "schema_version",
                    $"unsupported schema version {version.GetRawText()}; this library reads versions {CurrentSchemaVersion} and {MaxSchemaVersion}.");
            }
        }

        StrictObject o = StrictObject.Open(root, "", PropertyNames);
        int schemaVersion = o.GetInt32("schema_version");
        string id = o.GetString("id");
        string? description = o.GetOptionalString("description");
        string baseSha = o.GetString("base_model_sha256");
        InterventionOperatorKind operatorKind = schemaVersion == CurrentSchemaVersion
            ? InterventionNames.ParseOperator(o.GetString("operator"), "operator")
            : InterventionNames.ParseOperatorV2(o.GetString("operator"), "operator");

        List<InterventionSite> sites = [];
        int index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("sites"), "sites"))
        {
            if (index >= MaxSites)
            {
                throw new InterventionContractException("sites", $"more than {MaxSites} sites.");
            }

            sites.Add(InterventionSite.Read(item, StrictJson.Index("sites", index++)));
        }

        return new InterventionProfile(id, baseSha, operatorKind, sites, description, schemaVersion);
    }

    /// <summary>
    /// The canonical <c>profile.json</c> bytes: UTF-8 without byte order mark, two-space indentation, LF line ends, a
    /// final LF, properties in the documented order, sites sorted, floats in shortest round-trip form. Equal profiles
    /// serialize to equal bytes.
    /// </summary>
    public byte[] Serialize() => StrictJson.Write(writer =>
    {
        writer.WriteStartObject();
        writer.WriteNumber("schema_version", SchemaVersion);
        writer.WriteString("id", Id);
        if (Description is not null)
        {
            writer.WriteString("description", Description);
        }

        writer.WriteString("base_model_sha256", BaseModelSha256);
        writer.WriteString("operator", InterventionNames.Of(Operator));
        writer.WriteStartArray("sites");
        foreach (InterventionSite site in Sites)
        {
            site.Write(writer);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    });

    /// <summary>Whether this profile applies to the base graph with SHA-256 <paramref name="baseModelSha256"/>.</summary>
    public bool AppliesTo(string baseModelSha256) => string.Equals(BaseModelSha256, baseModelSha256, StringComparison.Ordinal);

    /// <summary>
    /// Checks that this profile applies to the base graph with SHA-256 <paramref name="baseModelSha256"/>. Call it
    /// before loading anything, so a profile for another graph never reaches a session.
    /// </summary>
    /// <exception cref="InterventionMismatchException">The profile names another base graph.</exception>
    public void VerifyBaseModel(string baseModelSha256)
    {
        InterventionHashes.RequireSha256(baseModelSha256, nameof(baseModelSha256));
        if (!AppliesTo(baseModelSha256))
        {
            throw new InterventionMismatchException(
                $"Profile '{Id}' applies to base graph {BaseModelSha256}, not {baseModelSha256}.");
        }
    }

    /// <summary>Checks <see cref="VerifyBaseModel(string)"/> against the SHA-256 of the base graph file at <paramref name="baseModelPath"/>.</summary>
    /// <exception cref="InterventionMismatchException">The profile names another base graph.</exception>
    public void VerifyBaseModelFile(string baseModelPath) => VerifyBaseModel(InterventionHashes.ComputeFileSha256(baseModelPath));

    /// <summary>The site at <paramref name="layer"/> and <paramref name="target"/>, or <see langword="null"/>.</summary>
    public InterventionSite? FindSite(int layer, InterventionTarget target)
    {
        foreach (InterventionSite site in Sites)
        {
            if (site.Layer == layer && site.Target == target)
            {
                return site;
            }
        }

        return null;
    }

    internal static int CompareSites(InterventionSite a, InterventionSite b)
    {
        int c = a.Layer.CompareTo(b.Layer);
        return c != 0 ? c : a.Target.CompareTo(b.Target);
    }
}
