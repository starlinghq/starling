using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>
/// The compiler manifest of a derived intervention graph (<c>compiler-manifest.json</c>, schema version 1): what it was
/// compiled from (base artifacts, profile), how (compiler and site-mapping versions, opset, precision), what it adds
/// (declared inputs, resolved sites) and what it produced (derived graph and files, with their SHA-256). Immutable;
/// construction and <see cref="Parse(ReadOnlyMemory{byte})"/> validate every value. The format is specified in
/// <c>docs/interventions/profile-format.md</c>.
/// </summary>
public sealed class InterventionManifest
{
    /// <summary>The only schema version this library reads and writes.</summary>
    public const int CurrentSchemaVersion = 1;

    internal static readonly string[] PropertyNames =
    [
        "schema_version", "compiler_version", "site_mapping_version", "base_artifacts", "profile_id", "profile_sha256",
        "operator", "opset", "precision", "inputs", "sites", "derived_graph", "derived_files", "validation",
    ];

    /// <summary>Creates a manifest.</summary>
    /// <param name="compilerVersion">Version label of the compiler.</param>
    /// <param name="siteMappingVersion">Version label of the model-specific site mapping that resolved the sites.</param>
    /// <param name="baseArtifacts">Exactly one <see cref="InterventionArtifactRole.Graph"/> entry and any external data files; paths unique.</param>
    /// <param name="profileId">Id of the compiled profile.</param>
    /// <param name="profileSha256">SHA-256 of the compiled <c>profile.json</c> bytes.</param>
    /// <param name="operatorKind">The profile's operator.</param>
    /// <param name="opset">Default-domain ONNX opset of the derived graph, 1 to 1000.</param>
    /// <param name="precision">Precision name of the base graph (for example <c>fp32</c>).</param>
    /// <param name="inputs">Declared extra inputs: exactly one <see cref="InterventionInputRole.Strength"/> input, shape [] or [1]; names unique.</param>
    /// <param name="sites">One binding per profile site; no two share a layer and target, or a tensor.</param>
    /// <param name="derivedGraph">The derived <c>model.onnx</c>, relative to the derived artifact directory.</param>
    /// <param name="derivedFiles">
    /// Other derived files; paths unique. In a derived-artifact cache entry these are exactly the hard links of the base
    /// graph's external data, with the base artifact's SHA-256 and size (<c>docs/interventions/cache.md</c>,
    /// "Verification"; <see cref="InterventionArtifact.Open"/> rejects anything else). A compiler that writes a file of
    /// its own here needs a reader change and a compiler-version bump. Only these entries may carry an
    /// <see cref="InterventionArtifactFile.Identity"/>.
    /// </param>
    /// <param name="validation">How the compiler validated the derived graph, or null when not recorded.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionManifest(
        string compilerVersion,
        string siteMappingVersion,
        IEnumerable<InterventionBaseArtifact> baseArtifacts,
        string profileId,
        string profileSha256,
        InterventionOperatorKind operatorKind,
        int opset,
        string precision,
        IEnumerable<InterventionGraphInput> inputs,
        IEnumerable<InterventionSiteBinding> sites,
        InterventionArtifactFile derivedGraph,
        IEnumerable<InterventionArtifactFile> derivedFiles,
        InterventionManifestValidation? validation = null)
    {
        ArgumentNullException.ThrowIfNull(baseArtifacts);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(sites);
        ArgumentNullException.ThrowIfNull(derivedGraph);
        ArgumentNullException.ThrowIfNull(derivedFiles);

        if (!StrictJson.IsVersionLabel(compilerVersion))
        {
            throw new InterventionContractException("compiler_version", $"'{compilerVersion}' is not a version label ([A-Za-z0-9._+/-], at most 64).");
        }

        if (!StrictJson.IsVersionLabel(siteMappingVersion))
        {
            throw new InterventionContractException("site_mapping_version", $"'{siteMappingVersion}' is not a version label ([A-Za-z0-9._+/-], at most 64).");
        }

        if (!StrictJson.IsProfileId(profileId))
        {
            throw new InterventionContractException("profile_id", $"'{profileId}' is not a valid profile id.");
        }

        StrictJson.RequireSha256(profileSha256 ?? "", "profile_sha256");
        if (!Enum.IsDefined(operatorKind))
        {
            throw new InterventionContractException("operator", $"unknown operator {operatorKind}.");
        }

        if (opset is < 1 or > 1000)
        {
            throw new InterventionContractException("opset", $"opset {opset} is outside 1..1000.");
        }

        if (!StrictJson.IsPrecision(precision))
        {
            throw new InterventionContractException("precision", $"'{precision}' is not a precision name ([a-z0-9_], at most 32).");
        }

        InterventionBaseArtifact[] baseList = [.. baseArtifacts];
        InterventionBaseArtifact? graph = null;
        HashSet<string> basePaths = new(StringComparer.Ordinal);
        foreach (InterventionBaseArtifact artifact in baseList)
        {
            ArgumentNullException.ThrowIfNull(artifact, nameof(baseArtifacts));
            if (!basePaths.Add(artifact.File.Path))
            {
                throw new InterventionContractException("base_artifacts", $"path '{artifact.File.Path}' appears twice.");
            }

            if (artifact.File.Identity is not null)
            {
                throw new InterventionContractException("base_artifacts", $"base artifact '{artifact.File.Path}' carries an identity; only derived files do.");
            }

            if (artifact.Role == InterventionArtifactRole.Graph)
            {
                graph = graph is null
                    ? artifact
                    : throw new InterventionContractException("base_artifacts", "more than one artifact has role 'graph'.");
            }
        }

        if (graph is null)
        {
            throw new InterventionContractException("base_artifacts", "no artifact has role 'graph'.");
        }

        InterventionGraphInput[] inputList = [.. inputs];
        HashSet<string> inputNames = new(StringComparer.Ordinal);
        InterventionGraphInput? strength = null;
        foreach (InterventionGraphInput input in inputList)
        {
            ArgumentNullException.ThrowIfNull(input, nameof(inputs));
            if (!inputNames.Add(input.Name))
            {
                throw new InterventionContractException("inputs", $"input '{input.Name}' appears twice.");
            }

            if (input.Role == InterventionInputRole.Strength)
            {
                strength = strength is null
                    ? input
                    : throw new InterventionContractException("inputs", "more than one input has role 'strength'.");
            }
        }

        if (strength is null)
        {
            throw new InterventionContractException("inputs", "no input has role 'strength'.");
        }

        if (!(strength.Shape.Count == 0 || (strength.Shape.Count == 1 && strength.Shape[0] == 1)))
        {
            throw new InterventionContractException("inputs", $"the strength input '{strength.Name}' must have shape [] or [1].");
        }

        InterventionSiteBinding[] siteList = [.. sites];
        if (siteList.Length is 0 or > InterventionProfile.MaxSites)
        {
            throw new InterventionContractException("sites", $"{siteList.Length} sites; expected 1..{InterventionProfile.MaxSites}.");
        }

        foreach (InterventionSiteBinding site in siteList)
        {
            ArgumentNullException.ThrowIfNull(site, nameof(sites));
        }

        Array.Sort(siteList, CompareBindings);
        for (int i = 1; i < siteList.Length; i++)
        {
            if (CompareBindings(siteList[i - 1], siteList[i]) == 0)
            {
                throw new InterventionContractException("sites", $"two sites at layer {siteList[i].Layer}, target '{InterventionNames.Of(siteList[i].Target)}'.");
            }
        }

        // Each site is a distinct tensor of the base graph; two sites resolved to one tensor mean a wrong site mapping.
        HashSet<string> tensors = new(StringComparer.Ordinal);
        foreach (InterventionSiteBinding site in siteList)
        {
            if (!tensors.Add(site.Tensor))
            {
                throw new InterventionContractException("sites", $"tensor '{site.Tensor}' is bound to more than one site.");
            }
        }

        if (derivedGraph.Identity is not null)
        {
            throw new InterventionContractException("derived_graph", "the derived graph carries an identity; only derived files do.");
        }

        InterventionArtifactFile[] derivedList = [.. derivedFiles];
        HashSet<string> derivedPaths = new(StringComparer.Ordinal) { derivedGraph.Path };
        foreach (InterventionArtifactFile file in derivedList)
        {
            ArgumentNullException.ThrowIfNull(file, nameof(derivedFiles));
            if (!derivedPaths.Add(file.Path))
            {
                throw new InterventionContractException("derived_files", $"path '{file.Path}' appears twice (or equals derived_graph.path).");
            }
        }

        CompilerVersion = compilerVersion;
        SiteMappingVersion = siteMappingVersion;
        BaseArtifacts = new ReadOnlyCollection<InterventionBaseArtifact>(baseList);
        BaseGraph = graph.File;
        ProfileId = profileId;
        ProfileSha256 = profileSha256!;
        Operator = operatorKind;
        Opset = opset;
        Precision = precision;
        Inputs = new ReadOnlyCollection<InterventionGraphInput>(inputList);
        StrengthInput = strength;
        Sites = new ReadOnlyCollection<InterventionSiteBinding>(siteList);
        DerivedGraph = derivedGraph;
        DerivedFiles = new ReadOnlyCollection<InterventionArtifactFile>(derivedList);
        Validation = validation;
    }

    /// <summary>Version label of the compiler.</summary>
    public string CompilerVersion { get; }

    /// <summary>Version label of the site mapping.</summary>
    public string SiteMappingVersion { get; }

    /// <summary>The base graph and its external weight files, in document order.</summary>
    public IReadOnlyList<InterventionBaseArtifact> BaseArtifacts { get; }

    /// <summary>The base graph entry of <see cref="BaseArtifacts"/>.</summary>
    public InterventionArtifactFile BaseGraph { get; }

    /// <summary>SHA-256 of the base graph (<see cref="BaseGraph"/>); equals the profile's <see cref="InterventionProfile.BaseModelSha256"/>.</summary>
    public string BaseModelSha256 => BaseGraph.Sha256;

    /// <summary>Id of the compiled profile.</summary>
    public string ProfileId { get; }

    /// <summary>SHA-256 of the compiled <c>profile.json</c> bytes.</summary>
    public string ProfileSha256 { get; }

    /// <summary>The profile's operator.</summary>
    public InterventionOperatorKind Operator { get; }

    /// <summary>Default-domain ONNX opset of the derived graph.</summary>
    public int Opset { get; }

    /// <summary>Precision name of the base graph.</summary>
    public string Precision { get; }

    /// <summary>Extra inputs the derived graph declares, in document order.</summary>
    public IReadOnlyList<InterventionGraphInput> Inputs { get; }

    /// <summary>The strength input of <see cref="Inputs"/>.</summary>
    public InterventionGraphInput StrengthInput { get; }

    /// <summary>Resolved sites, sorted by layer, then target.</summary>
    public IReadOnlyList<InterventionSiteBinding> Sites { get; }

    /// <summary>The derived <c>model.onnx</c>.</summary>
    public InterventionArtifactFile DerivedGraph { get; }

    /// <summary>Other derived files, in document order.</summary>
    public IReadOnlyList<InterventionArtifactFile> DerivedFiles { get; }

    /// <summary>
    /// How the compiler validated the derived graph, or null when the manifest does not record it. A value of
    /// <see cref="InterventionCheckerResult.SkippedBaseRejected"/> means onnx.checker could not judge the graph.
    /// </summary>
    public InterventionManifestValidation? Validation { get; }

    /// <summary>Parses and validates a <c>compiler-manifest.json</c> document (UTF-8, no byte order mark).</summary>
    /// <exception cref="InterventionContractException">The document violates the contract.</exception>
    public static InterventionManifest Parse(ReadOnlyMemory<byte> utf8Json)
    {
        using JsonDocument document = StrictJson.Parse(utf8Json, "compiler manifest");
        JsonElement root = document.RootElement;
        if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("schema_version", out JsonElement version))
        {
            if (version.ValueKind != JsonValueKind.Number || !version.TryGetInt32(out int v) || v != CurrentSchemaVersion)
            {
                throw new InterventionContractException(
                    "schema_version",
                    $"unsupported schema version {version.GetRawText()}; this library reads version {CurrentSchemaVersion}.");
            }
        }

        StrictObject o = StrictObject.Open(root, "", PropertyNames);
        _ = o.GetInt32("schema_version");

        List<InterventionBaseArtifact> baseArtifacts = [];
        int index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("base_artifacts"), "base_artifacts"))
        {
            baseArtifacts.Add(InterventionBaseArtifact.Read(item, StrictJson.Index("base_artifacts", index++)));
        }

        List<InterventionGraphInput> inputs = [];
        index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("inputs"), "inputs"))
        {
            inputs.Add(InterventionGraphInput.Read(item, StrictJson.Index("inputs", index++)));
        }

        List<InterventionSiteBinding> sites = [];
        index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("sites"), "sites"))
        {
            if (index >= InterventionProfile.MaxSites)
            {
                throw new InterventionContractException("sites", $"more than {InterventionProfile.MaxSites} sites.");
            }

            sites.Add(InterventionSiteBinding.Read(item, StrictJson.Index("sites", index++)));
        }

        InterventionArtifactFile derivedGraph = InterventionArtifactFile.Read(
            StrictObject.Open(o.Required("derived_graph"), "derived_graph", "path", "sha256", "size"));

        List<InterventionArtifactFile> derivedFiles = [];
        index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("derived_files"), "derived_files"))
        {
            derivedFiles.Add(InterventionArtifactFile.Read(
                StrictObject.Open(item, StrictJson.Index("derived_files", index++), "path", "sha256", "size", "identity")));
        }

        InterventionManifestValidation? validation = o.Has("validation")
            ? InterventionManifestValidation.Read(o.Required("validation"), "validation")
            : null;

        return new InterventionManifest(
            o.GetString("compiler_version"),
            o.GetString("site_mapping_version"),
            baseArtifacts,
            o.GetString("profile_id"),
            o.GetString("profile_sha256"),
            InterventionNames.ParseOperator(o.GetString("operator"), "operator"),
            o.GetInt32("opset"),
            o.GetString("precision"),
            inputs,
            sites,
            derivedGraph,
            derivedFiles,
            validation);
    }

    /// <summary>Reads and validates the manifest at <paramref name="path"/>.</summary>
    /// <exception cref="InterventionContractException">The document violates the contract.</exception>
    public static InterventionManifest Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        FileInfo info = new(path);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"Intervention compiler manifest not found: {path}", path);
        }

        if (info.Length > StrictJson.MaxDocumentBytes)
        {
            throw new InterventionContractException($"{path} is {info.Length} bytes; the limit is {StrictJson.MaxDocumentBytes}.");
        }

        return Parse(File.ReadAllBytes(path));
    }

    /// <summary>
    /// The canonical <c>compiler-manifest.json</c> bytes (UTF-8 without byte order mark, two-space indentation, LF line
    /// ends, a final LF, properties in the documented order, sites sorted; arrays otherwise in document order).
    /// </summary>
    public byte[] Serialize() => StrictJson.Write(writer =>
    {
        writer.WriteStartObject();
        writer.WriteNumber("schema_version", CurrentSchemaVersion);
        writer.WriteString("compiler_version", CompilerVersion);
        writer.WriteString("site_mapping_version", SiteMappingVersion);
        writer.WriteStartArray("base_artifacts");
        foreach (InterventionBaseArtifact artifact in BaseArtifacts)
        {
            artifact.Write(writer);
        }

        writer.WriteEndArray();
        writer.WriteString("profile_id", ProfileId);
        writer.WriteString("profile_sha256", ProfileSha256);
        writer.WriteString("operator", InterventionNames.Of(Operator));
        writer.WriteNumber("opset", Opset);
        writer.WriteString("precision", Precision);
        writer.WriteStartArray("inputs");
        foreach (InterventionGraphInput input in Inputs)
        {
            input.Write(writer);
        }

        writer.WriteEndArray();
        writer.WriteStartArray("sites");
        foreach (InterventionSiteBinding site in Sites)
        {
            site.Write(writer);
        }

        writer.WriteEndArray();
        writer.WritePropertyName("derived_graph");
        DerivedGraph.Write(writer);
        writer.WriteStartArray("derived_files");
        foreach (InterventionArtifactFile file in DerivedFiles)
        {
            file.Write(writer);
        }

        writer.WriteEndArray();
        if (Validation is not null)
        {
            writer.WritePropertyName("validation");
            Validation.Write(writer);
        }

        writer.WriteEndObject();
    });

    /// <summary>
    /// Checks that this manifest was compiled from <paramref name="profile"/> with SHA-256
    /// <paramref name="profileSha256"/>: same id, same profile hash, same operator, the base graph the profile names,
    /// and exactly the profile's sites (layer and target).
    /// </summary>
    /// <exception cref="InterventionMismatchException">Any of these differ.</exception>
    public void VerifyProfile(InterventionProfile profile, string profileSha256)
    {
        ArgumentNullException.ThrowIfNull(profile);
        InterventionHashes.RequireSha256(profileSha256, nameof(profileSha256));
        if (!string.Equals(ProfileId, profile.Id, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"Manifest was compiled from profile '{ProfileId}', not '{profile.Id}'.");
        }

        if (!string.Equals(ProfileSha256, profileSha256, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"Manifest records profile SHA-256 {ProfileSha256}, the profile file has {profileSha256}.");
        }

        if (Operator != profile.Operator)
        {
            throw new InterventionMismatchException(
                $"Manifest operator '{InterventionNames.Of(Operator)}' differs from the profile's '{InterventionNames.Of(profile.Operator)}'.");
        }

        if (!profile.AppliesTo(BaseModelSha256))
        {
            throw new InterventionMismatchException(
                $"Manifest base graph {BaseModelSha256} differs from the profile's base graph {profile.BaseModelSha256}.");
        }

        bool sameSites = Sites.Count == profile.Sites.Count;
        for (int i = 0; sameSites && i < Sites.Count; i++)
        {
            sameSites = Sites[i].Layer == profile.Sites[i].Layer && Sites[i].Target == profile.Sites[i].Target;
        }

        if (!sameSites)
        {
            throw new InterventionMismatchException($"Manifest sites differ from the sites of profile '{profile.Id}'.");
        }
    }

    /// <summary>Checks <see cref="VerifyProfile(InterventionProfile, string)"/> against a loaded profile file.</summary>
    /// <exception cref="InterventionMismatchException">The manifest belongs to another profile or base graph.</exception>
    public void VerifyProfile(InterventionProfileFile profileFile)
    {
        ArgumentNullException.ThrowIfNull(profileFile);
        VerifyProfile(profileFile.Profile, profileFile.Sha256);
    }

    /// <summary>Checks that the base graph of this manifest has SHA-256 <paramref name="baseModelSha256"/>.</summary>
    /// <exception cref="InterventionMismatchException">The manifest was compiled for another base graph.</exception>
    public void VerifyBaseModel(string baseModelSha256)
    {
        InterventionHashes.RequireSha256(baseModelSha256, nameof(baseModelSha256));
        if (!string.Equals(BaseModelSha256, baseModelSha256, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"Manifest was compiled for base graph {BaseModelSha256}, not {baseModelSha256}.");
        }
    }

    /// <summary>Checks the derived graph and every derived file under <paramref name="derivedDirectory"/> (size, then SHA-256).</summary>
    /// <exception cref="InterventionMismatchException">A size or SHA-256 differs.</exception>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    public void VerifyDerivedFiles(string derivedDirectory)
    {
        DerivedGraph.Verify(derivedDirectory);
        foreach (InterventionArtifactFile file in DerivedFiles)
        {
            file.Verify(derivedDirectory);
        }
    }

    /// <summary>
    /// Checks every base artifact under <paramref name="baseDirectory"/> (size, then SHA-256). This hashes the external
    /// weight files, which can take seconds for a multi-gigabyte model.
    /// </summary>
    /// <exception cref="InterventionMismatchException">A size or SHA-256 differs.</exception>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    public void VerifyBaseArtifacts(string baseDirectory)
    {
        foreach (InterventionBaseArtifact artifact in BaseArtifacts)
        {
            artifact.File.Verify(baseDirectory);
        }
    }

    private static int CompareBindings(InterventionSiteBinding a, InterventionSiteBinding b)
    {
        int c = a.Layer.CompareTo(b.Layer);
        return c != 0 ? c : a.Target.CompareTo(b.Target);
    }
}
