using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>One file with its identity: a path relative to a directory, its SHA-256 and its size in bytes.</summary>
public sealed class InterventionArtifactFile
{
    /// <summary>Creates the record.</summary>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionArtifactFile(string path, string sha256, long size)
        : this(path, sha256, size, null, "")
    {
    }

    /// <summary>Creates the record of a derived file with its identity on the machine that wrote it.</summary>
    /// <param name="path">Path relative to the directory the record belongs to, forward slashes.</param>
    /// <param name="sha256">SHA-256 of the file bytes.</param>
    /// <param name="size">Size in bytes.</param>
    /// <param name="identity">Device, inode and modification time (only <c>derived_files</c> entries carry one), or null.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionArtifactFile(string path, string sha256, long size, InterventionFileIdentity? identity)
        : this(path, sha256, size, identity, "")
    {
    }

    internal InterventionArtifactFile(string path, string sha256, long size, string jsonPath)
        : this(path, sha256, size, null, jsonPath)
    {
    }

    private InterventionArtifactFile(string path, string sha256, long size, InterventionFileIdentity? identity, string jsonPath)
    {
        if (!StrictJson.IsRelativePath(path))
        {
            throw new InterventionContractException(StrictObject.Join(jsonPath, "path"), $"'{path}' is not a relative path with forward slashes and no '.' or '..' segment.");
        }

        StrictJson.RequireSha256(sha256 ?? "", StrictObject.Join(jsonPath, "sha256"));
        if (size < 0)
        {
            throw new InterventionContractException(StrictObject.Join(jsonPath, "size"), $"size {size} is negative.");
        }

        Path = path;
        Sha256 = sha256!;
        Size = size;
        Identity = identity;
    }

    /// <summary>Path relative to the directory the record belongs to, forward slashes.</summary>
    public string Path { get; }

    /// <summary>SHA-256 of the file bytes.</summary>
    public string Sha256 { get; }

    /// <summary>Size in bytes.</summary>
    public long Size { get; }

    /// <summary>
    /// For a <c>derived_files</c> entry: the file's device, inode and modification time when the compiler linked it,
    /// which <see cref="InterventionArtifactVerification.Fast"/> compares; null when not recorded, and always null for
    /// the derived graph and base artifacts.
    /// </summary>
    public InterventionFileIdentity? Identity { get; }

    /// <summary>Checks the file under <paramref name="directory"/> against this record (size first, then SHA-256).</summary>
    /// <exception cref="InterventionMismatchException">The size or SHA-256 differs.</exception>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    public void Verify(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        InterventionHashes.VerifyFile(StrictJson.Resolve(directory, Path), Sha256, Size);
    }

    internal static InterventionArtifactFile Read(StrictObject o) =>
        new(
            o.GetString("path"),
            o.GetString("sha256"),
            o.GetInt64("size"),
            o.Has("identity") ? InterventionFileIdentity.Read(o.Required("identity"), o.Child("identity")) : null,
            o.Path);

    internal void WriteProperties(Utf8JsonWriter writer)
    {
        writer.WriteString("path", Path);
        writer.WriteString("sha256", Sha256);
        writer.WriteNumber("size", Size);
        if (Identity is not null)
        {
            writer.WritePropertyName("identity");
            Identity.Write(writer);
        }
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        WriteProperties(writer);
        writer.WriteEndObject();
    }
}

/// <summary>A base artifact a derived graph depends on: the base graph or one of its external weight files.</summary>
public sealed class InterventionBaseArtifact
{
    /// <summary>Creates the record.</summary>
    /// <param name="role">What the file is.</param>
    /// <param name="file">Path relative to the base model directory, SHA-256 and size.</param>
    public InterventionBaseArtifact(InterventionArtifactRole role, InterventionArtifactFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        if (!Enum.IsDefined(role))
        {
            throw new InterventionContractException($"unknown artifact role {role}.");
        }

        Role = role;
        File = file;
    }

    /// <summary>What the file is.</summary>
    public InterventionArtifactRole Role { get; }

    /// <summary>Path relative to the base model directory, SHA-256 and size.</summary>
    public InterventionArtifactFile File { get; }

    internal static readonly string[] PropertyNames = ["role", "path", "sha256", "size"];

    internal static InterventionBaseArtifact Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        InterventionArtifactRole role = InterventionNames.ParseArtifactRole(o.GetString("role"), o.Child("role"));
        return new InterventionBaseArtifact(role, InterventionArtifactFile.Read(o));
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("role", InterventionNames.Of(Role));
        File.WriteProperties(writer);
        writer.WriteEndObject();
    }
}

/// <summary>An extra input the derived graph declares beyond the base graph's inputs.</summary>
public sealed class InterventionGraphInput
{
    /// <summary>Largest accepted element count of a declared input (2^31 - 1).</summary>
    public const long MaxElementCount = int.MaxValue;

    private readonly long[] _shape;

    /// <summary>Creates the record.</summary>
    /// <param name="name">ONNX input name.</param>
    /// <param name="role">What the input carries.</param>
    /// <param name="dtype">Element type.</param>
    /// <param name="shape">Fixed shape; empty for a scalar. Every dimension is non-negative.</param>
    public InterventionGraphInput(string name, InterventionInputRole role, InterventionDType dtype, IEnumerable<long> shape)
        : this(name, role, dtype, [.. shape ?? throw new ArgumentNullException(nameof(shape))], "")
    {
    }

    private InterventionGraphInput(string name, InterventionInputRole role, InterventionDType dtype, long[] shape, string path)
    {
        if (!StrictJson.IsTensorName(name))
        {
            throw new InterventionContractException(StrictObject.Join(path, "name"), "input name is empty, too long or has a control character.");
        }

        if (!Enum.IsDefined(role) || !Enum.IsDefined(dtype))
        {
            throw new InterventionContractException(path, "unknown role or dtype.");
        }

        if (shape.Length > 8)
        {
            throw new InterventionContractException(StrictObject.Join(path, "shape"), $"rank {shape.Length} exceeds 8.");
        }

        long count = 1;
        foreach (long dim in shape)
        {
            if (dim < 0)
            {
                throw new InterventionContractException(StrictObject.Join(path, "shape"), $"dimension {dim} is negative; declared inputs have fixed shapes.");
            }

            // Bounded so ElementCount never wraps (an unchecked product of large dimensions can wrap to any value).
            count = dim == 0 || count <= MaxElementCount / dim
                ? count * dim
                : throw new InterventionContractException(StrictObject.Join(path, "shape"), $"more than {MaxElementCount} elements.");
        }

        Name = name;
        Role = role;
        DType = dtype;
        _shape = shape;
        Shape = new ReadOnlyCollection<long>(_shape);
    }

    /// <summary>ONNX input name.</summary>
    public string Name { get; }

    /// <summary>What the input carries.</summary>
    public InterventionInputRole Role { get; }

    /// <summary>Element type.</summary>
    public InterventionDType DType { get; }

    /// <summary>Fixed shape; empty for a scalar.</summary>
    public IReadOnlyList<long> Shape { get; }

    /// <summary>Number of elements (1 for a scalar).</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (long dim in _shape)
            {
                count *= dim;
            }

            return count;
        }
    }

    internal static readonly string[] PropertyNames = ["name", "role", "dtype", "shape"];

    internal static InterventionGraphInput Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        string name = o.GetString("name");
        InterventionInputRole role = InterventionNames.ParseInputRole(o.GetString("role"), o.Child("role"));
        InterventionDType dtype = InterventionNames.ParseDType(o.GetString("dtype"), o.Child("dtype"));
        List<long> shape = [];
        int index = 0;
        foreach (JsonElement dim in StrictJson.GetArray(o.Required("shape"), o.Child("shape")))
        {
            string dimPath = StrictJson.Index(o.Child("shape"), index++);
            if (dim.ValueKind != JsonValueKind.Number || !dim.TryGetInt64(out long value))
            {
                throw new InterventionContractException(dimPath, "expected an integer dimension.");
            }

            shape.Add(value);
        }

        return new InterventionGraphInput(name, role, dtype, [.. shape], path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("name", Name);
        writer.WriteString("role", InterventionNames.Of(Role));
        writer.WriteString("dtype", InterventionNames.Of(DType));
        writer.WriteStartArray("shape");
        foreach (long dim in _shape)
        {
            writer.WriteNumberValue(dim);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }
}

/// <summary>
/// Where the compiler applied a site: the site's layer and target, the base graph tensor it resolved to, and (schema
/// version 2) the profiles applied there.
/// </summary>
public sealed class InterventionSiteBinding
{
    private static readonly IReadOnlyList<int> NoProfiles = new ReadOnlyCollection<int>([]);

    /// <summary>Creates the record of a schema version 1 site (the manifest's one profile).</summary>
    /// <param name="layer">Zero-based decoder layer index.</param>
    /// <param name="target">Where relative to the layer.</param>
    /// <param name="tensor">Name of the base graph tensor the site mapping resolved the site to.</param>
    public InterventionSiteBinding(int layer, InterventionTarget target, string tensor)
        : this(layer, target, tensor, null, "")
    {
    }

    /// <summary>Creates the record of a schema version 2 site.</summary>
    /// <param name="layer">Zero-based decoder layer index.</param>
    /// <param name="target">Where relative to the layer.</param>
    /// <param name="tensor">Name of the base graph tensor.</param>
    /// <param name="profiles">
    /// Strength indices of the profiles applied here, strictly ascending, at least one. They are applied in this order:
    /// the first reads the base tensor, each later one the previous one's output.
    /// </param>
    public InterventionSiteBinding(int layer, InterventionTarget target, string tensor, IEnumerable<int> profiles)
        : this(layer, target, tensor, [.. profiles ?? throw new ArgumentNullException(nameof(profiles))], "")
    {
    }

    private InterventionSiteBinding(int layer, InterventionTarget target, string tensor, int[]? profiles, string path)
    {
        if (layer < 0)
        {
            throw new InterventionContractException(StrictObject.Join(path, "layer"), $"layer {layer} is negative.");
        }

        if (!Enum.IsDefined(target))
        {
            throw new InterventionContractException(StrictObject.Join(path, "target"), $"unknown target {target}.");
        }

        if (!StrictJson.IsTensorName(tensor))
        {
            throw new InterventionContractException(StrictObject.Join(path, "tensor"), "tensor name is empty, too long or has a control character.");
        }

        if (profiles is not null)
        {
            bool ascending = profiles.Length > 0 && profiles[0] >= 0;
            for (int i = 1; ascending && i < profiles.Length; i++)
            {
                ascending = profiles[i - 1] < profiles[i];
            }

            if (!ascending)
            {
                throw new InterventionContractException(StrictObject.Join(path, "profiles"), "expected one or more strength indices in strictly ascending order.");
            }
        }

        Layer = layer;
        Target = target;
        Tensor = tensor;
        Profiles = profiles is null ? NoProfiles : new ReadOnlyCollection<int>(profiles);
    }

    /// <summary>Zero-based decoder layer index.</summary>
    public int Layer { get; }

    /// <summary>Where relative to the layer.</summary>
    public InterventionTarget Target { get; }

    /// <summary>Name of the base graph tensor.</summary>
    public string Tensor { get; }

    /// <summary>
    /// Schema version 2: strength indices of the profiles applied at this site, in application order (ascending). Empty
    /// for a schema version 1 site, where the manifest's one profile applies.
    /// </summary>
    public IReadOnlyList<int> Profiles { get; }

    internal static readonly string[] PropertyNames = ["layer", "target", "tensor"];

    internal static readonly string[] PropertyNamesV2 = ["layer", "target", "tensor", "profiles"];

    internal static InterventionSiteBinding Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        return new InterventionSiteBinding(
            o.GetInt32("layer"),
            InterventionNames.ParseTarget(o.GetString("target"), o.Child("target")),
            o.GetString("tensor"),
            null,
            path);
    }

    internal static InterventionSiteBinding ReadV2(JsonElement element, string path, int profileCount)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNamesV2);
        List<int> profiles = [];
        int index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("profiles"), o.Child("profiles")))
        {
            string itemPath = StrictJson.Index(o.Child("profiles"), index++);
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int value) || value < 0 || value >= profileCount
                || item.GetRawText().AsSpan().IndexOfAny('.', 'e', 'E') >= 0)
            {
                throw new InterventionContractException(itemPath, $"expected an integer in [0, {profileCount - 1}].");
            }

            profiles.Add(value);
        }

        return new InterventionSiteBinding(
            o.GetInt32("layer"),
            InterventionNames.ParseTarget(o.GetString("target"), o.Child("target")),
            o.GetString("tensor"),
            [.. profiles],
            path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteNumber("layer", Layer);
        writer.WriteString("target", InterventionNames.Of(Target));
        writer.WriteString("tensor", Tensor);
        if (Profiles.Count > 0)
        {
            writer.WriteStartArray("profiles");
            foreach (int index in Profiles)
            {
                writer.WriteNumberValue(index);
            }

            writer.WriteEndArray();
        }

        writer.WriteEndObject();
    }
}

/// <summary>One compiled profile of a schema version 2 manifest; its position in <see cref="InterventionManifest.Profiles"/> is its strength index.</summary>
public sealed class InterventionManifestProfile
{
    /// <summary>Creates the record.</summary>
    /// <param name="id">The profile's id.</param>
    /// <param name="sha256">SHA-256 of the compiled <c>profile.json</c> bytes.</param>
    /// <param name="operatorKind">The profile's operator.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionManifestProfile(string id, string sha256, InterventionOperatorKind operatorKind)
        : this(id, sha256, operatorKind, "")
    {
    }

    private InterventionManifestProfile(string id, string sha256, InterventionOperatorKind operatorKind, string path)
    {
        if (!StrictJson.IsProfileId(id))
        {
            throw new InterventionContractException(StrictObject.Join(path, "id"), $"'{id}' is not a valid profile id.");
        }

        StrictJson.RequireSha256(sha256 ?? "", StrictObject.Join(path, "sha256"));
        if (!Enum.IsDefined(operatorKind))
        {
            throw new InterventionContractException(StrictObject.Join(path, "operator"), $"unknown operator {operatorKind}.");
        }

        Id = id;
        Sha256 = sha256!;
        Operator = operatorKind;
    }

    /// <summary>The profile's id.</summary>
    public string Id { get; }

    /// <summary>SHA-256 of the compiled <c>profile.json</c> bytes.</summary>
    public string Sha256 { get; }

    /// <summary>The profile's operator.</summary>
    public InterventionOperatorKind Operator { get; }

    internal static readonly string[] PropertyNames = ["id", "sha256", "operator"];

    internal static InterventionManifestProfile Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        return new InterventionManifestProfile(
            o.GetString("id"),
            o.GetString("sha256"),
            InterventionNames.ParseOperatorV2(o.GetString("operator"), o.Child("operator")),
            path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("id", Id);
        writer.WriteString("sha256", Sha256);
        writer.WriteString("operator", InterventionNames.Of(Operator));
        writer.WriteEndObject();
    }
}
