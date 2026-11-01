using System.Text.Json;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>
/// Where a derived file's bytes live on the machine that wrote the entry (manifest <c>derived_files[].identity</c>):
/// the <c>stat</c> device number, inode number and modification time in nanoseconds since the Unix epoch, symlinks
/// followed. A hard link shares all three with its base file. The values are machine-local and not part of the cache
/// key; <see cref="InterventionArtifactVerification.Fast"/> compares them so that a replaced link or a weights file
/// rewritten in place fails without hashing the weights.
/// </summary>
public sealed class InterventionFileIdentity : IEquatable<InterventionFileIdentity>
{
    internal static readonly string[] PropertyNames = ["device", "inode", "mtime_ns"];

    /// <summary>Creates the record.</summary>
    /// <param name="device">Device number (<c>st_dev</c>), 0 or more.</param>
    /// <param name="inode">Inode number (<c>st_ino</c>), 0 or more.</param>
    /// <param name="modifiedNanoseconds">Modification time (<c>st_mtime_ns</c>), nanoseconds since 1970-01-01 UTC.</param>
    /// <exception cref="InterventionContractException">The device or inode number is negative.</exception>
    public InterventionFileIdentity(long device, long inode, long modifiedNanoseconds)
        : this(device, inode, modifiedNanoseconds, "")
    {
    }

    private InterventionFileIdentity(long device, long inode, long modifiedNanoseconds, string jsonPath)
    {
        if (device < 0)
        {
            throw new InterventionContractException(StrictObject.Join(jsonPath, "device"), $"device {device} is negative.");
        }

        if (inode < 0)
        {
            throw new InterventionContractException(StrictObject.Join(jsonPath, "inode"), $"inode {inode} is negative.");
        }

        Device = device;
        Inode = inode;
        ModifiedNanoseconds = modifiedNanoseconds;
    }

    /// <summary>Device number (<c>st_dev</c>, as glibc encodes it).</summary>
    public long Device { get; }

    /// <summary>Inode number (<c>st_ino</c>).</summary>
    public long Inode { get; }

    /// <summary>Modification time in nanoseconds since 1970-01-01 UTC (<c>st_mtime_ns</c>).</summary>
    public long ModifiedNanoseconds { get; }

    /// <inheritdoc/>
    public bool Equals(InterventionFileIdentity? other) =>
        other is not null && Device == other.Device && Inode == other.Inode && ModifiedNanoseconds == other.ModifiedNanoseconds;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as InterventionFileIdentity);

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Device, Inode, ModifiedNanoseconds);

    /// <inheritdoc/>
    public override string ToString() => $"device {Device} inode {Inode} mtime {ModifiedNanoseconds} ns";

    internal static InterventionFileIdentity Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        return new InterventionFileIdentity(o.GetInt64("device"), o.GetInt64("inode"), o.GetInt64("mtime_ns"), path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteNumber("device", Device);
        writer.WriteNumber("inode", Inode);
        writer.WriteNumber("mtime_ns", ModifiedNanoseconds);
        writer.WriteEndObject();
    }
}

/// <summary>What onnx.checker said about a derived graph (manifest <c>validation.onnx_checker</c>).</summary>
public enum InterventionCheckerResult
{
    /// <summary>JSON <c>passed</c>: onnx.checker accepted the derived graph.</summary>
    Passed,

    /// <summary>
    /// JSON <c>skipped_base_rejected</c>: onnx.checker already rejects the base graph (for example operators only ONNX
    /// Runtime registers in the default domain, such as LFM3's <c>SimplifiedLayerNormalization</c>), so it could not
    /// judge the derived graph; the compiler's structural checks and the ONNX Runtime load stand in for it.
    /// </summary>
    SkippedBaseRejected,
}

/// <summary>How the compiler validated the derived graph (manifest <c>validation</c>, optional in the schema).</summary>
public sealed class InterventionManifestValidation
{
    /// <summary>Longest <see cref="Detail"/>, in UTF-16 code units.</summary>
    public const int MaxDetailLength = 512;

    internal static readonly string[] PropertyNames = ["onnx_checker", "detail"];

    /// <summary>Creates the record.</summary>
    /// <param name="onnxChecker">What onnx.checker said.</param>
    /// <param name="detail">One line, 1 to <see cref="MaxDetailLength"/> characters without control characters, or null.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionManifestValidation(InterventionCheckerResult onnxChecker, string? detail = null)
        : this(onnxChecker, detail, "")
    {
    }

    private InterventionManifestValidation(InterventionCheckerResult onnxChecker, string? detail, string jsonPath)
    {
        if (!Enum.IsDefined(onnxChecker))
        {
            throw new InterventionContractException(StrictObject.Join(jsonPath, "onnx_checker"), $"unknown onnx_checker result {onnxChecker}.");
        }

        if (detail is not null && (detail.Length is 0 or > MaxDetailLength || detail.Any(char.IsControl)))
        {
            throw new InterventionContractException(
                StrictObject.Join(jsonPath, "detail"), $"expected 1 to {MaxDetailLength} characters without control characters.");
        }

        OnnxChecker = onnxChecker;
        Detail = detail;
    }

    /// <summary>What onnx.checker said about the derived graph.</summary>
    public InterventionCheckerResult OnnxChecker { get; }

    /// <summary>For <see cref="InterventionCheckerResult.SkippedBaseRejected"/>: the first line of the checker's message about the base graph.</summary>
    public string? Detail { get; }

    internal static string Of(InterventionCheckerResult value) => value switch
    {
        InterventionCheckerResult.Passed => "passed",
        InterventionCheckerResult.SkippedBaseRejected => "skipped_base_rejected",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    internal static InterventionManifestValidation Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        string result = o.GetString("onnx_checker");
        InterventionCheckerResult checker = result switch
        {
            "passed" => InterventionCheckerResult.Passed,
            "skipped_base_rejected" => InterventionCheckerResult.SkippedBaseRejected,
            _ => throw new InterventionContractException(
                o.Child("onnx_checker"), $"unknown onnx_checker result '{result}'; expected 'passed' or 'skipped_base_rejected'."),
        };
        return new InterventionManifestValidation(checker, o.GetOptionalString("detail"), path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("onnx_checker", Of(OnnxChecker));
        if (Detail is not null)
        {
            writer.WriteString("detail", Detail);
        }

        writer.WriteEndObject();
    }
}
