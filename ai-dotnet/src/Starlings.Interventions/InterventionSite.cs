using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Interventions.Internal;

namespace Starlings.Interventions;

/// <summary>
/// One place in the base graph where the operator is applied: a decoder layer and a target relative to it, the hidden
/// size of the site tensor, the rank of the intervention, the directions file and its SHA-256, and the per-direction
/// scales. Immutable; the constructor validates every value.
/// </summary>
public sealed class InterventionSite
{
    /// <summary>Largest accepted hidden size (a guard against corrupt documents, not a model limit).</summary>
    public const int MaxHiddenSize = 1 << 20;

    private readonly float[] _scales;

    /// <summary>Creates a site.</summary>
    /// <param name="layer">Zero-based decoder layer index.</param>
    /// <param name="target">Where relative to the layer.</param>
    /// <param name="hiddenSize">Last dimension of the site tensor, 1 to <see cref="MaxHiddenSize"/>.</param>
    /// <param name="rank">Number of directions, 1 to <paramref name="hiddenSize"/>.</param>
    /// <param name="dtype">Element type of the directions file and the scales.</param>
    /// <param name="directionsFile">Path of the directions file, relative to the profile's directory, forward slashes.</param>
    /// <param name="directionsSha256">SHA-256 of the directions file.</param>
    /// <param name="scales">Per-direction scales, <paramref name="rank"/> finite values.</param>
    /// <exception cref="InterventionContractException">A value violates the contract.</exception>
    public InterventionSite(
        int layer,
        InterventionTarget target,
        int hiddenSize,
        int rank,
        InterventionDType dtype,
        string directionsFile,
        string directionsSha256,
        IEnumerable<float> scales)
        : this(layer, target, hiddenSize, rank, dtype, directionsFile, directionsSha256, [.. scales ?? throw new ArgumentNullException(nameof(scales))], "")
    {
    }

    private InterventionSite(
        int layer,
        InterventionTarget target,
        int hiddenSize,
        int rank,
        InterventionDType dtype,
        string directionsFile,
        string directionsSha256,
        float[] scales,
        string path)
    {
        if (layer < 0)
        {
            throw new InterventionContractException(StrictObject.Join(path, "layer"), $"layer {layer} is negative.");
        }

        if (!Enum.IsDefined(target))
        {
            throw new InterventionContractException(StrictObject.Join(path, "target"), $"unknown target {target}.");
        }

        if (hiddenSize is < 1 or > MaxHiddenSize)
        {
            throw new InterventionContractException(StrictObject.Join(path, "hidden_size"), $"hidden size {hiddenSize} is outside 1..{MaxHiddenSize}.");
        }

        if (rank < 1 || rank > hiddenSize)
        {
            throw new InterventionContractException(StrictObject.Join(path, "rank"), $"rank {rank} is outside 1..{hiddenSize} (the hidden size).");
        }

        if (!Enum.IsDefined(dtype))
        {
            throw new InterventionContractException(StrictObject.Join(path, "dtype"), $"unknown dtype {dtype}.");
        }

        if (!StrictJson.IsRelativePath(directionsFile))
        {
            throw new InterventionContractException(
                StrictObject.Join(path, "directions_file"),
                $"'{directionsFile}' is not a relative path with forward slashes and no '.' or '..' segment.");
        }

        StrictJson.RequireSha256(directionsSha256 ?? "", StrictObject.Join(path, "directions_sha256"));

        if (scales.Length != rank)
        {
            throw new InterventionContractException(StrictObject.Join(path, "scales"), $"{scales.Length} scales for rank {rank}.");
        }

        for (int i = 0; i < scales.Length; i++)
        {
            if (!float.IsFinite(scales[i]))
            {
                throw new InterventionContractException(StrictJson.Index(StrictObject.Join(path, "scales"), i), "scale is not finite.");
            }
        }

        Layer = layer;
        Target = target;
        HiddenSize = hiddenSize;
        Rank = rank;
        DType = dtype;
        DirectionsFile = directionsFile;
        DirectionsSha256 = directionsSha256!;
        _scales = scales;
        Scales = new ReadOnlyCollection<float>(_scales);
    }

    /// <summary>Zero-based decoder layer index.</summary>
    public int Layer { get; }

    /// <summary>Where relative to <see cref="Layer"/>.</summary>
    public InterventionTarget Target { get; }

    /// <summary>Last dimension of the site tensor.</summary>
    public int HiddenSize { get; }

    /// <summary>Number of directions (rows of <c>Q</c>).</summary>
    public int Rank { get; }

    /// <summary>Element type of the directions file and the scales.</summary>
    public InterventionDType DType { get; }

    /// <summary>Path of the directions file, relative to the profile's directory, forward slashes.</summary>
    public string DirectionsFile { get; }

    /// <summary>SHA-256 of the directions file.</summary>
    public string DirectionsSha256 { get; }

    /// <summary>Per-direction scales <c>s</c>, <see cref="Rank"/> values.</summary>
    public IReadOnlyList<float> Scales { get; }

    /// <summary>Byte length the directions file must have: <c>rank * hidden_size * 4</c>.</summary>
    public long DirectionsByteLength => (long)Rank * HiddenSize * sizeof(float);

    internal static readonly string[] PropertyNames =
        ["layer", "target", "hidden_size", "rank", "dtype", "directions_file", "directions_sha256", "scales"];

    internal static InterventionSite Read(JsonElement element, string path)
    {
        StrictObject o = StrictObject.Open(element, path, PropertyNames);
        int layer = o.GetInt32("layer");
        InterventionTarget target = InterventionNames.ParseTarget(o.GetString("target"), o.Child("target"));
        int hiddenSize = o.GetInt32("hidden_size");
        int rank = o.GetInt32("rank");
        InterventionDType dtype = InterventionNames.ParseDType(o.GetString("dtype"), o.Child("dtype"));
        string directionsFile = o.GetString("directions_file");
        string directionsSha256 = o.GetString("directions_sha256");

        List<float> scales = [];
        string scalesPath = o.Child("scales");
        int index = 0;
        foreach (JsonElement item in StrictJson.GetArray(o.Required("scales"), scalesPath))
        {
            scales.Add(StrictJson.GetSingle(item, StrictJson.Index(scalesPath, index++)));
        }

        return new InterventionSite(layer, target, hiddenSize, rank, dtype, directionsFile, directionsSha256, [.. scales], path);
    }

    internal void Write(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteNumber("layer", Layer);
        writer.WriteString("target", InterventionNames.Of(Target));
        writer.WriteNumber("hidden_size", HiddenSize);
        writer.WriteNumber("rank", Rank);
        writer.WriteString("dtype", InterventionNames.Of(DType));
        writer.WriteString("directions_file", DirectionsFile);
        writer.WriteString("directions_sha256", DirectionsSha256);
        writer.WriteStartArray("scales");
        foreach (float scale in _scales)
        {
            writer.WriteNumberValue(scale);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }
}
