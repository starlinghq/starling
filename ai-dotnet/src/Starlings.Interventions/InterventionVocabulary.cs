namespace Starlings.Interventions;

/// <summary>The intervention operators. V1 has exactly one; any other name in a document is rejected.</summary>
public enum InterventionOperatorKind
{
    /// <summary>
    /// Low-rank directional projection, JSON <c>directional_projection</c>:
    /// <c>y' = y - alpha * ((y Q^T) * s) Q</c> for a site tensor <c>y</c> of shape <c>[..., hidden]</c>, directions
    /// <c>Q</c> of shape <c>[rank, hidden]</c>, per-direction scales <c>s</c> of shape <c>[rank]</c> (element-wise), and
    /// the request-level strength <c>alpha</c>. With orthonormal rows of <c>Q</c>, <c>s = 1</c> and <c>alpha = 1</c> it
    /// removes the component of <c>y</c> in the span of <c>Q</c>; <c>alpha = 0</c> leaves <c>y</c> unchanged.
    /// </summary>
    DirectionalProjection,
}

/// <summary>Where in a model an intervention site sits, relative to its <see cref="InterventionSite.Layer"/>.</summary>
public enum InterventionTarget
{
    /// <summary>
    /// JSON <c>residual_post</c>: the residual stream after decoder layer <see cref="InterventionSite.Layer"/> (that
    /// layer's output, the next layer's input). A model's site mapping names the tensor that carries it.
    /// </summary>
    ResidualPost,
}

/// <summary>Element type of a site's directions and scales, and of declared graph inputs.</summary>
public enum InterventionDType
{
    /// <summary>JSON <c>float32</c>: IEEE 754 binary32, little-endian in files.</summary>
    Fp32,
}

/// <summary>What a declared extra input of a derived graph is for.</summary>
public enum InterventionInputRole
{
    /// <summary>JSON <c>strength</c>: the request-level strength <c>alpha</c> of the operator.</summary>
    Strength,
}

/// <summary>What a base artifact of a derived graph is.</summary>
public enum InterventionArtifactRole
{
    /// <summary>JSON <c>graph</c>: the base <c>model.onnx</c>; exactly one per manifest.</summary>
    Graph,

    /// <summary>JSON <c>external_data</c>: an external weight file the base graph references.</summary>
    ExternalData,
}

/// <summary>JSON spellings of the closed vocabularies. Parsing is ordinal and case-sensitive.</summary>
public static class InterventionNames
{
    /// <summary>The JSON name of an operator.</summary>
    public static string Of(InterventionOperatorKind value) => value switch
    {
        InterventionOperatorKind.DirectionalProjection => "directional_projection",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    /// <summary>The JSON name of a target.</summary>
    public static string Of(InterventionTarget value) => value switch
    {
        InterventionTarget.ResidualPost => "residual_post",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    /// <summary>The JSON name of a dtype.</summary>
    public static string Of(InterventionDType value) => value switch
    {
        InterventionDType.Fp32 => "float32",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    /// <summary>The JSON name of an input role.</summary>
    public static string Of(InterventionInputRole value) => value switch
    {
        InterventionInputRole.Strength => "strength",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    /// <summary>The JSON name of an artifact role.</summary>
    public static string Of(InterventionArtifactRole value) => value switch
    {
        InterventionArtifactRole.Graph => "graph",
        InterventionArtifactRole.ExternalData => "external_data",
        _ => throw new ArgumentOutOfRangeException(nameof(value), value, null),
    };

    internal static InterventionOperatorKind ParseOperator(string value, string path) => value switch
    {
        "directional_projection" => InterventionOperatorKind.DirectionalProjection,
        _ => throw new InterventionContractException(path, $"unknown operator '{value}'; V1 knows only 'directional_projection'."),
    };

    internal static InterventionTarget ParseTarget(string value, string path) => value switch
    {
        "residual_post" => InterventionTarget.ResidualPost,
        _ => throw new InterventionContractException(path, $"unknown target '{value}'; V1 knows only 'residual_post'."),
    };

    internal static InterventionDType ParseDType(string value, string path) => value switch
    {
        "float32" => InterventionDType.Fp32,
        _ => throw new InterventionContractException(path, $"unsupported dtype '{value}'; V1 knows only 'float32'."),
    };

    internal static InterventionInputRole ParseInputRole(string value, string path) => value switch
    {
        "strength" => InterventionInputRole.Strength,
        _ => throw new InterventionContractException(path, $"unknown input role '{value}'; V1 knows only 'strength'."),
    };

    internal static InterventionArtifactRole ParseArtifactRole(string value, string path) => value switch
    {
        "graph" => InterventionArtifactRole.Graph,
        "external_data" => InterventionArtifactRole.ExternalData,
        _ => throw new InterventionContractException(path, $"unknown artifact role '{value}'; expected 'graph' or 'external_data'."),
    };
}
