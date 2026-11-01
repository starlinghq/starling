namespace Starlings.Runtime.Onnx;

/// <summary>
/// Intervention identity of one loaded session: which original model it stands for, which graph was actually loaded,
/// whether that graph is a derived intervention graph, the profiles compiled into it, and the extra inputs it declares.
/// A companion to <see cref="OnnxSessionDiagnostics"/>, built by the model package that loaded the session; it holds
/// plain data only, so this package knows nothing about profiles or manifests (<c>docs/interventions/</c>). A derived
/// graph loads through the ordinary <see cref="OnnxModelSession.Create(string, OnnxSessionOptions?)"/>; nothing in
/// <see cref="OnnxSessionOptions"/> changes. Status: experimental.
/// </summary>
public sealed record OnnxInterventionDiagnostics
{
    private OnnxInterventionDiagnostics()
    {
    }

    /// <summary>Whether the loaded graph is a derived intervention graph.</summary>
    public bool Enabled { get; private init; }

    /// <summary>
    /// SHA-256 of the original (base) model graph the session stands for: the base graph a derived graph was compiled
    /// from, or the loaded graph itself when interventions are off (<see langword="null"/> when the session did not
    /// hash it and no digest was supplied).
    /// </summary>
    public string? OriginalModelSha256 { get; private init; }

    /// <summary>The loaded model file (<see cref="OnnxSessionDiagnostics.ModelSource"/>).</summary>
    public required string LoadedModelSource { get; init; }

    /// <summary>
    /// SHA-256 of the graph that was loaded: the derived graph's recorded SHA-256 when interventions are on, else the
    /// session's digest (<see langword="null"/> when there is none).
    /// </summary>
    public string? LoadedModelSha256 { get; private init; }

    /// <summary>
    /// Where the session's digest, which equals <see cref="LoadedModelSha256"/>, came from
    /// (<see cref="OnnxSessionDiagnostics.ModelSha256Source"/>): <see cref="OnnxModelDigestSource.Computed"/> when the
    /// session hashed the loaded file, so the bytes ONNX Runtime read are proven; <see cref="OnnxModelDigestSource.Known"/>
    /// when the caller supplied it as <see cref="OnnxSessionOptions.KnownModelSha256"/>, so the file was not re-read and
    /// the value is only as good as the caller's own check (for example the manifest verification before loading);
    /// <see langword="null"/> when the session has no digest and <see cref="LoadedModelSha256"/> is only the caller's
    /// record.
    /// </summary>
    public OnnxModelDigestSource? LoadedModelSha256Source { get; private init; }

    /// <summary>Ids of the intervention profiles compiled into the loaded graph; empty when interventions are off.</summary>
    public IReadOnlyList<string> ProfileIds { get; private init; } = [];

    /// <summary>The graph's intervention inputs as ONNX Runtime reports them, in graph order; empty when off.</summary>
    public IReadOnlyList<OnnxTensorInfo> InterventionInputs { get; private init; } = [];

    /// <summary>The other graph inputs (those of the original model), in graph order.</summary>
    public IReadOnlyList<OnnxTensorInfo> ModelInputs { get; private init; } = [];

    /// <summary>
    /// Describes a session of an original graph (no intervention): <see cref="Enabled"/> is false, the original and
    /// loaded identities are both the session's digest, and every input is a model input.
    /// </summary>
    public static OnnxInterventionDiagnostics ForOriginalGraph(OnnxModelSession session)
    {
        ArgumentNullException.ThrowIfNull(session);
        OnnxSessionDiagnostics d = session.Diagnostics;
        return new OnnxInterventionDiagnostics
        {
            Enabled = false,
            OriginalModelSha256 = d.ModelSha256,
            LoadedModelSource = d.ModelSource,
            LoadedModelSha256 = d.ModelSha256,
            LoadedModelSha256Source = d.ModelSha256Source,
            ModelInputs = d.Inputs,
        };
    }

    /// <summary>
    /// Describes a session of a derived intervention graph and checks it against what the caller verified before
    /// loading (typically the compiler manifest): the session's digest, when it has one, must equal
    /// <paramref name="derivedGraphSha256"/>, and every name in <paramref name="interventionInputNames"/> must be a graph
    /// input. The remaining inputs are reported as <see cref="ModelInputs"/>; comparing them with the original model's
    /// inputs is the model package's job.
    /// </summary>
    /// <param name="session">The loaded derived graph.</param>
    /// <param name="originalModelSha256">SHA-256 of the base graph the derived graph was compiled from.</param>
    /// <param name="derivedGraphSha256">SHA-256 of the derived graph as recorded (and verified) before loading.</param>
    /// <param name="profileIds">Ids of the compiled profiles; at least one, unique, none empty.</param>
    /// <param name="interventionInputNames">The extra inputs the derived graph declares; at least one, unique.</param>
    /// <exception cref="ArgumentException">A digest is malformed, the two digests are equal, a list is empty or has duplicates, or a declared input is not a graph input.</exception>
    /// <exception cref="OnnxModelIntegrityException">The session's digest differs from <paramref name="derivedGraphSha256"/>.</exception>
    public static OnnxInterventionDiagnostics ForDerivedGraph(
        OnnxModelSession session,
        string originalModelSha256,
        string derivedGraphSha256,
        IReadOnlyCollection<string> profileIds,
        IReadOnlyCollection<string> interventionInputNames)
    {
        ArgumentNullException.ThrowIfNull(session);
        RequireSha256(originalModelSha256, nameof(originalModelSha256));
        RequireSha256(derivedGraphSha256, nameof(derivedGraphSha256));
        ArgumentNullException.ThrowIfNull(profileIds);
        ArgumentNullException.ThrowIfNull(interventionInputNames);
        if (string.Equals(originalModelSha256, derivedGraphSha256, StringComparison.Ordinal))
        {
            throw new ArgumentException("A derived graph cannot have the original graph's SHA-256.", nameof(derivedGraphSha256));
        }

        string[] ids = RequireDistinct(profileIds, nameof(profileIds));
        string[] names = RequireDistinct(interventionInputNames, nameof(interventionInputNames));

        OnnxSessionDiagnostics d = session.Diagnostics;
        if (d.ModelSha256 is { } loaded && !string.Equals(loaded, derivedGraphSha256, StringComparison.Ordinal))
        {
            throw new OnnxModelIntegrityException(
                $"Model {d.ModelSource} has SHA-256 {loaded}; the derived intervention graph was recorded as {derivedGraphSha256}.");
        }

        HashSet<string> declared = new(names, StringComparer.Ordinal);
        foreach (string name in names)
        {
            if (!d.Inputs.Any(i => string.Equals(i.Name, name, StringComparison.Ordinal)))
            {
                throw new ArgumentException(
                    $"Declared intervention input '{name}' is not an input of {d.ModelSource} (inputs: {string.Join(", ", d.Inputs.Select(i => i.Name))}).",
                    nameof(interventionInputNames));
            }
        }

        return new OnnxInterventionDiagnostics
        {
            Enabled = true,
            OriginalModelSha256 = originalModelSha256,
            LoadedModelSource = d.ModelSource,
            LoadedModelSha256 = derivedGraphSha256,
            LoadedModelSha256Source = d.ModelSha256Source,
            ProfileIds = ids,
            InterventionInputs = [.. d.Inputs.Where(i => declared.Contains(i.Name))],
            ModelInputs = [.. d.Inputs.Where(i => !declared.Contains(i.Name))],
        };
    }

    private static string[] RequireDistinct(IReadOnlyCollection<string> values, string paramName)
    {
        string[] list = [.. values];
        if (list.Length == 0)
        {
            throw new ArgumentException("At least one value is required.", paramName);
        }

        HashSet<string> seen = new(StringComparer.Ordinal);
        foreach (string value in list)
        {
            if (string.IsNullOrEmpty(value))
            {
                throw new ArgumentException("Values must not be null or empty.", paramName);
            }

            if (!seen.Add(value))
            {
                throw new ArgumentException($"'{value}' appears twice.", paramName);
            }
        }

        return list;
    }

    private static void RequireSha256(string value, string paramName)
    {
        ArgumentNullException.ThrowIfNull(value, paramName);
        if (value.Length != 64 || !value.All(c => char.IsAsciiDigit(c) || c is >= 'a' and <= 'f'))
        {
            throw new ArgumentException("Expected a SHA-256 as 64 lower-case hexadecimal characters.", paramName);
        }
    }
}
