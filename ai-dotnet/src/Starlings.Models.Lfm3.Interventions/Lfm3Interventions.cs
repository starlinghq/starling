using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Interventions;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Lfm3.Interventions;

/// <summary>
/// Loads LFM3-1.7B with a derived intervention graph and runs requests with a per-request
/// <see cref="InterventionSelection"/>. The derived graph is a cache entry written by the Python compiler
/// (<c>docs/interventions/cache.md</c>): the original weights, hard-linked, plus a low-rank directional projection at the
/// profile's residual-stream sites and one float32 strength input. Requests change only that input.
/// </summary>
/// <remarks>
/// <para>Without an intervention, load the original graph with <see cref="Lfm3Model.Create(Lfm3ModelFiles, Lfm3Options?)"/>:
/// nothing here runs and nothing costs more. A selection on such a model is rejected.</para>
/// <para>The strength is fixed when a sequence is created and passed on every native call of that sequence; a request
/// cannot change it midway. No selection on an intervention graph means strength 0, the original model's output.</para>
/// <para>Which caller gets which profile is the host application's decision; this library ships the mechanism only.</para>
/// </remarks>
public static class Lfm3Interventions
{
    /// <summary>The site mapping of the fp32 graph this integration accepts (model-pipeline <c>injection_sites.py</c>).</summary>
    public const string Fp32SiteMappingVersion = "lfm3-1.7b/fp32/1";

    /// <summary>
    /// Opens the cache entry in <paramref name="artifactDirectory"/> (<see cref="InterventionArtifactCache.OpenEntry"/>:
    /// the directory name must be the key recomputed from its manifest), checks it against the LFM3 graph of
    /// <paramref name="files"/> and loads it.
    /// </summary>
    /// <param name="files">The verified original files; their precision selects the expected base graph.</param>
    /// <param name="artifactDirectory">The cache entry directory (<c>&lt;cache&gt;/lfm3/interventions/&lt;key&gt;</c>).</param>
    /// <param name="options">Load options. <see cref="OnnxSessionOptions.KnownModelSha256"/> is replaced by the derived graph's verified SHA-256, whatever the options set; an <see cref="OnnxSessionOptions.ExpectedModelSha256"/> equal to the original graph's SHA-256 is dropped (the derived graph was checked against it), any other value is kept and must match the derived graph.</param>
    /// <param name="verification">How much of the artifact to hash; <see cref="InterventionArtifactVerification.Full"/> also hashes the base files of <paramref name="files"/>.</param>
    /// <exception cref="InterventionMismatchException">The artifact was compiled for another graph, or a file does not match its manifest.</exception>
    /// <exception cref="InterventionContractException">The manifest or profile violates the contract, or declares inputs this integration does not know.</exception>
    public static Lfm3Model CreateModel(
        Lfm3ModelFiles files,
        string artifactDirectory,
        Lfm3Options? options = null,
        InterventionArtifactVerification verification = InterventionArtifactVerification.Fast)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentException.ThrowIfNullOrEmpty(artifactDirectory);
        InterventionArtifact artifact = InterventionArtifactCache.OpenEntry(artifactDirectory, verification, files.Snapshot.Directory);
        return CreateModel(files, artifact, options);
    }

    /// <summary>Checks an opened derived artifact against the LFM3 graph of <paramref name="files"/> and loads it.</summary>
    /// <exception cref="InterventionMismatchException">The artifact was compiled for another graph.</exception>
    /// <exception cref="InterventionContractException">The artifact declares inputs or sites this integration does not know.</exception>
    public static Lfm3Model CreateModel(Lfm3ModelFiles files, InterventionArtifact artifact, Lfm3Options? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(artifact);
        if (files.ModelSha256 is { } verified)
        {
            // The snapshot hashed the original graph: it must be the locked one (it always is for a verified snapshot).
            if (!string.Equals(verified, BaseModelSha256(files.Precision), StringComparison.Ordinal))
            {
                throw new InterventionMismatchException(
                    $"The original graph has SHA-256 {verified}; the LFM3 {files.Precision} lock records {BaseModelSha256(files.Precision)}.");
            }
        }

        CheckArtifact(artifact, files.Precision);
        options ??= new Lfm3Options();
        // The session's digest is the derived graph's, which InterventionArtifact.Open hashed at every level, never the
        // original graph's that Lfm3Model.Create(files) would pass.
        // A caller that reuses options pinning the original graph's digest (ExpectedModelSha256) would otherwise fail the
        // derived load with an integrity error: the manifest has just tied the derived graph to exactly that original.
        string? expected = options.Session.ExpectedModelSha256;
        if (string.Equals(expected, artifact.Manifest.BaseModelSha256, StringComparison.Ordinal))
        {
            expected = null;
        }

        options = options with
        {
            Session = options.Session with { KnownModelSha256 = artifact.Manifest.DerivedGraph.Sha256, ExpectedModelSha256 = expected },
        };
        var intervention = new Lfm3GraphIntervention(artifact.Manifest.ProfileId, artifact.Manifest.StrengthInput.Name, artifact);
        Lfm3Model model = Lfm3Model.Create(artifact.GraphPath, files.Precision, options, intervention);
        try
        {
            _ = model.GetInterventionDiagnostics();
            return model;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>As <see cref="CreateModel(Lfm3ModelFiles, string, Lfm3Options?, InterventionArtifactVerification)"/>, with the tokenizer; the returned chat model owns the graph.</summary>
    public static Lfm3ChatModel CreateChatModel(
        Lfm3ModelFiles files,
        string artifactDirectory,
        Lfm3Options? options = null,
        InterventionArtifactVerification verification = InterventionArtifactVerification.Fast)
    {
        ArgumentNullException.ThrowIfNull(files);
        Tokenizers.Tokenizer tokenizer = Tokenizers.Tokenizer.FromFile(files.TokenizerPath);
        return new Lfm3ChatModel(CreateModel(files, artifactDirectory, options, verification), tokenizer, ownsModel: true);
    }

    /// <summary>
    /// The SHA-256 of the original graph a derived artifact must have been compiled from: the lock's entry for
    /// <see cref="Lfm3ModelAssets.ModelFile"/> of <paramref name="precision"/>.
    /// </summary>
    public static string BaseModelSha256(Lfm3Precision precision) =>
        Lfm3ModelAssets.LockFor(precision).GetFile(Lfm3ModelAssets.ModelFile(precision)).Sha256;

    /// <summary>
    /// Checks a derived artifact's manifest against the LFM3 graph of <paramref name="precision"/> before any session
    /// exists: the base graph identity, every base artifact (graph and external data) against the precision's asset lock,
    /// the precision, the site mapping version (<see cref="Fp32SiteMappingVersion"/>), exactly one float32 strength input
    /// of shape [1] that is not a base input name, and sites inside the 24 decoder layers at hidden size 2048.
    /// </summary>
    /// <exception cref="InterventionMismatchException">The artifact was compiled for another graph or precision.</exception>
    /// <exception cref="InterventionContractException">The artifact declares inputs or sites this integration does not know.</exception>
    public static void CheckArtifact(InterventionArtifact artifact, Lfm3Precision precision)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        CheckManifest(artifact.Manifest, precision);
        foreach (InterventionSite site in artifact.Profile.Profile.Sites)
        {
            if (site.HiddenSize != Lfm3Checkpoint.HiddenSize || site.Layer >= Lfm3Checkpoint.Layers)
            {
                throw new InterventionContractException(
                    $"Profile site (layer {site.Layer}, hidden size {site.HiddenSize}) is not a residual-stream site of LFM3-1.7B " +
                    $"({Lfm3Checkpoint.Layers} layers, hidden size {Lfm3Checkpoint.HiddenSize}).");
            }
        }
    }

    internal static void CheckManifest(InterventionManifest manifest, Lfm3Precision precision)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        manifest.VerifyBaseModel(BaseModelSha256(precision));
        string expectedPrecision = precision == Lfm3Precision.Fp16 ? "fp16" : "fp32";
        if (!string.Equals(manifest.Precision, expectedPrecision, StringComparison.Ordinal))
        {
            throw new InterventionMismatchException($"The artifact was compiled at precision '{manifest.Precision}'; the LFM3 {precision} graph needs '{expectedPrecision}'.");
        }

        // The graph file holds only offsets into the external data, so its hash alone does not identify the weights: every
        // base artifact (graph and external data) must be the lock's file, and every locked graph file must be recorded.
        // Fast verification compares the linked weights with these entries by size only, so this is what ties them to
        // the lock's hashes.
        ModelAssetLock assetLock = Lfm3ModelAssets.LockFor(precision);
        var recorded = new HashSet<string>(StringComparer.Ordinal);
        foreach (InterventionBaseArtifact artifact in manifest.BaseArtifacts)
        {
            InterventionArtifactFile file = artifact.File;
            ModelAssetFile? locked = assetLock.Files.FirstOrDefault(f => string.Equals(f.Path, file.Path, StringComparison.Ordinal));
            if (locked is null || !string.Equals(locked.Sha256, file.Sha256, StringComparison.Ordinal) || locked.Size != file.Size)
            {
                throw new InterventionMismatchException(
                    $"Base artifact '{file.Path}' ({file.Sha256}, {file.Size} bytes) is not a file of the LFM3 {precision} lock {assetLock.Name}.");
            }

            recorded.Add(file.Path);
        }

        foreach (ModelAssetFile locked in assetLock.Files)
        {
            if (locked.Path.StartsWith(Lfm3ModelAssets.ModelFile(precision), StringComparison.Ordinal) && !recorded.Contains(locked.Path))
            {
                throw new InterventionMismatchException($"The artifact does not record the locked base file '{locked.Path}'.");
            }
        }

        // Site mappings are versioned; accept only the mapping this integration was verified with.
        string? mappingVersion = precision == Lfm3Precision.Fp32 ? Fp32SiteMappingVersion : null;
        if (!string.Equals(manifest.SiteMappingVersion, mappingVersion, StringComparison.Ordinal))
        {
            throw new InterventionContractException(
                mappingVersion is null
                    ? $"The LFM3 {precision} graph has no supported site mapping (the artifact names '{manifest.SiteMappingVersion}')."
                    : $"The artifact uses site mapping '{manifest.SiteMappingVersion}'; this integration supports '{mappingVersion}'.");
        }

        if (manifest.Inputs.Count != 1)
        {
            throw new InterventionContractException(
                $"The artifact declares {manifest.Inputs.Count} extra inputs ({string.Join(", ", manifest.Inputs.Select(i => i.Name))}); LFM3 accepts exactly one, the strength.");
        }

        InterventionGraphInput strength = manifest.StrengthInput;
        if (strength.DType != InterventionDType.Fp32 || strength.Shape.Count > 1 || (strength.Shape.Count == 1 && strength.Shape[0] != 1))
        {
            throw new InterventionContractException($"The strength input '{strength.Name}' must be float32 of shape [1] or [].");
        }

        if (strength.Shape.Count == 0)
        {
            // The sequence passes a [1] tensor; a scalar input would be rejected by ONNX Runtime at the first call.
            throw new InterventionContractException($"The strength input '{strength.Name}' is a scalar; LFM3 passes shape [1].");
        }

        if (strength.Name is Lfm3Model.InputIdsName or Lfm3Model.AttentionMaskName or Lfm3Model.PositionIdsName or Lfm3Model.LogitsToKeepName
            || strength.Name.StartsWith("past_", StringComparison.Ordinal))
        {
            throw new InterventionContractException($"The strength input '{strength.Name}' collides with an LFM3 graph input.");
        }

        foreach (InterventionSiteBinding site in manifest.Sites)
        {
            if (site.Layer >= Lfm3Checkpoint.Layers)
            {
                throw new InterventionContractException($"Site layer {site.Layer} is outside the {Lfm3Checkpoint.Layers} LFM3 decoder layers.");
            }
        }
    }

    /// <summary>
    /// Starts a request with <paramref name="selection"/> fixed for the sequence's lifetime. Null on an intervention graph
    /// means strength 0; null on the original graph is <see cref="Lfm3Model.CreateSequence()"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">A selection was given but the loaded graph has no intervention.</exception>
    /// <exception cref="InterventionMismatchException">The selection names another profile than the compiled one.</exception>
    public static Lfm3Sequence CreateSequence(this Lfm3Model model, InterventionSelection? selection)
    {
        ArgumentNullException.ThrowIfNull(model);
        return model.CreateSequence(Resolve(model, selection));
    }

    /// <summary>As <see cref="Lfm3ChatModel.Generate"/>, with <paramref name="selection"/> fixed for the request.</summary>
    /// <exception cref="InvalidOperationException">A selection was given but the loaded graph has no intervention.</exception>
    /// <exception cref="InterventionMismatchException">The selection names another profile than the compiled one.</exception>
    public static GenerationResult Generate(
        this Lfm3ChatModel chat, IReadOnlyList<int> promptIds, InterventionSelection? selection, GenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chat);
        return chat.GenerateWithStrength(promptIds, options, Resolve(chat.Model, selection), cancellationToken);
    }

    /// <summary>As <see cref="Lfm3ChatModel.GenerateStreaming"/>, with <paramref name="selection"/> fixed for the request. Arguments, including the selection, are checked when this method is called.</summary>
    /// <exception cref="InvalidOperationException">A selection was given but the loaded graph has no intervention.</exception>
    /// <exception cref="InterventionMismatchException">The selection names another profile than the compiled one.</exception>
    public static IEnumerable<GenerationUpdate> GenerateStreaming(
        this Lfm3ChatModel chat, IReadOnlyList<int> promptIds, InterventionSelection? selection, GenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chat);
        return chat.GenerateStreamingCore(promptIds, options, null, Resolve(chat.Model, selection), cancellationToken);
    }

    /// <summary>As <see cref="Lfm3ChatModel.Chat"/>, with <paramref name="selection"/> fixed for the request.</summary>
    public static GenerationResult Chat(
        this Lfm3ChatModel chat, IReadOnlyList<ChatTurn> turns, InterventionSelection? selection, GenerationOptions? options = null,
        bool enableThinking = true, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chat);
        float? strength = Resolve(chat.Model, selection);
        return chat.GenerateWithStrength(chat.EncodePrompt(turns, enableThinking), options, strength, cancellationToken);
    }

    /// <summary>The verified derived artifact the model was loaded from; null for the original graph.</summary>
    public static InterventionArtifact? GetInterventionArtifact(this Lfm3Model model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return model.Intervention?.State as InterventionArtifact;
    }

    /// <summary>
    /// Intervention diagnostics of the loaded session: for a derived graph the original and loaded identities, the
    /// profile id and the strength input (<see cref="OnnxInterventionDiagnostics.ForDerivedGraph"/>); for the original
    /// graph, interventions off.
    /// </summary>
    public static OnnxInterventionDiagnostics GetInterventionDiagnostics(this Lfm3Model model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.GetInterventionArtifact() is not { } artifact)
        {
            return OnnxInterventionDiagnostics.ForOriginalGraph(model.Session);
        }

        InterventionManifest manifest = artifact.Manifest;
        return OnnxInterventionDiagnostics.ForDerivedGraph(
            model.Session, manifest.BaseModelSha256, manifest.DerivedGraph.Sha256, [manifest.ProfileId], [.. manifest.Inputs.Select(i => i.Name)]);
    }

    private static float? Resolve(Lfm3Model model, InterventionSelection? selection)
    {
        if (selection is null)
        {
            return null;
        }

        if (model.GetInterventionArtifact() is not { } artifact)
        {
            throw new InvalidOperationException(
                $"The loaded LFM3 graph has no intervention; selection '{selection.ProfileId}' needs a derived graph (Lfm3Interventions.CreateModel).");
        }

        selection.EnsureMatches(artifact.Profile.Profile);
        return selection.Strength;
    }
}
