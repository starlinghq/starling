using System.Collections.Immutable;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Kev;

/// <summary>Immutable model configuration. Defaults are the profile values.</summary>
public sealed record KevOptions
{
    /// <summary>
    /// Session settings (provider, threads, digest check). <see cref="OnnxExecutionProvider.Cpu"/> runs the
    /// <c>onnx-fp32-cpu</c> profiles, <see cref="OnnxExecutionProvider.Cuda"/> (<c>use_tf32=0</c>, forced by
    /// Runtime.Onnx) the <c>onnx-fp32-cuda</c> profiles on the same bundle (<see cref="KevCapabilities"/>).
    /// </summary>
    public OnnxSessionOptions Session { get; init; } = new();

    /// <summary>
    /// Whether requests run their graph calls one request at a time. Null (default) serializes on CUDA and not on CPU:
    /// concurrent runs on one CUDA session returned wrong outputs for GLiNER2 and Qwen3.5 on ONNX Runtime 1.30, and the
    /// serialized mode is the CUDA profiles' contract. A call waiting for the gate stays cancellable.
    /// </summary>
    public bool? SerializeNativeCalls { get; init; }

    /// <summary>
    /// Calibration temperature. Null (default) uses the checkpoint's calibrated temperature (<c>head.pt</c>); 1.0 gives
    /// the raw probabilities, as upstream's <c>KEV_TEMPERATURE=1.0</c>.
    /// </summary>
    public double? Temperature { get; init; }

    /// <summary>
    /// Maximum state tokens, <c>&lt;state&gt;</c> included. Longer states are rejected with
    /// <see cref="KevInputTooLongException"/>, never truncated. Default and maximum
    /// <see cref="KevModelProfile.MaxStateTokens"/> (the validated context).
    /// </summary>
    public int MaxStateTokens { get; init; } = KevModelProfile.MaxStateTokens;

    /// <summary>Maximum tokens of one question's row (state plus branch). Default and maximum <see cref="KevModelProfile.MaxRowTokens"/>.</summary>
    public int MaxRowTokens { get; init; } = KevModelProfile.MaxRowTokens;

    /// <summary>
    /// Maximum tokens of one question's branch alone (<c>&lt;q&gt;</c> through <c>&lt;decide&gt;</c>, the state not
    /// counted), or null (default) for no branch limit beyond <see cref="MaxRowTokens"/>. A longer branch is over the
    /// limit like an over-long row (<see cref="KevTokenLimit.Branch"/>): thrown, or declined with
    /// <see cref="DeclineOverLongQuestions"/>. A budget, not a model limit: the validated context is
    /// <see cref="MaxRowTokens"/>. At most <see cref="KevModelProfile.MaxRowTokens"/>.
    /// </summary>
    public int? MaxBranchTokens { get; init; }

    /// <summary>
    /// What <see cref="KevDecisionModel.Decide(KevRequest, CancellationToken)"/> and
    /// <see cref="KevDecisionModel.Encode(KevRecord)"/> do with questions over <see cref="MaxRowTokens"/> or
    /// <see cref="MaxBranchTokens"/>. False (default): throw <see cref="KevInputTooLongException"/>, which lists every
    /// such question (<see cref="KevInputTooLongException.OverLongQuestions"/>). True: answer the other questions and
    /// list the over-long ones in <see cref="KevDecision.Declined"/> (and <see cref="KevEncoding.Declined"/>), all in
    /// one pass. A state over <see cref="MaxStateTokens"/> always throws. Nothing is truncated either way.
    /// </summary>
    public bool DeclineOverLongQuestions { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Session);
        if (Session.Provider is not (OnnxExecutionProvider.Cpu or OnnxExecutionProvider.Cuda))
        {
            throw new NotSupportedException($"The Kev profiles run on the CPU or the CUDA provider, not {Session.Provider}.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(MaxStateTokens, 2);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxStateTokens, KevModelProfile.MaxStateTokens);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxRowTokens, 2);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxRowTokens, KevModelProfile.MaxRowTokens);
        if (MaxBranchTokens is { } branch)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(branch, 1, nameof(MaxBranchTokens));
            ArgumentOutOfRangeException.ThrowIfGreaterThan(branch, KevModelProfile.MaxRowTokens, nameof(MaxBranchTokens));
        }
    }
}

/// <summary>
/// A Kev 1.0 decision model on ONNX Runtime: one state and typed questions in, a calibrated probability per option
/// out, as <c>kev.model.DecisionModel.probs</c> computes it on the fp32 path (<see cref="KevModelProfile.SourceRevision"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Pipeline.</b> The request is rendered and escaped as upstream does (<see cref="KevRequest.ToRecord"/>,
/// <see cref="KevText"/>), tokenized with the owned tokenizer and laid out as Kev's row form (<see cref="Encode"/>):
/// the state runs once through the backbone graph, then every question runs as its own causal row continuing from a
/// copy of the state's cache (attention keys and values, DeltaNet convolution and recurrent states), so questions
/// cannot influence one another. The pointer head reads the hidden states of each option's <c>&lt;/opt&gt;</c> and of
/// the question's <c>&lt;decide&gt;</c>.
/// </para>
/// <para>
/// <b>Providers.</b> The CPU provider runs the <c>onnx-fp32-cpu</c> profiles; the CUDA provider runs the same bundle as
/// the <c>onnx-fp32-cuda</c> profiles (<see cref="ProfileId"/>, <c>use_tf32=0</c>). The graph is the builder's CPU
/// export: inside a CUDA session ONNX Runtime keeps some of its nodes on the CPU provider, and the state cache passes
/// through host memory between rows.
/// </para>
/// <para>
/// <b>Limits.</b> States above <see cref="KevOptions.MaxStateTokens"/> are rejected
/// (<see cref="KevInputTooLongException"/>); questions whose row is above <see cref="KevOptions.MaxRowTokens"/> or
/// whose branch is above <see cref="KevOptions.MaxBranchTokens"/> are rejected together, or declined with
/// <see cref="KevOptions.DeclineOverLongQuestions"/>. Rows run one at a time. See <see cref="KevCapabilities"/>.
/// </para>
/// <para>
/// <b>State reuse.</b> <see cref="CreateState"/> runs a state once and keeps its cache in a <see cref="KevState"/>;
/// <see cref="Decide(KevState, IEnumerable{KeyValuePair{string, KevQuestion}}, CancellationToken)"/> answers any
/// number of question sets against it, with the same probabilities as
/// <see cref="Decide(KevRequest, CancellationToken)"/> on the same state and questions.
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Methods are synchronous and hold no per-call state on the model, so they are meant to
/// be callable concurrently (<see cref="KevCapabilities"/>, <c>concurrent_calls</c>). On CUDA requests run their graph
/// calls one request at a time by default (<see cref="SerializesNativeCalls"/>). Cancellation is checked before
/// each graph run and inside a run (between graph nodes); a cancelled call throws <see cref="OperationCanceledException"/>
/// and the model stays usable.
/// </para>
/// </remarks>
public sealed partial class KevDecisionModel : IDisposable
{
    /// <summary>Graph output: final hidden states, float [batch, sequence, hidden].</summary>
    public const string HiddenStatesName = "hidden_states";

    private const string InputIdsName = "input_ids";
    private const string AttentionMaskName = "attention_mask";
    private const string PositionIdsName = "position_ids";
    private const int AttentionHeadSize = 256;
    private const int LinearHeadSize = 128;

    private readonly OnnxModelSession _session;
    private readonly bool _ownsSession;
    private readonly KevEncoder _encoder;
    private readonly KevPointerHead _head;
    private readonly KevOptions _options;
    private readonly string[] _pastNames;
    private readonly string[] _presentNames;
    private readonly OrtValue[] _emptyPast;
    private readonly SemaphoreSlim? _runGate;
    private readonly bool _onCuda;
    private int _disposed;

    private KevDecisionModel(KevModelProfile profile, OnnxModelSession session, bool ownsSession, KevEncoder encoder, KevPointerHead head, KevOptions options)
    {
        Profile = profile;
        _session = session;
        _ownsSession = ownsSession;
        _encoder = encoder;
        _head = head;
        _options = options;
        (_pastNames, _presentNames, _emptyPast) = ValidateSignature(profile, session);
        _onCuda = session.Diagnostics.SessionProviders[0] == OnnxRuntimeInfo.CudaProviderName;
        _runGate = (options.SerializeNativeCalls ?? _onCuda) ? new SemaphoreSlim(1, 1) : null;
    }

    /// <summary>The profile.</summary>
    public KevModelProfile Profile { get; }

    /// <summary>Options in effect.</summary>
    public KevOptions Options => _options;

    /// <summary>The registry profile this model runs: <see cref="KevModelProfile.Id"/> on CPU, <see cref="KevModelProfile.CudaId"/> on CUDA.</summary>
    public string ProfileId => _onCuda ? Profile.CudaId : Profile.Id;

    /// <summary>
    /// True when this model runs one request's graph calls at a time (<see cref="KevOptions.SerializeNativeCalls"/>; the
    /// default on CUDA).
    /// </summary>
    public bool SerializesNativeCalls => _runGate is not null;

    /// <summary>The calibration temperature in effect.</summary>
    public double Temperature => _head.Temperature;

    /// <summary>
    /// True when <see cref="Temperature"/> is the checkpoint's calibrated temperature (<see cref="KevModelProfile.Temperature"/>):
    /// <see cref="KevOptions.Temperature"/> null or set to that value. Every answer repeats it (<see cref="KevAnswer.IsCalibrated"/>).
    /// </summary>
    public bool IsCalibrated => _head.Temperature == Profile.Temperature;

    /// <summary>
    /// Where this model runs: provider, CUDA device, profile, serialization, and the session's node placement when
    /// captured. Read from the session, so a consumer does not derive it from <see cref="OnnxSessionOptions"/>.
    /// </summary>
    public KevPlacement Placement => new(
        _onCuda ? OnnxExecutionProvider.Cuda : OnnxExecutionProvider.Cpu,
        _onCuda ? _options.Session.CudaDeviceId : null,
        ProfileId,
        SerializesNativeCalls,
        _session.Placement);

    /// <summary>The underlying session, for diagnostics.</summary>
    public OnnxModelSession Session => _session;

    /// <summary>
    /// Checks, before anything is loaded, whether <see cref="Create(KevModelFiles, KevOptions?)"/> with
    /// <paramref name="options"/> can start: the options are valid, the ONNX Runtime native library loads, and for
    /// <see cref="OnnxExecutionProvider.Cuda"/> the native library has the CUDA provider and it initializes for
    /// <see cref="OnnxSessionOptions.CudaDeviceId"/> (<see cref="OnnxRuntimeInfo.TryInitializeProvider(OnnxSessionOptions, out string?)"/>).
    /// No graph, head or tokenizer is read and no device memory is allocated, so it costs milliseconds instead of a
    /// 3 to 17 GB load; a missing, busy or full device, or too little memory, can still fail at
    /// <see cref="Create(KevModelFiles, KevOptions?)"/>.
    /// </summary>
    /// <param name="options">The options <see cref="Create(KevModelFiles, KevOptions?)"/> would get; null for the defaults (CPU).</param>
    /// <param name="reason">Why the model cannot load with these options; null when it can.</param>
    /// <returns>True when nothing found stops the load.</returns>
    public static bool CanLoad(KevOptions? options, [NotNullWhen(false)] out string? reason)
    {
        options ??= new KevOptions();
        try
        {
            options.Validate();
            return OnnxRuntimeInfo.TryInitializeProvider(options.Session, out reason);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            reason = ex.Message;
            return false;
        }
    }

    /// <summary>
    /// Loads a verified bundle (<see cref="KevModelAssets.OpenDirectory"/>) and owns the session. Unless
    /// <paramref name="options"/> sets <see cref="OnnxSessionOptions.ExpectedModelSha256"/>, the graph must match the lock.
    /// </summary>
    public static KevDecisionModel Create(KevModelFiles files, KevOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new KevOptions();
        if (options.Session is { ExpectedModelSha256: null } session)
        {
            options = options with { Session = session with { ExpectedModelSha256 = KevModelAssets.ModelSha256(files.Profile.Model) } };
        }

        // A snapshot verified with SHA-256 already hashed the graph: pass the digest instead of reading the file again.
        if (options.Session is { KnownModelSha256: null } known && files.ModelSha256 is { } verified)
        {
            options = options with { Session = known with { KnownModelSha256 = verified } };
        }

        return Create(files.Profile, files.ModelPath, files.HeadPath, files.HeadMetadataPath, files.TokenizerPath, options);
    }

    /// <summary>
    /// Loads a graph, the head and a <c>tokenizer.json</c> and owns the session. The tokenizer and the head metadata are
    /// checked against the profile; the graph's signature is checked always, its digest only when
    /// <see cref="OnnxSessionOptions.ExpectedModelSha256"/> is set.
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer or head is not the profile's.</exception>
    /// <exception cref="NotSupportedException">The graph does not have the profile's signature.</exception>
    public static KevDecisionModel Create(
        KevModelProfile profile, string modelPath, string headPath, string headMetadataPath, string tokenizerPath, KevOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        ArgumentException.ThrowIfNullOrEmpty(headPath);
        ArgumentException.ThrowIfNullOrEmpty(headMetadataPath);
        ArgumentException.ThrowIfNullOrEmpty(tokenizerPath);
        options ??= new KevOptions();
        options.Validate();
        var encoder = KevEncoder.FromFile(tokenizerPath);
        var head = KevPointerHead.Load(profile, headPath, headMetadataPath, options.Temperature);
        var session = OnnxModelSession.Create(modelPath, options.Session);
        try
        {
            return new KevDecisionModel(profile, session, ownsSession: true, encoder, head, options);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>Answers a request: one calibrated distribution per question, mapped to System One answers.</summary>
    /// <exception cref="KevInputTooLongException">
    /// The state is over the limit, or a question is and <see cref="KevOptions.DeclineOverLongQuestions"/> is false.
    /// </exception>
    public KevDecision Decide(KevRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var record = request.ToRecord();
        var encoding = Encode(record);
        var probs = encoding.Rows.Count == 0 ? [] : Run(encoding, cancellationToken);
        return ToDecision(request.Questions, encoding, probs);
    }

    /// <summary>
    /// Probabilities per row of an internal record (Kev's <c>encode</c> input): one per question, or, with
    /// <see cref="KevOptions.DeclineOverLongQuestions"/>, one per question that is not over a limit (map them with
    /// <see cref="Encode"/> and <see cref="KevEncoding.QuestionIndices"/>).
    /// </summary>
    public IReadOnlyList<float[]> Score(KevRecord record, CancellationToken cancellationToken = default)
    {
        var encoding = Encode(record);
        return encoding.Rows.Count == 0 ? [] : Run(encoding, cancellationToken);
    }

    /// <summary>
    /// Stage 1: Kev's row-form encoding of a record (escaping, tokenization, delimiters, positions, readouts). With
    /// <see cref="KevOptions.DeclineOverLongQuestions"/>, over-long questions are left out and listed in
    /// <see cref="KevEncoding.Declined"/>; the encoding has no rows when every question is over a limit.
    /// </summary>
    /// <exception cref="KevInputTooLongException">
    /// The state is over the limit, or a question is and <see cref="KevOptions.DeclineOverLongQuestions"/> is false.
    /// </exception>
    public KevEncoding Encode(KevRecord record)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _encoder.Encode(record, _options.MaxStateTokens, _options.MaxRowTokens, _options.MaxBranchTokens, _options.DeclineOverLongQuestions);
    }

    private KevDecision ToDecision(IReadOnlyList<KeyValuePair<string, KevQuestion>> questions, KevEncoding encoding, IReadOnlyList<float[]> probs)
    {
        var answers = new List<KevAnswer>(probs.Count);
        bool calibrated = IsCalibrated;
        for (int r = 0; r < probs.Count; r++)
        {
            var (id, question) = questions[encoding.QuestionIndexOf(r)];
            answers.Add(new KevAnswer(id, question, probs[r], _head.Temperature, calibrated));
        }

        return new KevDecision(answers, encoding.StateIds.Length, encoding.Rows.Sum(r => r.Ids.Length)) { Declined = encoding.Declined };
    }

    /// <summary>
    /// Stage 2: runs an encoding: the state once, then each question's branch from a copy of the state's cache.
    /// Returns calibrated probabilities per question.
    /// </summary>
    public IReadOnlyList<float[]> Run(KevEncoding encoding, CancellationToken cancellationToken = default)
    {
        CheckEncoding(encoding);
        // With the gate (the CUDA default), a request's state pass and rows run as one unit.
        _runGate?.Wait(cancellationToken);
        try
        {
            return RunCached(encoding, cancellationToken);
        }
        finally
        {
            _runGate?.Release();
        }
    }

    private float[][] RunCached(KevEncoding encoding, CancellationToken cancellationToken)
    {
        using var cache = RunState(encoding.StateIds, cancellationToken);
        return RunRows(encoding, [.. cache], cancellationToken);
    }

    /// <summary>The state pass: the cache (attention keys and values, DeltaNet states) the rows continue from.</summary>
    private IDisposableReadOnlyCollection<OrtValue> RunState(ImmutableArray<int> stateIds, CancellationToken cancellationToken) =>
        RunGraph([.. stateIds], Positions(0, stateIds.Length), 0, _emptyPast, _presentNames, cancellationToken);

    /// <summary>Each row from the state's cache <paramref name="past"/>, which no graph run modifies.</summary>
    private float[][] RunRows(KevEncoding encoding, OrtValue[] past, CancellationToken cancellationToken)
    {
        int stateLength = encoding.StateIds.Length;
        var result = new float[encoding.Rows.Count][];
        for (int k = 0; k < result.Length; k++)
        {
            var row = encoding.Rows[k];
            using var outputs = RunGraph([.. row.Ids], [.. row.Positions], stateLength, past, [HiddenStatesName], cancellationToken);
            result[k] = Readout(outputs[0], row.Decide, row.Options, offset: 0);
        }

        return result;
    }

    /// <summary>
    /// Runs every question as one full causal row (state followed by its branch) without a cache: upstream's
    /// <c>forward()</c> layout. Equal to <see cref="Run"/> within the profile's cached-versus-uncached bound; for evidence.
    /// </summary>
    public IReadOnlyList<float[]> RunUncached(KevEncoding encoding, CancellationToken cancellationToken = default)
    {
        CheckEncoding(encoding);
        _runGate?.Wait(cancellationToken);
        try
        {
            return RunFullRows(encoding, cancellationToken);
        }
        finally
        {
            _runGate?.Release();
        }
    }

    private float[][] RunFullRows(KevEncoding encoding, CancellationToken cancellationToken)
    {
        int stateLength = encoding.StateIds.Length;
        var result = new float[encoding.Rows.Count][];
        for (int k = 0; k < result.Length; k++)
        {
            var row = encoding.Rows[k];
            int[] ids = [.. encoding.StateIds, .. row.Ids];
            using var outputs = RunGraph(ids, Positions(0, ids.Length), 0, _emptyPast, [HiddenStatesName], cancellationToken);
            result[k] = Readout(outputs[0], row.Decide, row.Options, offset: stateLength);
        }

        return result;
    }

    /// <summary>Releases the session (when owned) and the cache templates. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var value in _emptyPast)
        {
            value.Dispose();
        }

        if (_ownsSession)
        {
            _session.Dispose();
        }

        _runGate?.Dispose();
    }

    private static int[] Positions(int start, int count) => [.. Enumerable.Range(start, count)];

    private void CheckEncoding(KevEncoding encoding)
    {
        ArgumentNullException.ThrowIfNull(encoding);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (encoding.StateIds.IsDefaultOrEmpty || encoding.StateIds[0] != KevModelProfile.StateTokenId || encoding.Rows.Count == 0)
        {
            throw new ArgumentException("An encoding needs a state starting with <state> and at least one row.", nameof(encoding));
        }

        foreach (var row in encoding.Rows)
        {
            if (row.Ids.Length == 0 || row.Positions.Length != row.Ids.Length || row.Decide != row.Ids.Length - 1 || row.Options.IsDefaultOrEmpty
                || row.Options.Any(o => o < 0 || o >= row.Decide))
            {
                throw new ArgumentException("A row's positions, <decide> or option offsets are inconsistent.", nameof(encoding));
            }
        }
    }

    private float[] Readout(OrtValue hiddenValue, int decide, IReadOnlyList<int> options, int offset)
    {
        var hidden = hiddenValue.GetTensorDataAsSpan<float>();
        int d = Profile.HiddenSize;
        var keys = new float[options.Count][];
        for (int i = 0; i < keys.Length; i++)
        {
            keys[i] = hidden.Slice((offset + options[i]) * d, d).ToArray();
        }

        var probs = _head.Probabilities(hidden.Slice((offset + decide) * d, d), keys);
        if (probs.Any(p => !float.IsFinite(p)))
        {
            throw new InvalidOperationException("The model produced a non-finite probability.");
        }

        return probs;
    }

    private IDisposableReadOnlyCollection<OrtValue> RunGraph(
        int[] ids, int[] positions, int pastLength, IReadOnlyList<OrtValue> past, IReadOnlyCollection<string> outputNames, CancellationToken cancellationToken)
    {
        int n = ids.Length;
        long[] idValues = [.. ids.Select(i => (long)i)];
        long[] mask = new long[pastLength + n];
        Array.Fill(mask, 1L);
        long[] pos = new long[3 * n];
        for (int axis = 0; axis < 3; axis++)
        {
            for (int t = 0; t < n; t++)
            {
                pos[axis * n + t] = positions[t];
            }
        }

        using var idsValue = OrtValue.CreateTensorValueFromMemory(idValues, [1, n]);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, [1, pastLength + n]);
        using var posValue = OrtValue.CreateTensorValueFromMemory(pos, [3, 1, n]);
        var names = new List<string>(3 + _pastNames.Length) { InputIdsName, AttentionMaskName, PositionIdsName };
        names.AddRange(_pastNames);
        var values = new List<OrtValue>(names.Count) { idsValue, maskValue, posValue };
        values.AddRange(past);
        return _session.Run(names, values, outputNames, cancellationToken);
    }

    private static (string[] Past, string[] Present, OrtValue[] Empty) ValidateSignature(KevModelProfile profile, OnnxModelSession session)
    {
        var inputs = session.Inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        var outputs = session.Outputs.ToDictionary(o => o.Name, StringComparer.Ordinal);
        void Require(bool ok, string what)
        {
            if (!ok)
            {
                throw new NotSupportedException($"The graph does not have the {profile.Id} signature: {what}.");
            }
        }

        Require(inputs.TryGetValue(InputIdsName, out var ids) && ids.ElementType == "Int64", "input_ids int64");
        Require(inputs.TryGetValue(AttentionMaskName, out var mask) && mask.ElementType == "Int64", "attention_mask int64");
        Require(inputs.TryGetValue(PositionIdsName, out var pos) && pos.ElementType == "Int64" && pos.Dimensions.Count == 3 && pos.Dimensions[0] == 3, "position_ids int64 [3, batch, sequence]");
        Require(outputs.TryGetValue(HiddenStatesName, out var hidden) && hidden.ElementType == "Float" && hidden.Dimensions.Count == 3 && hidden.Dimensions[2] == profile.HiddenSize,
            $"hidden_states float [batch, sequence, {profile.HiddenSize}]");

        var past = new List<string>();
        var present = new List<string>();
        var empty = new List<OrtValue>();
        try
        {
            for (int layer = 0; layer < profile.Layers; layer++)
            {
                string[] suffixes = KevModelProfile.IsFullAttention(layer) ? ["key", "value"] : ["conv", "recurrent"];
                foreach (string suffix in suffixes)
                {
                    string inName = KevModelProfile.IsFullAttention(layer)
                        ? string.Create(CultureInfo.InvariantCulture, $"past_key_values.{layer}.{suffix}")
                        : string.Create(CultureInfo.InvariantCulture, $"past.{layer}.{suffix}");
                    string outName = string.Create(CultureInfo.InvariantCulture, $"present.{layer}.{suffix}");
                    Require(inputs.TryGetValue(inName, out var info) && info.ElementType == "Float", $"input {inName}");
                    Require(outputs.ContainsKey(outName), $"output {outName}");
                    long[] shape = [.. info!.Dimensions];
                    int rank = suffix == "conv" ? 3 : 4; // conv: [batch, conv_dim, kernel - 1]
                    Require(shape.Length == rank, $"{inName} rank {rank}");
                    shape[0] = 1;
                    if (suffix is "key" or "value")
                    {
                        Require(shape[1] == profile.KeyValueHeads, $"{inName} with {profile.KeyValueHeads} key/value heads");
                        shape[2] = 0;
                        shape[3] = AttentionHeadSize;
                    }
                    else if (suffix == "recurrent")
                    {
                        Require(shape[1] == profile.LinearValueHeads && shape[2] == LinearHeadSize && shape[3] == LinearHeadSize,
                            $"{inName} [batch, {profile.LinearValueHeads}, {LinearHeadSize}, {LinearHeadSize}]");
                    }
                    else
                    {
                        Require(shape.Skip(1).All(dim => dim > 0), $"{inName} fixed dimensions");
                    }

                    past.Add(inName);
                    present.Add(outName);
                    empty.Add(OrtValue.CreateTensorValueFromMemory(new float[shape.Aggregate(1L, (a, b) => a * b)], shape));
                }
            }

            Require(inputs.Count == 3 + past.Count, $"exactly {3 + past.Count} inputs");
            return ([.. past], [.. present], [.. empty]);
        }
        catch
        {
            foreach (var value in empty)
            {
                value.Dispose();
            }

            throw;
        }
    }
}
