namespace Starlings.Models.Gliner2;

/// <summary>Score of one candidate label.</summary>
/// <param name="Label">The label, as passed.</param>
/// <param name="Logit">The classification head's logit (temperature 1).</param>
/// <param name="Score">
/// Softmax of the logits over the labels for single-label tasks, sigmoid of the logit for multi-label tasks (GLiNER2's
/// <c>class_act</c> "auto").
/// </param>
public sealed record Gliner2LabelScore(string Label, float Logit, float Score);

/// <summary>A classification of one text.</summary>
public sealed class Gliner2ClassificationResult
{
    internal Gliner2ClassificationResult(string task, bool multiLabel, float threshold, IReadOnlyList<Gliner2LabelScore> scores, IReadOnlyList<Gliner2LabelScore> selected)
    {
        Task = task;
        MultiLabel = multiLabel;
        Threshold = threshold;
        Scores = scores;
        Selected = selected;
    }

    /// <summary>The task name (the schema prompt).</summary>
    public string Task { get; }

    /// <summary>True for a multi-label task.</summary>
    public bool MultiLabel { get; }

    /// <summary>The multi-label threshold used (GLiNER2's <c>cls_threshold</c>); not used for single-label tasks.</summary>
    public float Threshold { get; }

    /// <summary>Every label with its logit and score, in the order passed.</summary>
    public IReadOnlyList<Gliner2LabelScore> Scores { get; }

    /// <summary>
    /// GLiNER2's answer: single-label, the highest-scoring label (first on ties); multi-label, every label whose score is
    /// at least <see cref="Threshold"/>, in label order, or the highest-scoring label alone when none is.
    /// </summary>
    public IReadOnlyList<Gliner2LabelScore> Selected { get; }

    /// <summary>The first selected label.</summary>
    public Gliner2LabelScore Top => Selected[0];
}

/// <summary>
/// Zero-shot text classification with GLiNER2's classification head (Experimental, <see cref="Gliner2Capabilities"/>
/// key <c>classification</c>): a task name and candidate labels in, a score per label out, as
/// <c>GLiNER2.classify_text(text, {task: {"labels": labels, "multi_label": multi_label, "cls_threshold": threshold}})</c>
/// computes them for English text.
/// </summary>
/// <remarks>
/// <para>
/// Uses an existing <see cref="Gliner2EntityExtractor"/>'s session (not owned): the schema
/// <c>( [P] task ( [L] label1 [L] label2 ... ) )</c> is encoded with the extractor's preprocessor, the locked graph's
/// <c>last_hidden_state</c> is read at the <c>[L]</c> markers, and <see cref="Gliner2ClassificationHead"/> maps each
/// to a logit. The same length limit applies (<see cref="Gliner2ExtractorOptions.MaxSequenceLength"/>, schema
/// included; nothing is truncated).
/// </para>
/// <para>
/// Not offered: several tasks in one call, label descriptions or examples, temperature, explicit activations,
/// batching, long-text chunking. Safe to call concurrently; on CUDA calls follow the extractor's serialization.
/// </para>
/// </remarks>
public sealed class Gliner2TextClassifier
{
    private readonly Gliner2EntityExtractor _extractor;
    private readonly Gliner2ClassificationHead _head;

    /// <summary>Creates a classifier over <paramref name="extractor"/>'s session and preprocessor.</summary>
    /// <exception cref="NotSupportedException">The extractor's session runs on CUDA (classification has no CUDA evidence).</exception>
    public Gliner2TextClassifier(Gliner2EntityExtractor extractor, Gliner2ClassificationHead head)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        ArgumentNullException.ThrowIfNull(head);
        if (extractor.Session.Diagnostics.SessionProviders[0] != Starlings.Runtime.Onnx.OnnxRuntimeInfo.CpuProviderName)
        {
            throw new NotSupportedException(
                $"Classification is experimental on the CPU provider only; this session runs on {extractor.Session.Diagnostics.SessionProviders[0]} (Gliner2Capabilities: classification.cuda).");
        }

        _extractor = extractor;
        _head = head;
    }

    /// <summary>Classifies one text.</summary>
    /// <param name="text">English text (valid Unicode).</param>
    /// <param name="task">Task name, for example "sentiment"; part of the model input.</param>
    /// <param name="labels">At least two distinct labels; order and case matter to the model.</param>
    /// <param name="multiLabel">False: softmax over the labels, one answer. True: sigmoid per label, answers above <paramref name="threshold"/>.</param>
    /// <param name="threshold">Multi-label threshold in [0, 1] (GLiNER2 default 0.5).</param>
    /// <param name="cancellationToken">Cancels before or during the run.</param>
    /// <exception cref="ArgumentException">An empty task, fewer than two labels, an empty or duplicate label, or text with an unpaired surrogate.</exception>
    /// <exception cref="Gliner2InputTooLongException">The text and schema need more than <see cref="Gliner2ExtractorOptions.MaxSequenceLength"/> tokens.</exception>
    public Gliner2ClassificationResult Classify(
        string text, string task, IReadOnlyList<string> labels, bool multiLabel = false, float threshold = 0.5f, CancellationToken cancellationToken = default)
    {
        var input = Prepare(text, task, labels);
        Gliner2ExtractorOptions.ValidateThreshold(threshold, nameof(threshold));
        _extractor.EnsureLength(input, 0);
        var markers = _extractor.RunLabelMarkers(input, cancellationToken);
        return Decode(task, input.Labels, _head.Logits(markers), multiLabel, threshold);
    }

    /// <summary>The model input GLiNER2 builds for this classification (no length check).</summary>
    public Gliner2ModelInput Prepare(string text, string task, IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(text);
        ValidateTask(task);
        var normalized = ValidateLabels(labels);
        return _extractor.Preprocessor.Build(text, normalized, null, null, task, Gliner2Preprocessor.ClassificationMarker);
    }

    /// <summary>GLiNER2's decision rule over the head's logits (temperature 1, <c>class_act</c> "auto").</summary>
    internal static Gliner2ClassificationResult Decode(string task, IReadOnlyList<string> labels, float[] logits, bool multiLabel, float threshold)
    {
        if (logits.Length != labels.Count)
        {
            throw new ArgumentException($"{logits.Length} logits for {labels.Count} labels.", nameof(logits));
        }

        var scores = new float[logits.Length];
        if (multiLabel)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = 1f / (1f + MathF.Exp(-logits[i]));
            }
        }
        else
        {
            float max = logits.Max();
            float sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = MathF.Exp(logits[i] - max);
                sum += scores[i];
            }

            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] /= sum;
            }
        }

        var all = new Gliner2LabelScore[labels.Count];
        int best = 0;
        for (int i = 0; i < all.Length; i++)
        {
            all[i] = new Gliner2LabelScore(labels[i], logits[i], scores[i]);
            if (scores[i] > scores[best])
            {
                best = i;
            }
        }

        var selected = multiLabel ? all.Where(s => s.Score >= threshold).ToList() : [];
        if (selected.Count == 0)
        {
            selected.Add(all[best]);
        }

        return new Gliner2ClassificationResult(task, multiLabel, threshold, all, selected.AsReadOnly());
    }

    internal static void ValidateTask(string task)
    {
        if (string.IsNullOrWhiteSpace(task))
        {
            throw new ArgumentException("The task name must not be null, empty or whitespace.", nameof(task));
        }
    }

    internal static IReadOnlyList<string> ValidateLabels(IReadOnlyList<string> labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (labels.Count < 2)
        {
            throw new ArgumentException("Classification needs at least two labels.", nameof(labels));
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < labels.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(labels[i]))
            {
                throw new ArgumentException($"Label {i} is null, empty or whitespace.", nameof(labels));
            }

            if (!seen.Add(labels[i]))
            {
                throw new ArgumentException($"Label '{labels[i]}' is repeated; labels must be unique.", nameof(labels));
            }
        }

        return labels.ToArray();
    }
}
