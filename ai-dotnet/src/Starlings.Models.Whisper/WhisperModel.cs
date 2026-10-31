using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Whisper;

/// <summary>
/// The <c>openai/whisper-large-v3-turbo</c> encoder and decoder over ONNX Runtime: encodes a log-mel spectrogram
/// (<see cref="WhisperMelSpectrogram"/>) to <c>last_hidden_state</c>, then greedily decodes token ids with the model's
/// suppression lists, and optionally the timestamp rules (<see cref="GenerateWithTimestamps"/>). Decodes one 30 s window.
/// </summary>
/// <remarks>
/// <para><b>Decoder loop.</b> With the <c>decoder_with_past</c> graph loaded (<see cref="Create(WhisperModelFiles, WhisperOptions?)"/>
/// or the three-graph path overload), the default is the KV cache (<see cref="WhisperDecoderMode.KvCache"/>): the
/// full-context decoder runs once over the prompt and returns every layer's self- and cross-attention keys and values,
/// then each step feeds one token and the cached keys and values to <c>decoder_with_past</c>. Without it, or with
/// <see cref="WhisperDecoderMode.FullContext"/>, every step re-runs the full-context decoder over all ids (the
/// reference path). The two loops differ in the last bits of the logits: the fp16 graphs sum the attention and
/// projections over the whole prefix in one product on the full-context path and one row at a time on the cached
/// path, so the float16 accumulations round differently (about 1e-3 to 1e-2 in the logits). The onboarding route bounds
/// that difference and checks that ids, segments and transcripts are identical.</para>
/// <para><b>Prompt.</b> The decode prompt is <c>[&lt;|startoftranscript|&gt;, language, &lt;|transcribe|&gt;,
/// &lt;|notimestamps|&gt;]</c>. The language token is the id of a language code (see <see cref="LanguageToId"/>).
/// With no language, the slot holds <see cref="StartOfLanguageModel"/>: that is a task token, not a language token, and
/// this auto-detect prompt is untested (transformers instead picks the most likely language token after
/// <c>&lt;|startoftranscript|&gt;</c>).</para>
/// <para><b>Suppression.</b> <see cref="SuppressTokens"/> are masked on every step; <see cref="BeginSuppressTokens"/>
/// ([220, eos]) on the first step only, mirroring transformers <c>SuppressTokensLogitsProcessor</c> +
/// <c>SuppressTokensAtBeginLogitsProcessor</c>.</para>
/// <para><b>Timestamps.</b> <see cref="GenerateWithTimestamps"/> then applies transformers 5.18.0
/// <c>WhisperTimeStampLogitsProcessor</c> with the generation config's values (<see cref="TimestampBegin"/>,
/// <see cref="MaxInitialTimestampIndex"/>): timestamps come in pairs except directly before end of text, never
/// decrease, the first generated token is a timestamp of at most <see cref="MaxInitialTimestampIndex"/> steps, and when
/// the timestamps' summed probability exceeds every text token's, only a timestamp may follow.</para>
/// </remarks>
public sealed class WhisperModel : IDisposable
{
    /// <summary>Encoder graph input name for the log-mel spectrogram.</summary>
    public const string InputFeaturesName = "input_features";
    /// <summary>Encoder graph output name for the last hidden state.</summary>
    public const string LastHiddenStateName = "last_hidden_state";
    /// <summary>Decoder graph input name for the token ids.</summary>
    public const string InputIdsName = "input_ids";
    /// <summary>Decoder graph input name for the encoder hidden state.</summary>
    public const string EncoderHiddenStatesName = "encoder_hidden_states";
    /// <summary>Decoder graph output name for the logits.</summary>
    public const string LogitsName = "logits";

    /// <summary>Number of mel bins (128).</summary>
    public const int NumMelBins = 128;
    /// <summary>Number of mel frames for 30 s of audio (3000).</summary>
    public const int NumFrames = 3000;
    /// <summary>Encoder output sequence length (1500, half the frames).</summary>
    public const int EncoderSequenceLength = 1500;
    /// <summary>Model hidden size (1280).</summary>
    public const int HiddenSize = 1280;
    /// <summary>Vocabulary size (51866).</summary>
    public const int VocabSize = 51866;
    /// <summary>Decoder positions (448): the prompt plus generated tokens never exceed this (transformers' <c>max_length</c>).</summary>
    public const int MaxTargetPositions = 448;

    /// <summary><c>&lt;|endoftext|&gt;</c> (also eos and pad).</summary>
    public const int EndOfText = 50257;
    /// <summary><c>&lt;|startoftranscript|&gt;</c>.</summary>
    public const int StartOfTranscript = 50258;
    /// <summary><c>&lt;|startoflm|&gt;</c>: a task token; <see cref="Prompt"/> puts it in the language slot when no language is given (untested).</summary>
    public const int StartOfLanguageModel = 50361;
    /// <summary><c>&lt;|transcribe|&gt;</c>.</summary>
    public const int Transcribe = 50360;
    /// <summary><c>&lt;|translate|&gt;</c>.</summary>
    public const int Translate = 50359;
    /// <summary><c>&lt;|notimestamps|&gt;</c>.</summary>
    public const int NoTimestamps = 50364;
    /// <summary><c>&lt;|en|&gt;</c> language token.</summary>
    public const int English = 50259;
    /// <summary><c>&lt;|0.00|&gt;</c>, the first timestamp token (<see cref="NoTimestamps"/> + 1); <c>&lt;|30.00|&gt;</c> is 51865.</summary>
    public const int TimestampBegin = 50365;
    /// <summary>The first generated timestamp is at most this many <see cref="TimePrecision"/> steps in (1.00 s), from <c>generation_config.json</c>.</summary>
    public const int MaxInitialTimestampIndex = 50;
    /// <summary>Seconds per timestamp step (0.02).</summary>
    public const double TimePrecision = 0.02;

    private static readonly int[] SuppressTokens =
    [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542,
        873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
        6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265, 15618,
        16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865,
        42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363,
    ];

    private static readonly bool[] SuppressMask = BuildSuppressMask();
    private static readonly int[] BeginSuppressTokens = [220, 50257];

    /// <summary>Decoder graph input name of the new token's position in the KV-cache graph.</summary>
    public const string CachePositionName = "cache_position";
    /// <summary>Decoder layers (4): one self- and one cross-attention key/value pair each.</summary>
    public const int DecoderLayers = 4;

    private static readonly string[] FullContextOutputNames = BuildFullContextOutputNames();
    private static readonly string[] WithPastInputNames = BuildWithPastInputNames();
    private static readonly string[] WithPastOutputNames = BuildWithPastOutputNames();

    private readonly OnnxModelSession _encoder;
    private readonly OnnxModelSession _decoder;
    private readonly OnnxModelSession? _decoderWithPast;
    private readonly bool _ownsSessions;
    private int _disposed;

    private WhisperModel(OnnxModelSession encoder, OnnxModelSession decoder, OnnxModelSession? decoderWithPast, WhisperDecoderMode mode, bool ownsSessions)
    {
        _encoder = encoder;
        _decoder = decoder;
        _decoderWithPast = decoderWithPast;
        DecoderMode = mode;
        _ownsSessions = ownsSessions;
    }

    /// <summary>The decoder loop <see cref="Generate"/> and <see cref="GenerateWithTimestamps"/> use.</summary>
    public WhisperDecoderMode DecoderMode { get; }

    /// <summary>
    /// Loads the graphs of verified files (<see cref="WhisperModelAssets.OpenLocal"/> or
    /// <see cref="WhisperModelAssets.OpenDirectory"/>) and owns the resulting sessions: the encoder, the full-context
    /// decoder, and the KV-cache decoder unless <see cref="WhisperOptions.DecoderMode"/> is
    /// <see cref="WhisperDecoderMode.FullContext"/>. When the snapshot hashed the graphs
    /// (<see cref="WhisperModelFiles.EncoderSha256"/> and the others), each session takes that digest as
    /// <see cref="OnnxSessionOptions.KnownModelSha256"/> (unless the options set one) instead of reading the graph again.
    /// </summary>
    public static WhisperModel Create(WhisperModelFiles files, WhisperOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new WhisperOptions();
        options = options with
        {
            EncoderSession = options.EncoderSession with { KnownModelSha256 = options.EncoderSession.KnownModelSha256 ?? files.EncoderSha256 },
            DecoderSession = options.DecoderSession with { KnownModelSha256 = options.DecoderSession.KnownModelSha256 ?? files.DecoderSha256 },
            DecoderWithPastSession = options.DecoderWithPastSession with
            {
                KnownModelSha256 = options.DecoderWithPastSession.KnownModelSha256 ?? files.DecoderWithPastSha256,
            },
        };
        return Create(files.EncoderPath, files.DecoderPath, files.DecoderWithPastPath, options);
    }

    /// <summary>
    /// Loads the encoder and the full-context decoder and owns the resulting sessions. Without the KV-cache graph the
    /// decoder loop is <see cref="WhisperDecoderMode.FullContext"/>.
    /// </summary>
    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="WhisperOptions.Language"/> is not a language code of the model (checked before any graph loads).</exception>
    /// <exception cref="ArgumentException"><see cref="WhisperOptions.DecoderMode"/> is <see cref="WhisperDecoderMode.KvCache"/>, which needs the KV-cache graph.</exception>
    public static WhisperModel Create(string encoderPath, string decoderPath, WhisperOptions? options = null) =>
        Create(encoderPath, decoderPath, null, options);

    /// <summary>
    /// Loads the encoder, the full-context decoder and, unless <see cref="WhisperOptions.DecoderMode"/> is
    /// <see cref="WhisperDecoderMode.FullContext"/>, the KV-cache decoder (<c>decoder_with_past_model_fp16.onnx</c>), and
    /// owns the resulting sessions. With <paramref name="decoderWithPastPath"/> null the loop is full-context.
    /// </summary>
    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="WhisperOptions.Language"/> is not a language code of the model (checked before any graph loads).</exception>
    /// <exception cref="ArgumentException"><see cref="WhisperOptions.DecoderMode"/> is <see cref="WhisperDecoderMode.KvCache"/> and <paramref name="decoderWithPastPath"/> is null.</exception>
    public static WhisperModel Create(string encoderPath, string decoderPath, string? decoderWithPastPath, WhisperOptions? options = null)
    {
        options ??= new WhisperOptions();
        options.Validate();
        if (options.DecoderMode == WhisperDecoderMode.KvCache && decoderWithPastPath is null)
        {
            throw new ArgumentException("DecoderMode.KvCache needs the decoder_with_past graph.", nameof(decoderWithPastPath));
        }

        bool cached = decoderWithPastPath is not null && options.DecoderMode != WhisperDecoderMode.FullContext;
        var sessions = new List<OnnxModelSession>();
        try
        {
            sessions.Add(OnnxModelSession.Create(encoderPath, options.EncoderSession));
            sessions.Add(OnnxModelSession.Create(decoderPath, options.DecoderSession));
            if (cached)
            {
                sessions.Add(OnnxModelSession.Create(decoderWithPastPath!, options.DecoderWithPastSession));
            }

            return new WhisperModel(
                sessions[0],
                sessions[1],
                cached ? sessions[2] : null,
                cached ? WhisperDecoderMode.KvCache : WhisperDecoderMode.FullContext,
                ownsSessions: true);
        }
        catch
        {
            foreach (var session in sessions)
            {
                session.Dispose();
            }

            throw;
        }
    }

    /// <summary>The two-letter language codes this model supports, in token-id order (from the generation config).</summary>
    public static IReadOnlyDictionary<string, int> LanguageToId { get; } = new Dictionary<string, int>
    {
        ["en"] = 50259, ["zh"] = 50260, ["de"] = 50261, ["es"] = 50262, ["ru"] = 50263, ["ko"] = 50264,
        ["fr"] = 50265, ["ja"] = 50266, ["pt"] = 50267, ["tr"] = 50268, ["pl"] = 50269, ["ca"] = 50270,
        ["nl"] = 50271, ["ar"] = 50272, ["sv"] = 50273, ["it"] = 50274, ["id"] = 50275, ["hi"] = 50276,
        ["fi"] = 50277, ["vi"] = 50278, ["he"] = 50279, ["uk"] = 50280, ["el"] = 50281, ["ms"] = 50282,
        ["cs"] = 50283, ["ro"] = 50284, ["da"] = 50285, ["hu"] = 50286, ["ta"] = 50287, ["no"] = 50288,
        ["th"] = 50289, ["ur"] = 50290, ["hr"] = 50291, ["bg"] = 50292, ["lt"] = 50293, ["la"] = 50294,
        ["mi"] = 50295, ["ml"] = 50296, ["cy"] = 50297, ["sk"] = 50298, ["te"] = 50299, ["fa"] = 50300,
        ["lv"] = 50301, ["bn"] = 50302, ["sr"] = 50303, ["az"] = 50304, ["sl"] = 50305, ["kn"] = 50306,
        ["et"] = 50307, ["mk"] = 50308, ["br"] = 50309, ["eu"] = 50310, ["is"] = 50311, ["hy"] = 50312,
        ["ne"] = 50313, ["mn"] = 50314, ["bs"] = 50315, ["kk"] = 50316, ["sq"] = 50317, ["sw"] = 50318,
        ["gl"] = 50319, ["mr"] = 50320, ["pa"] = 50321, ["si"] = 50322, ["km"] = 50323, ["sn"] = 50324,
        ["yo"] = 50325, ["so"] = 50326, ["af"] = 50327, ["oc"] = 50328, ["ka"] = 50329, ["be"] = 50330,
        ["tg"] = 50331, ["sd"] = 50332, ["gu"] = 50333, ["am"] = 50334, ["yi"] = 50335, ["lo"] = 50336,
        ["uz"] = 50337, ["fo"] = 50338, ["ht"] = 50339, ["ps"] = 50340, ["tk"] = 50341, ["nn"] = 50342,
        ["mt"] = 50343, ["sa"] = 50344, ["lb"] = 50345, ["my"] = 50346, ["bo"] = 50347, ["tl"] = 50348,
        ["mg"] = 50349, ["as"] = 50350, ["tt"] = 50351, ["haw"] = 50352, ["ln"] = 50353, ["ha"] = 50354,
        ["ba"] = 50355, ["jw"] = 50356, ["su"] = 50357, ["yue"] = 50358,
    };

    /// <summary>Runs the encoder: <c>[128, 3000]</c> mel to <c>[1500, 1280]</c> last hidden state.</summary>
    public float[] Encode(float[] mel, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (mel.Length != NumMelBins * NumFrames)
        {
            throw new ArgumentException($"Mel must be {NumMelBins * NumFrames} floats, got {mel.Length}.", nameof(mel));
        }

        long[] shape = [1, NumMelBins, NumFrames];
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        using var melValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, mel.AsMemory(0, NumMelBins * NumFrames), shape);
        var hidden = new float[EncoderSequenceLength * HiddenSize];
        using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, hidden.AsMemory(0, hidden.Length), [1, EncoderSequenceLength, HiddenSize]);
        _encoder.Run([InputFeaturesName], [melValue], [LastHiddenStateName], [hiddenValue], cancellationToken);
        return hidden;
    }

    /// <summary>Runs the decoder over <paramref name="ids"/> with the encoder's hidden state, returning logits [seq, vocab].</summary>
    public float[] DecodeLogits(long[] ids, float[] encoderHidden, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(ids);
        var logits = new float[ids.Length * VocabSize];
        DecodeLogitsInto(ids.AsMemory(), encoderHidden, logits.AsMemory(), cancellationToken);
        return logits;
    }

    /// <summary>
    /// Builds the decode prompt for the given language and task. A <see langword="null"/> language puts
    /// <see cref="StartOfLanguageModel"/> in the language slot (auto-detect, untested).
    /// </summary>
    /// <exception cref="WhisperUnsupportedLanguageException"><paramref name="language"/> is not a key of <see cref="LanguageToId"/> (case-insensitive).</exception>
    public static int[] Prompt(string? language = null, bool transcribe = true, bool noTimestamps = true)
    {
        int lang = LanguageTokenId(language, nameof(language));
        int task = transcribe ? Transcribe : Translate;
        var prompt = new List<int> { StartOfTranscript, lang, task };
        if (noTimestamps)
        {
            prompt.Add(NoTimestamps);
        }

        return prompt.ToArray();
    }

    /// <summary>
    /// Greedily decodes from <paramref name="prompt"/> until eos, <paramref name="maxLength"/> generated tokens, or
    /// <see cref="MaxTargetPositions"/> tokens in total (the decoder has no position beyond it). Returns ids including the prompt.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt fills every decoder position.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLength"/> is less than 1.</exception>
    public int[] Generate(float[] encoderHidden, int[] prompt, int maxLength = 448, CancellationToken cancellationToken = default) =>
        GenerateCore(encoderHidden, prompt, maxLength, timestamps: false, DecoderMode, null, cancellationToken);

    /// <summary>
    /// As <see cref="Generate"/>, with the timestamp rules applied after suppression on every step (transformers
    /// <c>WhisperTimeStampLogitsProcessor</c>; its begin index is the prompt length). Use a prompt without
    /// <see cref="NoTimestamps"/> (<see cref="Prompt"/> with <c>noTimestamps: false</c>). Returns ids including the prompt.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt fills every decoder position.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLength"/> is less than 1.</exception>
    public int[] GenerateWithTimestamps(float[] encoderHidden, int[] prompt, int maxLength = 448, CancellationToken cancellationToken = default) =>
        GenerateCore(encoderHidden, prompt, maxLength, timestamps: true, DecoderMode, null, cancellationToken);

    /// <summary>
    /// Diagnostics: <see cref="Generate"/> or <see cref="GenerateWithTimestamps"/> with an explicit decoder loop, adding
    /// each step's raw logits row (before suppression and the timestamp rules) to <paramref name="stepLogits"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException"><paramref name="mode"/> is <see cref="WhisperDecoderMode.KvCache"/> and the KV-cache graph is not loaded.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLength"/> is less than 1.</exception>
    public int[] GenerateRecordingLogits(
        float[] encoderHidden, int[] prompt, WhisperDecoderMode mode, bool timestamps, List<float[]> stepLogits,
        int maxLength = 448, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stepLogits);
        if (mode == WhisperDecoderMode.KvCache && _decoderWithPast is null)
        {
            throw new InvalidOperationException("The KV-cache graph is not loaded.");
        }

        return GenerateCore(encoderHidden, prompt, maxLength, timestamps, mode, stepLogits, cancellationToken);
    }

    /// <summary>Disposes the sessions if this model owns them. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsSessions)
        {
            _decoderWithPast?.Dispose();
            _decoder.Dispose();
            _encoder.Dispose();
        }
    }

    /// <summary>The language-slot token of <paramref name="language"/>: its <see cref="LanguageToId"/> id, or <see cref="StartOfLanguageModel"/> for null.</summary>
    /// <exception cref="WhisperUnsupportedLanguageException">The code is not a key of <see cref="LanguageToId"/> (case-insensitive).</exception>
    internal static int LanguageTokenId(string? language, string? paramName)
    {
        if (language is null)
        {
            return StartOfLanguageModel;
        }

        return LanguageToId.TryGetValue(language.ToLowerInvariant(), out int id)
            ? id
            : throw WhisperUnsupportedLanguageException.For(language, paramName);
    }

    internal static void ApplySuppression(Span<float> logits, bool firstStep)
    {
        for (int i = 0; i < logits.Length; i++)
        {
            if (SuppressMask[i])
            {
                logits[i] = float.NegativeInfinity;
            }
        }

        if (firstStep)
        {
            foreach (int t in BeginSuppressTokens)
            {
                logits[t] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>
    /// transformers 5.18.0 <c>WhisperTimeStampLogitsProcessor</c> on one suppressed logits row; <paramref name="generated"/>
    /// holds the tokens after the prompt. The masks follow the pinned source; its log-softmax test is computed in double
    /// with left-to-right sums (transformers computes it in float32), the same formula as the route's Python loop.
    /// </summary>
    internal static void ApplyTimestampRules(Span<float> logits, ReadOnlySpan<long> generated)
    {
        logits[NoTimestamps] = float.NegativeInfinity;
        bool last = generated.Length >= 1 && generated[^1] >= TimestampBegin;
        bool penultimate = generated.Length < 2 || generated[^2] >= TimestampBegin;
        if (last && penultimate)
        {
            // Two timestamps in a row (or one at the start): the next token is not a timestamp.
            logits[TimestampBegin..].Fill(float.NegativeInfinity);
        }
        else if (last)
        {
            // One timestamp after text: the next token is not text (ids below end of text).
            logits[..EndOfText].Fill(float.NegativeInfinity);
        }

        long lastTimestamp = -1;
        foreach (long token in generated)
        {
            if (token >= TimestampBegin)
            {
                lastTimestamp = token;
            }
        }

        if (lastTimestamp >= 0)
        {
            // Timestamps never decrease; after a closed pair the next one must also be later than the last.
            int limit = (int)(last && !penultimate ? lastTimestamp : lastTimestamp + 1);
            logits[TimestampBegin..limit].Fill(float.NegativeInfinity);
        }

        if (generated.Length == 0)
        {
            logits[..TimestampBegin].Fill(float.NegativeInfinity);
            logits[(TimestampBegin + MaxInitialTimestampIndex + 1)..].Fill(float.NegativeInfinity);
        }

        double peak = double.NegativeInfinity;
        foreach (float value in logits)
        {
            peak = Math.Max(peak, value);
        }

        double sum = 0;
        foreach (float value in logits)
        {
            sum += Math.Exp(value - peak);
        }

        double normalizer = peak + Math.Log(sum);
        double timestampPeak = double.NegativeInfinity;
        for (int i = TimestampBegin; i < logits.Length; i++)
        {
            timestampPeak = Math.Max(timestampPeak, logits[i] - normalizer);
        }

        double timestampLogProb = double.NegativeInfinity;
        if (!double.IsNegativeInfinity(timestampPeak))
        {
            double timestampSum = 0;
            for (int i = TimestampBegin; i < logits.Length; i++)
            {
                timestampSum += Math.Exp(logits[i] - normalizer - timestampPeak);
            }

            timestampLogProb = timestampPeak + Math.Log(timestampSum);
        }

        double textPeak = double.NegativeInfinity;
        for (int i = 0; i < TimestampBegin; i++)
        {
            textPeak = Math.Max(textPeak, logits[i] - normalizer);
        }

        if (timestampLogProb > textPeak)
        {
            logits[..TimestampBegin].Fill(float.NegativeInfinity);
        }
    }

    internal static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestValue)
            {
                bestValue = values[i];
                best = i;
            }
        }

        return best;
    }

    private int[] GenerateCore(
        float[] encoderHidden, int[] prompt, int maxLength, bool timestamps, WhisperDecoderMode mode, List<float[]>? stepLogits,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Length is 0 || prompt.Length >= MaxTargetPositions)
        {
            throw new ArgumentException($"The prompt must hold 1 to {MaxTargetPositions - 1} tokens, got {prompt.Length}.", nameof(prompt));
        }

        // Both loops emit at least one token once they start, so a non-positive budget is refused for either mode.
        ArgumentOutOfRangeException.ThrowIfLessThan(maxLength, 1);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (mode == WhisperDecoderMode.KvCache)
        {
            return GenerateCached(encoderHidden, prompt, maxLength, timestamps, stepLogits, cancellationToken);
        }

        // One ids buffer and one logits buffer per call, instead of fresh [seq, vocab] arrays every step. The logits
        // buffer grows by doubling up to the last step's length, so it never holds more than twice the rows needed.
        int maxIds = (int)Math.Min((long)prompt.Length + maxLength, MaxTargetPositions);
        var ids = new long[MaxTargetPositions];
        for (int i = 0; i < prompt.Length; i++)
        {
            ids[i] = prompt[i];
        }

        int count = prompt.Length;
        float[] logits = [];
        for (int step = 0; step < maxLength && count < MaxTargetPositions; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (logits.Length < count * VocabSize)
            {
                // count < maxIds here, so rows >= count.
                int rows = Math.Min(count * 2, maxIds);
                logits = new float[rows * VocabSize];
            }

            DecodeLogitsInto(ids.AsMemory(0, count), encoderHidden, logits.AsMemory(0, count * VocabSize), cancellationToken);
            var last = logits.AsSpan((count - 1) * VocabSize, VocabSize);
            stepLogits?.Add(last.ToArray());
            ApplySuppression(last, step == 0);
            if (timestamps)
            {
                ApplyTimestampRules(last, ids.AsSpan(prompt.Length, count - prompt.Length));
            }

            int next = ArgMax(last);
            ids[count++] = next;
            if (next == EndOfText)
            {
                break;
            }
        }

        var result = new int[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = (int)ids[i];
        }

        return result;
    }

    /// <summary>
    /// The KV-cache loop: the full-context decoder over the prompt (logits plus every layer's present keys and values),
    /// then one <c>decoder_with_past</c> run per generated token with the cached keys and values. Same stopping rule
    /// as the full-context loop.
    /// </summary>
    private int[] GenerateCached(
        float[] encoderHidden, int[] prompt, int maxLength, bool timestamps, List<float[]>? stepLogits, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(encoderHidden);
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        var ids = new List<long>(MaxTargetPositions);
        foreach (int token in prompt)
        {
            ids.Add(token);
        }

        var row = new float[VocabSize];
        var token1 = new long[1];
        var position = new long[1];
        using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, encoderHidden.AsMemory(0, encoderHidden.Length), [1, EncoderSequenceLength, HiddenSize]);
        using var promptValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, ids.ToArray().AsMemory(), [1, prompt.Length]);
        using var tokenValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, token1.AsMemory(), [1, 1]);
        using var positionValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, position.AsMemory(), [1]);

        // Outputs: logits, then per layer decoder key, decoder value, encoder key, encoder value.
        using var first = _decoder.Run([InputIdsName, EncoderHiddenStatesName], [promptValue, hiddenValue], FullContextOutputNames, cancellationToken);
        var encoderCache = new OrtValue[2 * DecoderLayers];
        var decoderCache = new OrtValue[2 * DecoderLayers];
        for (int layer = 0; layer < DecoderLayers; layer++)
        {
            decoderCache[2 * layer] = first[1 + 4 * layer];
            decoderCache[2 * layer + 1] = first[2 + 4 * layer];
            encoderCache[2 * layer] = first[3 + 4 * layer];
            encoderCache[2 * layer + 1] = first[4 + 4 * layer];
        }

        first[0].GetTensorDataAsSpan<float>().Slice((prompt.Length - 1) * VocabSize, VocabSize).CopyTo(row);
        IDisposableReadOnlyCollection<OrtValue>? previous = null;
        try
        {
            var inputs = new OrtValue[WithPastInputNames.Length];
            for (int step = 0; ; step++)
            {
                stepLogits?.Add((float[])row.Clone());
                ApplySuppression(row, step == 0);
                if (timestamps)
                {
                    ApplyTimestampRules(row, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[prompt.Length..]);
                }

                int next = ArgMax(row);
                ids.Add(next);
                if (next == EndOfText || step + 1 >= maxLength || ids.Count >= MaxTargetPositions)
                {
                    break;
                }

                cancellationToken.ThrowIfCancellationRequested();
                token1[0] = next;
                position[0] = ids.Count - 1;
                // Inputs: input_ids, per layer decoder key, decoder value, encoder key, encoder value, cache_position.
                inputs[0] = tokenValue;
                for (int layer = 0; layer < DecoderLayers; layer++)
                {
                    inputs[1 + 4 * layer] = decoderCache[2 * layer];
                    inputs[2 + 4 * layer] = decoderCache[2 * layer + 1];
                    inputs[3 + 4 * layer] = encoderCache[2 * layer];
                    inputs[4 + 4 * layer] = encoderCache[2 * layer + 1];
                }

                inputs[^1] = positionValue;
                var current = _decoderWithPast!.Run(WithPastInputNames, inputs, WithPastOutputNames, cancellationToken);
                previous?.Dispose();
                previous = current;
                for (int i = 0; i < decoderCache.Length; i++)
                {
                    decoderCache[i] = current[1 + i];
                }

                current[0].GetTensorDataAsSpan<float>()[..VocabSize].CopyTo(row);
            }
        }
        finally
        {
            previous?.Dispose();
        }

        var result = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            result[i] = (int)ids[i];
        }

        return result;
    }

    private static string[] BuildFullContextOutputNames()
    {
        var names = new List<string> { LogitsName };
        for (int layer = 0; layer < DecoderLayers; layer++)
        {
            names.AddRange([
                $"present.{layer}.decoder.key", $"present.{layer}.decoder.value",
                $"present.{layer}.encoder.key", $"present.{layer}.encoder.value",
            ]);
        }

        return names.ToArray();
    }

    private static string[] BuildWithPastInputNames()
    {
        var names = new List<string> { InputIdsName };
        for (int layer = 0; layer < DecoderLayers; layer++)
        {
            names.AddRange([
                $"past_key_values.{layer}.decoder.key", $"past_key_values.{layer}.decoder.value",
                $"past_key_values.{layer}.encoder.key", $"past_key_values.{layer}.encoder.value",
            ]);
        }

        names.Add(CachePositionName);
        return names.ToArray();
    }

    private static string[] BuildWithPastOutputNames()
    {
        var names = new List<string> { LogitsName };
        for (int layer = 0; layer < DecoderLayers; layer++)
        {
            names.AddRange([$"present.{layer}.decoder.key", $"present.{layer}.decoder.value"]);
        }

        return names.ToArray();
    }

    private void DecodeLogitsInto(Memory<long> ids, float[] encoderHidden, Memory<float> logits, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(encoderHidden);
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        using var idsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, ids, [1, ids.Length]);
        using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, encoderHidden.AsMemory(0, encoderHidden.Length), [1, EncoderSequenceLength, HiddenSize]);
        using var logitsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, logits, [1, ids.Length, VocabSize]);
        _decoder.Run([InputIdsName, EncoderHiddenStatesName], [idsValue, hiddenValue], [LogitsName], [logitsValue], cancellationToken);
    }

    private static bool[] BuildSuppressMask()
    {
        var mask = new bool[VocabSize];
        foreach (int t in SuppressTokens)
        {
            mask[t] = true;
        }

        return mask;
    }
}
