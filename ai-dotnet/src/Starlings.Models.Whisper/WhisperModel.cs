using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Whisper;

/// <summary>
/// The <c>openai/whisper-large-v3-turbo</c> encoder and decoder over ONNX Runtime: encodes a log-mel spectrogram
/// (<see cref="WhisperMelSpectrogram"/>) to <c>last_hidden_state</c>, then greedily decodes token ids with the model's
/// suppression lists. Full-context decoder (no KV cache) for the first cut; transcribes at most 30 s of audio.
/// </summary>
/// <remarks>
/// <para><b>Prompt.</b> The decode prompt is <c>[&lt;|startoftranscript|&gt;, language, &lt;|transcribe|&gt;,
/// &lt;|notimestamps|&gt;]</c>. The language token is a two-letter code id (see <see cref="LanguageToId"/>) or
/// <see cref="StartOfLanguageModel"/> (auto-detect).</para>
/// <para><b>Suppression.</b> <see cref="SuppressTokens"/> are masked on every step; <see cref="BeginSuppressTokens"/>
/// ([220, eos]) on the first step only, mirroring transformers <c>SuppressTokensLogitsProcessor</c> +
/// <c>SuppressTokensAtBeginLogitsProcessor</c>.</para>
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
    /// <summary><c>&lt;|startoflm|&gt;</c> — auto-detect language.</summary>
    public const int StartOfLanguageModel = 50361;
    /// <summary><c>&lt;|transcribe|&gt;</c>.</summary>
    public const int Transcribe = 50360;
    /// <summary><c>&lt;|translate|&gt;</c>.</summary>
    public const int Translate = 50359;
    /// <summary><c>&lt;|notimestamps|&gt;</c>.</summary>
    public const int NoTimestamps = 50364;
    /// <summary><c>&lt;|en|&gt;</c> language token.</summary>
    public const int English = 50259;

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

    private readonly OnnxModelSession _encoder;
    private readonly OnnxModelSession _decoder;
    private readonly bool _ownsSessions;
    private int _disposed;

    private WhisperModel(OnnxModelSession encoder, OnnxModelSession decoder, bool ownsSessions)
    {
        _encoder = encoder;
        _decoder = decoder;
        _ownsSessions = ownsSessions;
    }

    /// <summary>Loads the encoder and decoder graphs and owns the resulting sessions.</summary>
    public static WhisperModel Create(string encoderPath, string decoderPath, WhisperOptions? options = null)
    {
        options ??= new WhisperOptions();
        options.Validate();
        var encoder = OnnxModelSession.Create(encoderPath, options.EncoderSession);
        try
        {
            var decoder = OnnxModelSession.Create(decoderPath, options.DecoderSession);
            try
            {
                return new WhisperModel(encoder, decoder, ownsSessions: true);
            }
            catch
            {
                decoder.Dispose();
                throw;
            }
        }
        catch
        {
            encoder.Dispose();
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
        var memoryInfo = OrtMemoryInfo.DefaultInstance;
        using var idsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, ids.AsMemory(0, ids.Length), [1, ids.Length]);
        using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, encoderHidden.AsMemory(0, encoderHidden.Length), [1, EncoderSequenceLength, HiddenSize]);
        var logits = new float[ids.Length * VocabSize];
        using var logitsValue = OrtValue.CreateTensorValueFromMemory(memoryInfo, logits.AsMemory(0, logits.Length), [1, ids.Length, VocabSize]);
        _decoder.Run([InputIdsName, EncoderHiddenStatesName], [idsValue, hiddenValue], [LogitsName], [logitsValue], cancellationToken);
        return logits;
    }

    /// <summary>Builds the decode prompt for the given language (or auto) and task.</summary>
    public static int[] Prompt(string? language = null, bool transcribe = true, bool noTimestamps = true)
    {
        int lang = language is null ? StartOfLanguageModel : LanguageToId[language.ToLowerInvariant()];
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
    public int[] Generate(float[] encoderHidden, int[] prompt, int maxLength = 448, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Length is 0 || prompt.Length >= MaxTargetPositions)
        {
            throw new ArgumentException($"The prompt must hold 1 to {MaxTargetPositions - 1} tokens, got {prompt.Length}.", nameof(prompt));
        }

        var ids = new List<int>(prompt);
        for (int step = 0; step < maxLength && ids.Count < MaxTargetPositions; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var logits = DecodeLogits(ids.Select(i => (long)i).ToArray(), encoderHidden, cancellationToken);
            var last = logits.AsSpan((ids.Count - 1) * VocabSize, VocabSize);
            ApplySuppression(last, step == 0);
            ids.Add(ArgMax(last));
            if (ids[^1] == EndOfText)
            {
                break;
            }
        }

        return ids.ToArray();
    }

    /// <summary>Disposes the sessions if this model owns them. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsSessions)
        {
            _decoder.Dispose();
            _encoder.Dispose();
        }
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
