namespace Starlings.Models.Whisper;

/// <summary>
/// Speech-to-text for <c>openai/whisper-large-v3-turbo</c>: 16 kHz audio to the log-mel spectrogram
/// (<see cref="WhisperMelSpectrogram"/>), through the encoder and greedy decoder (<see cref="WhisperModel"/>), and back
/// to text (<see cref="WhisperTokenizer"/>), as plain text (<see cref="Transcribe"/>) or as segments with timestamps
/// (<see cref="TranscribeWithTimestamps"/>). Audio longer than 30 s is transcribed sequentially in 30 s windows, as
/// transformers long-form generation does.
/// </summary>
public sealed class WhisperTranscriber : IDisposable
{
    private readonly WhisperModel _model;
    private readonly WhisperTokenizer _tokenizer;
    private readonly WhisperOptions _options;
    private readonly bool _ownsModel;
    private int _disposed;

    private WhisperTranscriber(WhisperModel model, WhisperTokenizer tokenizer, WhisperOptions options, bool ownsModel)
    {
        _model = model;
        _tokenizer = tokenizer;
        _options = options;
        _ownsModel = ownsModel;
    }

    /// <summary>
    /// Loads the encoder, decoder and tokenizer of verified files (<see cref="WhisperModelAssets.OpenLocal"/>) and owns
    /// them. The tokenizer must be the pinned file (<see cref="WhisperModelAssets.TokenizerSha256"/>); it is checked before
    /// the graphs load. Graph digests the snapshot already verified are passed to the sessions
    /// (<see cref="WhisperModel.Create(WhisperModelFiles, WhisperOptions?)"/>), so the graphs are not read again.
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer file is not the pinned one.</exception>
    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="WhisperOptions.Language"/> is not a language code of the model.</exception>
    public static WhisperTranscriber Create(WhisperModelFiles files, WhisperOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new WhisperOptions();
        options.Validate();
        var tokenizer = WhisperTokenizer.FromFile(files.TokenizerPath, WhisperModelAssets.TokenizerSha256);
        var model = WhisperModel.Create(files, options);
        return new WhisperTranscriber(model, tokenizer, options, ownsModel: true);
    }

    /// <summary>
    /// Loads the encoder, the full-context decoder and the tokenizer and owns them (the decoder loop is full-context;
    /// the four-path overload adds the KV-cache graph). The tokenizer must be the pinned file
    /// (<see cref="WhisperModelAssets.TokenizerSha256"/>); it is checked before the graphs load. The graphs are checked
    /// only when <see cref="Starlings.Runtime.Onnx.OnnxSessionOptions.ExpectedModelSha256"/> is set on the session options.
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer file is not the pinned one.</exception>
    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="WhisperOptions.Language"/> is not a language code of the model.</exception>
    public static WhisperTranscriber Create(string encoderPath, string decoderPath, string tokenizerPath, WhisperOptions? options = null)
    {
        options ??= new WhisperOptions();
        options.Validate();
        var tokenizer = WhisperTokenizer.FromFile(tokenizerPath, WhisperModelAssets.TokenizerSha256);
        var model = WhisperModel.Create(encoderPath, decoderPath, options);
        return new WhisperTranscriber(model, tokenizer, options, ownsModel: true);
    }

    /// <summary>
    /// As <see cref="Create(string, string, string, WhisperOptions?)"/>, with the KV-cache decoder graph
    /// (<c>decoder_with_past_model_fp16.onnx</c>), which the decoder loop uses by default
    /// (<see cref="WhisperOptions.DecoderMode"/>).
    /// </summary>
    /// <exception cref="InvalidDataException">The tokenizer file is not the pinned one.</exception>
    /// <exception cref="WhisperUnsupportedLanguageException"><see cref="WhisperOptions.Language"/> is not a language code of the model.</exception>
    public static WhisperTranscriber Create(
        string encoderPath, string decoderPath, string decoderWithPastPath, string tokenizerPath, WhisperOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(decoderWithPastPath);
        options ??= new WhisperOptions();
        options.Validate();
        var tokenizer = WhisperTokenizer.FromFile(tokenizerPath, WhisperModelAssets.TokenizerSha256);
        var model = WhisperModel.Create(encoderPath, decoderPath, decoderWithPastPath, options);
        return new WhisperTranscriber(model, tokenizer, options, ownsModel: true);
    }

    /// <summary>
    /// Transcribes one 16 kHz float32 clip. A clip whose spectrogram fits one window (at most 3000 frames: up to
    /// 480,159 samples) is decoded in one window: up to 30 s it is zero-padded to 30 s, and the 159 samples past 30 s
    /// only extend the last frame (<see cref="WhisperMelSpectrogram.ExtractLongForm"/> still yields 3000 frames), as
    /// transformers treats them (short-form, <c>frames &lt;= 3000</c>). A longer clip is long-form: the text of
    /// <see cref="TranscribeWithTimestamps"/>, because sequential long-form decoding needs the timestamps (transformers
    /// enables them for long-form input and refuses <c>return_timestamps=False</c>).
    /// </summary>
    /// <exception cref="ArgumentException">The clip is empty.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled, also during a long clip's spectrogram.</exception>
    public string Transcribe(float[] waveform, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        // Refuse before the spectrogram, the costliest CPU step ahead of the encoder.
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (IsLongForm(waveform.Length))
        {
            return TranscribeWithTimestamps(waveform, cancellationToken).Text;
        }

        return TranscribeMel(Features(waveform, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Whether a clip of <paramref name="samples"/> is long-form: its spectrogram has more than one window of frames
    /// (<c>samples / 160 &gt; 3000</c>, from 480,160 samples), transformers' <c>is_shortform = frames &lt;= 3000</c>.
    /// </summary>
    internal static bool IsLongForm(int samples) => samples / WhisperMelSpectrogram.HopLength > WhisperMelSpectrogram.NumFrames;

    /// <summary>The clip's spectrogram: zero-padded to 30 s up to 480,000 samples, otherwise over the whole clip.</summary>
    private static float[] Features(float[] waveform, CancellationToken cancellationToken) =>
        waveform.Length > WhisperMelSpectrogram.MaxSamples
            ? WhisperMelSpectrogram.ExtractLongForm(waveform, out _, cancellationToken)
            : WhisperMelSpectrogram.ExtractChunk(waveform);

    /// <summary>
    /// Transcribes a log-mel spectrogram (<c>[128, 3000]</c>) to text. With <see cref="WhisperOptions.NoTimestamps"/>
    /// false this is the text of <see cref="TranscribeMelWithTimestamps"/>.
    /// </summary>
    public string TranscribeMel(float[] mel, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_options.NoTimestamps)
        {
            return TranscribeMelWithTimestamps(mel, cancellationToken).Text;
        }

        var hidden = _model.Encode(mel, cancellationToken);
        var prompt = WhisperModel.Prompt(_options.Language, transcribe: true, noTimestamps: true);
        var ids = _model.Generate(hidden, prompt, _options.MaxLength, cancellationToken);
        return _tokenizer.Decode(ids);
    }

    /// <summary>
    /// Transcribes one 16 kHz float32 clip with segment timestamps on the clip's timeline. Up to 30 s the clip is
    /// zero-padded to one window (<see cref="WhisperMelSpectrogram.ExtractChunk"/>, as <see cref="Transcribe"/> pads it);
    /// a longer clip gets one spectrogram over all its samples (<see cref="WhisperMelSpectrogram.ExtractLongForm"/>), which
    /// the sequential decode walks in 30 s windows. See <see cref="TranscribeMelWithTimestamps"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The clip is empty.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled, also during a long clip's spectrogram.</exception>
    public WhisperTranscription TranscribeWithTimestamps(float[] waveform, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        return TranscribeMelWithTimestamps(Features(waveform, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Transcribes a log-mel spectrogram (<c>[128, frames]</c>, row-major: one 30 s window of 3000 frames, or a
    /// long-form spectrogram of any length) with segment timestamps, as transformers 5.18.0
    /// <c>generate(return_timestamps=True)</c> does with greedy decoding (sequential long-form): decode the window at
    /// the current frame with the timestamp rules (<see cref="WhisperModel.GenerateWithTimestamps"/>; frames past the end
    /// are zero), split it into segments at timestamp pairs and offset them by the window's start, then move to the last
    /// closing timestamp when the window ends inside a segment, or past the whole window otherwise, until every frame
    /// is covered. No temperature fallback, no conditioning on earlier windows, no compression-ratio, log-probability
    /// or no-speech threshold. Word-level timestamps are unsupported.
    /// </summary>
    /// <exception cref="ArgumentException">The spectrogram is empty or not a whole number of 128-bin frames.</exception>
    public WhisperTranscription TranscribeMelWithTimestamps(float[] mel, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mel);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (mel.Length == 0 || mel.Length % WhisperModel.NumMelBins != 0)
        {
            throw new ArgumentException($"Mel must hold a positive multiple of {WhisperModel.NumMelBins} floats, got {mel.Length}.", nameof(mel));
        }

        int totalFrames = mel.Length / WhisperModel.NumMelBins;
        var prompt = WhisperModel.Prompt(_options.Language, transcribe: true, noTimestamps: false);
        var windows = new List<WhisperWindow>();
        var segments = new List<WhisperSegment>();
        var allIds = new List<int>();
        var window = new float[WhisperModel.NumMelBins * WhisperModel.NumFrames];
        int seek = 0;
        while (seek < totalFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int frames = Math.Min(totalFrames - seek, WhisperModel.NumFrames);
            Array.Clear(window);
            for (int m = 0; m < WhisperModel.NumMelBins; m++)
            {
                Array.Copy(mel, m * totalFrames + seek, window, m * WhisperModel.NumFrames, frames);
            }

            var hidden = _model.Encode(window, cancellationToken);
            int[] ids = _model.GenerateWithTimestamps(hidden, prompt, _options.MaxLength, cancellationToken);
            int end = ids[^1] == WhisperModel.EndOfText ? ids.Length - 1 : ids.Length;
            int[] tokens = ids[prompt.Length..end];
            var pieces = WhisperSegmenter.Split(tokens, seek, frames, out int advance);
            foreach (var (start, stop, piece) in pieces)
            {
                segments.Add(new WhisperSegment(start, stop, _tokenizer.Decode(piece), piece));
                allIds.AddRange(piece);
            }

            windows.Add(new WhisperWindow(seek, ids));
            seek += advance;
        }

        return new WhisperTranscription(_tokenizer.Decode(allIds), segments, windows);
    }

    /// <summary>The underlying model, for diagnostics.</summary>
    public WhisperModel Model => _model;

    /// <summary>Transcription options in effect.</summary>
    public WhisperOptions Options => _options;

    /// <summary>Disposes the model if this transcriber owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsModel)
        {
            _model.Dispose();
        }
    }
}
