namespace Starlings.Models.Whisper;

/// <summary>
/// Speech-to-text for <c>openai/whisper-large-v3-turbo</c>: 16 kHz audio to the log-mel spectrogram
/// (<see cref="WhisperMelSpectrogram"/>), through the encoder and greedy decoder (<see cref="WhisperModel"/>), and back
/// to text (<see cref="WhisperTokenizer"/>). Transcribes at most 30 s of audio per call, as plain text
/// (<see cref="Transcribe"/>) or as segments with timestamps (<see cref="TranscribeWithTimestamps"/>).
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
    /// Loads the encoder, decoder and tokenizer and owns them. The tokenizer must be the pinned file
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

    /// <summary>Transcribes one 16 kHz float32 clip of up to 30 s (at most 480,000 samples; shorter clips are zero-padded).</summary>
    /// <exception cref="ArgumentException">The clip is empty or longer than 30 s (chunking is unsupported).</exception>
    public string Transcribe(float[] waveform, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        // Refuse before the 30 s spectrogram, the costliest CPU step ahead of the encoder.
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var mel = WhisperMelSpectrogram.ExtractChunk(waveform);
        return TranscribeMel(mel, cancellationToken);
    }

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
    /// Transcribes one 16 kHz float32 clip of up to 30 s with segment timestamps (shorter clips are zero-padded, as
    /// <see cref="Transcribe"/> pads them). See <see cref="TranscribeMelWithTimestamps"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The clip is empty or longer than 30 s (chunking is unsupported).</exception>
    public WhisperTranscription TranscribeWithTimestamps(float[] waveform, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var mel = WhisperMelSpectrogram.ExtractChunk(waveform);
        return TranscribeMelWithTimestamps(mel, cancellationToken);
    }

    /// <summary>
    /// Transcribes a log-mel spectrogram (<c>[128, 3000]</c>) with segment timestamps, as transformers 5.18.0
    /// <c>generate(return_timestamps=True)</c> does with greedy decoding: decode the window with the timestamp rules
    /// (<see cref="WhisperModel.GenerateWithTimestamps"/>), split it into segments at timestamp pairs, and when the
    /// window ends inside a segment, decode again from the last closing timestamp (the remaining frames, zero-padded)
    /// until the frames are covered. No temperature fallback, no conditioning on earlier text, no thresholds.
    /// Word-level timestamps are unsupported.
    /// </summary>
    public WhisperTranscription TranscribeMelWithTimestamps(float[] mel, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mel);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (mel.Length != WhisperModel.NumMelBins * WhisperModel.NumFrames)
        {
            throw new ArgumentException($"Mel must be {WhisperModel.NumMelBins * WhisperModel.NumFrames} floats, got {mel.Length}.", nameof(mel));
        }

        const int totalFrames = WhisperModel.NumFrames;
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
