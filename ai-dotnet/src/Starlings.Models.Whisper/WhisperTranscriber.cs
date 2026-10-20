namespace Starlings.Models.Whisper;

/// <summary>
/// Speech-to-text for <c>openai/whisper-large-v3-turbo</c>: 16 kHz audio to the log-mel spectrogram
/// (<see cref="WhisperMelSpectrogram"/>), through the encoder and greedy decoder (<see cref="WhisperModel"/>), and back
/// to text (<see cref="WhisperTokenizer"/>). Transcribes at most 30 s of audio per call.
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

    /// <summary>Loads the encoder, decoder and tokenizer and owns them.</summary>
    public static WhisperTranscriber Create(string encoderPath, string decoderPath, string tokenizerPath, WhisperOptions? options = null)
    {
        options ??= new WhisperOptions();
        options.Validate();
        var model = WhisperModel.Create(encoderPath, decoderPath, options);
        try
        {
            var tokenizer = WhisperTokenizer.FromFile(tokenizerPath);
            return new WhisperTranscriber(model, tokenizer, options, ownsModel: true);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>Transcribes one 16 kHz float32 waveform (480,000 samples for a 30 s clip).</summary>
    public string Transcribe(float[] waveform, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        var mel = WhisperMelSpectrogram.Extract(waveform);
        return TranscribeMel(mel, cancellationToken);
    }

    /// <summary>Transcribes a log-mel spectrogram (<c>[128, 3000]</c>) to text.</summary>
    public string TranscribeMel(float[] mel, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var hidden = _model.Encode(mel, cancellationToken);
        var prompt = WhisperModel.Prompt(_options.Language, transcribe: true, noTimestamps: _options.NoTimestamps);
        var ids = _model.Generate(hidden, prompt, _options.MaxLength, cancellationToken);
        return _tokenizer.Decode(ids);
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
