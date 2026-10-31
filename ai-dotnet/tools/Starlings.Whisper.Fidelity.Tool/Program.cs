// Whisper development-parity evidence for model-pipeline's whisper-onnx-cpu-v1 route. For each reference clip:
//   dotnet-mel:   the reference log-mel spectrogram through WhisperModel: encoder hidden state, greedy ids, decoder
//                 logits teacher-forced on the reference ids, and the owned tokenizer's transcript of those ids;
//   dotnet-audio: the clip's WAV samples through WhisperMelSpectrogram (the .NET mel) and the public
//                 WhisperTranscriber.Transcribe path.
// Both paths also decode with segment timestamps (TranscribeMelWithTimestamps from the reference mel,
// TranscribeWithTimestamps from the WAV samples): windows, segments and transcript. A clip over 30 s is long-form: the
// .NET whole-clip mel (WhisperMelSpectrogram.ExtractLongForm), the timestamped decodes and the public Transcribe text.
// The mel path runs the full-context decoder loop (the reference path); the KV-cache loop (the library default) decodes
// the reference mel again (kv_ids with each step's logits, kv_timestamped) and serves the public audio paths.
// Writes dotnet.json plus a float32 sidecar in the layout run_whisper.py uses; the evaluator compares them.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.Whisper;
using Starlings.Runtime.Onnx;

return ToolMain.Run(ToolBody);

int ToolBody()
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || args[i] is not ("--profile" or "--snapshot" or "--corpus" or "--reference" or "--out" or "--threads")
            || !options.TryAdd(args[i], args[i + 1]))
        {
            Console.Error.WriteLine("usage: --profile <candidate.json> --snapshot <dir> --corpus <asr-corpus.jsonl> --reference <reference.json> --out <dir> [--threads 2]");
            return 2;
        }
    }

    if (ToolSettings.RequiredOptions.Any(key => !options.ContainsKey(key)))
    {
        Console.Error.WriteLine("missing --profile, --snapshot, --corpus, --reference, or --out");
        return 2;
    }

    // The WAV reader and the tensor sidecars are little-endian.
    if (!BitConverter.IsLittleEndian)
    {
        throw new PlatformNotSupportedException("WAV samples and fidelity tensor sidecars require a little-endian host.");
    }

    int threads = options.TryGetValue("--threads", out var rawThreads) ? int.Parse(rawThreads, System.Globalization.CultureInfo.InvariantCulture) : 2;
    string profilePath = options["--profile"];
    using var profileDocument = JsonDocument.Parse(File.ReadAllText(profilePath));
    var profile = profileDocument.RootElement;
    Contract.Check(profile);
    string profileId = profile.GetProperty("id").GetString()!;
    string profileHash = Sha256(profilePath);

    using var referenceDocument = JsonDocument.Parse(File.ReadAllText(options["--reference"]));
    var reference = referenceDocument.RootElement;
    if (reference.GetProperty("profile").GetString() != profileId || reference.GetProperty("candidate_sha256").GetString() != profileHash)
    {
        throw new InvalidDataException("Reference inputs do not belong to the selected candidate contract.");
    }

    if (reference.GetProperty("cases").GetArrayLength() < 2)
    {
        throw new InvalidDataException("Reference inputs require at least two clips.");
    }

    var corpus = ReadCorpus(options["--corpus"]);
    var referenceMel = ReadReferenceMel(options["--reference"], reference);
    var files = WhisperModelAssets.OpenDirectory(options["--snapshot"]);
    string output = Path.GetFullPath(options["--out"]);
    Directory.CreateDirectory(output);

    var session = new OnnxSessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = 1 };
    var tokenizer = WhisperTokenizer.FromFile(files.TokenizerPath, WhisperModelAssets.TokenizerSha256);
    var rows = new List<Dictionary<string, object>>();
    var layout = new List<Dictionary<string, object>>();
    string tensorPath = Path.Combine(output, "dotnet-tensors.f32");
    float[]? cancellationClip = null;
    // Two transcribers per language, full-context and KV-cache: each owns its encoder and decoder sessions (about 2 GB).
    var transcribers = new Dictionary<string, WhisperTranscriber>(StringComparer.Ordinal);
    var cachedTranscribers = new Dictionary<string, WhisperTranscriber>(StringComparer.Ordinal);
    try
    {
        using var tensors = File.Create(tensorPath);
        foreach (var source in reference.GetProperty("cases").EnumerateArray())
        {
            string id = source.GetProperty("id").GetString()!;
            string language = source.GetProperty("language").GetString()!;
            if (!corpus.TryGetValue(id, out var clip) || clip.Sha256 != source.GetProperty("audio_sha256").GetString())
            {
                throw new InvalidDataException($"{id}: the reference clip is missing from the corpus or differs from it.");
            }

            if (!transcribers.TryGetValue(language, out var transcriber))
            {
                // Default MaxLength: the library stops at max_target_positions tokens, the prompt included, as transformers does.
                var whisperOptions = new WhisperOptions
                {
                    Language = language,
                    EncoderSession = session,
                    DecoderSession = session,
                    DecoderWithPastSession = session,
                    DecoderMode = WhisperDecoderMode.FullContext,
                };
                transcriber = WhisperTranscriber.Create(files.EncoderPath, files.DecoderPath, files.TokenizerPath, whisperOptions);
                transcribers.Add(language, transcriber);
                var cached = WhisperTranscriber.Create(
                    files.EncoderPath, files.DecoderPath, files.DecoderWithPastPath!, files.TokenizerPath, whisperOptions with { DecoderMode = null });
                if (cached.Model.DecoderMode != WhisperDecoderMode.KvCache)
                {
                    throw new InvalidOperationException("The default decoder loop with the KV-cache graph is not the KV cache.");
                }

                cachedTranscribers.Add(language, cached);
            }

            var cachedTranscriber = cachedTranscribers[language];

            if (source.GetProperty("long_form").GetBoolean())
            {
                float[] longMel = WhisperMelSpectrogram.ExtractLongForm(clip.Waveform, out int frames);
                Write(tensors, layout, id, "mel", longMel, [WhisperModel.NumMelBins, frames]);
                rows.Add(new Dictionary<string, object>
                {
                    ["id"] = id,
                    ["language"] = language,
                    ["audio_sha256"] = clip.Sha256,
                    ["long_form"] = true,
                    ["timestamped"] = Timestamped(transcriber.TranscribeMelWithTimestamps(referenceMel[id])),
                    ["kv_timestamped"] = Timestamped(cachedTranscriber.TranscribeMelWithTimestamps(referenceMel[id])),
                    ["audio_timestamped"] = Timestamped(cachedTranscriber.TranscribeWithTimestamps(clip.Waveform)),
                    ["audio_transcript"] = cachedTranscriber.Transcribe(clip.Waveform),
                });
                continue;
            }

            int[] forced = source.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
            var model = transcriber.Model;
            float[] hidden = model.Encode(referenceMel[id]);
            int[] ids = model.Generate(hidden, WhisperModel.Prompt(language), transcriber.Options.MaxLength);
            float[] logits = model.DecodeLogits(forced[..^1].Select(value => (long)value).ToArray(), hidden);
            float[] mel = WhisperMelSpectrogram.ExtractChunk(clip.Waveform);
            Write(tensors, layout, id, "mel", mel, [WhisperModel.NumMelBins, WhisperModel.NumFrames]);
            Write(tensors, layout, id, "hidden", hidden, [WhisperModel.EncoderSequenceLength, WhisperModel.HiddenSize]);
            Write(tensors, layout, id, "logits", logits, [forced.Length - 1, WhisperModel.VocabSize]);
            var steps = new List<float[]>();
            int[] cachedIds = cachedTranscriber.Model.GenerateRecordingLogits(
                hidden, WhisperModel.Prompt(language), WhisperDecoderMode.KvCache, timestamps: false, steps, transcriber.Options.MaxLength);
            Write(tensors, layout, id, "kv_logits", steps.SelectMany(step => step).ToArray(), [steps.Count, WhisperModel.VocabSize]);
            rows.Add(new Dictionary<string, object>
            {
                ["id"] = id,
                ["language"] = language,
                ["audio_sha256"] = clip.Sha256,
                ["ids"] = ids,
                ["forced_ids"] = forced,
                ["transcript"] = tokenizer.Decode(ids),
                ["long_form"] = false,
                ["audio_transcript"] = cachedTranscriber.Transcribe(clip.Waveform),
                ["timestamped"] = Timestamped(transcriber.TranscribeMelWithTimestamps(referenceMel[id])),
                ["audio_timestamped"] = Timestamped(cachedTranscriber.TranscribeWithTimestamps(clip.Waveform)),
                ["kv_ids"] = cachedIds,
                ["kv_timestamped"] = Timestamped(cachedTranscriber.TranscribeMelWithTimestamps(referenceMel[id])),
            });
            cancellationClip ??= clip.Waveform;
        }
    }
    finally
    {
        foreach (var transcriber in transcribers.Values.Concat(cachedTranscribers.Values))
        {
            transcriber.Dispose();
        }
    }

    bool cancellationVerified = false;
    using (var transcriber = WhisperTranscriber.Create(files.EncoderPath, files.DecoderPath, files.TokenizerPath,
        new WhisperOptions { Language = "en", EncoderSession = session, DecoderSession = session }))
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        try
        {
            transcriber.Transcribe(cancellationClip!, cancellation.Token);
        }
        catch (OperationCanceledException)
        {
            cancellationVerified = true;
        }
    }

    var result = new Dictionary<string, object>
    {
        ["format"] = "starlings.whisper-candidate-results",
        ["format_version"] = 1,
        ["profile"] = profileId,
        ["task"] = profile.GetProperty("task").GetString()!,
        ["candidate_sha256"] = profileHash,
        ["stage"] = "dotnet",
        ["reference_status"] = "proposed",
        ["reference_sha256"] = Sha256(options["--reference"]),
        ["assets"] = new[] { WhisperModelAssets.EncoderFile, WhisperModelAssets.DecoderFile, WhisperModelAssets.DecoderWithPastFile, WhisperModelAssets.TokenizerFile }
            .Select(path => new { path, sha256 = WhisperModelAssets.Lock.Files.Single(file => file.Path == path).Sha256 }).ToArray(),
        ["environment"] = new
        {
            machine = Environment.MachineName,
            dotnet = RuntimeInformation.FrameworkDescription,
            onnxruntime = OnnxRuntimeInfo.NativeVersion,
            provider = "CPUExecutionProvider",
            threads,
            inter_op_threads = 1,
        },
        ["cases"] = rows,
        ["tensors"] = new { file = Path.GetFileName(tensorPath), sha256 = Sha256(tensorPath), cases = layout },
        ["checks"] = new { pre_cancelled_request = cancellationVerified },
    };
    File.WriteAllText(Path.Combine(output, "dotnet.json"), JsonSerializer.Serialize(result, ToolSettings.Json) + "\n");
    Console.WriteLine($"wrote {rows.Count} whisper clips (mel and audio paths) to {output}");
    return 0;

    static Dictionary<string, object> Timestamped(WhisperTranscription transcription) => new()
    {
        ["windows"] = transcription.Windows.Select(window => new Dictionary<string, object> { ["seek"] = window.SeekFrame, ["ids"] = window.Ids }).ToArray(),
        ["segments"] = transcription.Segments.Select(segment => new Dictionary<string, object>
        {
            ["start"] = segment.Start,
            ["end"] = segment.End,
            ["ids"] = segment.Ids,
            ["text"] = segment.Text,
        }).ToArray(),
        ["transcript"] = transcription.Text,
    };

    static void Write(FileStream tensors, List<Dictionary<string, object>> layout, string id, string name, float[] values, int[] shape)
    {
        if (values.Length != shape.Aggregate(1, (product, size) => product * size))
        {
            throw new InvalidDataException($"{id}/{name}: {values.Length} values do not fill shape [{string.Join(", ", shape)}].");
        }

        layout.Add(new Dictionary<string, object> { ["id"] = id, ["name"] = name, ["offset"] = tensors.Position / sizeof(float), ["shape"] = shape });
        tensors.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    }

    static Dictionary<string, float[]> ReadReferenceMel(string referencePath, JsonElement reference)
    {
        var tensors = reference.GetProperty("tensors");
        string file = tensors.GetProperty("file").GetString()!;
        if (Path.GetFileName(file) != file)
        {
            throw new InvalidDataException("Unsafe reference tensor sidecar path.");
        }

        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(referencePath))!, file);
        if (Sha256(path) != tensors.GetProperty("sha256").GetString())
        {
            throw new InvalidDataException("Reference tensor sidecar differs from its recorded SHA-256.");
        }

        byte[] blob = File.ReadAllBytes(path);
        var mel = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var entry in tensors.GetProperty("cases").EnumerateArray().Where(entry => entry.GetProperty("name").GetString() == "mel"))
        {
            long offset = entry.GetProperty("offset").GetInt64();
            int count = entry.GetProperty("shape").EnumerateArray().Aggregate(1, (product, size) => product * size.GetInt32());
            if (count <= 0 || count % WhisperModel.NumMelBins != 0 || (offset + count) * sizeof(float) > blob.Length)
            {
                throw new InvalidDataException("Reference mel tensor has the wrong shape or lies outside the sidecar.");
            }

            mel.Add(entry.GetProperty("id").GetString()!, MemoryMarshal.Cast<byte, float>(blob.AsSpan((int)(offset * sizeof(float)), count * sizeof(float))).ToArray());
        }

        return mel;
    }

    static Dictionary<string, Clip> ReadCorpus(string corpusPath)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(corpusPath))!;
        var clips = new Dictionary<string, Clip>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(corpusPath).Where(line => line.Length > 0))
        {
            using var row = JsonDocument.Parse(line);
            string id = row.RootElement.GetProperty("id").GetString()!;
            string relative = row.RootElement.GetProperty("audio").GetString()!;
            string path = Path.GetFullPath(Path.Combine(directory, relative));
            if (!path.StartsWith(directory + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{id}: audio path escapes the corpus directory.");
            }

            string sha256 = row.RootElement.GetProperty("sha256").GetString()!;
            if (Sha256(path) != sha256)
            {
                throw new InvalidDataException($"{id}: audio differs from its pinned SHA-256.");
            }

            float[] waveform = Wav.ReadPcm16Mono(path, WhisperMelSpectrogram.SampleRate);
            if (waveform.Length != row.RootElement.GetProperty("samples").GetInt32())
            {
                throw new InvalidDataException($"{id}: the WAV holds {waveform.Length} samples, not the indexed count.");
            }

            clips.Add(id, new Clip(waveform, sha256));
        }

        return clips;
    }

    static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}

internal sealed record Clip(float[] Waveform, string Sha256);

/// <summary>The candidate contract against the library's fixed constants: Python and C# must agree exactly.</summary>
internal static class Contract
{
    public static void Check(JsonElement profile)
    {
        var origin = profile.GetProperty("origin");
        Same("task", profile.GetProperty("task").GetString(), "transcription");
        Same("origin.repo", origin.GetProperty("repo").GetString(), WhisperModelAssets.Repository);
        Same("origin.revision", origin.GetProperty("revision").GetString(), WhisperModelAssets.Revision);
        var pinned = origin.GetProperty("files").EnumerateArray()
            .ToDictionary(file => file.GetProperty("path").GetString()!, file => (file.GetProperty("size").GetInt64(), file.GetProperty("sha256").GetString()!));
        foreach (var file in WhisperModelAssets.Lock.Files)
        {
            Same($"origin.files[{file.Path}]", pinned.GetValueOrDefault(file.Path), (file.Size, file.Sha256));
        }

        var runtime = profile.GetProperty("runtime");
        Same("runtime.backend", runtime.GetProperty("backend").GetString(), "onnx");
        Same("runtime.precision", runtime.GetProperty("precision").GetString(), "fp16");
        Same("runtime.provider", runtime.GetProperty("provider").GetString(), "CPUExecutionProvider");
        Same("runtime.version", runtime.GetProperty("version").GetString(), OnnxRuntimeInfo.NativeVersion);

        var behavior = profile.GetProperty("behavior");
        Same("behavior.encoder_file", behavior.GetProperty("encoder_file").GetString(), WhisperModelAssets.EncoderFile);
        Same("behavior.decoder_file", behavior.GetProperty("decoder_file").GetString(), WhisperModelAssets.DecoderFile);
        Same("behavior.decoder_with_past_file", behavior.GetProperty("decoder_with_past_file").GetString(), WhisperModelAssets.DecoderWithPastFile);
        Same("behavior.decoder_layers", behavior.GetProperty("decoder_layers").GetInt32(), WhisperModel.DecoderLayers);
        Same("behavior.tokenizer_file", behavior.GetProperty("tokenizer_file").GetString(), WhisperModelAssets.TokenizerFile);
        Same("behavior.sample_rate", behavior.GetProperty("sample_rate").GetInt32(), WhisperMelSpectrogram.SampleRate);
        Same("behavior.n_fft", behavior.GetProperty("n_fft").GetInt32(), WhisperMelSpectrogram.Nfft);
        Same("behavior.hop_length", behavior.GetProperty("hop_length").GetInt32(), WhisperMelSpectrogram.HopLength);
        Same("behavior.num_mel_bins", behavior.GetProperty("num_mel_bins").GetInt32(), WhisperModel.NumMelBins);
        Same("behavior.max_samples", behavior.GetProperty("max_samples").GetInt32(), WhisperMelSpectrogram.MaxSamples);
        Same("behavior.num_frames", behavior.GetProperty("num_frames").GetInt32(), WhisperModel.NumFrames);
        Same("behavior.encoder_positions", behavior.GetProperty("encoder_positions").GetInt32(), WhisperModel.EncoderSequenceLength);
        Same("behavior.hidden_size", behavior.GetProperty("hidden_size").GetInt32(), WhisperModel.HiddenSize);
        Same("behavior.vocabulary_size", behavior.GetProperty("vocabulary_size").GetInt32(), WhisperModel.VocabSize);
        Same("behavior.max_target_positions", behavior.GetProperty("max_target_positions").GetInt32(), WhisperModel.MaxTargetPositions);
        // Transcribe always prompts with <|transcribe|>; the suppression lists are private to WhisperModel and are compared
        // with the contract from WhisperModel.cs by model-pipeline's tests.
        Same("behavior.task", behavior.GetProperty("task").GetString(), "transcribe");
        var timestamps = behavior.GetProperty("timestamps");
        Same("behavior.timestamps.level", timestamps.GetProperty("level").GetString(), "segment");
        Same("behavior.timestamps.timestamp_begin", timestamps.GetProperty("timestamp_begin").GetInt32(), WhisperModel.TimestampBegin);
        Same("behavior.timestamps.max_initial_timestamp_index", timestamps.GetProperty("max_initial_timestamp_index").GetInt32(), WhisperModel.MaxInitialTimestampIndex);
        Same("behavior.timestamps.time_precision", timestamps.GetProperty("time_precision").GetDouble(), WhisperModel.TimePrecision);
        Same("behavior.timestamps.feature_precision", timestamps.GetProperty("feature_precision").GetDouble(), 0.01);
        Same("behavior.timestamps.input_stride", timestamps.GetProperty("input_stride").GetInt32(), 2);
        // TranscribeMelWithTimestamps: sequential windows over a whole-clip spectrogram, greedy, no fallback, no
        // conditioning on earlier windows, no thresholds.
        var longForm = behavior.GetProperty("long_form");
        Same("behavior.long_form.strategy", longForm.GetProperty("strategy").GetString(), "sequential");
        Same("behavior.long_form.features", longForm.GetProperty("features").GetString(), "full-clip");
        Same("behavior.long_form.temperature", longForm.GetProperty("temperature").GetDouble(), 0.0);
        Same("behavior.long_form.condition_on_prev_tokens", longForm.GetProperty("condition_on_prev_tokens").GetBoolean(), false);
        foreach (string threshold in new[] { "compression_ratio_threshold", "logprob_threshold", "no_speech_threshold" })
        {
            Same($"behavior.long_form.{threshold}", longForm.GetProperty(threshold).ValueKind, JsonValueKind.Null);
        }
        Same("behavior.decoding", behavior.GetProperty("decoding").GetString(), "greedy");
        var special = behavior.GetProperty("special_tokens");
        Same("special_tokens.end_of_text", special.GetProperty("end_of_text").GetInt32(), WhisperModel.EndOfText);
        Same("special_tokens.start_of_transcript", special.GetProperty("start_of_transcript").GetInt32(), WhisperModel.StartOfTranscript);
        Same("special_tokens.translate", special.GetProperty("translate").GetInt32(), WhisperModel.Translate);
        Same("special_tokens.transcribe", special.GetProperty("transcribe").GetInt32(), WhisperModel.Transcribe);
        Same("special_tokens.start_of_lm", special.GetProperty("start_of_lm").GetInt32(), WhisperModel.StartOfLanguageModel);
        Same("special_tokens.no_timestamps", special.GetProperty("no_timestamps").GetInt32(), WhisperModel.NoTimestamps);
        foreach (var language in behavior.GetProperty("languages").EnumerateObject())
        {
            Same($"languages.{language.Name}", language.Value.GetInt32(), WhisperModel.LanguageToId[language.Name]);
        }
    }

    private static void Same<T>(string field, T actual, T expected)
    {
        if (!EqualityComparer<T>.Default.Equals(actual, expected))
        {
            throw new InvalidDataException($"Unsupported Whisper contract field {field}: expected {expected}, found {actual}.");
        }
    }
}

/// <summary>Minimal RIFF/WAVE reader for 16-bit PCM mono clips: samples / 32768, exact in float32 (as run_whisper.py reads them).</summary>
internal static class Wav
{
    public static float[] ReadPcm16Mono(string path, int sampleRate)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var span = bytes.AsSpan();
        if (bytes.Length < 12 || !span[..4].SequenceEqual("RIFF"u8) || !span[8..12].SequenceEqual("WAVE"u8))
        {
            throw new InvalidDataException($"{path}: not a RIFF/WAVE file.");
        }

        bool format = false;
        for (int position = 12; position + 8 <= bytes.Length;)
        {
            var id = span.Slice(position, 4);
            int size = BitConverter.ToInt32(bytes, position + 4);
            int body = position + 8;
            if (size < 0 || body + size > bytes.Length)
            {
                throw new InvalidDataException($"{path}: truncated chunk.");
            }

            if (id.SequenceEqual("fmt "u8))
            {
                format = size >= 16 && BitConverter.ToInt16(bytes, body) == 1 && BitConverter.ToInt16(bytes, body + 2) == 1
                    && BitConverter.ToInt32(bytes, body + 4) == sampleRate && BitConverter.ToInt16(bytes, body + 14) == 16;
                if (!format)
                {
                    throw new InvalidDataException($"{path}: expected 16-bit mono {sampleRate} Hz PCM.");
                }
            }
            else if (id.SequenceEqual("data"u8))
            {
                if (!format || size % 2 != 0)
                {
                    throw new InvalidDataException($"{path}: data before format, or an incomplete sample.");
                }

                var samples = MemoryMarshal.Cast<byte, short>(span.Slice(body, size));
                var waveform = new float[samples.Length];
                for (int i = 0; i < samples.Length; i++)
                {
                    waveform[i] = samples[i] / 32768f;
                }

                return waveform;
            }

            position = body + size + (size & 1);
        }

        throw new InvalidDataException($"{path}: no data chunk.");
    }
}

internal static class ToolSettings
{
    public static readonly string[] RequiredOptions = ["--profile", "--snapshot", "--corpus", "--reference", "--out"];
    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
}
