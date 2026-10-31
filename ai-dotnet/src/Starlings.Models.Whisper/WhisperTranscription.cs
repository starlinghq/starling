namespace Starlings.Models.Whisper;

/// <summary>
/// One segment of a timestamped transcription (<see cref="WhisperTranscriber.TranscribeWithTimestamps"/>): its start and
/// end on the clip's timeline in seconds, its text, and its token ids (timestamp tokens included, end of text excluded).
/// </summary>
/// <param name="Start">Start in seconds: the window's offset plus the opening timestamp (multiples of 0.02 s).</param>
/// <param name="End">End in seconds: the window's offset plus the closing timestamp, or the window's end when the
/// decoder closed no segment.</param>
/// <param name="Text">The segment's text: the ids decoded without special or timestamp tokens.</param>
/// <param name="Ids">The segment's token ids, as the decoder emitted them in its window.</param>
public sealed record WhisperSegment(double Start, double End, string Text, IReadOnlyList<int> Ids);

/// <summary>One decoded window of a timestamped transcription, for diagnostics.</summary>
/// <param name="SeekFrame">The window's first mel frame (10 ms per frame).</param>
/// <param name="Ids">The decoded ids, the prompt included, ending in end of text unless the decoder ran out of positions.</param>
public sealed record WhisperWindow(int SeekFrame, IReadOnlyList<int> Ids);

/// <summary>A timestamped transcription: the full text, its segments in order, and the decoded windows.</summary>
/// <param name="Text">The transcript: every segment's ids decoded together without special or timestamp tokens.</param>
/// <param name="Segments">The segments in time order.</param>
/// <param name="Windows">The windows the sequential decode ran, in order.</param>
public sealed record WhisperTranscription(string Text, IReadOnlyList<WhisperSegment> Segments, IReadOnlyList<WhisperWindow> Windows);

/// <summary>
/// The split of one decoded window into segments and the seek advance, restated from transformers 5.18.0
/// <c>WhisperGenerationMixin._retrieve_segment</c> (no token-level timestamps).
/// </summary>
internal static class WhisperSegmenter
{
    /// <summary>Mel frames per encoder position (the encoder's two convolutions have strides 1 and 2).</summary>
    public const int InputStride = 2;

    /// <summary>Seconds per mel frame.</summary>
    public const float FeaturePrecision = 0.01f;

    /// <summary>
    /// Splits a window's generated tokens (no prompt, no end of text) into segments with absolute times, and returns
    /// how many mel frames the next window starts after this one's first frame.
    /// </summary>
    public static List<(double Start, double End, int[] Ids)> Split(int[] tokens, int seek, int seekFrames, out int advance)
    {
        // Same order of float64 operations as transformers: seek * time_precision / input_stride.
        double offset = seek * WhisperModel.TimePrecision / InputStride;
        var segments = new List<(double, double, int[])>();
        bool singleEnding = tokens.Length >= 2 && tokens[^2] < WhisperModel.TimestampBegin && tokens[^1] >= WhisperModel.TimestampBegin;
        var slices = new List<int>();
        for (int i = 0; i + 1 < tokens.Length; i++)
        {
            if (tokens[i] >= WhisperModel.TimestampBegin && tokens[i + 1] >= WhisperModel.TimestampBegin)
            {
                slices.Add(i + 1);
            }
        }

        if (slices.Count > 0)
        {
            if (singleEnding)
            {
                slices.Add(tokens.Length);
            }
            else
            {
                // Keep the closing timestamp of the last complete segment; the text after it is dropped.
                slices[^1] += 1;
            }

            int lastSlice = 0;
            for (int k = 0; k < slices.Count; k++)
            {
                int[] piece = tokens[lastSlice..slices[k]];
                int endIndex = k != slices.Count - 1 || singleEnding ? piece.Length - 1 : piece.Length - 2;
                segments.Add((
                    offset + (piece[0] - WhisperModel.TimestampBegin) * WhisperModel.TimePrecision,
                    offset + (piece[endIndex] - WhisperModel.TimestampBegin) * WhisperModel.TimePrecision,
                    piece));
                lastSlice = slices[k];
            }

            advance = singleEnding ? seekFrames : (tokens[lastSlice - 2] - WhisperModel.TimestampBegin) * InputStride;
            return segments;
        }

        // No closed segment: the whole window is one segment ending at its last timestamp, or at the window's end.
        // transformers computes the window's end position in float32 (a long tensor times Python floats).
        int lastPosition = (int)((float)seekFrames * FeaturePrecision / (float)WhisperModel.TimePrecision);
        double end = offset + lastPosition * WhisperModel.TimePrecision;
        for (int i = tokens.Length - 1; i >= 0; i--)
        {
            if (tokens[i] >= WhisperModel.TimestampBegin)
            {
                if (tokens[i] != WhisperModel.TimestampBegin)
                {
                    end = offset + (tokens[i] - WhisperModel.TimestampBegin) * WhisperModel.TimePrecision;
                }

                break;
            }
        }

        segments.Add((offset, end, tokens));
        advance = seekFrames;
        return segments;
    }
}
