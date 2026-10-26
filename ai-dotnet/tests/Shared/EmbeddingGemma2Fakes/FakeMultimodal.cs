using Starlings.Models.EmbeddingGemma2;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// The multimodal embedder over the weight-free graphs (TestModels: FakeEmbeddingGemma2, FakeEmbeddingGemma2Vision,
/// FakeEmbeddingGemma2Multimodal) and <see cref="MarkerTokenizer"/>. Runs in CI.
/// </summary>
internal static class FakeMultimodal
{
    private static readonly OnnxSessionOptions Session = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    public static EmbeddingGemma2MultimodalEmbedder Create(EmbeddingGemma2MultimodalOptions? options = null, bool withTextEncoder = true)
    {
        options ??= new EmbeddingGemma2MultimodalOptions();
        var vision = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Vision, Session);
        var multimodal = OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2Multimodal, Session);
        var text = withTextEncoder
            ? new EmbeddingGemma2Encoder(OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, Session), new EmbeddingGemma2EncoderOptions { Dimension = options.Dimension, MaxSequenceLength = options.MaxSequenceLength }, ownsSession: true)
            : null;
        return new EmbeddingGemma2MultimodalEmbedder(new MarkerTokenizer(), vision, multimodal, text, options, ownsSessions: true);
    }

    /// <summary>A deterministic image whose bytes depend on <paramref name="seed"/>.</summary>
    public static EmbeddingGemma2Image Image(int width, int height, int seed)
    {
        var bytes = new byte[width * height * 3];
        uint state = (uint)(seed * 2654435761u) | 1;
        for (int i = 0; i < bytes.Length; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bytes[i] = (byte)state;
        }

        return EmbeddingGemma2Image.FromRgb24(width, height, bytes);
    }
}

/// <summary>
/// A stand-in tokenizer: <c>&lt;bos&gt;</c>, then the media markers as their ids and every other character as its code
/// point plus 10, then <c>&lt;eos&gt;</c>.
/// </summary>
internal sealed class MarkerTokenizer : IEmbeddingGemma2Tokenizer
{
    private static readonly (string Marker, long Id)[] Markers = [("<|image|>", 258_880), ("<|image>", 255_999), ("<image|>", 258_882), ("<|video|>", 258_884)];

    public EmbeddingGemma2Input Tokenize(string text)
    {
        var ids = new List<long> { 2 };
        for (int i = 0; i < text.Length;)
        {
            bool matched = false;
            foreach (var (marker, id) in Markers)
            {
                if (string.CompareOrdinal(text, i, marker, 0, marker.Length) == 0)
                {
                    ids.Add(id);
                    i += marker.Length;
                    matched = true;
                    break;
                }
            }

            if (!matched)
            {
                ids.Add(text[i] + 10);
                i++;
            }
        }

        ids.Add(1);
        return new EmbeddingGemma2Input(ids.ToArray());
    }
}
