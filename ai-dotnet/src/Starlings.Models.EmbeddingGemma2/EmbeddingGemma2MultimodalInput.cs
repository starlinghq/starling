using System.Text;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// One input to embed: text, images and videos in order. It renders as upstream's chat-message format does: the text
/// parts concatenated with no separator, each image as <c>&lt;|image&gt;</c> + n x <c>&lt;|image|&gt;</c> +
/// <c>&lt;image|&gt;</c> at its position (n is the image's soft-token count), each video as one such block per sampled
/// frame with <c>&lt;|video|&gt;</c> in place of <c>&lt;|image|&gt;</c> (no separator, no timestamps), then
/// <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>. Mean pooling covers every token, the media tokens included.
/// </summary>
/// <remarks>
/// <para>
/// Mapping from sentence-transformers inputs: an image alone is <c>FromImage(image)</c>; a dict
/// <c>{"text": "A cat on a blanket. &lt;|image|&gt;", "image": img}</c> is
/// <c>Create("A cat on a blanket. ", image)</c>; a dict without a placeholder in its text appends the images after
/// the text in the dict's key order (<c>{"text", "image"}</c>: <c>Create("Product photo:", image)</c>). Videos map the
/// same way: <c>{"text": "The clip &lt;|video|&gt; shows ...", "video": v}</c> is <c>Create("The clip ", video, " shows ...")</c>,
/// and a video alone is <c>FromVideo(video)</c>. Text parts must not contain media placeholders or boundary tokens
/// (<c>&lt;|image|&gt;</c>, <c>&lt;|image&gt;</c>, <c>&lt;image|&gt;</c>, <c>&lt;|audio|&gt;</c>, <c>&lt;|audio&gt;</c>,
/// <c>&lt;audio|&gt;</c>, <c>&lt;|video|&gt;</c>): media are positioned by the parts, never by placeholder text.
/// </para>
/// <para>
/// The model card recommends no prompt for media ("Prefixes apply to text only"). A prompt passed with an image or video
/// input is rendered first, as upstream does, but is not recommended.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2MultimodalInput
{
    private static readonly string[] MediaMarkers = ["<|image|>", "<|image>", "<image|>", "<|audio|>", "<|audio>", "<audio|>", "<|video|>"];

    private EmbeddingGemma2MultimodalInput(IReadOnlyList<object> parts)
    {
        Parts = parts;
        int images = 0, videos = 0;
        foreach (var part in parts)
        {
            images += part is EmbeddingGemma2Image ? 1 : 0;
            videos += part is EmbeddingGemma2Video ? 1 : 0;
        }

        ImageCount = images;
        VideoCount = videos;
    }

    /// <summary>The parts in order: <see cref="string"/> (text), <see cref="EmbeddingGemma2Image"/> or <see cref="EmbeddingGemma2Video"/>.</summary>
    public IReadOnlyList<object> Parts { get; }

    /// <summary>Number of image parts.</summary>
    public int ImageCount { get; }

    /// <summary>Number of video parts.</summary>
    public int VideoCount { get; }

    /// <summary>True when the input has at least one image.</summary>
    public bool HasImages => ImageCount > 0;

    /// <summary>True when the input has at least one video.</summary>
    public bool HasVideos => VideoCount > 0;

    /// <summary>True when the input has an image or a video (it then runs on the vision and multimodal graphs).</summary>
    public bool HasMedia => ImageCount + VideoCount > 0;

    /// <summary>The images in order.</summary>
    public IEnumerable<EmbeddingGemma2Image> Images => Parts.OfType<EmbeddingGemma2Image>();

    /// <summary>The videos in order.</summary>
    public IEnumerable<EmbeddingGemma2Video> Videos => Parts.OfType<EmbeddingGemma2Video>();

    /// <summary>An image alone (upstream renders it as <c>&lt;|image|&gt;</c>).</summary>
    public static EmbeddingGemma2MultimodalInput FromImage(EmbeddingGemma2Image image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return new EmbeddingGemma2MultimodalInput([image]);
    }

    /// <summary>A video alone (upstream renders it as <c>&lt;|video|&gt;</c>).</summary>
    public static EmbeddingGemma2MultimodalInput FromVideo(EmbeddingGemma2Video video)
    {
        ArgumentNullException.ThrowIfNull(video);
        return new EmbeddingGemma2MultimodalInput([video]);
    }

    /// <summary>Text alone (embedded on the released text graph, exactly as <see cref="EmbeddingGemma2TextEmbedder"/>).</summary>
    public static EmbeddingGemma2MultimodalInput FromText(string text) => Create(text);

    /// <summary>
    /// Text, images and videos in order. Each part is a <see cref="string"/>, an <see cref="EmbeddingGemma2Image"/> or an
    /// <see cref="EmbeddingGemma2Video"/>; empty strings are dropped.
    /// </summary>
    /// <exception cref="ArgumentException">A part is null or of another type, a text part contains a media placeholder, or there are no parts.</exception>
    public static EmbeddingGemma2MultimodalInput Create(params IEnumerable<object> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        var list = new List<object>();
        int index = 0;
        foreach (var part in parts)
        {
            switch (part)
            {
                case string text:
                    foreach (var marker in MediaMarkers)
                    {
                        if (text.Contains(marker, StringComparison.Ordinal))
                        {
                            throw new ArgumentException($"Text part {index} contains '{marker}'. Images and videos are positioned by their parts; media placeholders and boundary tokens are not accepted in text.", nameof(parts));
                        }
                    }

                    if (text.Length > 0)
                    {
                        list.Add(text);
                    }

                    break;
                case EmbeddingGemma2Image image:
                    list.Add(image);
                    break;
                case EmbeddingGemma2Video video:
                    list.Add(video);
                    break;
                case null:
                    throw new ArgumentException($"Part {index} is null.", nameof(parts));
                default:
                    throw new ArgumentException($"Part {index} is a {part.GetType().Name}; expected string, {nameof(EmbeddingGemma2Image)} or {nameof(EmbeddingGemma2Video)}.", nameof(parts));
            }

            index++;
        }

        if (index == 0)
        {
            throw new ArgumentException("An input needs at least one part.", nameof(parts));
        }

        return new EmbeddingGemma2MultimodalInput(list);
    }

    /// <summary>
    /// The text the tokenizer sees for this input with <paramref name="prompt"/>: the prompt text, the text parts, and for
    /// each media part in order its soft-token blocks (an image: <c>&lt;|image&gt;</c> + n x <c>&lt;|image|&gt;</c> +
    /// <c>&lt;image|&gt;</c>; a video: per sampled frame <c>&lt;|image&gt;</c> + n x <c>&lt;|video|&gt;</c> +
    /// <c>&lt;image|&gt;</c>). <paramref name="media"/> gives, per media part in order, the frame count (1 for an image)
    /// and the soft tokens per frame. <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c> are added by the tokenizer.
    /// </summary>
    internal string Render(EmbeddingGemma2Prompt prompt, IReadOnlyList<(int Frames, int SoftTokens)> media)
    {
        var sb = new StringBuilder(prompt.Text);
        int k = 0;
        foreach (var part in Parts)
        {
            if (part is string text)
            {
                sb.Append(text);
                continue;
            }

            string token = part is EmbeddingGemma2Video ? "<|video|>" : "<|image|>";
            var (frames, soft) = media[k++];
            for (int f = 0; f < frames; f++)
            {
                sb.Append("<|image>");
                for (int i = 0; i < soft; i++)
                {
                    sb.Append(token);
                }

                sb.Append("<image|>");
            }
        }

        return sb.ToString();
    }
}
