using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Models.EmbeddingGemma2;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>One video of a video fixture row: upstream's sampled indices and the reference digests of its frames.</summary>
internal sealed record VideoBenchVideo(string Video, int[] SampledIndices, int SoftTokensPerFrame, string FramesSha256, string PixelValuesSha256);

/// <summary>One row of <c>video-fp32-cpu/video-embeddings.jsonl</c> with what the harness needs.</summary>
internal sealed record VideoBenchRow(string Id, string Prompt, string? Text, bool HasImage, IReadOnlyList<VideoBenchVideo> Videos, long[] InputIds, float[] Embedding)
{
    public EmbeddingGemma2Prompt PackagePrompt => EmbeddingGemma2Prompt.FromName(Prompt);
}

/// <summary>
/// Decoded-frame video fixtures for the benchmark, without a video decoder: the sampled RGB frames of every video row
/// come from the reference tensors (<c>run_embeddinggemma2_video.py --tensors</c>: <c>&lt;row id&gt;.npz</c>,
/// <c>video&lt;k&gt;_frames_uint8</c>, torchcodec on FFmpeg 8), each array checked against the protected
/// <c>tensors.json</c> SHA-256 and the row's <c>frames_uint8</c> digest. The harness times what follows decoding
/// (preprocessing, the vision graph, the multimodal text graph), the same on both sides of the comparison.
/// </summary>
internal static class EmbeddingGemma2VideoFrames
{
    /// <summary>Directory of the reference <c>&lt;row id&gt;.npz</c> tensors (the variable the video tests use).</summary>
    public const string TensorsEnvironmentVariable = "STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS";

    public const string Reference = "model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/video-embeddings.jsonl";

    public static string FixtureDirectory => Path.Combine(EmbeddingGemma2TestAssets.ProtectedRoot, "video-fp32-cpu");

    /// <summary><c>$STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS</c>, else <c>&lt;models&gt;/embeddinggemma-2-video-tensors</c>.</summary>
    public static string TensorsDirectory =>
        Environment.GetEnvironmentVariable(TensorsEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(EmbeddingGemma2TestAssets.ModelsRoot, "embeddinggemma-2-video-tensors");

    public static Dictionary<string, VideoBenchRow> LoadRows() =>
        EmbeddingGemma2TestAssets.ReadJsonLines(Path.Combine(FixtureDirectory, "video-embeddings.jsonl"))
            .Select(r => new VideoBenchRow(
                r.GetProperty("id").GetString()!,
                r.GetProperty("prompt").GetString()!,
                r.GetProperty("text").ValueKind == JsonValueKind.Null ? null : r.GetProperty("text").GetString(),
                r.GetProperty("image").ValueKind != JsonValueKind.Null,
                r.GetProperty("videos").EnumerateArray().Select(v => new VideoBenchVideo(
                    v.GetProperty("video").GetString()!,
                    v.GetProperty("sampled_indices").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                    v.GetProperty("soft_tokens_per_frame").GetInt32(),
                    v.GetProperty("digests").GetProperty("frames_uint8").GetString()!,
                    v.GetProperty("digests").GetProperty("pixel_values").GetString()!)).ToList(),
                r.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                r.GetProperty("embedding").EnumerateArray().Select(e => e.GetSingle()).ToArray()))
            .ToDictionary(r => r.Id, StringComparer.Ordinal);

    private static readonly Lazy<JsonElement> TensorDigests = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(FixtureDirectory, "tensors.json"))).RootElement);

    /// <summary>
    /// The sampled frames of every video of <paramref name="row"/>, as pre-sampled videos
    /// (<see cref="EmbeddingGemma2Video.FromSampledFrames"/> with upstream's indices). Throws
    /// <see cref="BenchAssetsMissingException"/> when the tensors are absent and <see cref="InvalidDataException"/> when
    /// an array is not the reference one.
    /// </summary>
    public static EmbeddingGemma2Video[] LoadVideos(VideoBenchRow row)
    {
        string path = Path.Combine(TensorsDirectory, row.Id + ".npz");
        if (!File.Exists(path))
        {
            throw new BenchAssetsMissingException(
                $"decoded-frame video fixtures not found ({path}); set {TensorsEnvironmentVariable} to the directory of " +
                "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_video.py --tensors output (<row id>.npz, pinned by video-fp32-cpu/tensors.json).");
        }

        using var zip = ZipFile.OpenRead(path);
        var videos = new EmbeddingGemma2Video[row.Videos.Count];
        for (int k = 0; k < videos.Length; k++)
        {
            string name = $"video{k}_frames_uint8";
            var (shape, data) = ReadUInt8(zip, row.Id, name);
            string sha = Convert.ToHexStringLower(SHA256.HashData(data));
            if (sha != TensorDigests.Value.GetProperty(row.Id).GetProperty(name).GetProperty("sha256").GetString() || sha != row.Videos[k].FramesSha256)
            {
                throw new InvalidDataException($"{path}/{name}: SHA-256 {sha} differs from tensors.json and the row's frames digest; not the reference frames.");
            }

            if (shape.Length != 4 || shape[3] != 3 || shape[0] != row.Videos[k].SampledIndices.Length)
            {
                throw new InvalidDataException($"{path}/{name}: shape [{string.Join(", ", shape)}], expected [{row.Videos[k].SampledIndices.Length}, H, W, 3].");
            }

            int frameBytes = shape[1] * shape[2] * 3;
            var frames = new EmbeddingGemma2Image[shape[0]];
            for (int f = 0; f < frames.Length; f++)
            {
                frames[f] = EmbeddingGemma2Image.FromRgb24(shape[2], shape[1], data.AsMemory(f * frameBytes, frameBytes));
            }

            videos[k] = EmbeddingGemma2Video.FromSampledFrames(frames, row.Videos[k].SampledIndices);
        }

        return videos;
    }

    /// <summary>
    /// The input of a video row (upstream's layout): <c>&lt;|video|&gt;</c> placeholders in the text mark the video
    /// positions; without any, the videos follow the text; a row without text is its videos alone.
    /// </summary>
    public static EmbeddingGemma2MultimodalInput BuildInput(VideoBenchRow row, EmbeddingGemma2Video[] videos)
    {
        if (row.HasImage)
        {
            throw new ArgumentException($"Video row '{row.Id}' has a still image; the video workloads take rows of text and videos only.");
        }

        if (row.Text is null)
        {
            return EmbeddingGemma2MultimodalInput.Create(videos);
        }

        string[] pieces = row.Text.Split("<|video|>");
        if (pieces.Length == 1)
        {
            return EmbeddingGemma2MultimodalInput.Create([row.Text, .. videos]);
        }

        if (pieces.Length - 1 != videos.Length)
        {
            throw new InvalidDataException($"{row.Id}: {videos.Length} videos for {pieces.Length - 1} placeholders.");
        }

        var parts = new List<object>();
        for (int i = 0; i < pieces.Length; i++)
        {
            parts.Add(pieces[i]);
            if (i < videos.Length)
            {
                parts.Add(videos[i]);
            }
        }

        return EmbeddingGemma2MultimodalInput.Create(parts);
    }

    /// <summary>A C-ordered uint8 array of an <c>.npz</c> entry: its shape and bytes.</summary>
    private static (int[] Shape, byte[] Data) ReadUInt8(ZipArchive zip, string rowId, string name)
    {
        var entry = zip.GetEntry(name + ".npy") ?? throw new InvalidDataException($"{rowId}.npz has no {name}.");
        using var stream = entry.Open();
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        var bytes = buffer.ToArray();
        if (bytes.Length < 10 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException($"{rowId}/{name}: not an .npy file");
        }

        int headerLength = bytes[6] == 1 ? BitConverter.ToUInt16(bytes, 8) : BitConverter.ToInt32(bytes, 8);
        int start = bytes[6] == 1 ? 10 : 12;
        string header = Encoding.ASCII.GetString(bytes, start, headerLength);
        if (!header.Contains("'descr': '|u1'", StringComparison.Ordinal) || header.Contains("'fortran_order': True", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{rowId}/{name}: expected C-ordered uint8, header {header}");
        }

        int open = header.IndexOf("'shape': (", StringComparison.Ordinal) + "'shape': (".Length;
        int close = header.IndexOf(')', open);
        int[] shape = header[open..close].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        return (shape, bytes[(start + headerLength)..]);
    }
}
