namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// The image preprocessing of the pinned revision (<c>Gemma4ImageProcessor</c>, the torchvision backend that
/// transformers 5.19.0 runs), restated from already-decoded RGB: aspect-preserving target size within the patch budget,
/// PyTorch's uint8 antialiased bicubic resize, rescale by <c>float32(1/255)</c>, 16x16 patches and their (x, y)
/// positions. On the 28 supported fixture images every step equals the reference bit for bit (resized bytes, pixel
/// values, positions).
/// </summary>
/// <remarks>
/// <para>
/// The resize is a port of PyTorch's CPU uint8 antialias kernel (what <c>torchvision.transforms.v2.functional.resize(...,
/// BICUBIC, antialias=True)</c> runs on a uint8 tensor): separable, horizontal pass then vertical pass, Keys cubic with
/// a = -0.5 and a support of 2 times the downscale factor, int16 fixed-point weights with a per-pass precision, uint8
/// intermediate. It is not Pillow's resize (22-bit weights) nor any library "bicubic": one or two levels of difference
/// on under 1% of pixels move the embedding by up to 2.6e-3.
/// </para>
/// <para>
/// Frame-ready: video frames use the same resize, rescale and patching with a budget of 140 soft tokens (1,260
/// patches). Budgets other than 280 are untested in this package version.
/// </para>
/// </remarks>
public static class EmbeddingGemma2ImageProcessor
{
    /// <summary>Patch side in pixels.</summary>
    public const int PatchSize = 16;

    /// <summary>Patches per side of one pooled soft token (the vision tower's 3x3 average pooling).</summary>
    public const int PoolingKernelSize = 3;

    /// <summary>Values per patch: 16 x 16 pixels x 3 channels, in (row, column, channel) order.</summary>
    public const int PatchValues = PatchSize * PatchSize * 3;

    /// <summary>Side multiple of the resized image: <c>PoolingKernelSize * PatchSize</c> = 48 pixels.</summary>
    public const int SideMultiple = PoolingKernelSize * PatchSize;

    /// <summary>The checkpoint's default image budget (<c>max_soft_tokens</c> 280, so 2,520 patches).</summary>
    public const int DefaultMaxSoftTokens = 280;

    /// <summary>
    /// Rescale factor of <c>processor_config.json</c> (0.00392156862745098) as float32. A pixel value is
    /// <c>byte * RescaleFactor</c> in float32, which is not always <c>byte / 255f</c>.
    /// </summary>
    public const float RescaleFactor = (float)0.00392156862745098;

    /// <summary>Budgets the processor accepts (<c>_SUPPORTED_SOFT_TOKENS</c>). Only 280 is tested.</summary>
    public static IReadOnlyList<int> SupportedSoftTokenBudgets { get; } = [70, 140, 280, 560, 1_120];

    /// <summary>Patches for a budget: <c>maxSoftTokens * 9</c>.</summary>
    public static int MaxPatches(int maxSoftTokens = DefaultMaxSoftTokens)
    {
        ValidateBudget(maxSoftTokens);
        return maxSoftTokens * PoolingKernelSize * PoolingKernelSize;
    }

    /// <summary>
    /// The resized size (<c>get_aspect_ratio_preserving_size</c>): the largest size with sides that are multiples of 48
    /// and at most <c>maxSoftTokens * 9</c> patches, scaled by <c>sqrt(budget pixels / pixels)</c> (upscaling allowed);
    /// a side that rounds to 0 becomes 48 and the other <c>min(floor(long / short) * 48, maxSoftTokens / 9 * 48)</c>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive, or the budget is not supported.</exception>
    public static (int Width, int Height) GetTargetSize(int width, int height, int maxSoftTokens = DefaultMaxSoftTokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        int maxPatches = MaxPatches(maxSoftTokens);
        long totalPx = (long)height * width;
        long targetPx = (long)maxPatches * PatchSize * PatchSize;
        double factor = Math.Sqrt((double)targetPx / totalPx);
        double idealHeight = factor * height;
        double idealWidth = factor * width;
        long targetHeight = (long)Math.Floor(idealHeight / SideMultiple) * SideMultiple;
        long targetWidth = (long)Math.Floor(idealWidth / SideMultiple) * SideMultiple;
        if (targetHeight == 0 && targetWidth == 0)
        {
            throw new ArgumentException($"A {width}x{height} image resizes to 0x0.", nameof(width));
        }

        long maxSide = (long)(maxPatches / (PoolingKernelSize * PoolingKernelSize)) * SideMultiple;
        if (targetHeight == 0)
        {
            targetHeight = SideMultiple;
            targetWidth = Math.Min((long)Math.Floor((double)width / height) * SideMultiple, maxSide);
        }
        else if (targetWidth == 0)
        {
            targetWidth = SideMultiple;
            targetHeight = Math.Min((long)Math.Floor((double)height / width) * SideMultiple, maxSide);
        }

        if (targetHeight * targetWidth > targetPx)
        {
            throw new ArgumentException($"Resizing {width}x{height} to {targetWidth}x{targetHeight} exceeds {maxPatches} patches.", nameof(width));
        }

        return ((int)targetWidth, (int)targetHeight);
    }

    /// <summary>Soft tokens the image becomes: <c>(target height / 48) * (target width / 48)</c>.</summary>
    public static int GetSoftTokenCount(int width, int height, int maxSoftTokens = DefaultMaxSoftTokens)
    {
        var (tw, th) = GetTargetSize(width, height, maxSoftTokens);
        return (th / SideMultiple) * (tw / SideMultiple);
    }

    /// <summary>
    /// Resizes tightly packed RGB24 pixels with PyTorch's uint8 antialiased bicubic kernel (see the type remarks).
    /// Returns a new <c>targetWidth * targetHeight * 3</c> buffer; equal sizes return a copy.
    /// </summary>
    public static byte[] Resize(ReadOnlySpan<byte> rgb, int width, int height, int targetWidth, int targetHeight)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(targetWidth, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(targetHeight, 1);
        if (rgb.Length != (long)width * height * 3)
        {
            throw new ArgumentException($"A {width}x{height} RGB24 image needs {(long)width * height * 3} bytes; got {rgb.Length}.", nameof(rgb));
        }

        return TorchAntialiasResampler.Resize(rgb, width, height, targetWidth, targetHeight);
    }

    /// <summary>
    /// Runs the whole processor on one image: target size, resize (skipped when the size already matches), rescale,
    /// patches and positions, unpadded (the vision graph pads; padding rows are zero pixels at (-1, -1)).
    /// </summary>
    public static EmbeddingGemma2ImagePatches Preprocess(EmbeddingGemma2Image image, int maxSoftTokens = DefaultMaxSoftTokens)
    {
        ArgumentNullException.ThrowIfNull(image);
        var (tw, th) = GetTargetSize(image.Width, image.Height, maxSoftTokens);
        byte[] resized = tw == image.Width && th == image.Height
            ? image.Pixels.ToArray()
            : TorchAntialiasResampler.Resize(image.Pixels.Span, image.Width, image.Height, tw, th);
        int patches = (tw / PatchSize) * (th / PatchSize);
        var pixelValues = new float[patches * PatchValues];
        var positions = new long[patches * 2];
        WritePatches(resized, tw, th, pixelValues, positions);
        return new EmbeddingGemma2ImagePatches(tw, th, resized, pixelValues, positions);
    }

    /// <summary>
    /// Writes the patches of a resized image (sides multiples of 16) row-major over the patch grid, each patch as (row,
    /// column, channel) values times <see cref="RescaleFactor"/>, and their (x, y) positions.
    /// </summary>
    internal static void WritePatches(ReadOnlySpan<byte> resized, int width, int height, Span<float> pixelValues, Span<long> positions)
    {
        int pw = width / PatchSize, ph = height / PatchSize;
        for (int py = 0; py < ph; py++)
        {
            for (int px = 0; px < pw; px++)
            {
                int r = py * pw + px;
                positions[2 * r] = px;
                positions[2 * r + 1] = py;
                var patch = pixelValues.Slice(r * PatchValues, PatchValues);
                for (int dy = 0; dy < PatchSize; dy++)
                {
                    var src = resized.Slice(((py * PatchSize + dy) * width + px * PatchSize) * 3, PatchSize * 3);
                    var dst = patch.Slice(dy * PatchSize * 3, PatchSize * 3);
                    for (int k = 0; k < src.Length; k++)
                    {
                        dst[k] = src[k] * RescaleFactor;
                    }
                }
            }
        }
    }

    internal static void ValidateBudget(int maxSoftTokens)
    {
        if (maxSoftTokens is not (70 or 140 or 280 or 560 or 1_120))
        {
            throw new ArgumentOutOfRangeException(nameof(maxSoftTokens), maxSoftTokens, "Supported image budgets are 70, 140, 280, 560 and 1,120 soft tokens (280 is the tested default).");
        }
    }
}

/// <summary>One preprocessed image: the resized bytes and the vision graph's unpadded inputs.</summary>
public sealed class EmbeddingGemma2ImagePatches
{
    internal EmbeddingGemma2ImagePatches(int resizedWidth, int resizedHeight, byte[] resizedPixels, float[] pixelValues, long[] positions)
    {
        ResizedWidth = resizedWidth;
        ResizedHeight = resizedHeight;
        ResizedPixels = resizedPixels;
        PixelValues = pixelValues;
        Positions = positions;
    }

    /// <summary>Width after resizing (a multiple of 48).</summary>
    public int ResizedWidth { get; }

    /// <summary>Height after resizing (a multiple of 48).</summary>
    public int ResizedHeight { get; }

    /// <summary>The resized RGB24 bytes (the reference's <c>resized_uint8</c>, height x width x 3).</summary>
    public ReadOnlyMemory<byte> ResizedPixels { get; }

    /// <summary>Patch values, <c>PatchCount x 768</c> float32 (the reference's <c>pixel_values</c> without padding rows).</summary>
    public ReadOnlyMemory<float> PixelValues { get; }

    /// <summary>Patch positions, <c>PatchCount x 2</c> as (x, y) (the reference's <c>image_position_ids</c> without padding rows).</summary>
    public ReadOnlyMemory<long> Positions { get; }

    /// <summary>Number of patches: <c>(ResizedWidth / 16) * (ResizedHeight / 16)</c>, always a multiple of 9.</summary>
    public int PatchCount => Positions.Length / 2;

    /// <summary>Soft tokens this image becomes: <c>PatchCount / 9</c>.</summary>
    public int SoftTokenCount => PatchCount / (EmbeddingGemma2ImageProcessor.PoolingKernelSize * EmbeddingGemma2ImageProcessor.PoolingKernelSize);
}
