using Starlings.Models.PaddleOcr.Geometry;

namespace Starlings.Models.PaddleOcr;

/// <summary>The detector input tensor and the geometry needed to map its probability map back to the image.</summary>
/// <param name="Tensor">Normalized BGR, CHW, <c>[1, 3, Height, Width]</c>.</param>
/// <param name="Height">Resized height (a multiple of 32).</param>
/// <param name="Width">Resized width (a multiple of 32).</param>
/// <param name="SourceHeight">Height of the input image.</param>
/// <param name="SourceWidth">Width of the input image.</param>
internal sealed record DetectorInput(float[] Tensor, int Height, int Width, int SourceHeight, int SourceWidth);

/// <summary>A DB box before recognition: four (x, y) corners in image pixels and its box score.</summary>
internal sealed record DetectedBox(int[] Points, double Score);

/// <summary>
/// The PP-OCRv5 pipeline steps of paddlex 3.7.2, restated: detector preprocessing (<c>DetResizeForTest</c>,
/// <c>NormalizeImage</c>, <c>ToCHWImage</c>), DB post-processing (<c>DBPostProcess</c>, quad boxes, fast score),
/// box order (<c>SortQuadBoxes</c>), crops (<c>CropByPolys</c>, quad), recognizer preprocessing
/// (<c>OCRReisizeNormImg</c>) and greedy CTC decoding (<c>CTCLabelDecode</c>). The arithmetic follows the Python
/// (NumPy float32 and float64 steps as upstream evaluates them) so the tensors equal the reference's.
/// </summary>
internal static class PaddleOcrProcessing
{
    /// <summary>Detector input: pad tiny images, resize (longer side at most 960, sides multiples of 32), normalize.</summary>
    public static DetectorInput Detector(PaddleOcrImage image)
    {
        int sourceHeight = image.Height, sourceWidth = image.Width;
        byte[] pixels = image.Bgr;
        int h = sourceHeight, w = sourceWidth;
        if (h + w < 64)
        {
            // DetResizeForTest.image_padding: zero-pad to at least 32 x 32, the image at the top left.
            int ph = Math.Max(32, h), pw = Math.Max(32, w);
            var padded = new byte[ph * pw * 3];
            for (int y = 0; y < h; y++)
            {
                Array.Copy(pixels, y * w * 3, padded, y * pw * 3, w * 3);
            }

            (pixels, h, w) = (padded, ph, pw);
        }

        double ratio = 1.0;
        if (Math.Max(h, w) > PaddleOcrContract.LimitSideLength)
        {
            ratio = h > w ? (double)PaddleOcrContract.LimitSideLength / h : (double)PaddleOcrContract.LimitSideLength / w;
        }

        int resizeH = (int)(h * ratio), resizeW = (int)(w * ratio);
        if (Math.Max(resizeH, resizeW) > PaddleOcrContract.MaxSideLimit)
        {
            ratio = (double)PaddleOcrContract.MaxSideLimit / Math.Max(resizeH, resizeW);
            (resizeH, resizeW) = ((int)(resizeH * ratio), (int)(resizeW * ratio));
        }

        resizeH = Math.Max((int)(Math.Round(resizeH / 32.0, MidpointRounding.ToEven) * 32), 32);
        resizeW = Math.Max((int)(Math.Round(resizeW / 32.0, MidpointRounding.ToEven) * 32), 32);
        byte[] resized = resizeH == h && resizeW == w ? pixels : OpenCvResize.Linear(pixels, h, w, 3, resizeH, resizeW);

        // NormalizeImage: per channel, float32(pixel) * float32(scale / std) + float32(-mean / std).
        var tensor = new float[3 * resizeH * resizeW];
        int plane = resizeH * resizeW;
        for (int c = 0; c < 3; c++)
        {
            float alpha = (float)(1.0 / 255.0 / StdDouble[c]);
            float beta = (float)(-MeanDouble[c] / StdDouble[c]);
            for (int i = 0; i < plane; i++)
            {
                float value = resized[i * 3 + c];
                value *= alpha;
                value += beta;
                tensor[c * plane + i] = value;
            }
        }

        return new DetectorInput(tensor, resizeH, resizeW, sourceHeight, sourceWidth);
    }

    // The yml's mean and std as Python floats (double); paddlex computes alpha = scale / std and beta = -mean / std in
    // double, and NumPy applies them to the float32 image as float32 scalars.
    private static readonly double[] MeanDouble = [0.485, 0.456, 0.406];
    private static readonly double[] StdDouble = [0.229, 0.224, 0.225];

    /// <summary>
    /// DB post-processing of a probability map (<c>[height, width]</c>) to boxes in image pixels, in contour order:
    /// threshold, contours, min-area boxes, mean-probability score, unclip, rescale.
    /// </summary>
    public static List<DetectedBox> PostProcess(ReadOnlySpan<float> probability, int height, int width, int sourceHeight, int sourceWidth)
    {
        if (probability.Length != height * width)
        {
            throw new ArgumentException("Probability map size does not match.", nameof(probability));
        }

        var bitmap = new bool[height * width];
        for (int i = 0; i < bitmap.Length; i++)
        {
            bitmap[i] = probability[i] > PaddleOcrContract.Threshold;
        }

        var contours = OpenCvContours.Find(bitmap, height, width);
        double widthScale = (double)sourceWidth / width, heightScale = (double)sourceHeight / height;
        var boxes = new List<DetectedBox>();
        int count = Math.Min(contours.Count, PaddleOcrContract.MaxCandidates);
        for (int index = 0; index < count; index++)
        {
            var (points, side) = MiniBox(OpenCvShapes.MinAreaRect(contours[index]));
            if (side < PaddleOcrContract.MinSize)
            {
                continue;
            }

            double score = BoxScoreFast(probability, height, width, points);
            if (PaddleOcrContract.BoxThreshold > score)
            {
                continue;
            }

            double distance = OpenCvShapes.ContourArea(points) * PaddleOcrContract.UnclipRatio / OpenCvShapes.ArcLength(points);
            var expanded = ClipperOffset.OffsetRound(points, distance);
            if (expanded.Count < 2)
            {
                continue;
            }

            var expandedPoints = new int[expanded.Count];
            for (int i = 0; i < expanded.Count; i++)
            {
                expandedPoints[i] = checked((int)expanded[i]);
            }

            var (box, expandedSide) = MiniBox(OpenCvShapes.MinAreaRect(expandedPoints));
            if (expandedSide < PaddleOcrContract.MinSize + 2)
            {
                continue;
            }

            var scaled = new int[8];
            for (int i = 0; i < 4; i++)
            {
                // float32 coordinate times the float64 scale, rounded half to even, clipped; then int16.
                double x = Math.Max(0, Math.Min(Math.Round(box[i * 2] * widthScale, MidpointRounding.ToEven), sourceWidth));
                double y = Math.Max(0, Math.Min(Math.Round(box[i * 2 + 1] * heightScale, MidpointRounding.ToEven), sourceHeight));
                scaled[i * 2] = (short)(float)x;
                scaled[i * 2 + 1] = (short)(float)y;
            }

            boxes.Add(new DetectedBox(scaled, score));
        }

        return boxes;
    }

    /// <summary>
    /// <c>DBPostProcess.get_mini_boxes</c>: the rectangle's corners sorted by x (stable), then ordered top-left,
    /// top-right, bottom-right, bottom-left; and the shorter side.
    /// </summary>
    internal static (float[] Points, float Side) MiniBox(RotatedRect rectangle)
    {
        var corners = OpenCvShapes.BoxPoints(rectangle);
        var order = new[] { 0, 1, 2, 3 };
        // Python's sorted() is stable; a stable insertion sort by x keeps ties in boxPoints order.
        for (int i = 1; i < 4; i++)
        {
            for (int j = i; j > 0 && corners[order[j] * 2] < corners[order[j - 1] * 2]; j--)
            {
                (order[j], order[j - 1]) = (order[j - 1], order[j]);
            }
        }

        float Y(int k) => corners[order[k] * 2 + 1];
        int i1, i2, i3, i4;
        if (Y(1) > Y(0))
        {
            (i1, i4) = (0, 1);
        }
        else
        {
            (i1, i4) = (1, 0);
        }

        if (Y(3) > Y(2))
        {
            (i2, i3) = (2, 3);
        }
        else
        {
            (i2, i3) = (3, 2);
        }

        var box = new float[8];
        int[] picks = [i1, i2, i3, i4];
        for (int k = 0; k < 4; k++)
        {
            box[k * 2] = corners[order[picks[k]] * 2];
            box[k * 2 + 1] = corners[order[picks[k]] * 2 + 1];
        }

        return (box, Math.Min(rectangle.Width, rectangle.Height));
    }

    /// <summary><c>DBPostProcess.box_score_fast</c>: mean probability inside the box's filled polygon.</summary>
    internal static double BoxScoreFast(ReadOnlySpan<float> probability, int height, int width, ReadOnlySpan<float> box)
    {
        float minX = Math.Min(Math.Min(box[0], box[2]), Math.Min(box[4], box[6]));
        float maxX = Math.Max(Math.Max(box[0], box[2]), Math.Max(box[4], box[6]));
        float minY = Math.Min(Math.Min(box[1], box[3]), Math.Min(box[5], box[7]));
        float maxY = Math.Max(Math.Max(box[1], box[3]), Math.Max(box[5], box[7]));
        int xmin = Math.Max(0, Math.Min((int)Math.Floor(minX), width - 1));
        int xmax = Math.Max(0, Math.Min((int)Math.Ceiling(maxX), width - 1));
        int ymin = Math.Max(0, Math.Min((int)Math.Floor(minY), height - 1));
        int ymax = Math.Max(0, Math.Min((int)Math.Ceiling(maxY), height - 1));
        int mh = ymax - ymin + 1, mw = xmax - xmin + 1;
        var polygon = new int[8];
        for (int i = 0; i < 4; i++)
        {
            // float32 minus the integer origin, then astype(int32) (truncation toward zero).
            polygon[i * 2] = (int)(box[i * 2] - xmin);
            polygon[i * 2 + 1] = (int)(box[i * 2 + 1] - ymin);
        }

        var mask = new bool[mh * mw];
        OpenCvFill.FillPolygon(mask, mh, mw, polygon);
        double sum = 0;
        int n = 0;
        for (int y = 0; y < mh; y++)
        {
            for (int x = 0; x < mw; x++)
            {
                if (mask[y * mw + x])
                {
                    sum += probability[(ymin + y) * width + xmin + x];
                    n++;
                }
            }
        }

        return n == 0 ? 0 : sum / n;
    }

    /// <summary>
    /// <c>SortQuadBoxes</c>: stable sort by the first point's (y, x), then a pass that moves a box before its
    /// predecessors while their first points' y differ by less than 10 and its x is smaller.
    /// </summary>
    public static List<DetectedBox> Sort(List<DetectedBox> boxes)
    {
        var sorted = boxes
            .Select((box, index) => (box, index))
            .OrderBy(item => item.box.Points[1])
            .ThenBy(item => item.box.Points[0])
            .ThenBy(item => item.index)
            .Select(item => item.box)
            .ToList();
        for (int i = 0; i < sorted.Count - 1; i++)
        {
            for (int j = i; j >= 0; j--)
            {
                if (Math.Abs(sorted[j + 1].Points[1] - sorted[j].Points[1]) < PaddleOcrContract.SortRowTolerance
                    && sorted[j + 1].Points[0] < sorted[j].Points[0])
                {
                    (sorted[j], sorted[j + 1]) = (sorted[j + 1], sorted[j]);
                }
                else
                {
                    break;
                }
            }
        }

        return sorted;
    }

    /// <summary>
    /// <c>CropByPolys.get_minarea_rect_crop</c>: the box's min-area rectangle, perspective-warped upright
    /// (bicubic, replicated border) and rotated 90 degrees counter-clockwise when its height is at least 1.5 times its
    /// width. Returns BGR pixels with their size, or null for an empty crop (the pipeline drops it).
    /// </summary>
    public static (byte[] Pixels, int Height, int Width)? Crop(PaddleOcrImage image, ReadOnlySpan<int> box)
    {
        var (points, _) = MiniBox(OpenCvShapes.MinAreaRect(box));
        static float Norm(float ax, float ay, float bx, float by)
        {
            float dx = ax - bx, dy = ay - by;
            return MathF.Sqrt(dx * dx + dy * dy);
        }

        int cropW = (int)Math.Max(Norm(points[0], points[1], points[2], points[3]), Norm(points[4], points[5], points[6], points[7]));
        int cropH = (int)Math.Max(Norm(points[0], points[1], points[6], points[7]), Norm(points[2], points[3], points[4], points[5]));
        if (cropW <= 0 || cropH <= 0)
        {
            return null;
        }

        float[] target = [0, 0, cropW, 0, cropW, cropH, 0, cropH];
        var matrix = OpenCvWarp.PerspectiveTransform(points, target);
        var pixels = OpenCvWarp.PerspectiveCubic(image.Bgr, image.Height, image.Width, 3, matrix, cropH, cropW);
        if (cropH * 1.0 / cropW >= PaddleOcrContract.TallRatio)
        {
            // np.rot90: out[i, j] = in[j, W - 1 - i], shape (W, H).
            var rotated = new byte[pixels.Length];
            for (int i = 0; i < cropW; i++)
            {
                for (int j = 0; j < cropH; j++)
                {
                    int s = (j * cropW + (cropW - 1 - i)) * 3, d = (i * cropH + j) * 3;
                    rotated[d] = pixels[s];
                    rotated[d + 1] = pixels[s + 1];
                    rotated[d + 2] = pixels[s + 2];
                }
            }

            return (rotated, cropW, cropH);
        }

        return (pixels, cropH, cropW);
    }

    /// <summary>
    /// <c>OCRReisizeNormImg.resize</c> for one crop: height 48, width from the aspect ratio (at least 320, at most 3200),
    /// <c>(x / 255 - 0.5) / 0.5</c>, zero-padded on the right. Returns <c>[1, 3, 48, width]</c> and the width.
    /// </summary>
    public static (float[] Tensor, int Width) Recognizer(byte[] crop, int height, int width)
    {
        const int ImageH = PaddleOcrContract.RecognizerHeight;
        double maxRatio = (double)PaddleOcrContract.RecognizerWidth / ImageH;
        maxRatio = Math.Max(maxRatio, width * 1.0 / height);
        int imageW = (int)(ImageH * maxRatio);
        int resizedW;
        if (imageW > PaddleOcrContract.RecognizerMaxWidth)
        {
            resizedW = imageW = PaddleOcrContract.RecognizerMaxWidth;
        }
        else
        {
            double ratio = width / (double)height;
            double ceiling = Math.Ceiling(ImageH * ratio);
            resizedW = ceiling > imageW ? imageW : (int)ceiling;
        }

        var resized = OpenCvResize.Linear(crop, height, width, 3, ImageH, resizedW);
        var tensor = new float[3 * ImageH * imageW];
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < ImageH; y++)
            {
                for (int x = 0; x < resizedW; x++)
                {
                    float value = resized[(y * resizedW + x) * 3 + c];
                    value /= 255f;
                    value -= 0.5f;
                    value /= 0.5f;
                    tensor[(c * ImageH + y) * imageW + x] = value;
                }
            }
        }

        return (tensor, imageW);
    }

    /// <summary>
    /// <c>CTCLabelDecode</c>: per step the first maximum, repeats collapsed, the blank dropped; the confidence is NumPy's
    /// float32 mean of the kept maxima (0 when nothing is kept).
    /// </summary>
    public static (string Text, float Confidence) Decode(ReadOnlySpan<float> probabilities, int steps, IReadOnlyList<string> characters)
    {
        int classes = characters.Count;
        if (probabilities.Length != steps * classes)
        {
            throw new ArgumentException($"Expected {steps} x {classes} probabilities, got {probabilities.Length}.", nameof(probabilities));
        }

        var text = new System.Text.StringBuilder();
        var kept = new List<float>();
        int previous = -1;
        for (int t = 0; t < steps; t++)
        {
            var row = probabilities.Slice(t * classes, classes);
            int best = 0;
            float max = row[0];
            for (int c = 1; c < classes; c++)
            {
                if (row[c] > max)
                {
                    max = row[c];
                    best = c;
                }
            }

            if (best != previous && best != 0)
            {
                text.Append(characters[best]);
                kept.Add(max);
            }

            previous = best;
        }

        return (text.ToString(), kept.Count == 0 ? 0f : PairwiseSum(kept, 0, kept.Count) / kept.Count);
    }

    /// <summary>NumPy's pairwise float32 summation (<c>pairwise_sum</c>, block 128, eight accumulators).</summary>
    internal static float PairwiseSum(List<float> values, int start, int n)
    {
        if (n < 8)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++)
            {
                sum += values[start + i];
            }

            return sum;
        }

        if (n <= 128)
        {
            Span<float> r = stackalloc float[8];
            for (int k = 0; k < 8; k++)
            {
                r[k] = values[start + k];
            }

            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                for (int k = 0; k < 8; k++)
                {
                    r[k] += values[start + i + k];
                }
            }

            float result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++)
            {
                result += values[start + i];
            }

            return result;
        }

        int half = n / 2;
        half -= half % 8;
        return PairwiseSum(values, start, half) + PairwiseSum(values, start + half, n - half);
    }
}
