using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Models.PaddleOcr.Geometry;

namespace Starlings.Models.PaddleOcr.Tests;

/// <summary>
/// The owned OpenCV and Clipper restatements against the pinned implementations: fixtures written by
/// <c>model-pipeline/reference/ppocr/make_geometry_fixtures.py</c> with opencv-contrib-python 4.10.0.84 and
/// pyclipper 1.4.0.
/// </summary>
public sealed class GeometryConformanceTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "TestData", "opencv-geometry-v1.json"))).RootElement;

    internal static byte[] XorShiftBytes(uint seed, int count)
    {
        uint state = seed == 0 ? 1 : seed;
        var output = new byte[count];
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            output[i] = (byte)(state >> 24);
        }

        return output;
    }

    private static int[] Ints(JsonElement points) => points.EnumerateArray().SelectMany(p => p.EnumerateArray().Select(v => v.GetInt32())).ToArray();

    private static float[] Floats(JsonElement points) => points.EnumerateArray().SelectMany(p => p.EnumerateArray().Select(v => v.GetSingle())).ToArray();

    private static bool[] Bitmap(JsonElement rows, out int height, out int width)
    {
        var lines = rows.EnumerateArray().Select(r => r.GetString()!).ToArray();
        height = lines.Length;
        width = lines[0].Length;
        return lines.SelectMany(line => line.Select(c => c == '1')).ToArray();
    }

    [Fact]
    public void ResizeEqualsOpenCvInterLinear()
    {
        foreach (var item in Fixture.GetProperty("resize").EnumerateArray())
        {
            int h = item.GetProperty("size")[0].GetInt32(), w = item.GetProperty("size")[1].GetInt32();
            int th = item.GetProperty("target")[0].GetInt32(), tw = item.GetProperty("target")[1].GetInt32();
            var source = XorShiftBytes(item.GetProperty("seed").GetUInt32(), h * w * 3);
            var output = OpenCvResize.Linear(source, h, w, 3, th, tw);
            Assert.True(
                Convert.ToHexStringLower(SHA256.HashData(output)) == item.GetProperty("sha256").GetString(),
                $"resize {h}x{w} -> {th}x{tw}");
        }
    }

    [Fact]
    public void PerspectiveTransformAndCubicWarpEqualOpenCv()
    {
        foreach (var item in Fixture.GetProperty("warp").EnumerateArray())
        {
            int h = item.GetProperty("size")[0].GetInt32(), w = item.GetProperty("size")[1].GetInt32();
            int ch = item.GetProperty("crop")[0].GetInt32(), cw = item.GetProperty("crop")[1].GetInt32();
            var points = Floats(item.GetProperty("points"));
            float[] target = [0, 0, cw, 0, cw, ch, 0, ch];
            var matrix = OpenCvWarp.PerspectiveTransform(points, target);
            var expected = item.GetProperty("matrix").EnumerateArray().SelectMany(r => r.EnumerateArray().Select(v => v.GetDouble())).ToArray();
            for (int i = 0; i < 9; i++)
            {
                Assert.True(Math.Abs(matrix[i] - expected[i]) <= 1e-12 * Math.Max(1, Math.Abs(expected[i])), $"matrix[{i}] {matrix[i]} != {expected[i]} ({h}x{w})");
            }

            var source = XorShiftBytes(item.GetProperty("seed").GetUInt32(), h * w * 3);
            var output = OpenCvWarp.PerspectiveCubic(source, h, w, 3, matrix, ch, cw);
            Assert.True(
                Convert.ToHexStringLower(SHA256.HashData(output)) == item.GetProperty("sha256").GetString(),
                $"warp {h}x{w} -> {ch}x{cw}");
        }
    }

    [Fact]
    public void ContoursEqualOpenCvListSimple()
    {
        foreach (var item in Fixture.GetProperty("contours").EnumerateArray())
        {
            var mask = Bitmap(item.GetProperty("bitmap"), out int h, out int w);
            var contours = OpenCvContours.Find(mask, h, w);
            var expected = item.GetProperty("contours").EnumerateArray().Select(Ints).ToList();
            Assert.Equal(expected.Count, contours.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.Equal(expected[i], contours[i]);
            }
        }
    }

    [Fact]
    public void MinAreaRectAndBoxPointsEqualOpenCv()
    {
        foreach (var item in Fixture.GetProperty("min_area_rect").EnumerateArray())
        {
            var rect = item.GetProperty("dtype").GetString() == "int32"
                ? OpenCvShapes.MinAreaRect(Ints(item.GetProperty("points")))
                : OpenCvShapes.MinAreaRect(Floats(item.GetProperty("points")));
            float[] expected =
            [
                item.GetProperty("center")[0].GetSingle(), item.GetProperty("center")[1].GetSingle(),
                item.GetProperty("size")[0].GetSingle(), item.GetProperty("size")[1].GetSingle(), item.GetProperty("angle").GetSingle(),
            ];
            float[] actual = [rect.CenterX, rect.CenterY, rect.Width, rect.Height, rect.Angle];
            Assert.True(expected.SequenceEqual(actual), $"{item.GetProperty("points")}: expected {string.Join(", ", expected)}, got {string.Join(", ", actual)}");
            Assert.Equal(Floats(item.GetProperty("box")), OpenCvShapes.BoxPoints(rect));
        }
    }

    [Fact]
    public void ContourAreaAndArcLengthEqualOpenCv()
    {
        foreach (var item in Fixture.GetProperty("measure").EnumerateArray())
        {
            var points = Floats(item.GetProperty("points"));
            Assert.Equal(item.GetProperty("area").GetDouble(), OpenCvShapes.ContourArea(points));
            Assert.Equal(item.GetProperty("length").GetDouble(), OpenCvShapes.ArcLength(points));
        }
    }

    [Fact]
    public void FillPolyAndMaskedMeanEqualOpenCv()
    {
        int index = 0;
        foreach (var item in Fixture.GetProperty("fill").EnumerateArray())
        {
            int h = item.GetProperty("size")[0].GetInt32(), w = item.GetProperty("size")[1].GetInt32();
            var expected = Bitmap(item.GetProperty("mask"), out _, out _);
            var mask = new bool[h * w];
            OpenCvFill.FillPolygon(mask, h, w, Ints(item.GetProperty("points")));
            Assert.True(expected.SequenceEqual(mask), $"fill case {index}: {item.GetProperty("points")}: " + string.Join(" ", Enumerable.Range(0, mask.Length).Where(i => mask[i] != expected[i]).Select(i => $"({i % w},{i / w}:{(mask[i] ? "extra" : "missing")})")));
            var values = XorShiftBytes(item.GetProperty("values_seed").GetUInt32(), h * w).Select(v => v / 255f).ToArray();
            double sum = 0;
            int n = 0;
            for (int i = 0; i < mask.Length; i++)
            {
                if (mask[i])
                {
                    sum += values[i];
                    n++;
                }
            }

            Assert.Equal(item.GetProperty("mean").GetDouble(), n == 0 ? 0 : sum / n, 12);
            index++;
        }
    }

    [Fact]
    public void ClipperOffsetHasPyclipperHull()
    {
        foreach (var item in Fixture.GetProperty("offset").EnumerateArray())
        {
            var points = Floats(item.GetProperty("points"));
            var output = ClipperOffset.OffsetRound(points, item.GetProperty("distance").GetDouble());
            var expected = item.GetProperty("paths").EnumerateArray().SelectMany(Ints).ToArray();
            var expectedHull = Hull(expected);
            var actualHull = Hull(output.Select(v => checked((int)v)).ToArray());
            Assert.True(expectedHull.SetEquals(actualHull), $"offset of {item.GetProperty("points")}: hull differs: expected {string.Join(" ", expectedHull.Order())}, got {string.Join(" ", actualHull.Order())}; raw {string.Join(",", output)}");
            // Every vertex pyclipper keeps is one of ours (the union only removes vertices).
            for (int i = 0; i < expected.Length; i += 2)
            {
                Assert.Contains((expected[i], expected[i + 1]), Pairs(output.Select(v => (int)v).ToArray()));
            }
        }
    }

    private static HashSet<(int, int)> Pairs(int[] xy) => Enumerable.Range(0, xy.Length / 2).Select(i => (xy[i * 2], xy[i * 2 + 1])).ToHashSet();

    private static HashSet<(int, int)> Hull(int[] xy)
    {
        var hull = OpenCvShapes.ConvexHull(xy.Select(v => (double)v).ToArray(), isFloat: false);
        return Enumerable.Range(0, hull.Length / 2).Select(i => ((int)hull[i * 2], (int)hull[i * 2 + 1])).ToHashSet();
    }
}
