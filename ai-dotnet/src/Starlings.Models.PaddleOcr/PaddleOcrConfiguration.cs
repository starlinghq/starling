using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Starlings.Models.PaddleOcr;

/// <summary>
/// The constants of the <c>ppocr-onnx-cpu-v1</c> contract (<c>model-pipeline/candidates/ppocrv5-mobile/profile.json</c>):
/// every value the pipeline uses, as the pinned <c>inference.yml</c> files and paddlex 3.7.2 define them.
/// <see cref="PaddleOcrConfiguration"/> checks the files against these values field by field.
/// </summary>
internal static class PaddleOcrContract
{
    public const string DetectorModel = "PP-OCRv5_mobile_det";
    public const string RecognizerModel = "PP-OCRv5_mobile_rec";
    public const string InputName = "x";
    public const string OutputName = "fetch_name_0";

    // Detector resize: paddlex maps DetResizeForTest {resize_long: 960} of the PP-OCRv5 detectors to limit_side_len 960,
    // limit_type "max", and caps the longer side at max_side_limit 4000; sides round half to even to multiples of 32.
    public const int LimitSideLength = 960;
    public const int MaxSideLimit = 4000;
    public const int SideMultiple = 32;
    public static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    public static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    // DBPostProcess.
    public const float Threshold = 0.3f;
    public const double BoxThreshold = 0.6;
    public const int MaxCandidates = 1000;
    public const double UnclipRatio = 1.5;
    public const int MinSize = 3;

    // Crop: tall crops (height / width >= 1.5) are rotated 90 degrees counter-clockwise.
    public const double TallRatio = 1.5;

    // Recognizer: RecResizeImg image_shape [3, 48, 320], widths capped at 3200 (paddlex OCRReisizeNormImg).
    public const int RecognizerHeight = 48;
    public const int RecognizerWidth = 320;
    public const int RecognizerMaxWidth = 3200;

    // CTC: blank, the 18,383 dictionary entries, then the space character.
    public const int DictionarySize = 18383;
    public const string DictionarySha256 = "17665d27ed39f0deb82007859992d626d3105d0ee4578c120b7c72138dc04d05";
    public const int Classes = DictionarySize + 2;

    // Pipeline: boxes sort by top-left (y, x), then same-row boxes (top-left y within 10) by x.
    public const int SortRowTolerance = 10;

    /// <summary>Largest accepted input image (pixels), a guard against unbounded allocations.</summary>
    public const long MaxPixels = 64L * 1024 * 1024;

    /// <summary>
    /// Longest accepted image side. Upstream stores box corners as int16 (<c>DBPostProcess</c>: <c>astype(np.int16)</c>),
    /// so a corner past 32,767 wraps there; this package refuses such images instead of returning wrapped boxes.
    /// </summary>
    public const int MaxSide = short.MaxValue;
}

/// <summary>The recognizer dictionary and the checked constants of the two pinned <c>inference.yml</c> files.</summary>
public sealed class PaddleOcrConfiguration
{
    private PaddleOcrConfiguration(string[] characters) => Characters = characters;

    /// <summary>
    /// The CTC label map: index 0 is the blank, then the 18,383 dictionary entries of the recognizer's
    /// <c>inference.yml</c>, then the space character (paddlex <c>CTCLabelDecode</c> with <c>use_space_char</c>). An
    /// entry may hold more than one UTF-16 unit (for example flag emoji).
    /// </summary>
    public IReadOnlyList<string> Characters { get; }

    /// <summary>
    /// Reads both configurations and checks every preprocessing and post-processing constant against the contract,
    /// field by field; anything else is refused.
    /// </summary>
    /// <exception cref="PaddleOcrConfigurationException">A configuration differs from the contract or is unreadable.</exception>
    public static PaddleOcrConfiguration Load(string detectorConfigPath, string recognizerConfigPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(detectorConfigPath);
        ArgumentException.ThrowIfNullOrEmpty(recognizerConfigPath);
        return Parse(ReadText(detectorConfigPath), ReadText(recognizerConfigPath));
    }

    internal static PaddleOcrConfiguration Parse(string detectorYaml, string recognizerYaml)
    {
        var detector = AsMap(MiniYaml.Parse(detectorYaml), "detector");
        CheckKeys(detector, "detector");
        Expect("detector Global.model_name", Get(AsMap(detector["Global"], "detector Global"), "model_name"), PaddleOcrContract.DetectorModel);
        CheckDetector(detector);
        var recognizer = AsMap(MiniYaml.Parse(recognizerYaml), "recognizer");
        CheckKeys(recognizer, "recognizer");
        Expect("recognizer Global.model_name", Get(AsMap(recognizer["Global"], "recognizer Global"), "model_name"), PaddleOcrContract.RecognizerModel);
        Expect(
            "recognizer PreProcess",
            MiniYaml.Canonical(recognizer["PreProcess"]),
            "{transform_ops:[{DecodeImage:{channel_first:false,img_mode:BGR}},{MultiLabelEncode:{gtc_encode:NRTRLabelEncode}},"
            + "{RecResizeImg:{image_shape:[3,48,320]}},{KeepKeys:{keep_keys:[image,label_ctc,label_gtc,length,valid_ratio]}}]}");
        var post = AsMap(recognizer["PostProcess"], "recognizer PostProcess");
        Expect("recognizer PostProcess keys", string.Join(",", post.Keys.Order(StringComparer.Ordinal)), "character_dict,name");
        Expect("recognizer PostProcess.name", Get(post, "name"), "CTCLabelDecode");
        if (post["character_dict"] is not List<object?> entries || entries.Any(entry => entry is not string { Length: > 0 } text || text.Contains('\n', StringComparison.Ordinal)))
        {
            throw new PaddleOcrConfigurationException("recognizer character_dict must be a list of nonempty single-line strings.");
        }

        var dictionary = entries.Cast<string>().ToArray();
        Expect("recognizer character_dict size", dictionary.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), PaddleOcrContract.DictionarySize.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Expect("recognizer character_dict SHA-256", DictionaryDigest(dictionary), PaddleOcrContract.DictionarySha256);
        if (dictionary.Distinct(StringComparer.Ordinal).Count() != dictionary.Length || dictionary.Contains(" "))
        {
            throw new PaddleOcrConfigurationException("recognizer character_dict must hold distinct entries without the space character.");
        }

        return new PaddleOcrConfiguration(["blank", .. dictionary, " "]);
    }

    private static void CheckDetector(Dictionary<string, object?> detector)
    {
        Expect(
            "detector PreProcess",
            MiniYaml.Canonical(detector["PreProcess"]),
            "{transform_ops:[{DecodeImage:{channel_first:false,img_mode:BGR}},{DetLabelEncode:null},{DetResizeForTest:{resize_long:960}},"
            + "{NormalizeImage:{mean:[0.485,0.456,0.406],order:hwc,scale:1./255.,std:[0.229,0.224,0.225]}},{ToCHWImage:null},"
            + "{KeepKeys:{keep_keys:[image,shape,polys,ignore_tags]}}]}");
        Expect(
            "detector PostProcess",
            MiniYaml.Canonical(detector["PostProcess"]),
            "{box_thresh:0.6,max_candidates:1000,name:DBPostProcess,thresh:0.3,unclip_ratio:1.5}");
    }

    /// <summary>SHA-256 (lowercase hex) of the dictionary entries joined by line feeds, as the contract pins it.</summary>
    internal static string DictionaryDigest(IEnumerable<string> entries) =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\n', entries))));

    private static string ReadText(string path)
    {
        try
        {
            return File.ReadAllText(path, Encoding.UTF8);
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            throw new PaddleOcrConfigurationException($"Cannot read {path}: {exception.Message}", exception);
        }
    }

    private static Dictionary<string, object?> AsMap(object? value, string where) =>
        value as Dictionary<string, object?> ?? throw new PaddleOcrConfigurationException($"{where} must be a mapping.");

    private static string? Get(Dictionary<string, object?> map, string key) => map.TryGetValue(key, out var value) ? value as string : null;

    private static void CheckKeys(Dictionary<string, object?> map, string where) =>
        Expect($"{where} top-level keys", string.Join(",", map.Keys.Order(StringComparer.Ordinal)), "Global,Hpi,PostProcess,PreProcess");

    private static void Expect(string field, string? actual, string expected)
    {
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new PaddleOcrConfigurationException($"{field} is {actual ?? "missing"}, the contract requires {expected}.");
        }
    }
}

/// <summary>
/// The YAML subset of the pinned <c>inference.yml</c> files: block mappings and sequences (a sequence may sit at its
/// key's indentation; an item may open a nested sequence or mapping on its own line), plain and single-quoted scalars, <c>null</c>, <c>true</c>/<c>false</c>, and anchors and
/// aliases (kept as text). Anything else is refused.
/// </summary>
internal static partial class MiniYaml
{
    private readonly record struct Line(int Indent, string Text, int Number);

    public static object? Parse(string text)
    {
        var lines = new List<Line>();
        int number = 0;
        foreach (var raw in text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n'))
        {
            number++;
            if (raw.Length == 0 || raw.TrimStart(' ').StartsWith('#'))
            {
                continue;
            }

            if (raw.Contains('\t', StringComparison.Ordinal))
            {
                throw Error(number, "tabs are not supported");
            }

            int indent = raw.Length - raw.TrimStart(' ').Length;
            lines.Add(new Line(indent, raw[indent..].TrimEnd(' '), number));
        }

        int index = 0;
        var value = lines.Count == 0 ? null : Block(lines, ref index, lines[0].Indent);
        if (index != lines.Count)
        {
            throw Error(lines[index].Number, "unexpected indentation");
        }

        return value;
    }

    /// <summary>Canonical text of a parsed value: maps with sorted keys, no spaces.</summary>
    public static string Canonical(object? value) => value switch
    {
        null => "null",
        string text => text,
        bool flag => flag ? "true" : "false",
        List<object?> list => "[" + string.Join(",", list.Select(Canonical)) + "]",
        Dictionary<string, object?> map => "{" + string.Join(",", map.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => pair.Key + ":" + Canonical(pair.Value))) + "}",
        _ => throw new InvalidOperationException(),
    };

    private static object? Block(List<Line> lines, ref int index, int indent) =>
        IsItem(lines[index].Text) ? Sequence(lines, ref index, indent) : Mapping(lines, ref index, indent);

    private static bool IsItem(string text) => text == "-" || text.StartsWith("- ", StringComparison.Ordinal);

    private static List<object?> Sequence(List<Line> lines, ref int index, int indent)
    {
        var items = new List<object?>();
        while (index < lines.Count && lines[index].Indent == indent && IsItem(lines[index].Text))
        {
            var line = lines[index];
            string content = line.Text.Length > 2 ? line.Text[2..] : string.Empty;
            if (IsItem(content))
            {
                // "- - value": a nested sequence whose first item shares the line.
                lines[index] = new Line(indent + 2, content, line.Number);
                items.Add(Sequence(lines, ref index, indent + 2));
            }
            else if (KeyPattern().IsMatch(content))
            {
                // "- key: value": a mapping whose first entry shares the item's line.
                lines[index] = new Line(indent + 2, content, line.Number);
                items.Add(Mapping(lines, ref index, indent + 2));
            }
            else
            {
                items.Add(Scalar(content, line.Number));
                index++;
            }
        }

        return items;
    }

    private static Dictionary<string, object?> Mapping(List<Line> lines, ref int index, int indent)
    {
        var map = new Dictionary<string, object?>(StringComparer.Ordinal);
        while (index < lines.Count && lines[index].Indent == indent && !IsItem(lines[index].Text))
        {
            var line = lines[index];
            var match = KeyPattern().Match(line.Text);
            if (!match.Success)
            {
                throw Error(line.Number, "expected 'key: value'");
            }

            string key = match.Groups[1].Value;
            string rest = match.Groups[2].Value.Trim();
            if (rest.StartsWith('&'))
            {
                rest = rest.Contains(' ', StringComparison.Ordinal) ? rest[(rest.IndexOf(' ', StringComparison.Ordinal) + 1)..] : string.Empty;
            }

            index++;
            object? value;
            if (rest.Length > 0)
            {
                value = Scalar(rest, line.Number);
            }
            else if (index < lines.Count && (lines[index].Indent > indent || (lines[index].Indent == indent && IsItem(lines[index].Text))))
            {
                value = Block(lines, ref index, lines[index].Indent);
            }
            else
            {
                value = null;
            }

            if (!map.TryAdd(key, value))
            {
                throw Error(line.Number, $"duplicate key {key}");
            }
        }

        return map;
    }

    private static object? Scalar(string text, int number)
    {
        if (text.StartsWith('\''))
        {
            if (text.Length < 2 || !text.EndsWith('\''))
            {
                throw Error(number, "unterminated single-quoted scalar");
            }

            string inner = text[1..^1];
            if (inner.Replace("''", string.Empty, StringComparison.Ordinal).Contains('\'', StringComparison.Ordinal))
            {
                throw Error(number, "unescaped quote in a single-quoted scalar");
            }

            return inner.Replace("''", "'", StringComparison.Ordinal);
        }

        if (text.StartsWith('"') || text.StartsWith('[') || text.StartsWith('{') || text.StartsWith('|') || text.StartsWith('>'))
        {
            throw Error(number, "unsupported scalar form");
        }

        return text switch
        {
            "null" or "~" => null,
            "true" => true,
            "false" => false,
            _ => text,
        };
    }

    private static PaddleOcrConfigurationException Error(int line, string message) => new($"inference.yml line {line}: {message}.");

    [GeneratedRegex(@"^([A-Za-z_][A-Za-z0-9_]*):(?: (.*))?$")]
    private static partial Regex KeyPattern();
}
