using System.Buffers;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Interventions.Internal;

/// <summary>
/// Strict reading of the contract's JSON documents: every object property must be known, present when required, and
/// appear once; numbers must have the declared type. Errors carry the JSON path of the offending value.
/// </summary>
internal readonly struct StrictObject
{
    private readonly Dictionary<string, JsonElement> _properties;

    private StrictObject(string path, Dictionary<string, JsonElement> properties)
    {
        Path = path;
        _properties = properties;
    }

    public string Path { get; }

    public static StrictObject Open(JsonElement element, string path, params ReadOnlySpan<string> allowed)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InterventionContractException(path, $"expected an object, found {Kind(element)}.");
        }

        Dictionary<string, JsonElement> properties = new(StringComparer.Ordinal);
        foreach (JsonProperty property in element.EnumerateObject())
        {
            string name = property.Name;
            if (!Contains(allowed, name))
            {
                throw new InterventionContractException(Join(path, name), "unknown property.");
            }

            if (!properties.TryAdd(name, property.Value))
            {
                throw new InterventionContractException(Join(path, name), "duplicate property.");
            }
        }

        return new StrictObject(path, properties);
    }

    public bool Has(string name) => _properties.ContainsKey(name);

    public JsonElement Required(string name) =>
        _properties.TryGetValue(name, out JsonElement value)
            ? value
            : throw new InterventionContractException(Join(Path, name), "required property is missing.");

    public string Child(string name) => Join(Path, name);

    public int GetInt32(string name)
    {
        JsonElement value = Required(name);
        return value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int result)
            ? result
            : throw new InterventionContractException(Child(name), $"expected a 32-bit integer, found {Kind(value)} {Raw(value)}.");
    }

    public long GetInt64(string name)
    {
        JsonElement value = Required(name);
        return value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out long result)
            ? result
            : throw new InterventionContractException(Child(name), $"expected a 64-bit integer, found {Kind(value)} {Raw(value)}.");
    }

    public string GetString(string name) => StrictJson.GetString(Required(name), Child(name));

    public string? GetOptionalString(string name) => Has(name) ? GetString(name) : null;

    public static string Join(string path, string name) => string.IsNullOrEmpty(path) ? name : $"{path}.{name}";

    private static bool Contains(ReadOnlySpan<string> names, string name)
    {
        foreach (string candidate in names)
        {
            if (string.Equals(candidate, name, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    internal static string Kind(JsonElement value) => value.ValueKind.ToString().ToLowerInvariant();

    private static string Raw(JsonElement value)
    {
        string raw = value.GetRawText();
        return raw.Length <= 40 ? raw : raw[..40] + "...";
    }
}

internal static partial class StrictJson
{
    /// <summary>Largest accepted profile or manifest document; both are small by construction.</summary>
    public const int MaxDocumentBytes = 16 * 1024 * 1024;

    private static readonly JsonDocumentOptions _documentOptions = new()
    {
        AllowTrailingCommas = false,
        CommentHandling = JsonCommentHandling.Disallow,
        MaxDepth = 32,
    };

    private static readonly JsonWriterOptions _writerOptions = new()
    {
        Indented = true,
        IndentCharacter = ' ',
        IndentSize = 2,
        NewLine = "\n",
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        SkipValidation = false,
    };

    public static JsonDocument Parse(ReadOnlyMemory<byte> utf8Json, string what)
    {
        if (utf8Json.Length > MaxDocumentBytes)
        {
            throw new InterventionContractException($"{what} is {utf8Json.Length} bytes; the limit is {MaxDocumentBytes}.");
        }

        ReadOnlySpan<byte> span = utf8Json.Span;
        if (span.StartsWith((ReadOnlySpan<byte>)[0xEF, 0xBB, 0xBF]))
        {
            throw new InterventionContractException($"{what} starts with a UTF-8 byte order mark; write UTF-8 without one.");
        }

        try
        {
            return JsonDocument.Parse(utf8Json, _documentOptions);
        }
        catch (JsonException e)
        {
            throw new InterventionContractException($"{what} is not valid JSON: {e.Message}", e);
        }
    }

    public static byte[] Write(Action<Utf8JsonWriter> write)
    {
        ArrayBufferWriter<byte> buffer = new();
        using (Utf8JsonWriter writer = new(buffer, _writerOptions))
        {
            write(writer);
        }

        byte[] bytes = new byte[buffer.WrittenCount + 1];
        buffer.WrittenSpan.CopyTo(bytes);
        bytes[^1] = (byte)'\n';
        return bytes;
    }

    public static string GetString(JsonElement value, string path) =>
        value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : throw new InterventionContractException(path, $"expected a string, found {StrictObject.Kind(value)}.");

    public static JsonElement.ArrayEnumerator GetArray(JsonElement value, string path) =>
        value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray()
            : throw new InterventionContractException(path, $"expected an array, found {StrictObject.Kind(value)}.");

    public static float GetSingle(JsonElement value, string path)
    {
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result) || !float.IsFinite(result))
        {
            throw new InterventionContractException(path, "expected a finite float32 number.");
        }

        return result;
    }

    public static string Index(string path, int index) => $"{path}[{index}]";

    /// <summary>Profile ids: lower-case ASCII letters, digits, '.', '_' and '-', starting with a letter or digit, at most 128.</summary>
    public static bool IsProfileId(string? value) => value is not null && ProfileIdPattern().IsMatch(value);

    /// <summary>Version labels (compiler, site mapping): ASCII letters, digits and <c>. _ + / -</c>, at most 64.</summary>
    public static bool IsVersionLabel(string? value) => value is not null && VersionPattern().IsMatch(value);

    /// <summary>Precision names: lower-case ASCII letters, digits and '_', at most 32.</summary>
    public static bool IsPrecision(string? value) => value is not null && PrecisionPattern().IsMatch(value);

    /// <summary>ONNX tensor and input names: non-empty, at most 1,024 characters, no control characters.</summary>
    public static bool IsTensorName(string? value)
    {
        if (string.IsNullOrEmpty(value) || value.Length > 1024)
        {
            return false;
        }

        foreach (char c in value)
        {
            if (char.IsControl(c))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Relative artifact paths: forward slashes only, not rooted, no empty, '.' or '..' segment, no backslash, colon or
    /// control character, at most 512 characters.
    /// </summary>
    public static bool IsRelativePath(string? value)
    {
        if (string.IsNullOrEmpty(value) || value.Length > 512 || value[0] == '/')
        {
            return false;
        }

        foreach (char c in value)
        {
            if (c is '\\' or ':' || char.IsControl(c))
            {
                return false;
            }
        }

        foreach (string segment in value.Split('/'))
        {
            if (segment is "" or "." or "..")
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Resolves a contract-relative path under <paramref name="directory"/>; the result never leaves it.</summary>
    public static string Resolve(string directory, string relativePath)
    {
        if (!IsRelativePath(relativePath))
        {
            throw new InterventionContractException($"'{relativePath}' is not a valid relative artifact path.");
        }

        string root = System.IO.Path.GetFullPath(directory);
        string full = System.IO.Path.GetFullPath(System.IO.Path.Combine(root, relativePath.Replace('/', System.IO.Path.DirectorySeparatorChar)));
        string prefix = System.IO.Path.EndsInDirectorySeparator(root) ? root : root + System.IO.Path.DirectorySeparatorChar;
        return full.StartsWith(prefix, StringComparison.Ordinal)
            ? full
            : throw new InterventionContractException($"'{relativePath}' resolves outside {root}.");
    }

    public static string RequireSha256(string value, string path) =>
        InterventionHashes.IsSha256(value)
            ? value
            : throw new InterventionContractException(path, "expected a SHA-256 as 64 lower-case hexadecimal characters.");

    [GeneratedRegex("^[a-z0-9][a-z0-9._-]{0,127}\\z", RegexOptions.CultureInvariant)]
    private static partial Regex ProfileIdPattern();

    [GeneratedRegex("^[A-Za-z0-9][A-Za-z0-9._+/-]{0,63}\\z", RegexOptions.CultureInvariant)]
    private static partial Regex VersionPattern();

    [GeneratedRegex("^[a-z0-9][a-z0-9_]{0,31}\\z", RegexOptions.CultureInvariant)]
    private static partial Regex PrecisionPattern();
}
