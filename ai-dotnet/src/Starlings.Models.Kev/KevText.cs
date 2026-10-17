using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Models.Kev;

/// <summary>
/// Text the model sees, as Kev 1.0 builds it (<c>kev/api.py</c> <c>render</c> and <c>option_text</c>,
/// <c>kev/model.py</c> <c>user_tokens</c> at <see cref="KevModelProfile.SourceRevision"/>). Python's <c>str()</c> of
/// JSON values is reproduced exactly: <c>True</c>/<c>False</c>, integers of any size, and floats in Python's shortest
/// round-trip <c>repr</c> (<c>1e-05</c>, <c>100000.0</c>, <c>1e+16</c>).
/// </summary>
public static partial class KevText
{
    /// <summary>
    /// <c>render</c>: flattens a JSON value into the text the model sees. Strings as they are, null as empty, numbers and
    /// booleans as Python prints them, arrays as <c>- item</c> lines and objects as <c>key: value</c> lines (nested values
    /// on following lines, indented two spaces per level).
    /// </summary>
    public static string Render(JsonElement value) => Render(value, 0);

    /// <summary>
    /// <c>option_text</c>: the option name alone when the description is absent, JSON null or the empty string, else
    /// <c>name: rendered description</c>.
    /// </summary>
    public static string OptionText(string name, KevContent? description)
    {
        ArgumentNullException.ThrowIfNull(name);
        return description is null || description.IsNullOrEmptyString ? name : $"{name}: {description.Render()}";
    }

    /// <summary>
    /// <c>user_tokens</c>' rewrite before tokenizing caller text: every <c>&lt;|name|&gt;</c> (ASCII letters, digits,
    /// underscore) becomes <c>&lt;¦name¦&gt;</c>, so user text can never produce a delimiter or control token.
    /// </summary>
    public static string EscapeSpecialTokens(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return SpecialToken().Replace(text, "<¦$1¦>");
    }

    [GeneratedRegex("<\\|([A-Za-z0-9_]+)\\|>", RegexOptions.CultureInvariant)]
    private static partial Regex SpecialToken();

    private static string Render(JsonElement v, int indent)
    {
        string pad = new(' ', 2 * indent);
        switch (v.ValueKind)
        {
            case JsonValueKind.Null:
            case JsonValueKind.Undefined:
                return string.Empty;
            case JsonValueKind.String:
                return v.GetString()!;
            case JsonValueKind.True:
                return "True";
            case JsonValueKind.False:
                return "False";
            case JsonValueKind.Number:
                return PythonNumber(v.GetRawText());
            case JsonValueKind.Array:
                return string.Join("\n", v.EnumerateArray().Select(x => $"{pad}- {PythonLStrip(Render(x, indent + 1))}"));
            case JsonValueKind.Object:
                // A Python dict keeps a repeated key at its first position with the last value.
                var entries = new List<(string Key, JsonElement Value)>();
                var index = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var p in v.EnumerateObject())
                {
                    if (index.TryGetValue(p.Name, out int at))
                    {
                        entries[at] = (p.Name, p.Value);
                    }
                    else
                    {
                        index[p.Name] = entries.Count;
                        entries.Add((p.Name, p.Value));
                    }
                }

                return string.Join("\n", entries.Select(e => e.Value.ValueKind is JsonValueKind.Object or JsonValueKind.Array
                    ? $"{pad}{e.Key}:\n{Render(e.Value, indent + 1)}"
                    : $"{pad}{e.Key}: {Render(e.Value, 0)}"));
            default:
                throw new ArgumentException($"Unsupported JSON value kind {v.ValueKind}.", nameof(v));
        }
    }

    /// <summary>Python <c>str()</c> of a JSON number: <c>json.loads</c> makes an int without '.', 'e' or 'E', else a float.</summary>
    internal static string PythonNumber(string raw)
    {
        if (raw.AsSpan().IndexOfAny('.', 'e', 'E') < 0)
        {
            return BigInteger.Parse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
        }

        return PythonFloatRepr(double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture));
    }

    /// <summary>Python's <c>repr(float)</c>: shortest round-trip digits; exponent form when the decimal exponent is below -4 or at least 16.</summary>
    internal static string PythonFloatRepr(double d)
    {
        if (double.IsNaN(d))
        {
            return "nan";
        }

        if (double.IsInfinity(d))
        {
            return d > 0 ? "inf" : "-inf";
        }

        bool negative = d < 0 || (d == 0 && double.IsNegative(d));
        if (d == 0)
        {
            return negative ? "-0.0" : "0.0";
        }

        // .NET's "R" is the shortest round-trip string (as Python's repr); only the layout differs.
        string r = Math.Abs(d).ToString("R", CultureInfo.InvariantCulture);
        int e = r.IndexOfAny(['E', 'e']);
        string mantissa = e < 0 ? r : r[..e];
        int exponent = e < 0 ? 0 : int.Parse(r[(e + 1)..], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
        int point = mantissa.IndexOf('.', StringComparison.Ordinal);
        string digits = point < 0 ? mantissa : mantissa.Remove(point, 1);
        int decpt = (point < 0 ? mantissa.Length : point) + exponent;
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0')
        {
            lead++;
        }

        digits = digits[lead..];
        decpt -= lead;
        digits = digits.TrimEnd('0');
        if (digits.Length == 0)
        {
            digits = "0";
        }

        var sb = new StringBuilder();
        if (negative)
        {
            sb.Append('-');
        }

        if (decpt > -4 && decpt <= 16)
        {
            if (decpt <= 0)
            {
                sb.Append("0.").Append('0', -decpt).Append(digits);
            }
            else if (decpt >= digits.Length)
            {
                sb.Append(digits).Append('0', decpt - digits.Length).Append(".0");
            }
            else
            {
                sb.Append(digits, 0, decpt).Append('.').Append(digits, decpt, digits.Length - decpt);
            }
        }
        else
        {
            int exp = decpt - 1;
            sb.Append(digits[0]);
            if (digits.Length > 1)
            {
                sb.Append('.').Append(digits, 1, digits.Length - 1);
            }

            sb.Append('e').Append(exp < 0 ? '-' : '+').Append(Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture));
        }

        return sb.ToString();
    }

    /// <summary>Python <c>str.lstrip()</c>: strips characters for which <c>str.isspace()</c> is true.</summary>
    internal static string PythonLStrip(string s)
    {
        int i = 0;
        while (i < s.Length && IsPythonSpace(s[i]))
        {
            i++;
        }

        return s[i..];
    }

    // Python's isspace: bidirectional class WS, B or S, or category Zs. That is .NET's char.IsWhiteSpace plus U+001C..U+001F.
    private static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || c is >= '\u001c' and <= '\u001f';
}
