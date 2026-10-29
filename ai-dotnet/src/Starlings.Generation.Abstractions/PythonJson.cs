using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace Starlings.Generation;

/// <summary>
/// The two value-to-text conversions chat templates apply to JSON data, as Python does them on the values
/// <c>json.loads</c> gives: transformers' <c>tojson</c> filter (<c>json.dumps(x, ensure_ascii=False)</c>: separators
/// <c>", "</c> and <c>": "</c>, keys in order) and Jinja's <c>string</c> filter (<c>str(x)</c>) for scalars. A JSON
/// number with a fraction or an exponent is a Python <c>float</c> (shortest round-trip <c>repr</c>), any other number an
/// <c>int</c> (arbitrary precision). Data Python could not hold the same way (a duplicate object key, a lone UTF-16
/// surrogate, an undefined element) is rejected with <see cref="PythonJsonException"/>.
/// </summary>
public static class PythonJson
{
    /// <summary><c>json.dumps(value, ensure_ascii=False)</c>.</summary>
    public static string Dumps(JsonElement value)
    {
        var sb = new StringBuilder();
        Dumps(sb, value);
        return sb.ToString();
    }

    /// <summary>Appends <c>json.dumps(value, ensure_ascii=False)</c> to <paramref name="sb"/>.</summary>
    public static void Dumps(StringBuilder sb, JsonElement value)
    {
        ArgumentNullException.ThrowIfNull(sb);
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                {
                    sb.Append('{');
                    HashSet<string>? seen = null;
                    bool first = true;
                    foreach (var property in value.EnumerateObject())
                    {
                        string key = PropertyName(property);
                        if (!(seen ??= new HashSet<string>(StringComparer.Ordinal)).Add(key))
                        {
                            // A Python dict keeps one value per key; which one depends on how the JSON was loaded.
                            throw new PythonJsonException($"Duplicate JSON object key '{key}' in tool data.");
                        }

                        if (!first)
                        {
                            sb.Append(", ");
                        }

                        first = false;
                        AppendString(sb, key);
                        sb.Append(": ");
                        Dumps(sb, property.Value);
                    }

                    sb.Append('}');
                    break;
                }

            case JsonValueKind.Array:
                {
                    sb.Append('[');
                    bool first = true;
                    foreach (var item in value.EnumerateArray())
                    {
                        if (!first)
                        {
                            sb.Append(", ");
                        }

                        first = false;
                        Dumps(sb, item);
                    }

                    sb.Append(']');
                    break;
                }

            case JsonValueKind.String:
                AppendString(sb, GetString(value));
                break;
            case JsonValueKind.Number:
                sb.Append(NumberText(value, json: true));
                break;
            case JsonValueKind.True:
                sb.Append("true");
                break;
            case JsonValueKind.False:
                sb.Append("false");
                break;
            case JsonValueKind.Null:
                sb.Append("null");
                break;
            default:
                throw new PythonJsonException("Undefined JSON value in tool data.");
        }
    }

    /// <summary>
    /// Jinja's <c>string</c> filter on a value as a template writes one tool-call argument: <c>tojson</c> for an object
    /// or array, else <c>str(value)</c> (a string as is, <c>True</c>, <c>False</c>, <c>None</c>, Python number text).
    /// </summary>
    public static string Str(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Object or JsonValueKind.Array => Dumps(value),
        JsonValueKind.String => CheckedString(GetString(value)),
        JsonValueKind.Number => NumberText(value, json: false),
        JsonValueKind.True => "True",
        JsonValueKind.False => "False",
        JsonValueKind.Null => "None",
        _ => throw new PythonJsonException("Undefined JSON value in a tool-call argument."),
    };

    /// <summary>Python <c>repr(float)</c> of a finite double (shortest round-trip digits).</summary>
    public static string FloatRepr(double d)
    {
        if (double.IsNaN(d))
        {
            return "nan";
        }

        if (double.IsInfinity(d))
        {
            return d > 0 ? "inf" : "-inf";
        }

        if (d == 0)
        {
            return double.IsNegative(d) ? "-0.0" : "0.0";
        }

        // .NET's "R" is the shortest round-trip representation, like Python's repr; only the layout differs.
        string r = Math.Abs(d).ToString("R", CultureInfo.InvariantCulture);
        int exponentPart = 0;
        int e = r.IndexOfAny(['E', 'e']);
        string mantissa = r;
        if (e >= 0)
        {
            exponentPart = int.Parse(r.AsSpan(e + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            mantissa = r[..e];
        }

        int dot = mantissa.IndexOf('.', StringComparison.Ordinal);
        string intPart = dot >= 0 ? mantissa[..dot] : mantissa;
        string all = dot >= 0 ? intPart + mantissa[(dot + 1)..] : intPart;
        int leadingZeros = all.Length - all.TrimStart('0').Length;
        string digits = all.Trim('0');
        int exponent = intPart.Length - leadingZeros - 1 + exponentPart;

        var sb = new StringBuilder();
        if (d < 0)
        {
            sb.Append('-');
        }

        if (exponent < -4 || exponent >= 16)
        {
            sb.Append(digits[0]);
            if (digits.Length > 1)
            {
                sb.Append('.').Append(digits.AsSpan(1));
            }

            sb.Append('e').Append(exponent < 0 ? '-' : '+').Append(Math.Abs(exponent).ToString("00", CultureInfo.InvariantCulture));
        }
        else if (exponent >= 0)
        {
            if (digits.Length <= exponent + 1)
            {
                sb.Append(digits).Append('0', exponent + 1 - digits.Length).Append(".0");
            }
            else
            {
                sb.Append(digits.AsSpan(0, exponent + 1)).Append('.').Append(digits.AsSpan(exponent + 1));
            }
        }
        else
        {
            sb.Append("0.").Append('0', -exponent - 1).Append(digits);
        }

        return sb.ToString();
    }

    private static string NumberText(JsonElement value, bool json)
    {
        string raw = value.GetRawText();
        if (raw.AsSpan().IndexOfAny(".eE") < 0)
        {
            return BigInteger.Parse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
        }

        double d = double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (json && double.IsInfinity(d))
        {
            return d > 0 ? "Infinity" : "-Infinity";
        }

        return FloatRepr(d);
    }

    // json.dumps(ensure_ascii=False): escapes '"', '\\' and U+0000 to U+001F only.
    private static void AppendString(StringBuilder sb, string s)
    {
        CheckedString(s);
        sb.Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"':
                    sb.Append("\\\"");
                    break;
                case '\\':
                    sb.Append("\\\\");
                    break;
                case '\n':
                    sb.Append("\\n");
                    break;
                case '\r':
                    sb.Append("\\r");
                    break;
                case '\t':
                    sb.Append("\\t");
                    break;
                case '\b':
                    sb.Append("\\b");
                    break;
                case '\f':
                    sb.Append("\\f");
                    break;
                case < ' ':
                    sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    break;
                default:
                    sb.Append(c);
                    break;
            }
        }

        sb.Append('"');
    }

    private static string PropertyName(JsonProperty property)
    {
        try
        {
            return property.Name;
        }
        catch (InvalidOperationException ex)
        {
            throw new PythonJsonException("Tool data contains a lone UTF-16 surrogate.", ex);
        }
    }

    private static string GetString(JsonElement value)
    {
        try
        {
            return value.GetString()!;
        }
        catch (InvalidOperationException ex)
        {
            throw new PythonJsonException("Tool data contains a lone UTF-16 surrogate.", ex);
        }
    }

    /// <summary>Returns <paramref name="s"/>, or throws when it holds a lone UTF-16 surrogate (no UTF-8 form; the reference cannot tokenize it).</summary>
    /// <exception cref="PythonJsonException">A lone surrogate.</exception>
    public static string CheckedString(string s)
    {
        ArgumentNullException.ThrowIfNull(s);
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(s[i]))
            {
                throw new PythonJsonException("Tool data contains a lone UTF-16 surrogate.");
            }
        }

        return s;
    }
}

/// <summary>JSON data that <see cref="PythonJson"/> cannot format as Python would (the message says why).</summary>
public sealed class PythonJsonException : Exception
{
    /// <summary>Creates the exception.</summary>
    public PythonJsonException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public PythonJsonException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public PythonJsonException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
