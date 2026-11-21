using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Starlings.Guardrails;

/// <summary>Hidden-character findings of one text (before anything is stripped).</summary>
/// <param name="TagCharacters">Unicode tag characters outside a subdivision flag (U+1F3F4, tag letters, U+E007F).</param>
/// <param name="SmuggledSelectors">Variation selectors in runs of two or more, or a supplementary selector not after a Han ideograph.</param>
/// <param name="BidiControls">Bidirectional embeddings and overrides (U+202A..U+202E); isolates and marks are ordinary.</param>
/// <param name="InvisibleSeparators">U+200B, U+2060..U+2064, U+180E, and U+FEFF after the first character (joiners excluded).</param>
/// <param name="TagPayload">The tag characters read as ASCII, or null.</param>
/// <param name="SelectorPayload">The smuggled selectors read as bytes (U+FE00..FE0F 0..15, U+E0100..E01EF 16..255) and decoded as UTF-8, or null.</param>
internal readonly record struct HiddenText(int TagCharacters, int SmuggledSelectors, int BidiControls, int InvisibleSeparators, string? TagPayload, string? SelectorPayload);

/// <summary>One text to scan: the plain text, or a payload decoded from it.</summary>
/// <param name="Source">What the text is: <c>plain</c>, <c>tags</c>, <c>selectors</c>, <c>base64</c>, <c>hex</c>, <c>rot13</c>, <c>reversed</c>, <c>upside-down</c>.</param>
/// <param name="Text">The text.</param>
internal readonly record struct DecodedText(string Source, string Text);

/// <summary>
/// The obfuscation handling of <see cref="PatternGuard"/>: finds hidden characters before normalization removes them,
/// decodes smuggled and encoded payloads, and builds the views the rules run on (cleaned, folded, leet-mapped, squashed).
/// </summary>
internal static partial class TextViews
{
    private const int MaxBlobs = 8;
    private const int MaxBlobChars = 4096;

    /// <summary>Counts and decodes hidden characters, exempting their ordinary uses.</summary>
    public static HiddenText FindHidden(string text)
    {
        int tagCount = 0, selectorCount = 0, bidi = 0, separators = 0;
        StringBuilder? tagPayload = null;
        List<byte>? selectorBytes = null;
        var runes = text.EnumerateRunes().ToArray();
        for (int i = 0; i < runes.Length; i++)
        {
            int v = runes[i].Value;
            if (v is >= 0xE0000 and <= 0xE007F)
            {
                // One pass over the whole run of tag characters (linear on hostile input): a subdivision flag (black
                // flag, 2 to 7 tag letters or digits, cancel tag) is ordinary; anything else is smuggled text.
                int end = i;
                while (end < runes.Length && runes[end].Value is >= 0xE0000 and <= 0xE007F)
                {
                    end++;
                }

                if (!IsFlag(runes, i, end))
                {
                    for (int k = i; k < end; k++)
                    {
                        tagCount++;
                        if (runes[k].Value is >= 0xE0020 and <= 0xE007E)
                        {
                            (tagPayload ??= new()).Append((char)(runes[k].Value - 0xE0000));
                        }
                    }
                }

                i = end - 1;
                continue;
            }

            if (IsSelector(v))
            {
                int start = i;
                while (i + 1 < runes.Length && IsSelector(runes[i + 1].Value))
                {
                    i++;
                }

                int length = i - start + 1;
                bool supplementaryAfterNonHan = length == 1 && v >= 0xE0100 && (start == 0 || !IsHan(runes[start - 1].Value));
                if (length >= 2 || supplementaryAfterNonHan)
                {
                    selectorCount += length;
                    selectorBytes ??= [];
                    for (int k = start; k <= i; k++)
                    {
                        int s = runes[k].Value;
                        selectorBytes.Add((byte)(s <= 0xFE0F ? s - 0xFE00 : s - 0xE0100 + 16));
                    }
                }

                continue;
            }

            if (v is >= 0x202A and <= 0x202E)
            {
                bidi++;
            }
            else if (v is 0x200B or (>= 0x2060 and <= 0x2064) or 0x180E || (v == 0xFEFF && i > 0))
            {
                separators++;
            }
        }

        string? selectorPayload = selectorBytes is null ? null : Encoding.UTF8.GetString(selectorBytes.ToArray());
        return new(tagCount, selectorCount, bidi, separators, tagPayload?.ToString(), selectorPayload);
    }

    /// <summary>
    /// The plain view: hidden and formatting characters removed (tags, selectors, bidi controls and marks, zero-width
    /// characters and joiners, soft hyphen, combining grapheme joiner), then NFKC (full-width, mathematical alphanumerics
    /// and other compatibility forms folded).
    /// </summary>
    public static string Clean(string text)
    {
        // Fast path: no hidden character (tags and supplementary selectors start with the high surrogate U+DB40) and no lone
        // surrogate (normalization throws on one; the slow path replaces it with U+FFFD).
        if (text.AsSpan().IndexOfAny(HiddenStarts) < 0 && IsWellFormed(text))
        {
            return text.Normalize(NormalizationForm.FormKC);
        }

        var builder = new StringBuilder(text.Length);
        Span<char> buffer = stackalloc char[2];
        foreach (var rune in text.EnumerateRunes())
        {
            int v = rune.Value;
            bool hidden = v is (>= 0xE0000 and <= 0xE007F) or (>= 0xE0100 and <= 0xE01EF) or (>= 0xFE00 and <= 0xFE0F)
                or (>= 0x200B and <= 0x200F) or (>= 0x202A and <= 0x202E) or (>= 0x2060 and <= 0x2069)
                or 0xFEFF or 0x00AD or 0x034F or 0x180E;
            if (!hidden)
            {
                builder.Append(buffer[..rune.EncodeToUtf16(buffer)]);
            }
        }

        return builder.ToString().Normalize(NormalizationForm.FormKC);
    }

    private static bool IsWellFormed(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(text[i]))
            {
                return false;
            }
        }

        return true;
    }

    // Every BMP character Clean's slow path removes, plus the high surrogate of the tag and supplementary-selector planes.
    private static readonly SearchValues<char> HiddenStarts = SearchValues.Create(
        "\uDB40\u00AD\u034F\u180E\u200B\u200C\u200D\u200E\u200F"
        + "\u202A\u202B\u202C\u202D\u202E"
        + "\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069"
        + "\uFE00\uFE01\uFE02\uFE03\uFE04\uFE05\uFE06\uFE07\uFE08\uFE09\uFE0A\uFE0B\uFE0C\uFE0D\uFE0E\uFE0F\uFEFF");

    /// <summary>
    /// The folded view of a cleaned text: combining marks removed (diacritics, underlines, zalgo) and common Cyrillic,
    /// Greek and Armenian homoglyphs mapped to Latin. For the English rules only: it garbles text in those scripts.
    /// </summary>
    public static string Fold(string clean)
    {
        var decomposed = clean.Normalize(NormalizationForm.FormD);
        var builder = new StringBuilder(decomposed.Length);
        foreach (var c in decomposed)
        {
            var category = CharUnicodeInfo.GetUnicodeCategory(c);
            if (category is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark)
            {
                continue;
            }

            builder.Append(Homoglyphs.TryGetValue(c, out var latin) ? latin : c);
        }

        return builder.ToString().Normalize(NormalizationForm.FormC);
    }

    /// <summary>
    /// True when a cleaned text shows a character trick that folding undoes: a combining mark on a Latin letter that NFKC
    /// left uncomposed (underline, zalgo, stacked accents), or a Cyrillic, Greek or Armenian look-alike inside a word with
    /// Latin letters. Ordinary accented text and text in those scripts return false.
    /// </summary>
    public static bool HasCharacterTricks(string clean)
    {
        // One pass: the last non-mark character of the word is carried, not searched for (linear on long mark runs).
        bool latin = false, lookAlike = false;
        char? lastBase = null;
        for (int i = 0; i <= clean.Length; i++)
        {
            if (i == clean.Length || char.IsWhiteSpace(clean[i]))
            {
                if (latin && lookAlike)
                {
                    return true;
                }

                latin = lookAlike = false;
                lastBase = null;
                continue;
            }

            char c = clean[i];
            latin |= char.IsAsciiLetter(c);
            lookAlike |= Homoglyphs.ContainsKey(c);
            if (CharUnicodeInfo.GetUnicodeCategory(c) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark)
            {
                if (lastBase is { } b && b < 0x250 && char.IsLetter(b))
                {
                    return true;
                }
            }
            else
            {
                lastBase = c;
            }
        }

        return false;
    }

    /// <summary>
    /// The leet view of a folded text: in words that mix letters with digits or symbols, those that stand for letters are
    /// mapped back (1 as <paramref name="one"/>: i or l); words of digits alone stay. A word spelled entirely in digits
    /// ("1337") is not mapped. Returns null when fewer than <paramref name="minWords"/> words change (ordinary text has the
    /// odd "24px" or "Q3"; leet attacks spell many words), so the caller can skip the view.
    /// </summary>
    public static string? Leet(string folded, char one = 'i', int minWords = 3)
    {
        char[]? chars = null;
        int changed = 0;
        int i = 0;
        while (i < folded.Length)
        {
            if (char.IsWhiteSpace(folded[i]))
            {
                i++;
                continue;
            }

            int start = i;
            bool letter = false, leet = false;
            while (i < folded.Length && !char.IsWhiteSpace(folded[i]))
            {
                letter |= char.IsLetter(folded[i]);
                leet |= LeetLetter(folded[i], one) != folded[i];
                i++;
            }

            if (letter && leet)
            {
                changed++;
                chars ??= folded.ToCharArray();
                for (int k = start; k < i; k++)
                {
                    chars[k] = LeetLetter(chars[k], one);
                }
            }
        }

        return changed < minWords ? null : new string(chars!);
    }

    /// <summary>Every leet digit and symbol mapped back, wherever it is (for the squashed view, where spacing is gone anyway).</summary>
    public static string LeetAll(string folded, char one = 'i') => new([.. folded.Select(c => LeetLetter(c, one))]);

    private static char LeetLetter(char c, char one) => c switch
    {
        '0' => 'o', '1' => one, '3' => 'e', '4' => 'a', '5' => 's', '7' => 't', '8' => 'b', '9' => 'g',
        '@' => 'a', '$' => 's', '!' => 'i', '|' => 'l', '+' => 't', '€' => 'e',
        _ => c,
    };

    /// <summary>The squashed view of a leet view: letters only, lower case. Defeats spacing, punctuation, emoji between letters.</summary>
    public static string Squash(string leet)
    {
        var builder = new StringBuilder(leet.Length);
        foreach (var c in leet)
        {
            if (char.IsLetter(c))
            {
                builder.Append(char.ToLowerInvariant(c));
            }
        }

        return builder.ToString();
    }

    /// <summary>Squashes any text the same way the views do (clean, fold, leet, squash), for phrases and canaries.</summary>
    public static string SquashAll(string text)
    {
        return Squash(LeetAll(Fold(Clean(text))));
    }

    /// <summary>
    /// Payloads decoded from a cleaned text, one level: base64 and hex blobs anywhere in it (at most 8), and ROT13, reversed
    /// and upside-down copies of its first <paramref name="maxChars"/> characters.
    /// </summary>
    public static IEnumerable<DecodedText> Decode(string clean, int maxChars)
    {
        int blobs = 0;
        foreach (Match m in Base64Blob().Matches(clean))
        {
            if (blobs >= MaxBlobs)
            {
                break;
            }

            if (m.Length <= MaxBlobChars && TryBase64(m.Value) is { } decoded)
            {
                blobs++;
                yield return new("base64", decoded);
            }
        }

        foreach (Match m in HexBlob().Matches(clean))
        {
            if (blobs >= MaxBlobs)
            {
                break;
            }

            if (m.Length <= MaxBlobChars && TryHex(m.Value) is { } decoded)
            {
                blobs++;
                yield return new("hex", decoded);
            }
        }

        var text = clean.Length > maxChars ? clean[..maxChars] : clean;
        if (text.Length < clean.Length && char.IsHighSurrogate(text[^1]))
        {
            text = text[..^1]; // the cut split a surrogate pair; normalization throws on the lone half
        }

        if (text.Any(char.IsAsciiLetter))
        {
            yield return new("rot13", Rot13(text));
        }

        yield return new("reversed", Reverse(text));
        if (text.Count(UpsideDown.ContainsKey) >= 3)
        {
            yield return new("upside-down", Reverse(new string([.. text.Select(FlipBack)])));
        }
    }

    /// <summary>The tag run [start, end) after a black flag is 2 to 7 tag letters or digits and a cancel tag (for example gbsct).</summary>
    private static bool IsFlag(Rune[] runes, int start, int end)
    {
        if (start == 0 || runes[start - 1].Value != 0x1F3F4 || end - start is < 3 or > 8 || runes[end - 1].Value != 0xE007F)
        {
            return false;
        }

        for (int k = start; k < end - 1; k++)
        {
            if (runes[k].Value is not ((>= 0xE0030 and <= 0xE0039) or (>= 0xE0061 and <= 0xE007A)))
            {
                return false;
            }
        }

        return true;
    }

    private static bool IsSelector(int v) => v is (>= 0xFE00 and <= 0xFE0F) or (>= 0xE0100 and <= 0xE01EF);

    private static bool IsHan(int v) => v is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0x20000 and <= 0x323AF) or (>= 0xF900 and <= 0xFAFF);

    private static string? TryBase64(string blob)
    {
        var s = blob.Replace('-', '+').Replace('_', '/').TrimEnd('=');
        s = s.PadRight(s.Length + ((4 - (s.Length % 4)) % 4), '=');
        var buffer = new byte[s.Length * 3 / 4];
        return Convert.TryFromBase64String(s, buffer, out int written) ? Printable(buffer.AsSpan(0, written)) : null;
    }

    private static string? TryHex(string blob)
    {
        try
        {
            return Printable(Convert.FromHexString(blob));
        }
        catch (FormatException)
        {
            return null;
        }
    }

    /// <summary>The bytes as UTF-8 text when they read as text (valid, at least 85% printable, some letters); else null.</summary>
    private static string? Printable(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 8)
        {
            return null;
        }

        string text;
        try
        {
            text = new UTF8Encoding(false, true).GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            return null;
        }

        int printable = text.Count(c => !char.IsControl(c) || c is '\n' or '\r' or '\t');
        int letters = text.Count(char.IsLetter);
        return printable >= text.Length * 0.85 && letters >= text.Length * 0.4 ? text : null;
    }

    private static string Rot13(string text) => new([.. text.Select(c => c switch
    {
        >= 'a' and <= 'z' => (char)('a' + ((c - 'a' + 13) % 26)),
        >= 'A' and <= 'Z' => (char)('A' + ((c - 'A' + 13) % 26)),
        _ => c,
    })]);

    // By code point, so surrogate pairs stay valid (normalization throws on a lone surrogate).
    private static string Reverse(string text)
    {
        var runes = text.EnumerateRunes().ToArray();
        Array.Reverse(runes);
        return string.Concat(runes.Select(r => r.ToString()));
    }

    private static readonly Dictionary<char, char> Homoglyphs = new()
    {
        // Cyrillic
        ['А'] = 'A', ['а'] = 'a', ['В'] = 'B', ['Е'] = 'E', ['е'] = 'e', ['К'] = 'K', ['к'] = 'k', ['М'] = 'M', ['Н'] = 'H',
        ['О'] = 'O', ['о'] = 'o', ['Р'] = 'P', ['р'] = 'p', ['С'] = 'C', ['с'] = 'c', ['Т'] = 'T', ['у'] = 'y', ['Х'] = 'X',
        ['х'] = 'x', ['І'] = 'I', ['і'] = 'i', ['Ј'] = 'J', ['ј'] = 'j', ['Ѕ'] = 'S', ['ѕ'] = 's', ['ԁ'] = 'd', ['ӏ'] = 'l',
        ['Ү'] = 'Y', ['ԛ'] = 'q', ['ԝ'] = 'w', ['ɡ'] = 'g',
        // Greek
        ['Α'] = 'A', ['Β'] = 'B', ['Ε'] = 'E', ['Ζ'] = 'Z', ['Η'] = 'H', ['Ι'] = 'I', ['Κ'] = 'K', ['Μ'] = 'M', ['Ν'] = 'N',
        ['Ο'] = 'O', ['Ρ'] = 'P', ['Τ'] = 'T', ['Υ'] = 'Y', ['Χ'] = 'X', ['α'] = 'a', ['ο'] = 'o', ['ρ'] = 'p', ['ι'] = 'i',
        ['ν'] = 'v', ['κ'] = 'k', ['τ'] = 't', ['υ'] = 'u', ['ε'] = 'e',
        // Armenian
        ['օ'] = 'o', ['ս'] = 'u', ['ց'] = 'g', ['հ'] = 'h',
    };

    // In text that has upside-down letters, the Latin letters that are each other's flips swap back too.
    private static char FlipBack(char c) => UpsideDown.TryGetValue(c, out var r) ? r : c switch
    {
        'q' => 'b', 'b' => 'q', 'd' => 'p', 'p' => 'd', 'n' => 'u', 'u' => 'n',
        _ => c,
    };

    // Upside-down letters that are not Latin letters themselves (their presence marks upside-down text).
    private static readonly Dictionary<char, char> UpsideDown = new()
    {
        ['ɐ'] = 'a', ['ɔ'] = 'c', ['ǝ'] = 'e', ['ɟ'] = 'f', ['ƃ'] = 'g', ['ɥ'] = 'h', ['ı'] = 'i', ['ɾ'] = 'j', ['ʞ'] = 'k',
        ['ן'] = 'l', ['ɯ'] = 'm', ['ɹ'] = 'r', ['ʇ'] = 't', ['ʌ'] = 'v', ['ʍ'] = 'w', ['ʎ'] = 'y',
    };

    [GeneratedRegex(@"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}", RegexOptions.CultureInvariant, 100)]
    private static partial Regex Base64Blob();

    [GeneratedRegex(@"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Fa-f])", RegexOptions.CultureInvariant, 100)]
    private static partial Regex HexBlob();
}
