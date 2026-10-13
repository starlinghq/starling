using System.Globalization;
using System.Text;

namespace Starlings.Models.Gliner2;

/// <summary>One word of GLiNER2's whitespace splitter: the lower-cased word and its span in Unicode scalar values.</summary>
/// <param name="Word">The matched text, lower-cased as Python <c>str.lower()</c> does.</param>
/// <param name="Start">Start, in Unicode scalar values of the normalized text.</param>
/// <param name="End">Exclusive end, in Unicode scalar values of the normalized text.</param>
internal readonly record struct Gliner2Word(string Word, int Start, int End);

/// <summary>
/// GLiNER2's text handling with the reference interpreter's semantics (CPython 3.11, Unicode 14.0.0): text
/// normalization (<c>processor.py:537</c>), the <c>WhitespaceTokenSplitter</c> regular expression
/// (<c>processing/word_splitter.py:20</c>) and per-word <c>str.lower()</c> (<c>processor.py:635</c>).
/// </summary>
/// <remarks>
/// <para>
/// The splitter is a hand-written scanner over Unicode scalar values, not a .NET <see cref="System.Text.RegularExpressions.Regex"/>:
/// .NET character classes work on UTF-16 code units (an astral letter is two non-letters to <c>\p{L}</c>), .NET's
/// <c>\s</c> lacks U+001C to U+001F, and .NET's <c>\w</c> includes combining marks, all unlike Python's <c>re</c>.
/// The pattern, alternatives tried in order at each position, leftmost match first:
/// <c>https?://\S+ | www\.\S+ | [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,} | @[a-z0-9_]+ | \w+(?:[-_]\w+)* | \S</c>
/// with IGNORECASE.
/// </para>
/// <para>
/// Character properties come from .NET's Unicode data, which is newer than Unicode 14. Characters unassigned in
/// Unicode 14 may therefore split or lower-case differently from the reference; for every character assigned in
/// Unicode 14 the predicates here equal CPython 3.11's (unit-tested over all scalar values).
/// </para>
/// </remarks>
internal static class Gliner2Text
{
    /// <summary>
    /// GLiNER2 normalization: empty text becomes <c>"."</c>; otherwise <c>"."</c> is appended unless the text ends in
    /// <c>.</c>, <c>!</c> or <c>?</c>. The original text is always a prefix of the result.
    /// </summary>
    public static string Normalize(string text)
    {
        if (text.Length == 0)
        {
            return ".";
        }

        return text[^1] is '.' or '!' or '?' ? text : text + ".";
    }

    /// <summary>Python <c>re</c> <c>\s</c> for str patterns (<c>str.isspace()</c>).</summary>
    public static bool IsSpace(int c) => c is (>= 0x9 and <= 0xD) or (>= 0x1C and <= 0x20) or 0x85 or 0xA0 or 0x1680
        or (>= 0x2000 and <= 0x200A) or 0x2028 or 0x2029 or 0x202F or 0x205F or 0x3000;

    /// <summary>Python <c>re</c> <c>\w</c> for str patterns: <c>str.isalnum()</c> or <c>_</c> (L*, Nd, Nl, No, and <c>_</c>).</summary>
    public static bool IsWord(int c)
    {
        if (c == '_')
        {
            return true;
        }

        return Rune.GetUnicodeCategory(new Rune(c)) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
            or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter or UnicodeCategory.OtherLetter
            or UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber;
    }

    /// <summary>Python <c>[a-z]</c> under IGNORECASE: ASCII letters plus U+0130, U+0131, U+017F and U+212A.</summary>
    public static bool IsAsciiLetterIgnoreCase(int c) => c is (>= 'a' and <= 'z') or (>= 'A' and <= 'Z') or 0x130 or 0x131 or 0x17F or 0x212A;

    /// <summary>
    /// Splits normalized text into GLiNER2 words. Offsets are Unicode scalar values of <paramref name="scalars"/>.
    /// </summary>
    public static List<Gliner2Word> SplitWords(ReadOnlySpan<int> scalars)
    {
        var words = new List<Gliner2Word>();
        int i = 0;
        while (i < scalars.Length)
        {
            if (IsSpace(scalars[i]))
            {
                i++;
                continue;
            }

            int end = MatchUrl(scalars, i);
            if (end < 0)
            {
                end = MatchEmail(scalars, i);
            }

            if (end < 0)
            {
                end = MatchHandle(scalars, i);
            }

            if (end < 0)
            {
                end = MatchWord(scalars, i);
            }

            if (end < 0)
            {
                end = i + 1;
            }

            words.Add(new Gliner2Word(Lower(scalars[i..end]), i, end));
            i = end;
        }

        return words;
    }

    /// <summary>The Unicode scalar values of <paramref name="text"/>; rejects unpaired surrogates.</summary>
    /// <exception cref="ArgumentException">The text contains an unpaired surrogate (not valid Unicode text).</exception>
    public static int[] ToScalars(string text, string parameterName)
    {
        var scalars = new int[text.Length];
        int count = 0;
        for (int i = 0; i < text.Length;)
        {
            var status = Rune.DecodeFromUtf16(text.AsSpan(i), out var rune, out int consumed);
            if (status != System.Buffers.OperationStatus.Done)
            {
                throw new ArgumentException($"Text contains an unpaired surrogate at UTF-16 index {i}; only valid Unicode text is accepted.", parameterName);
            }

            scalars[count++] = rune.Value;
            i += consumed;
        }

        return scalars.AsSpan(0, count).ToArray();
    }

    /// <summary>
    /// Python <c>str.lower()</c>: the simple lower-case mapping, except U+0130 (full mapping, two scalar values) and
    /// U+03A3 (Final_Sigma: <c>ς</c> after a cased letter not followed by one, ignoring case-ignorable characters).
    /// </summary>
    public static string Lower(ReadOnlySpan<int> scalars)
    {
        var builder = new StringBuilder(scalars.Length + 2);
        Span<char> buffer = stackalloc char[2];
        for (int i = 0; i < scalars.Length; i++)
        {
            int c = scalars[i];
            if (c == 0x130)
            {
                builder.Append('i').Append('̇');
                continue;
            }

            if (c == 0x3A3)
            {
                builder.Append(IsFinalSigma(scalars, i) ? 'ς' : 'σ');
                continue;
            }

            int written = Rune.ToLowerInvariant(new Rune(c)).EncodeToUtf16(buffer);
            builder.Append(buffer[..written]);
        }

        return builder.ToString();
    }

    /// <summary>Python <c>str.strip()</c> bounds over scalar values (whitespace as <see cref="IsSpace"/>).</summary>
    public static (int Start, int End) Strip(ReadOnlySpan<int> scalars, int start, int end)
    {
        while (start < end && IsSpace(scalars[start]))
        {
            start++;
        }

        while (end > start && IsSpace(scalars[end - 1]))
        {
            end--;
        }

        return (start, end);
    }

    /// <summary>The UTF-16 string of scalar values <paramref name="start"/>..<paramref name="end"/>.</summary>
    public static string ToText(ReadOnlySpan<int> scalars, int start, int end)
    {
        var builder = new StringBuilder(end - start);
        Span<char> buffer = stackalloc char[2];
        for (int i = start; i < end; i++)
        {
            int written = new Rune(scalars[i]).EncodeToUtf16(buffer);
            builder.Append(buffer[..written]);
        }

        return builder.ToString();
    }

    /// <summary>CPython's <c>handle_capital_sigma</c> over the word being lower-cased.</summary>
    internal static bool IsFinalSigma(ReadOnlySpan<int> scalars, int i)
    {
        int j = i - 1;
        while (j >= 0 && IsCaseIgnorable(scalars[j]))
        {
            j--;
        }

        if (j < 0 || !IsCased(scalars[j]))
        {
            return false;
        }

        j = i + 1;
        while (j < scalars.Length && IsCaseIgnorable(scalars[j]))
        {
            j++;
        }

        return j == scalars.Length || !IsCased(scalars[j]);
    }

    /// <summary>
    /// Unicode Case_Ignorable: Mn, Me, Cf, Lm, Sk, and Word_Break MidLetter, MidNumLet or Single_Quote.
    /// </summary>
    internal static bool IsCaseIgnorable(int c)
    {
        // U+1171E (AHOM CONSONANT SIGN MEDIAL RA) is Mn in Unicode 14 (the reference) and Mc in later versions.
        if (c is 0x27 or 0x2E or 0x3A or 0xB7 or 0x387 or 0x55F or 0x5F4 or 0x2018 or 0x2019 or 0x2024 or 0x2027
            or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A or 0x1171E)
        {
            return true;
        }

        return Rune.GetUnicodeCategory(new Rune(c)) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark
            or UnicodeCategory.Format or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol;
    }

    /// <summary>
    /// Unicode Cased: Lu, Ll, Lt, plus the Other_Lowercase and Other_Uppercase characters that are not case-ignorable
    /// (ordinal indicators, Roman numerals, circled and squared Latin letters).
    /// </summary>
    internal static bool IsCased(int c)
    {
        if (c is 0xAA or 0xBA or (>= 0x2160 and <= 0x217F) or (>= 0x24B6 and <= 0x24E9)
            or (>= 0x1F130 and <= 0x1F149) or (>= 0x1F150 and <= 0x1F169) or (>= 0x1F170 and <= 0x1F189))
        {
            return true;
        }

        return Rune.GetUnicodeCategory(new Rune(c)) is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter
            or UnicodeCategory.TitlecaseLetter;
    }

    /// <summary>Python IGNORECASE for the ASCII letters of the URL prefixes: only <c>s</c> has a non-ASCII match (U+017F).</summary>
    private static bool MatchesLetterIgnoreCase(int c, char lower) =>
        c == lower || c == lower - 32 || (lower == 's' && c == 0x17F);

    private static bool StartsWithIgnoreCase(ReadOnlySpan<int> t, int i, string prefix)
    {
        if (i + prefix.Length > t.Length)
        {
            return false;
        }

        for (int j = 0; j < prefix.Length; j++)
        {
            char p = prefix[j];
            bool ok = p is >= 'a' and <= 'z' ? MatchesLetterIgnoreCase(t[i + j], p) : t[i + j] == p;
            if (!ok)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary><c>https?://\S+</c> or <c>www\.\S+</c>; returns the end or -1.</summary>
    private static int MatchUrl(ReadOnlySpan<int> t, int i)
    {
        int j;
        if (StartsWithIgnoreCase(t, i, "https://"))
        {
            j = i + 8;
        }
        else if (StartsWithIgnoreCase(t, i, "http://"))
        {
            j = i + 7;
        }
        else if (StartsWithIgnoreCase(t, i, "www."))
        {
            j = i + 4;
        }
        else
        {
            return -1;
        }

        int k = j;
        while (k < t.Length && !IsSpace(t[k]))
        {
            k++;
        }

        return k > j ? k : -1;
    }

    private static bool IsAsciiDigit(int c) => c is >= '0' and <= '9';

    private static bool IsEmailLocal(int c) => IsAsciiLetterIgnoreCase(c) || IsAsciiDigit(c) || c is '.' or '_' or '%' or '+' or '-';

    private static bool IsEmailDomain(int c) => IsAsciiLetterIgnoreCase(c) || IsAsciiDigit(c) || c is '.' or '-';

    private static bool IsHandle(int c) => IsAsciiLetterIgnoreCase(c) || IsAsciiDigit(c) || c == '_';

    /// <summary>
    /// <c>[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}</c> with Python's backtracking: the local part cannot contain
    /// <c>@</c>, so it must run up to one; the greedy domain gives back characters from the right until a <c>.</c>
    /// followed by at least two letters is found (the letters then extend greedily).
    /// </summary>
    private static int MatchEmail(ReadOnlySpan<int> t, int i)
    {
        int j = i;
        while (j < t.Length && IsEmailLocal(t[j]))
        {
            j++;
        }

        if (j == i || j >= t.Length || t[j] != '@')
        {
            return -1;
        }

        int domainStart = j + 1;
        int domainEnd = domainStart;
        while (domainEnd < t.Length && IsEmailDomain(t[domainEnd]))
        {
            domainEnd++;
        }

        for (int dot = domainEnd - 1; dot >= domainStart + 1; dot--)
        {
            if (t[dot] != '.')
            {
                continue;
            }

            int q = dot + 1;
            while (q < t.Length && IsAsciiLetterIgnoreCase(t[q]))
            {
                q++;
            }

            if (q - (dot + 1) >= 2)
            {
                return q;
            }
        }

        return -1;
    }

    /// <summary><c>@[a-z0-9_]+</c>.</summary>
    private static int MatchHandle(ReadOnlySpan<int> t, int i)
    {
        if (t[i] != '@')
        {
            return -1;
        }

        int j = i + 1;
        while (j < t.Length && IsHandle(t[j]))
        {
            j++;
        }

        return j > i + 1 ? j : -1;
    }

    /// <summary><c>\w+(?:[-_]\w+)*</c>.</summary>
    private static int MatchWord(ReadOnlySpan<int> t, int i)
    {
        if (!IsWord(t[i]))
        {
            return -1;
        }

        int j = i;
        while (j < t.Length && IsWord(t[j]))
        {
            j++;
        }

        while (j + 1 < t.Length && t[j] is '-' or '_' && IsWord(t[j + 1]))
        {
            j++;
            while (j < t.Length && IsWord(t[j]))
            {
                j++;
            }
        }

        return j;
    }
}
