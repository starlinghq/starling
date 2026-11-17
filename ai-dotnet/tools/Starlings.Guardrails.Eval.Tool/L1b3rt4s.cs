using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

/// <summary>
/// Reads the L1B3RT4S jailbreak collection fetched by tools/fetch-l1b3rt4s.sh (AGPL-3.0; evaluation data only, never
/// committed). Vendor files are split into one item per prompt section; text is never printed by this tool.
/// </summary>
internal static partial class L1b3rt4s
{
    /// <summary>The pinned commit of tools/fetch-l1b3rt4s.sh.</summary>
    public const string Commit = "f441d65d901b29ca92ac20ee1afb5c9d0b6f8ef6";

    // Not attacks (leaked vendor system prompts, the project README, token and shortcut lists) or not prompts.
    private static readonly HashSet<string> Excluded = new(StringComparer.Ordinal)
    {
        "SYSTEMPROMPTS.mkd", "README.md", "LICENSE", "!SHORTCUTS.json", "*SPECIAL_TOKENS.json", ".verified",
    };

    // Token bombs: megabytes of variation selectors. Timed, not counted in detection rates.
    private static readonly HashSet<string> Bombs = new(StringComparer.Ordinal) { "TOKEN80M8.mkd", "TOKENADE.mkd" };

    /// <summary>The extraction directory: --eval-data, else $STARLING_EVAL_DATA, else &lt;repo&gt;/.eval-data.</summary>
    public static string Directory(string? evalData)
    {
        var root = evalData is { Length: > 0 } ? evalData
            : Environment.GetEnvironmentVariable("STARLING_EVAL_DATA") is { Length: > 0 } env ? env
            : Path.Combine(RepositoryRoot(), ".eval-data");
        var dir = Path.Combine(root, "elder-plinius", "L1B3RT4S", Commit);
        return File.Exists(Path.Combine(dir, ".verified"))
            ? dir
            : throw new FileNotFoundException($"L1B3RT4S not found at {dir}: run tools/fetch-l1b3rt4s.sh.");
    }

    /// <summary>Items of a set: <c>l1b3rt4s-dev</c>, <c>l1b3rt4s-heldout</c> (by file-name hash, 1 in 3 files held out) or <c>l1b3rt4s-bombs</c>.</summary>
    public static Item[] Load(string set, string directory)
    {
        var items = new List<Item>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var path in System.IO.Directory.GetFiles(directory).Order(StringComparer.Ordinal))
        {
            var name = Path.GetFileName(path);
            if (Excluded.Contains(name))
            {
                continue;
            }

            bool bomb = Bombs.Contains(name);
            if ((set == "l1b3rt4s-bombs") != bomb)
            {
                continue;
            }

            if (!bomb && (set == "l1b3rt4s-heldout") != IsHeldOut(name))
            {
                continue;
            }

            var text = File.ReadAllText(path, Encoding.UTF8);
            var sections = bomb ? [text] : Sections(text);
            for (int i = 0; i < sections.Count; i++)
            {
                var section = sections[i].Trim();
                if (section.Count(c => !char.IsWhiteSpace(c)) < 20)
                {
                    continue;
                }

                var hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(section)));
                if (seen.Add(hash))
                {
                    items.Add(new Item(section, 1, name, string.Create(CultureInfo.InvariantCulture, $"{name}#{i}:{hash[..8]}"), Sensitive: true));
                }
            }
        }

        return [.. items];
    }

    /// <summary>One file in three is held out, chosen by the first byte of the SHA-256 of its name.</summary>
    public static bool IsHeldOut(string fileName) => SHA256.HashData(Encoding.UTF8.GetBytes(fileName))[0] % 3 == 0;

    /// <summary>Splits on heading lines with text (# to ###); lines of # alone are dividers and stay in the text.</summary>
    private static List<string> Sections(string text)
    {
        var sections = new List<string>();
        var current = new StringBuilder();
        foreach (var line in text.Split('\n'))
        {
            if (Heading().IsMatch(line))
            {
                sections.Add(current.ToString());
                current.Clear();
                continue;
            }

            current.Append(line).Append('\n');
        }

        sections.Add(current.ToString());
        return sections;
    }

    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (System.IO.Directory.Exists(Path.Combine(dir.FullName, "model-pipeline")) && System.IO.Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        return Environment.CurrentDirectory;
    }

    [GeneratedRegex(@"^#{1,3}\s*[^#\s]")]
    private static partial Regex Heading();
}
