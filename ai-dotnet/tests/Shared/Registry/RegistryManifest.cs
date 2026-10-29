using System.Text.RegularExpressions;

namespace Starlings.TestSupport;

/// <summary>
/// Reads the fields of a registry manifest (<c>model-pipeline/registry/&lt;family&gt;/&lt;profile&gt;/manifest.yaml</c>)
/// that the packages' capability tables and generated capability records mirror: <c>maturity</c>, the
/// <c>capabilities</c> map, the <c>compatibility</c> rows and the contract fields of <c>behavior</c>
/// (<c>max_input_tokens</c>, <c>truncation</c>, <c>defaults.dimension</c>, <c>defaults.stop_token_ids</c>). Line based, for the manifests' own layout (two-space indentation, one key per line); the
/// schema validator (<c>model-pipeline validate</c>) checks the full document.
/// </summary>
internal sealed partial class RegistryManifest
{
    private RegistryManifest(string id, string maturity, IReadOnlyDictionary<string, string> capabilities, IReadOnlyDictionary<string, string> compatibility,
        IReadOnlyList<IReadOnlyDictionary<string, string>> compatibilityRows, IReadOnlyDictionary<string, string> behavior)
    {
        Id = id;
        Maturity = maturity;
        Capabilities = capabilities;
        Compatibility = compatibility;
        CompatibilityRows = compatibilityRows;
        Behavior = behavior;
    }

    /// <summary>The profile id (<c>id:</c>).</summary>
    public string Id { get; }

    /// <summary>The profile maturity (<c>maturity:</c>).</summary>
    public string Maturity { get; }

    /// <summary>Capability name to status, in file order.</summary>
    public IReadOnlyDictionary<string, string> Capabilities { get; }

    /// <summary>Platform (<c>os-arch provider</c>, plus the browser on browser rows, as the catalog labels it) to status.</summary>
    public IReadOnlyDictionary<string, string> Compatibility { get; }

    /// <summary>Every <c>compatibility</c> row in file order: field name (<c>os</c>, <c>arch</c>, <c>provider</c>, <c>status</c>, optional <c>browser</c> and <c>verified</c>) to value.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, string>> CompatibilityRows { get; }

    /// <summary>
    /// Scalar <c>behavior</c> fields: <c>max_input_tokens</c>, <c>truncation</c>, and from <c>defaults</c>,
    /// <c>dimension</c> and <c>stop_token_ids</c> (the flow list as written, without brackets), when present.
    /// </summary>
    public IReadOnlyDictionary<string, string> Behavior { get; }

    /// <summary>The manifests of a registry family, by profile id.</summary>
    public static IReadOnlyList<RegistryManifest> Family(string repositoryRoot, string family) =>
        Directory.GetDirectories(Path.Combine(repositoryRoot, "model-pipeline", "registry", family))
            .Select(d => Path.Combine(d, "manifest.yaml"))
            .Where(File.Exists)
            .Select(Load)
            .OrderBy(m => m.Id, StringComparer.Ordinal)
            .ToList();

    /// <summary>Parses one manifest.</summary>
    public static RegistryManifest Load(string path)
    {
        string[] lines = File.ReadAllText(path).Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n');
        string? id = null;
        string? maturity = null;
        var capabilities = new Dictionary<string, string>(StringComparer.Ordinal);
        var compatibility = new Dictionary<string, string>(StringComparer.Ordinal);
        var rows = new List<IReadOnlyDictionary<string, string>>();
        var behavior = new Dictionary<string, string>(StringComparer.Ordinal);
        string? section = null;
        string? subsection = null;
        Dictionary<string, string>? row = null;

        void FlushRow()
        {
            if (row is not null)
            {
                // Browser rows (providers wasm and webgpu) also name the browser, as the catalog's platform label does.
                string browser = row.TryGetValue("browser", out var b) ? " " + b.Trim('"') : string.Empty;
                compatibility.Add($"{row["os"]}-{row["arch"]} {row["provider"]}{browser}", row["status"]);
                rows.Add(row.ToDictionary(e => e.Key, e => e.Value.Trim('"'), StringComparer.Ordinal));
                row = null;
            }
        }

        foreach (string raw in lines)
        {
            string line = StripComment(raw);
            if (line.Trim().Length == 0)
            {
                continue;
            }

            if (!line.StartsWith(' '))
            {
                FlushRow();
                var top = TopLevel().Match(line);
                section = top.Success ? top.Groups[1].Value : null;
                if (section == "id")
                {
                    id = top.Groups[2].Value.Trim();
                }
                else if (section == "maturity")
                {
                    maturity = top.Groups[2].Value.Trim();
                }

                continue;
            }

            if (section == "behavior")
            {
                var field = BehaviorField().Match(line);
                if (field.Success)
                {
                    string indent = field.Groups[1].Value, key = field.Groups[2].Value, value = field.Groups[3].Value.Trim();
                    if (indent.Length == 2)
                    {
                        subsection = key;
                        if (key is "max_input_tokens" or "truncation")
                        {
                            behavior[key] = value;
                        }
                    }
                    else if (indent.Length == 4 && subsection == "defaults" && key is "dimension" or "stop_token_ids")
                    {
                        behavior[key] = value.Trim('[', ']').Trim();
                    }
                }
            }
            else if (section == "capabilities")
            {
                var entry = Entry().Match(line);
                if (!entry.Success)
                {
                    throw new FormatException($"{path}: unexpected capabilities line '{raw}'");
                }

                capabilities.Add(entry.Groups[1].Value, entry.Groups[2].Value);
            }
            else if (section == "compatibility")
            {
                var item = ListItem().Match(line);
                if (item.Success)
                {
                    FlushRow();
                    row = new Dictionary<string, string>(StringComparer.Ordinal) { [item.Groups[1].Value] = item.Groups[2].Value.Trim() };
                    continue;
                }

                var field = RowField().Match(line);
                if (field.Success && row is not null)
                {
                    row[field.Groups[1].Value] = field.Groups[2].Value.Trim();
                }
            }
        }

        FlushRow();
        return new RegistryManifest(
            id ?? throw new FormatException($"{path}: no id"),
            maturity ?? throw new FormatException($"{path}: no maturity"),
            capabilities,
            compatibility,
            rows,
            behavior);
    }

    private static string StripComment(string line)
    {
        int hash = line.IndexOf(" #", StringComparison.Ordinal);
        if (line.TrimStart().StartsWith('#'))
        {
            return string.Empty;
        }

        return hash >= 0 ? line[..hash] : line;
    }

    [GeneratedRegex(@"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")]
    private static partial Regex TopLevel();

    [GeneratedRegex(@"^  ([a-z0-9_]+):\s*([a-z-]+)\s*$")]
    private static partial Regex Entry();

    [GeneratedRegex(@"^( +)([a-z_]+):\s*(.*)$")]
    private static partial Regex BehaviorField();

    [GeneratedRegex(@"^  - ([a-z_]+):\s*(.*)$")]
    private static partial Regex ListItem();

    [GeneratedRegex(@"^    ([a-z_]+):\s*(.*)$")]
    private static partial Regex RowField();
}
