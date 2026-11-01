using System.Globalization;
using System.Text;

namespace Starlings.Interventions;

/// <summary>
/// The cache key of a derived intervention artifact: the SHA-256 of a canonical text over everything the derived graph
/// depends on (compiler version, site-mapping version, opset, precision, profile hash, base graph and external weight
/// identities, declared inputs). Every component is recorded in the compiler manifest, so the key is recomputed from the
/// manifest alone and compared with the cache directory name. The text is specified in
/// <c>docs/interventions/cache.md</c>; the Python writer (<c>model_pipeline.interventions.cache</c>) produces the same.
/// </summary>
public static class InterventionCacheKey
{
    /// <summary>First line of the key text; it changes when the key format changes.</summary>
    public const string Header = "starling-intervention-cache-key 1";

    /// <summary>
    /// The canonical key text of <paramref name="manifest"/>: one field per line, fields separated by a tab, every line
    /// ending in LF; base artifacts sorted by path and inputs by name, both ordinally over UTF-8 bytes.
    /// </summary>
    public static string GetText(InterventionManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        StringBuilder text = new();
        Line(text, Header);
        Line(text, "compiler_version", manifest.CompilerVersion);
        Line(text, "site_mapping_version", manifest.SiteMappingVersion);
        Line(text, "opset", manifest.Opset.ToString(CultureInfo.InvariantCulture));
        Line(text, "precision", manifest.Precision);
        Line(text, "profile_sha256", manifest.ProfileSha256);

        InterventionBaseArtifact[] artifacts = [.. manifest.BaseArtifacts];
        Array.Sort(artifacts, (a, b) => CompareUtf8(a.File.Path, b.File.Path));
        foreach (InterventionBaseArtifact artifact in artifacts)
        {
            Line(
                text,
                "base",
                InterventionNames.Of(artifact.Role),
                artifact.File.Path,
                artifact.File.Sha256,
                artifact.File.Size.ToString(CultureInfo.InvariantCulture));
        }

        InterventionGraphInput[] inputs = [.. manifest.Inputs];
        Array.Sort(inputs, (a, b) => CompareUtf8(a.Name, b.Name));
        foreach (InterventionGraphInput input in inputs)
        {
            string shape = "[" + string.Join(',', input.Shape.Select(d => d.ToString(CultureInfo.InvariantCulture))) + "]";
            Line(text, "input", input.Name, InterventionNames.Of(input.Role), InterventionNames.Of(input.DType), shape);
        }

        return text.ToString();
    }

    /// <summary>The cache key of <paramref name="manifest"/>: SHA-256 of <see cref="GetText"/> as UTF-8, lower-case hex.</summary>
    public static string Compute(InterventionManifest manifest) => InterventionHashes.ComputeSha256(Encoding.UTF8.GetBytes(GetText(manifest)));

    private static void Line(StringBuilder text, params ReadOnlySpan<string> fields)
    {
        for (int i = 0; i < fields.Length; i++)
        {
            // The contract forbids control characters in every component, so this never fires for a valid manifest.
            if (fields[i].AsSpan().IndexOfAny('\n', '\r') >= 0)
            {
                throw new InterventionContractException($"cache key component contains a line break: '{fields[i]}'.");
            }

            if (i > 0)
            {
                text.Append('\t');
            }

            text.Append(fields[i]);
        }

        text.Append('\n');
    }

    private static int CompareUtf8(string a, string b) =>
        Encoding.UTF8.GetBytes(a).AsSpan().SequenceCompareTo(Encoding.UTF8.GetBytes(b));
}
