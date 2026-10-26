using System.Globalization;
using Starlings.Models;

namespace Starlings.TestSupport;

/// <summary>
/// Checks a package's generated capability records (<see cref="ModelProfileCapabilities"/>, from
/// <c>model-pipeline catalog</c>) against the registry manifests they were generated from, field for field and
/// capability key for key, read by an independent parser (<see cref="RegistryManifest"/>).
/// </summary>
internal static class CapabilityRecordChecks
{
    /// <summary>
    /// <paramref name="records"/> holds exactly the profiles of <paramref name="manifests"/>, and every record equals
    /// its manifest: maturity, max input tokens, truncation, dimension, qualified providers, platform rows, stop token
    /// ids and the whole capabilities map.
    /// </summary>
    public static void AssertRecordsEqualManifests(IReadOnlyDictionary<string, ModelProfileCapabilities> records, IEnumerable<RegistryManifest> manifests)
    {
        var expected = manifests.ToList();
        Assert.Equal(expected.Select(m => m.Id).Order(StringComparer.Ordinal), records.Keys.Order(StringComparer.Ordinal));
        foreach (var manifest in expected)
        {
            var record = records[manifest.Id];
            string id = manifest.Id;
            Assert.Equal(id, record.ProfileId);
            Assert.True(manifest.Maturity == Name(record.Maturity), $"{id}: maturity {manifest.Maturity}, record {record.Maturity}");
            Assert.Equal(int.Parse(manifest.Behavior["max_input_tokens"], CultureInfo.InvariantCulture), record.MaxInputTokens);
            Assert.True(manifest.Behavior["truncation"] == Truncation(record.Truncation), $"{id}: truncation {manifest.Behavior["truncation"]}, record {record.Truncation}");
            int? dimension = manifest.Behavior.TryGetValue("dimension", out var d) ? int.Parse(d, CultureInfo.InvariantCulture) : null;
            Assert.Equal(dimension, record.Dimension);
            int[] stops = manifest.Behavior.TryGetValue("stop_token_ids", out var s) && s.Length > 0
                ? [.. s.Split(',').Select(x => int.Parse(x.Trim(), CultureInfo.InvariantCulture))]
                : [];
            Assert.Equal(stops, record.StopTokenIds);

            Assert.Equal(manifest.CompatibilityRows.Count, record.Platforms.Count);
            for (int i = 0; i < record.Platforms.Count; i++)
            {
                var row = manifest.CompatibilityRows[i];
                var platform = record.Platforms[i];
                Assert.Equal(row["os"], platform.Os);
                Assert.Equal(row["arch"], platform.Arch);
                Assert.Equal(row["provider"], platform.Provider);
                Assert.Equal(row["status"], Name(platform.Status));
                Assert.Equal(row.GetValueOrDefault("browser"), platform.Browser);
                Assert.Equal(row.TryGetValue("verified", out var v) ? DateOnly.Parse(v, CultureInfo.InvariantCulture) : null, platform.Verified);
            }

            Assert.Equal(
                manifest.CompatibilityRows.Where(r => r["status"] == "supported").Select(r => r["provider"]).Distinct().Order(StringComparer.Ordinal),
                record.QualifiedProviders);

            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), record.Capabilities.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(record.Capabilities[key]), $"{id} {key}: manifest {status}, record {Name(record.Capabilities[key])}");
            }

            if (record.Maturity == ProfileMaturity.Stable)
            {
                // The validator refuses Stable with an untested capability; the record follows the same rule.
                Assert.DoesNotContain(CapabilityStatus.Untested, record.Capabilities.Values);
            }
        }
    }

    /// <summary>A status as the manifests spell it.</summary>
    public static string Name(CapabilityStatus status) => status.ToString().ToLowerInvariant();

    private static string Name(ProfileMaturity maturity) => maturity.ToString().ToLowerInvariant();

    private static string Truncation(TruncationPolicy policy) => policy switch
    {
        TruncationPolicy.Rejects => "error",
        TruncationPolicy.TruncatesRight => "truncate-right",
        TruncationPolicy.TruncatesLeft => "truncate-left",
        TruncationPolicy.Chunks => "chunk",
        _ => throw new ArgumentOutOfRangeException(nameof(policy), policy, null),
    };
}
