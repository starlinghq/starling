using System.Runtime.InteropServices;
using System.Text;

namespace Starlings.Interventions.Tests;

/// <summary>Synthetic profiles and directions. No real model or behavioural direction is involved.</summary>
internal static class TestData
{
    public const string BaseSha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    public const string OtherSha = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";
    public const string DirSha = "1111111111111111111111111111111111111111111111111111111111111111";

    /// <summary>The canonical serialization of <see cref="Profile"/>, written out by hand.</summary>
    public const string CanonicalProfile = """
        {
          "schema_version": 1,
          "id": "synthetic-demo",
          "description": "Synthetic random directions for tests",
          "base_model_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
          "operator": "directional_projection",
          "sites": [
            {
              "layer": 1,
              "target": "residual_post",
              "hidden_size": 8,
              "rank": 2,
              "dtype": "float32",
              "directions_file": "directions/layer1.f32",
              "directions_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
              "scales": [
                1,
                0.5
              ]
            },
            {
              "layer": 3,
              "target": "residual_post",
              "hidden_size": 8,
              "rank": 1,
              "dtype": "float32",
              "directions_file": "directions/layer3.f32",
              "directions_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
              "scales": [
                0.1
              ]
            }
          ]
        }

        """;

    public static InterventionSite Site(int layer, int rank = 1, int hidden = 8, float[]? scales = null, string? file = null, string sha = DirSha) =>
        new(layer, InterventionTarget.ResidualPost, hidden, rank, InterventionDType.Fp32,
            file ?? $"directions/layer{layer}.f32", sha, scales ?? [.. Enumerable.Repeat(1f, rank)]);

    public static InterventionProfile Profile() => new(
        "synthetic-demo",
        BaseSha,
        InterventionOperatorKind.DirectionalProjection,
        [Site(3, scales: [0.1f]), Site(1, rank: 2, scales: [1f, 0.5f])],
        "Synthetic random directions for tests");

    public static byte[] Utf8(string text) => Encoding.UTF8.GetBytes(text);

    public static string Text(byte[] bytes) => Encoding.UTF8.GetString(bytes);

    /// <summary>Random orthonormal rows (Gram-Schmidt in double, then rounded to float32).</summary>
    public static float[] Orthonormal(int rank, int hidden, int seed)
    {
        Random random = new(seed);
        double[][] rows = new double[rank][];
        for (int r = 0; r < rank; r++)
        {
            double[] v = new double[hidden];
            for (int k = 0; k < hidden; k++)
            {
                v[k] = (random.NextDouble() * 2) - 1;
            }

            for (int p = 0; p < r; p++)
            {
                double dot = 0;
                for (int k = 0; k < hidden; k++)
                {
                    dot += v[k] * rows[p][k];
                }

                for (int k = 0; k < hidden; k++)
                {
                    v[k] -= dot * rows[p][k];
                }
            }

            double norm = Math.Sqrt(v.Sum(x => x * x));
            for (int k = 0; k < hidden; k++)
            {
                v[k] /= norm;
            }

            rows[r] = v;
        }

        return [.. rows.SelectMany(row => row.Select(x => (float)x))];
    }

    /// <summary>Hard-links <paramref name="target"/> to <paramref name="source"/> (Linux); copies elsewhere.</summary>
    public static void HardLink(string source, string target)
    {
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(target)!);
        if (!OperatingSystem.IsLinux())
        {
            File.Copy(source, target);
            return;
        }

        if (Link(source, target) != 0)
        {
            throw new IOException($"link({source}, {target}) failed with errno {Marshal.GetLastPInvokeError()}.");
        }
    }

    [DllImport("libc", EntryPoint = "link", SetLastError = true)]
#pragma warning disable SYSLIB1054, CA2101 // a test helper; runtime marshalling of two paths is fine here
    private static extern int Link([MarshalAs(UnmanagedType.LPUTF8Str)] string source, [MarshalAs(UnmanagedType.LPUTF8Str)] string target);
#pragma warning restore SYSLIB1054, CA2101

    /// <summary>A temporary directory deleted on dispose.</summary>
    public sealed class TempDir : IDisposable
    {
        public TempDir() => Path = Directory.CreateTempSubdirectory("starlings-interventions-").FullName;

        public string Path { get; }

        public string Write(string relative, byte[] bytes)
        {
            string full = System.IO.Path.Combine(Path, relative.Replace('/', System.IO.Path.DirectorySeparatorChar));
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(full)!);
            File.WriteAllBytes(full, bytes);
            return full;
        }

        public void Dispose()
        {
            try
            {
                Directory.Delete(Path, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }
}
