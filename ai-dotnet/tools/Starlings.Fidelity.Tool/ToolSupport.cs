using System.Diagnostics;
using System.Security.Cryptography;

/// <summary>Helpers shared by the e5 and GLiNER2 stage dumps.</summary>
internal static class ToolSupport
{
    public static string? CpuModel()
    {
        try
        {
            return File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal))?.Split(':', 2)[1].Trim();
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    public static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    public static string? FindRepositoryRoot()
    {
        foreach (var start in new[] { System.Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            {
                if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "protected")) && Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
                {
                    return dir.FullName;
                }
            }
        }

        return null;
    }

    public static string? GitHead(string? repoRoot)
    {
        if (repoRoot is null)
        {
            return null;
        }

        try
        {
            string head = RunGit(repoRoot, "rev-parse HEAD");
            string dirty = RunGit(repoRoot, "status --porcelain --untracked-files=no");
            return head.Length == 0 ? null : head + (dirty.Length > 0 ? "-dirty" : string.Empty);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    private static string RunGit(string dir, string arguments)
    {
        var info = new ProcessStartInfo("git", arguments) { WorkingDirectory = dir, RedirectStandardOutput = true, RedirectStandardError = true };
        using var process = Process.Start(info)!;
        string output = process.StandardOutput.ReadToEnd().Trim();
        process.WaitForExit();
        return process.ExitCode == 0 ? output : string.Empty;
    }
}
