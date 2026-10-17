namespace Starlings.Benchmarks;

/// <summary>Repository paths for the harnesses.</summary>
internal static class BenchPaths
{
    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string Workload(string fileName) => Path.Combine(RepositoryRoot, "ai-dotnet", "benchmarks", "workloads", fileName);

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "protected")) &&
                Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Repository root not found above " + AppContext.BaseDirectory);
    }
}
