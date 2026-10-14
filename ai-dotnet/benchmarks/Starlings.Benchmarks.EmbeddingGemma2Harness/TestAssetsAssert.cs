namespace Starlings.TestSupport;

/// <summary>
/// The two xunit calls the shared EmbeddingGemma 2 test-asset files make (<c>tests/Shared/EmbeddingGemma2*/</c>), for
/// this harness, which links those files but is not a test project: a skip becomes a missing-asset exit, an equality
/// failure an exception.
/// </summary>
internal static class Assert
{
    public static void SkipWhen(bool condition, string reason)
    {
        if (condition)
        {
            throw new Starlings.Benchmarks.BenchAssetsMissingException(reason);
        }
    }

    public static void Equal<T>(T expected, T actual)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new InvalidOperationException($"Expected {expected}, got {actual}.");
        }
    }
}
