// Shared entry-point wrapper for the ai-dotnet tools and benchmark harnesses (linked as source, the way
// ai-dotnet/tests/Shared is). An exception that escapes Main makes .NET abort the process (SIGABRT), and
// systemd-coredump then writes a core dump of up to the process size; an expected, typed error must never do that.
//
//   return ToolMain.Run(ToolBody);          // or: return await ToolMain.RunAsync(ToolBody);
//
// Exit codes: the body's own return value passes through unchanged (each tool keeps its documented codes: 0 success,
// 1 a failed check, 2 usage or input error, 3 or 4 tool-specific outcomes). An exception that escapes the body is
// written to stderr in full (type, message, stack, inner exceptions), prefixed "error: ", and the process exits with
// UnhandledErrorExitCode (70, EX_SOFTWARE from sysexits.h), a value no tool uses for anything else. A genuine
// OutOfMemoryException or StackOverflowException is not caught: the process state is not trustworthy then, so the
// runtime's own failure (and its dump, if enabled) is the right outcome.
using System.Diagnostics.CodeAnalysis;

/// <summary>Runs a tool's body and turns an escaping exception into <c>error:</c> on stderr and exit code 70.</summary>
internal static class ToolMain
{
    /// <summary>Exit code of a run whose body threw: 70 (EX_SOFTWARE), not used by any tool for another outcome.</summary>
    public const int UnhandledErrorExitCode = 70;

    /// <summary>Runs <paramref name="body"/> and returns its exit code, or <see cref="UnhandledErrorExitCode"/> if it threw.</summary>
    public static int Run(Func<int> body) => Run(body, Console.Error);

    /// <summary>As <see cref="Run(Func{int})"/>, for an asynchronous body.</summary>
    public static Task<int> RunAsync(Func<Task<int>> body) => RunAsync(body, Console.Error);

    /// <summary>As <see cref="Run(Func{int})"/>, writing the error to <paramref name="stderr"/> (tests).</summary>
    [SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "The entry point reports every exception except the fatal ones.")]
    public static int Run(Func<int> body, TextWriter stderr)
    {
        ArgumentNullException.ThrowIfNull(body);
        try
        {
            return body();
        }
        catch (Exception ex) when (!IsFatal(ex))
        {
            return Report(ex, stderr);
        }
    }

    /// <summary>As <see cref="RunAsync(Func{Task{int}})"/>, writing the error to <paramref name="stderr"/> (tests).</summary>
    [SuppressMessage("Design", "CA1031:Do not catch general exception types", Justification = "The entry point reports every exception except the fatal ones.")]
    public static async Task<int> RunAsync(Func<Task<int>> body, TextWriter stderr)
    {
        ArgumentNullException.ThrowIfNull(body);
        try
        {
            return await body().ConfigureAwait(false);
        }
        catch (Exception ex) when (!IsFatal(ex))
        {
            return Report(ex, stderr);
        }
    }

    /// <summary>Exceptions left to the runtime: the process cannot be trusted to report them.</summary>
    public static bool IsFatal(Exception ex) => ex is OutOfMemoryException or StackOverflowException;

    private static int Report(Exception ex, TextWriter stderr)
    {
        stderr.WriteLine($"error: {ex}");
        stderr.Flush();
        return UnhandledErrorExitCode;
    }
}
