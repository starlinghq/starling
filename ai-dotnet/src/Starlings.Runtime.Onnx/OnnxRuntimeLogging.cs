using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx;

/// <summary>Severity of an ONNX Runtime log message, in increasing order (<c>OrtLoggingLevel</c>).</summary>
public enum OnnxLogSeverity
{
    /// <summary>Verbose (<c>ORT_LOGGING_LEVEL_VERBOSE</c>). Most verbose messages are compiled only into debug builds of ONNX Runtime.</summary>
    Verbose,

    /// <summary>Informational (<c>ORT_LOGGING_LEVEL_INFO</c>).</summary>
    Info,

    /// <summary>Warning (<c>ORT_LOGGING_LEVEL_WARNING</c>), ONNX Runtime's own default.</summary>
    Warning,

    /// <summary>Error (<c>ORT_LOGGING_LEVEL_ERROR</c>), this package's default for sessions.</summary>
    Error,

    /// <summary>Fatal (<c>ORT_LOGGING_LEVEL_FATAL</c>).</summary>
    Fatal,
}

/// <summary>One message ONNX Runtime logged, as delivered to the sink set with <see cref="OnnxRuntimeInfo.ConfigureLogging"/>.</summary>
/// <param name="Severity">Message severity.</param>
/// <param name="Category">ONNX Runtime's category, for example <c>onnxruntime</c>.</param>
/// <param name="LogId">The logger id: <see cref="OnnxSessionOptions.LogId"/> for a session's messages, the environment's id otherwise.</param>
/// <param name="CodeLocation">Source location in ONNX Runtime that logged the message.</param>
/// <param name="Message">Message text.</param>
public readonly record struct OnnxRuntimeLogMessage(
    OnnxLogSeverity Severity,
    string Category,
    string LogId,
    string CodeLocation,
    string Message);

internal static class OnnxLogging
{
    internal static OrtLoggingLevel ToOrt(OnnxLogSeverity severity) => severity switch
    {
        OnnxLogSeverity.Verbose => OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
        OnnxLogSeverity.Info => OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO,
        OnnxLogSeverity.Warning => OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
        OnnxLogSeverity.Fatal => OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL,
        _ => OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
    };

    internal static OnnxLogSeverity FromOrt(OrtLoggingLevel level) => level switch
    {
        OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE => OnnxLogSeverity.Verbose,
        OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO => OnnxLogSeverity.Info,
        OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING => OnnxLogSeverity.Warning,
        OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL => OnnxLogSeverity.Fatal,
        _ => OnnxLogSeverity.Error,
    };

    /// <summary>
    /// Wraps a sink in the delegate ONNX Runtime calls. The sink runs on ONNX Runtime's threads, possibly concurrently;
    /// an exception it throws is dropped, because it would otherwise unwind through native frames.
    /// </summary>
    internal static DOrtLoggingFunction Adapt(Action<OnnxRuntimeLogMessage> sink) =>
        (_, severity, category, logId, codeLocation, message) =>
        {
            try
            {
                sink(new OnnxRuntimeLogMessage(FromOrt(severity), category ?? string.Empty, logId ?? string.Empty, codeLocation ?? string.Empty, message ?? string.Empty));
            }
#pragma warning disable CA1031 // A sink exception must not cross into native code.
            catch (Exception)
#pragma warning restore CA1031
            {
            }
        };
}
