using System.Text;
using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>
/// <see cref="PythonJson"/> for the chat template: the same text, with data the template cannot render reported as
/// <see cref="Qwen35TemplateException"/> (same message).
/// </summary>
internal static class Qwen35PythonJson
{
    /// <summary><c>json.dumps(value, ensure_ascii=False)</c>.</summary>
    public static string Dumps(JsonElement value)
    {
        try
        {
            return PythonJson.Dumps(value);
        }
        catch (PythonJsonException ex)
        {
            throw new Qwen35TemplateException(ex.Message, ex);
        }
    }

    public static void Dumps(StringBuilder sb, JsonElement value)
    {
        try
        {
            PythonJson.Dumps(sb, value);
        }
        catch (PythonJsonException ex)
        {
            throw new Qwen35TemplateException(ex.Message, ex);
        }
    }

    /// <summary>What the template writes for one tool-call argument (<see cref="PythonJson.Str"/>).</summary>
    public static string ArgumentText(JsonElement value)
    {
        try
        {
            return PythonJson.Str(value);
        }
        catch (PythonJsonException ex)
        {
            throw new Qwen35TemplateException(ex.Message, ex);
        }
    }

    /// <summary>Python <c>repr(float)</c> (<see cref="PythonJson.FloatRepr"/>).</summary>
    public static string FloatRepr(double d) => PythonJson.FloatRepr(d);
}
