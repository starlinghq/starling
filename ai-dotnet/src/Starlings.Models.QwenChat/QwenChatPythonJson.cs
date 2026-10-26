using System.Text;
using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.QwenChat;

/// <summary>
/// <see cref="PythonJson"/> for the Qwen3 and Qwen2.5 templates (transformers' <c>tojson</c>): the same text, with data
/// the templates cannot render reported as <see cref="NotSupportedException"/> (same message).
/// </summary>
internal static class QwenChatPythonJson
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
            throw new NotSupportedException(ex.Message, ex);
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
            throw new NotSupportedException(ex.Message, ex);
        }
    }

    /// <summary>Returns <paramref name="s"/>, or throws <see cref="NotSupportedException"/> for a lone UTF-16 surrogate.</summary>
    public static string CheckedString(string s)
    {
        try
        {
            return PythonJson.CheckedString(s);
        }
        catch (PythonJsonException ex)
        {
            throw new NotSupportedException(ex.Message, ex);
        }
    }
}
