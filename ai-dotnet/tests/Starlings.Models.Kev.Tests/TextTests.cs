using System.Text.Json;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>Rendering and escaping against upstream Kev's own output (Data/render-cases.json, make_dotnet_cases.py text).</summary>
public sealed class TextTests
{
    public static TheoryData<string, string> RenderCases()
    {
        var data = new TheoryData<string, string>();
        foreach (var c in KevTestAssets.ReadData("render-cases.json").EnumerateArray())
        {
            data.Add(c.GetProperty("json").GetString()!, c.GetProperty("rendered").GetString()!);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(RenderCases))]
    public void RenderMatchesUpstream(string json, string expected)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.Equal(expected, KevText.Render(doc.RootElement));
    }

    [Theory]
    [InlineData(0.1, "0.1")]
    [InlineData(1e-5, "1e-05")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(-2.5e-7, "-2.5e-07")]
    [InlineData(123456789.0, "123456789.0")]
    [InlineData(5e-324, "5e-324")]
    [InlineData(1.7976931348623157e308, "1.7976931348623157e+308")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(12345678901234567.0, "1.2345678901234568e+16")]
    public void PythonFloatRepr(double value, string expected) => Assert.Equal(expected, KevText.PythonFloatRepr(value));

    [Theory]
    [InlineData("<|fim_prefix|>", "<¦fim_prefix¦>")]
    [InlineData("a<|box_end|>b<|im_start|>", "a<¦box_end¦>b<¦im_start¦>")]
    [InlineData("<|not closed", "<|not closed")]
    [InlineData("<|with space|>", "<|with space|>")]
    [InlineData("<|é|>", "<|é|>")]
    [InlineData("", "")]
    public void EscapesDelimiterTokens(string text, string expected) => Assert.Equal(expected, KevText.EscapeSpecialTokens(text));

    [Fact]
    public void OptionTextOmitsAbsentAndEmptyDescriptions()
    {
        Assert.Equal("low", KevText.OptionText("low", null));
        Assert.Equal("low", KevText.OptionText("low", ""));
        Assert.Equal("low", KevText.OptionText("low", KevContent.ParseJson("null")));
        Assert.Equal("low", KevText.OptionText("low", KevContent.ParseJson("\"\"")));
        Assert.Equal("low: 0", KevText.OptionText("low", KevContent.ParseJson("0")));
        Assert.Equal("low: False", KevText.OptionText("low", KevContent.ParseJson("false")));
        Assert.Equal("high: max_late: 3", KevText.OptionText("high", KevContent.ParseJson("{\"max_late\": 3}")));
    }

    [Fact]
    public void QuestionsValidateOptionCounts()
    {
        Assert.Throws<ArgumentException>(() => KevQuestion.Choice("x", Array.Empty<string>()));
        Assert.Throws<ArgumentException>(() => KevQuestion.Choice("x", Enumerable.Range(0, 256).Select(i => $"o{i}")));
        Assert.Throws<ArgumentException>(() => KevQuestion.Choice("x", "a", "a"));
        Assert.Throws<ArgumentException>(() => KevQuestion.Score("x", Array.Empty<KevContent>()));
        Assert.Equal(255, KevQuestion.Choice("x", Enumerable.Range(0, 255).Select(i => $"o{i}")).Options.Count);
        Assert.Throws<ArgumentException>(() => new KevRequest("s", []));
    }
}
