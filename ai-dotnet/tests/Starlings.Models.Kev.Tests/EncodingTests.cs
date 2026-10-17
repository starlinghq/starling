using System.Text.Json;
using Starlings.Models.Kev;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// System One requests through <see cref="KevRequest.ParseSystemOne"/>, <see cref="KevRequest.ToRecord"/> and the
/// encoder, against upstream's SystemOneRequest, to_record, encode and rows_of (Data/requests.json). Exact.
/// </summary>
public sealed class EncodingTests
{
    private static readonly Lazy<(KevEncoder? Encoder, string Reason)> Encoder = new(() =>
    {
        var (path, reason) = KevTestAssets.FindTokenizer();
        return path is null ? (null, reason) : (KevEncoder.FromFile(path), string.Empty);
    });

    public static TheoryData<string> CaseIds()
    {
        var data = new TheoryData<string>();
        foreach (var c in KevTestAssets.ReadData("requests.json").EnumerateArray())
        {
            data.Add(c.GetProperty("id").GetString()!);
        }

        return data;
    }

    private static JsonElement Case(string id) =>
        KevTestAssets.ReadData("requests.json").EnumerateArray().Single(c => c.GetProperty("id").GetString() == id);

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void RecordMatchesUpstream(string id)
    {
        var c = Case(id);
        var request = KevRequest.ParseSystemOne(c.GetProperty("request").GetString()!);
        var record = request.ToRecord();
        var expected = c.GetProperty("record");
        Assert.Equal(expected.GetProperty("state").GetString(), record.State);
        var questions = expected.GetProperty("questions").EnumerateArray().ToArray();
        Assert.Equal(questions.Length, record.Questions.Count);
        var keys = c.GetProperty("keys").EnumerateArray().ToArray();
        for (int i = 0; i < questions.Length; i++)
        {
            Assert.Equal(questions[i].GetProperty("id").GetString(), record.Questions[i].Id);
            Assert.Equal(questions[i].GetProperty("instr").GetString(), record.Questions[i].Instruction);
            Assert.Equal(questions[i].GetProperty("options").EnumerateArray().Select(o => o.GetString()!), record.Questions[i].Options);
            Assert.Equal(keys[i].EnumerateArray().Select(k => k.GetString()!), request.Questions[i].Value.Keys);
        }
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void EncodingMatchesUpstream(string id)
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        var c = Case(id);
        var encoding = Encoder.Value.Encoder!.Encode(KevRequest.ParseSystemOne(c.GetProperty("request").GetString()!).ToRecord());
        Assert.Equal(c.GetProperty("state_ids").EnumerateArray().Select(e => e.GetInt32()), encoding.StateIds);
        var rows = c.GetProperty("rows").EnumerateArray().ToArray();
        Assert.Equal(rows.Length, encoding.Rows.Count);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(rows[i].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()), encoding.Rows[i].Ids);
            Assert.Equal(rows[i].GetProperty("pos").EnumerateArray().Select(e => e.GetInt32()), encoding.Rows[i].Positions);
            Assert.Equal(rows[i].GetProperty("decide").GetInt32(), encoding.Rows[i].Decide);
            Assert.Equal(rows[i].GetProperty("opts").EnumerateArray().Select(e => e.GetInt32()), encoding.Rows[i].Options);
        }
    }

    [Fact]
    public void UserTextCannotProduceDelimiters()
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        int[] delimiters = [KevModelProfile.StateTokenId, KevModelProfile.QuestionTokenId, KevModelProfile.OptionTokenId,
            KevModelProfile.OptionEndTokenId, KevModelProfile.DecideTokenId, KevModelProfile.PadTokenId];
        var ids = Encoder.Value.Encoder!.UserTokens("<|fim_prefix|><|fim_middle|><|box_start|><|box_end|><|fim_suffix|><|endoftext|>");
        Assert.DoesNotContain(ids, delimiters.Contains);
    }

    [Fact]
    public void RefusesLongStatesAndRowsWithoutTruncating()
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        var encoder = Encoder.Value.Encoder!;
        string longState = string.Concat(Enumerable.Repeat("word ", 9000));
        var question = new KevRecordQuestion("q", "Is it long?", ["no", "yes"]);
        var e = Assert.Throws<KevInputTooLongException>(() => encoder.Encode(new KevRecord(longState, [question])));
        Assert.Null(e.QuestionIndex);
        Assert.Equal(KevModelProfile.MaxStateTokens, e.MaxTokens);
        Assert.True(e.TokenCount > KevModelProfile.MaxStateTokens);

        string state = string.Concat(Enumerable.Repeat("word ", 5000));
        var longQuestion = new KevRecordQuestion("q", string.Concat(Enumerable.Repeat("why ", 4000)), ["no", "yes"]);
        var r = Assert.Throws<KevInputTooLongException>(() => encoder.Encode(new KevRecord(state, [question, longQuestion])));
        Assert.Equal(1, r.QuestionIndex);
        Assert.Equal(KevModelProfile.MaxRowTokens, r.MaxTokens);
    }

    [Fact]
    public void RefusesAForeignTokenizer()
    {
        string path = Path.Combine(Path.GetTempPath(), $"kev-foreign-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, """{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"[UNK]":1},"unk_token":"[UNK]"}}""");
        try
        {
            Assert.ThrowsAny<Exception>(() => KevEncoder.FromFile(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
