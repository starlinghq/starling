// Scratch experiment (not product code): ONNX Runtime GenAI 0.17.1 static batch (batch_size 2) vs single greedy runs
// on the Qwen3.5-0.8B fp32x bundle, CPU provider. Prompt ids come from the reference fixture (no tokenizer needed).
using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntimeGenAI;

string modelDir = args[0];
string fixture = args[1];
int newTokens = int.Parse(args[2]);
int threads = int.Parse(args[3]);
const int Pad = 248319; // Qwen35ChatModel.PadTokenIdOverlay: an embedding row the tokenizer never produces.
int[] eos = [248046, 248044];

var cases = new Dictionary<string, (int[] Prompt, int[] Reference)>();
foreach (string line in File.ReadLines(fixture))
{
    using var doc = JsonDocument.Parse(line);
    var r = doc.RootElement;
    cases[r.GetProperty("id").GetString()!] = (
        r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
        r.GetProperty("output_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray());
}

using var handle = new OgaHandle();
using var config = new Config(modelDir);
config.ClearProviders();
config.Overlay($"{{\"model\":{{\"decoder\":{{\"session_options\":{{\"intra_op_num_threads\":{threads}}}}}}}}}");
config.Overlay($"{{\"model\":{{\"pad_token_id\":{Pad}}}}}");
var loadClock = Stopwatch.StartNew();
using var model = new Model(config);
Console.WriteLine($"loaded in {loadClock.Elapsed.TotalSeconds:F1}s, threads {threads}, new tokens {newTokens}");

var singles = new Dictionary<string, (int[] Ids, float[] Step0, double Seconds)>();
(int[] Ids, float[] Step0, double Seconds) Single(string id)
{
    if (singles.TryGetValue(id, out var cached))
    {
        return cached;
    }

    int[] prompt = cases[id].Prompt;
    var clock = Stopwatch.StartNew();
    using var p = new GeneratorParams(model);
    p.SetSearchOption("max_length", prompt.Length + newTokens);
    p.SetSearchOption("do_sample", false);
    using var g = new Generator(model, p);
    g.AppendTokens(prompt);
    float[] step0 = LastLogits(g, 1)[0];
    var ids = new List<int>();
    while (ids.Count < newTokens && !g.IsDone())
    {
        g.GenerateNextToken();
        int t = g.GetNextTokens()[0];
        ids.Add(t);
        if (eos.Contains(t))
        {
            break;
        }
    }

    var result = (ids.ToArray(), step0, clock.Elapsed.TotalSeconds);
    singles[id] = result;
    return result;
}

static float[][] LastLogits(Generator g, int batch)
{
    using var tensor = g.GetOutput("logits");
    long[] shape = tensor.Shape();
    var data = tensor.GetData<float>();
    int vocab = (int)shape[^1];
    long perRow = data.Length / batch;
    var rows = new float[batch][];
    for (int b = 0; b < batch; b++)
    {
        int start = checked((int)((b + 1) * perRow - vocab));
        rows[b] = data.Slice(start, vocab).ToArray();
    }

    return rows;
}

void Batch(string label, string a, string b, bool leftPad)
{
    int[][] prompts = [cases[a].Prompt, cases[b].Prompt];
    int width = prompts.Max(p => p.Length);
    int[] flat = new int[2 * width];
    for (int r = 0; r < 2; r++)
    {
        int padCount = width - prompts[r].Length;
        Array.Fill(flat, Pad, r * width, width);
        prompts[r].CopyTo(flat, r * width + (leftPad ? padCount : 0));
    }

    var clock = Stopwatch.StartNew();
    using var p = new GeneratorParams(model);
    p.SetSearchOption("batch_size", 2);
    p.SetSearchOption("max_length", width + newTokens);
    p.SetSearchOption("do_sample", false);
    using var g = new Generator(model, p);
    string error = string.Empty;
    var rows = new List<int>[] { [], [] };
    bool[] finished = new bool[2];
    float[][] step0 = [];
    try
    {
        g.AppendTokens(flat);
        step0 = LastLogits(g, 2);
        int steps = 0;
        while (steps < newTokens && !g.IsDone() && !(finished[0] && finished[1]))
        {
            g.GenerateNextToken();
            var next = g.GetNextTokens();
            steps++;
            for (int r = 0; r < 2; r++)
            {
                if (!finished[r])
                {
                    rows[r].Add(next[r]);
                    finished[r] = eos.Contains(next[r]);
                }
            }
        }
    }
    catch (OnnxRuntimeGenAIException ex)
    {
        error = ex.Message;
    }

    double batchSeconds = clock.Elapsed.TotalSeconds;
    var sa = Single(a);
    var sb = Single(b);
    Console.WriteLine($"== {label}: [{a} {prompts[0].Length}] + [{b} {prompts[1].Length}], width {width}, {(leftPad ? "left" : "right")} padding");
    if (error.Length > 0)
    {
        Console.WriteLine($"   ERROR {error}");
        return;
    }

    for (int r = 0; r < 2; r++)
    {
        string id = r == 0 ? a : b;
        var single = r == 0 ? sa : sb;
        int[] batched = rows[r].ToArray();
        int agree = 0;
        while (agree < Math.Min(batched.Length, single.Ids.Length) && batched[agree] == single.Ids[agree])
        {
            agree++;
        }

        int[] reference = cases[id].Reference;
        int agreeRef = 0;
        while (agreeRef < Math.Min(batched.Length, reference.Length) && batched[agreeRef] == reference[agreeRef])
        {
            agreeRef++;
        }

        int singleRef = 0;
        while (singleRef < Math.Min(single.Ids.Length, reference.Length) && single.Ids[singleRef] == reference[singleRef])
        {
            singleRef++;
        }

        float maxAbs = 0;
        for (int i = 0; i < single.Step0.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(single.Step0[i] - step0[r][i]));
        }

        bool exact = batched.SequenceEqual(single.Ids);
        Console.WriteLine($"   row {r} {id}: batched {batched.Length} ids, single {single.Ids.Length} ids, identical {exact}, " +
                          $"agreeing prefix {agree}; vs reference: batched {agreeRef}, single {singleRef}; step-0 logits max abs {maxAbs:E2}");
        if (!exact)
        {
            Console.WriteLine($"      single : {string.Join(' ', single.Ids.Take(16))}");
            Console.WriteLine($"      batched: {string.Join(' ', batched.Take(16))}");
        }
    }

    Console.WriteLine($"   time: batch {batchSeconds:F2}s, singles {sa.Seconds:F2}s + {sb.Seconds:F2}s = {sa.Seconds + sb.Seconds:F2}s (sequential)");
}

// Warm-up single so the first timed run does not pay session initialization.
Single("hi");
singles.Clear();

if (args.Length > 4 && args[4] == "timing")
{
    // Wall time for one pair: two singles in sequence, two singles on two threads (today's CPU path), one batch of 2.
    foreach (var (a, b) in new[] { ("en-list", "en-list"), ("en-list", "en-math") })
    {
        var seq = new List<double>();
        var conc = new List<double>();
        var bat = new List<double>();
        for (int rep = 0; rep < 2; rep++)
        {
            singles.Clear();
            var c = Stopwatch.StartNew();
            Single(a);
            singles.Clear();
            Single(b);
            seq.Add(c.Elapsed.TotalSeconds);

            singles.Clear();
            c.Restart();
            var t1 = Task.Run(() => SingleUncached(a));
            var t2 = Task.Run(() => SingleUncached(b));
            Task.WaitAll(t1, t2);
            conc.Add(c.Elapsed.TotalSeconds);

            c.Restart();
            BatchQuiet(a, b);
            bat.Add(c.Elapsed.TotalSeconds);
        }

        Console.WriteLine($"timing [{a} {cases[a].Prompt.Length}] + [{b} {cases[b].Prompt.Length}], {newTokens} new tokens, min of 2: " +
                          $"sequential {seq.Min():F2}s, concurrent x2 {conc.Min():F2}s, batch 2 (left pad) {bat.Min():F2}s");
    }

    return;
}

int[] SingleUncached(string id)
{
    int[] prompt = cases[id].Prompt;
    using var p = new GeneratorParams(model);
    p.SetSearchOption("max_length", prompt.Length + newTokens);
    p.SetSearchOption("do_sample", false);
    using var g = new Generator(model, p);
    g.AppendTokens(prompt);
    var ids = new List<int>();
    while (ids.Count < newTokens && !g.IsDone())
    {
        g.GenerateNextToken();
        ids.Add(g.GetNextTokens()[0]);
    }

    return ids.ToArray();
}

void BatchQuiet(string a, string b)
{
    int[][] prompts = [cases[a].Prompt, cases[b].Prompt];
    int width = prompts.Max(p => p.Length);
    int[] flat = new int[2 * width];
    Array.Fill(flat, Pad);
    for (int r = 0; r < 2; r++)
    {
        prompts[r].CopyTo(flat, r * width + width - prompts[r].Length);
    }

    using var p = new GeneratorParams(model);
    p.SetSearchOption("batch_size", 2);
    p.SetSearchOption("max_length", width + newTokens);
    p.SetSearchOption("do_sample", false);
    using var g = new Generator(model, p);
    g.AppendTokens(flat);
    for (int s = 0; s < newTokens && !g.IsDone(); s++)
    {
        g.GenerateNextToken();
    }
}

Batch("same prompt twice", "en-list", "en-list", leftPad: true);
Batch("equal length, different prompts", "en-short", "special-endoftext", leftPad: true);
Batch("unequal (22 vs 40), left padding", "en-list", "en-math", leftPad: true);
Batch("unequal (22 vs 40), right padding", "en-list", "en-math", leftPad: false);
Batch("unequal (24 vs 80), left padding", "en-short", "code-cs", leftPad: true);
Batch("unequal (24 vs 80), right padding", "en-short", "code-cs", leftPad: false);
