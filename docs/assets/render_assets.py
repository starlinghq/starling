"""Render docs/assets/demo.gif and social-preview.png from the README quickstart outputs (verified against 0.1.0-preview.6).

Usage: python3 docs/assets/render_assets.py docs/assets   (needs Pillow, JetBrains Mono and Noto Sans CJK fonts).
The terminal text is hard-coded from real runs; rerun the quickstarts and update it when the API or output changes.
"""

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

MONO = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 17)
MONO_B = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf", 17)
CJK = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 17)

BG = (22, 27, 34)
BAR = (33, 38, 45)
FG = (230, 237, 243)
DIM = (125, 133, 144)
GREEN = (126, 231, 135)
BLUE = (121, 192, 255)
PURPLE = (210, 168, 255)
ORANGE = (255, 166, 87)
YELLOW = (242, 204, 96)

W, H = 960, 540
PAD_X, TOP, LINE = 24, 56, 25
CHAR_W = MONO.getlength("M")

KEYWORDS = {"using", "await", "foreach", "var", "in", "string"}


def is_wide(ch):
    return ord(ch) >= 0x2E80


def draw_text(d, x, y, text, color, font=MONO):
    for ch in text:
        f = CJK if is_wide(ch) else font
        d.text((x, y + (-3 if f is CJK else 0)), ch, font=f, fill=color)
        x += CHAR_W * (2 if is_wide(ch) else 1)
    return x


def code_spans(line):
    """Tiny C# colorizer: comments, strings, keywords, type-like names."""
    if line.lstrip().startswith("//"):
        return [(line, DIM)]
    spans = []
    for tok in re.findall(r'"[^"]*"|\w+|\s+|.', line):
        if tok.startswith('"'):
            spans.append((tok, GREEN))
        elif tok in KEYWORDS:
            spans.append((tok, PURPLE))
        elif tok[:1].isupper() and tok.isidentifier():
            spans.append((tok, BLUE))
        else:
            spans.append((tok, FG))
    return spans


def frame(lines, title):
    """lines: list of span lists [(text, color, bold?)]."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 36], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + i * 22, 12, 28 + i * 22, 24], fill=c)
    tw = MONO.getlength(title)
    d.text(((W - tw) / 2, 9), title, font=MONO, fill=DIM)
    y = TOP
    for spans in lines[-19:]:
        x = PAD_X
        for span in spans:
            text, color = span[0], span[1]
            bold = len(span) > 2 and span[2]
            x = draw_text(d, x, y, text, color, MONO_B if bold else MONO)
        y += LINE
    return img


PROMPT = ("$ ", GREEN, True)

SEARCH_CODE = """\
// Program.cs: multilingual semantic search
E5ModelFiles files = await E5ModelAssets.DownloadAsync();
using var embedder = E5EmbeddingGenerator.Create(files);

string[] docs = ["passage: The Eiffel Tower is in Paris.",
                 "passage: Der Rhein fließt durch Köln.",
                 "passage: 富士山は日本一高い山です。"];
var docVectors = await embedder.GenerateAsync(docs);
var query = await embedder.GenerateVectorAsync(
    "query: Which river runs through Cologne?");""".splitlines()

SEARCH_OUT = [
    [("0.680", DIM), ("  passage: The Eiffel Tower is in Paris.", FG)],
    [("0.807", GREEN, True), ("  passage: Der Rhein fließt durch Köln.", FG, True)],
    [("0.688", DIM), ("  passage: 富士山は日本一高い山です。", FG)],
]

CHAT_CODE = """\
// Program.cs: local chat through Microsoft.Extensions.AI
var files = await Qwen35ModelAssets.DownloadAsync(Qwen35Variant.Fp32x);
using IChatClient chat = Qwen35ChatClient.Create(files);

await foreach (var update in chat.GetStreamingResponseAsync(
    "In one sentence, why is the sky blue?"))
    Console.Write(update.Text);""".splitlines()

CHAT_ANSWER = (
    "The sky is blue because sunlight hits the Earth's atmosphere, where the shorter blue wavelengths "
    "of light are scattered more than the longer red wavelengths, making the sky appear blue to our eyes."
)


def wrap(text, width):
    words, rows, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            rows.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    rows.append(cur)
    return rows


frames, durations = [], []


def add(lines, title, ms):
    frames.append(frame(lines, title))
    durations.append(ms)


def typed(base, cmd, title):
    for i in range(0, len(cmd) + 1, 3):
        add(base + [[PROMPT, (cmd[:i], FG)]], title, 40)
    add(base + [[PROMPT, (cmd, FG)]], title, 350)


def scene(title, code, run_cmd, out_fn):
    typed([], "cat Program.cs", title)
    base = [[PROMPT, ("cat Program.cs", FG)]] + [code_spans(l) for l in code] + [[]]
    add(base, title, 1800)
    typed(base, run_cmd, title)
    base = base + [[PROMPT, (run_cmd, FG)]]
    out_fn(base, title)


def search_out(base, title):
    add(base, title, 700)
    for i in range(1, len(SEARCH_OUT) + 1):
        add(base + SEARCH_OUT[:i], title, 250)
    add(base + SEARCH_OUT, title, 3500)


def chat_out(base, title):
    add(base, title, 900)
    words = CHAT_ANSWER.split(" ")
    for i in range(1, len(words) + 1, 2):
        rows = wrap(" ".join(words[:i]), 74)
        add(base + [[(r, YELLOW)] for r in rows], title, 70)
    rows = wrap(CHAT_ANSWER, 74)
    add(base + [[(r, YELLOW)] for r in rows], title, 4000)


scene("e5 search  ·  Starlings.Models.E5.Extensions.AI", SEARCH_CODE, "dotnet run", search_out)
scene("Qwen3.5 chat  ·  Starlings.Models.Qwen35.Extensions.AI", CHAT_CODE, "dotnet run", chat_out)

# Fixed palette: every theme color blended with the background at 16 levels (antialiasing), so rare colors survive.
theme = [FG, DIM, GREEN, BLUE, PURPLE, ORANGE, YELLOW, (255, 95, 86), (255, 189, 46), (39, 201, 63)]
colors = [BG, BAR]
for c in theme:
    for k in range(1, 17):
        t = k / 16
        colors.append(tuple(round(BG[j] + (c[j] - BG[j]) * t) for j in range(3)))
colors = list(dict.fromkeys(colors))[:256]
palette = Image.new("P", (1, 1))
palette.putpalette([v for c in colors for v in c] + [0] * (768 - 3 * len(colors)))
pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
pal[0].save(OUT / "demo.gif", save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
print("frames", len(frames), "bytes", (OUT / "demo.gif").stat().st_size)

# Social preview, 1280x640 (GitHub's recommended size).
SW, SH = 1280, 640
img = Image.new("RGB", (SW, SH), BG)
d = ImageDraw.Draw(img)
big = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf", 84)
mid = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 34)
small = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 26)
d.text((90, 120), "Starling", font=big, fill=FG)
d.text((90, 235), "Local AI models for .NET,", font=mid, fill=FG)
d.text((90, 285), "with the evidence that they match upstream.", font=mid, fill=FG)
y = 380
for label, color in [
    ("embeddings  ·  entity extraction  ·  image & video search  ·  chat", BLUE),
    ("Microsoft.Extensions.AI  ·  ONNX Runtime  ·  CPU & CUDA  ·  no cloud", DIM),
]:
    d.text((90, y), label, font=small, fill=color)
    y += 46
d.text((90, 520), "dotnet add package Starlings.Models.E5.Extensions.AI --prerelease", font=small, fill=GREEN)
d.rectangle([0, SH - 10, SW, SH], fill=GREEN)
img.save(OUT / "social-preview.png", optimize=True)
print("social", (OUT / "social-preview.png").stat().st_size)
