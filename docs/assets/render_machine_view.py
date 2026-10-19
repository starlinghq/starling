#!/usr/bin/env python3
"""Render docs/assets/machine-view.svg: an animated "machine view" of the model delivery pipeline.

Usage: python3 docs/assets/render_machine_view.py docs/assets/machine-view.svg --font <JetBrains Mono .woff2>
--font embeds the font (SIL OFL 1.1) so the SVG renders the same inside a GitHub <img>, where
external fonts cannot load. Without it, text falls back to the viewer's monospace font.

Analogy: a precision foundry with a golden-master gauge. The twelve stations are the
twelve stages of the model delivery pipeline (docs/plan/fast-faithful-ai-dotnet.md,
section 8), with gates G0-G7 (section 10) and the four artifact identities (section 9).

Animation is SMIL only (no script). The model part's look never toggles on a timer:
each look is drawn only inside a fixed zone of the belt, so it changes as the part
passes under a machine.
"""
import argparse
import base64
import math
from pathlib import Path

ARGS = argparse.ArgumentParser()
ARGS.add_argument("out", type=Path)
ARGS.add_argument("--font", type=Path, help="JetBrains Mono woff2 to embed")
ARGS = ARGS.parse_args()
FONT_FACE = ""
if ARGS.font:
    b64 = base64.b64encode(ARGS.font.read_bytes()).decode()
    FONT_FACE = ("/* JetBrains Mono, SIL Open Font License 1.1 */ @font-face { font-family: 'JetBrains Mono'; "
                 f"font-weight: 100 800; src: url(data:font/woff2;base64,{b64}) format('woff2'); }}")

W, H = 2400, 1350
BG, GRID, PANEL = "#0a1628", "#112240", "#0d1f3c"
CYAN, CYAN2, TEXT, TEXT2 = "#00b4d8", "#48cae4", "#caf0f8", "#90e0ef"
AMBER, ORE, GREEN, DIM, WHITE, RED = "#f2a93b", "#b8692a", "#06d6a0", "#1b3350", "#ffffff", "#ef476f"
FONT = "'JetBrains Mono','JetBrainsMono Nerd Font','Courier New',monospace"

out = []


def a(s):
    out.append(s)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls, anchor="middle"):
    a(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{esc(s)}</text>')


def box(x, y, w, h, stroke=CYAN, fill=PANEL, sw=2, rx=0):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def gear(cx, cy, r, teeth, dur, reverse=False, color=CYAN):
    pts = []
    ri = r * 0.78
    for i in range(teeth * 2):
        ang0, ang1 = math.pi * i / teeth, math.pi * (i + 1) / teeth
        rr = r if i % 2 == 0 else ri
        for ang in (ang0 + 0.08, ang1 - 0.08):
            pts.append(f"{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}")
    frm, to = ("360", "0") if reverse else ("0", "360")
    a(f'<g><polygon points="{" ".join(pts)}" fill="{PANEL}" stroke="{color}" stroke-width="2"/>'
      f'<circle cx="{cx}" cy="{cy}" r="{r * 0.32:.1f}" fill="{BG}" stroke="{color}" stroke-width="2"/>'
      f'<line x1="{cx - r * 0.5:.1f}" y1="{cy}" x2="{cx + r * 0.5:.1f}" y2="{cy}" stroke="{color}" stroke-width="2"/>'
      f'<animateTransform attributeName="transform" type="rotate" from="{frm} {cx} {cy}" to="{to} {cx} {cy}" '
      f'dur="{dur}s" repeatCount="indefinite"/></g>')


def lamp_cycle(cx, cy, r, on, dur, off=0.92, attr="fill"):
    """A lamp that lights at fraction `on` of a shared cycle and resets at `off`."""
    a(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{DIM}">'
      f'<animate attributeName="{attr}" values="{DIM};{DIM};{GREEN};{GREEN};{DIM}" '
      f'keyTimes="0;{on:.3f};{on + 0.005:.3f};{off};1" dur="{dur}s" repeatCount="indefinite"/></circle>')


def station(x, y, num, name, analogy, repo_y, lines):
    text(x, y, f"{num}  {name}", "ttl")
    text(x, y + 24, analogy, "ana")
    for i, ln in enumerate(lines):
        text(x, repo_y + i * 20, ln, "repo")


# ================================================================ canvas
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
  f'role="img" aria-labelledby="t d">')
a('<title id="t">Starling Foundry: machine view of the model delivery pipeline</title>')
a('<desc id="d">An animated factory analogy for the twelve-stage model delivery pipeline in '
  'github.com/starlinghq/starling: discover, admit, pin, establish reference, execution route, .NET '
  'behavior, verify fidelity, optimize, integrate, qualify (gates G0 to G7), publish, maintain; then the '
  'published package runs offline in your app.</desc>')
a(f"""<defs>
<style>
  {FONT_FACE}
  text {{ font-family: {FONT}; }}
  .h0 {{ font-size: 40px; font-weight: 700; fill: {WHITE}; letter-spacing: .08em; }}
  .h1 {{ font-size: 18px; fill: {TEXT2}; }}
  .ttl {{ font-size: 19px; font-weight: 700; fill: {WHITE}; letter-spacing: .05em; }}
  .ana {{ font-size: 15px; font-style: italic; fill: {TEXT2}; }}
  .repo {{ font-size: 14px; fill: {CYAN2}; }}
  .lbl {{ font-size: 13px; font-weight: 700; fill: {TEXT}; }}
  .sm {{ font-size: 12px; fill: {TEXT2}; }}
  .amb {{ font-size: 13px; fill: {AMBER}; }}
  .ambb {{ font-size: 13px; font-weight: 700; fill: {AMBER}; }}
  .red {{ font-size: 12px; fill: {RED}; }}
  .lg {{ font-size: 15px; fill: {TEXT}; }}
</style>
<pattern id="g1" width="30" height="30" patternUnits="userSpaceOnUse">
  <path d="M 30 0 L 0 0 0 30" fill="none" stroke="{GRID}" stroke-width="1"/></pattern>
<pattern id="g2" width="150" height="150" patternUnits="userSpaceOnUse">
  <path d="M 150 0 L 0 0 0 150" fill="none" stroke="#17305a" stroke-width="1.2"/></pattern>
<marker id="ac" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
  <polygon points="0 0, 10 4, 0 8" fill="{CYAN}"/></marker>
<marker id="aa" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
  <polygon points="0 0, 10 4, 0 8" fill="{AMBER}"/></marker>
<marker id="ag" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
  <polygon points="0 0, 10 4, 0 8" fill="{GREEN}"/></marker>
</defs>""")
a(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
a(f'<rect width="{W}" height="{H}" fill="url(#g1)"/>')
a(f'<rect width="{W}" height="{H}" fill="url(#g2)"/>')
a(f'<rect x="14" y="14" width="{W - 28}" height="{H - 28}" fill="none" stroke="{CYAN}" stroke-width="1.5" opacity=".55"/>')

# ================================================================ header + legend
text(56, 70, "STARLING FOUNDRY", "h0", "start")
text(56, 102, "machine view of the 12-stage model delivery pipeline (docs/plan §8) in github.com/starlinghq/starling",
     "h1", "start")
lx = 1480
a(f'<rect x="{lx}" y="48" width="18" height="18" rx="3" fill="{AMBER}"/>')
text(lx + 28, 62, "the model, in transit", "lg", "start")
a(f'<rect x="{lx + 250}" y="48" width="18" height="18" fill="{PANEL}" stroke="{CYAN}" stroke-width="2"/>')
text(lx + 278, 62, "tooling (repo code)", "lg", "start")
a(f'<circle cx="{lx + 500}" cy="57" r="9" fill="{GREEN}"/>')
text(lx + 518, 62, "gate passed", "lg", "start")
a(f'<line x1="{lx + 650}" y1="57" x2="{lx + 694}" y2="57" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6"/>')
text(lx + 704, 62, "golden master", "lg", "start")
text(lx, 94, "analogy in italics · real repo paths and commands in blue", "sm", "start")

# ================================================================ geometry
TOP, BOT, TURN = 540, 1020, 2300
BELT = f"M 60 {TOP} H {TURN - 60} A 60 60 0 0 1 {TURN} {TOP + 60} V {BOT - 60} A 60 60 0 0 1 {TURN - 60} {BOT} H 640"
TX = [200, 560, 920, 1280, 1640, 2000]          # stations 0-5, left to right
BX = {6: 2080, 7: 1740, 8: 1410, 9: 1090, 10: 770}  # stations 6-10, right to left
TT, BT = 168, 716        # title baselines
TR, BR = 592, 1072       # repo-line baselines

# ================================================================ 11 maintain loop (behind everything)
a(f'<path d="M 1860 1190 H 2385 V 128 H 40 V 330 H 136" fill="none" stroke="{GREEN}" stroke-width="2.5" '
  f'stroke-dasharray="12 8" marker-end="url(#ag)">'
  f'<animate attributeName="stroke-dashoffset" values="0;-40" dur="1.6s" repeatCount="indefinite"/></path>')
text(1200, 122, "11 MAINTAIN feeds back: next candidate, or an explicit support-status change", "sm")

# ================================================================ golden master feed: 03 vault -> 06 verify
for d, mk in (("M 1400 252 H 2215 V 508", ""), ("M 2215 572 V 786", ' marker-end="url(#aa)"')):
    a(f'<path d="{d}" fill="none" stroke="{AMBER}" stroke-width="3" stroke-dasharray="10 8"{mk}>'
      f'<animate attributeName="stroke-dashoffset" values="0;-36" dur="1.2s" repeatCount="indefinite"/></path>')
for yy in (504, 576):
    a(f'<line x1="2203" y1="{yy}" x2="2227" y2="{yy}" stroke="{AMBER}" stroke-width="3"/>')
text(1800, 243, "golden master: protected fixtures + tolerances", "amb")

# ================================================================ 00 discover
x = TX[0]
station(x, TT, "00", "DISCOVER", "radar on upstream", TR,
        ["model-pipeline upstream", "releases · requests · deps", "→ candidate or update", "  (impact + effort)"])
a(f'<circle cx="{x}" cy="340" r="62" fill="{BG}" stroke="{CYAN}" stroke-width="2"/>')
for rr in (40, 20):
    a(f'<circle cx="{x}" cy="340" r="{rr}" fill="none" stroke="{DIM}" stroke-width="2"/>')
a(f'<line x1="{x - 62}" y1="340" x2="{x + 62}" y2="340" stroke="{DIM}" stroke-width="1.5"/>')
a(f'<line x1="{x}" y1="278" x2="{x}" y2="402" stroke="{DIM}" stroke-width="1.5"/>')
a(f'<g><path d="M {x} 340 L {x} 280 A 60 60 0 0 1 {x + 42} 297 Z" fill="{GREEN}" opacity=".35"/>'
  f'<line x1="{x}" y1="340" x2="{x}" y2="280" stroke="{GREEN}" stroke-width="2"/>'
  f'<animateTransform attributeName="transform" type="rotate" from="0 {x} 340" to="360 {x} 340" dur="4s" repeatCount="indefinite"/></g>')
for (bx, by, ph) in ((x + 30, 318, 0.0), (x - 26, 362, 0.45), (x + 14, 378, 0.7)):
    a(f'<circle cx="{bx}" cy="{by}" r="5" fill="{AMBER}" opacity="0">'
      f'<animate attributeName="opacity" values="0;0;1;0" keyTimes="0;{ph:.2f};{ph + .05:.2f};{min(ph + .35, 1):.2f}" dur="4s" repeatCount="indefinite"/></circle>')

# ================================================================ 01 admit
x = TX[1]
station(x, TT, "01", "ADMIT", "no work order, no entry", TR,
        ["admissions/<model>.md", "value · access · licences", "hardware · route · owner", "→ model contract (or reason)"])
box(x - 66, 268, 132, 16)
box(x - 64, 284, 16, 242)
box(x + 48, 284, 16, 242)
box(x - 42, 296, 84, 150)
text(x, 314, "WORK ORDER", "sm")
for i, item in enumerate(["value", "access", "licence", "hardware", "route", "owner"]):
    yy = 334 + i * 18
    a(f'<polyline points="{x - 34},{yy - 5} {x - 30},{yy - 1} {x - 24},{yy - 9}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
    text(x - 16, yy, item, "sm", "start")
box(x + 72, 300, 50, 62)
a(f'<circle cx="{x + 97}" cy="318" r="9" fill="{GREEN}"/>')
a(f'<rect x="{x + 84}" y="338" width="26" height="5" fill="{TEXT2}"/>')
box(x + 72, 420, 64, 70, stroke=RED)
text(x + 104, 446, "BLOCKED", "red")
text(x + 104, 464, "+ reason", "red")

# ================================================================ 02 pin
x = TX[2]
station(x, TT, "02", "PIN", "every part numbered + sealed", TR,
        ["code + weights: full commit SHA", "tokenizer · configs · templates", "deps · evaluation data", "→ source lock + digests"])
box(x + 40, 260, 24, 266)
box(x - 70, 260, 134, 28)
a('<g>')
box(x - 22, 288, 28, 64)
box(x - 40, 352, 64, 34, stroke=AMBER)
text(x - 8, 375, "SHA", "ambb")
a('<animateTransform attributeName="transform" type="translate" values="0 0;0 0;0 104;0 0" keyTimes="0;.55;.7;1" dur="2.4s" repeatCount="indefinite"/>')
a('</g>')
box(x + 74, 300, 112, 150)
text(x + 130, 318, "LOCK", "lbl")
for i, (k, v) in enumerate([("code", "@sha"), ("weights", "@sha"), ("tok", "#256"), ("eval", "#256")]):
    yy = 340 + i * 28
    a(f'<rect x="{x + 82}" y="{yy - 13}" width="96" height="20" fill="{BG}" stroke="{AMBER}" stroke-width="1.5"/>')
    text(x + 130, yy + 2, f"{k} {v}", "sm")

# ================================================================ 03 establish reference
x = TX[3]
station(x, TT, "03", "REFERENCE", "the original cuts the master", TR,
        ["reference/<model>/ runners", "original code, pinned env", "→ protected/ fixtures +", "  tolerance policy"])
box(x - 90, 214, 210, 92, stroke=AMBER, sw=3)
a(f'<circle cx="{x - 50}" cy="260" r="26" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>')
a(f'<g><line x1="{x - 50}" y1="238" x2="{x - 50}" y2="282" stroke="{AMBER}" stroke-width="3"/>'
  f'<line x1="{x - 72}" y1="260" x2="{x - 28}" y2="260" stroke="{AMBER}" stroke-width="3"/>'
  f'<animateTransform attributeName="transform" type="rotate" values="0 {x - 50} 260;90 {x - 50} 260;90 {x - 50} 260;0 {x - 50} 260" '
  f'keyTimes="0;.3;.7;1" dur="6s" repeatCount="indefinite"/></g>')
text(x - 14, 244, "VAULT", "ambb", "start")
text(x - 14, 264, "fixtures", "sm", "start")
text(x - 14, 282, "tolerances", "sm", "start")
text(x - 14, 300, "key: maintainer", "sm", "start")
a(f'<line x1="{x}" y1="378" x2="{x}" y2="312" stroke="{AMBER}" stroke-width="3" marker-end="url(#aa)"/>')
text(x + 8, 350, "outputs", "amb", "start")
box(x - 76, 380, 152, 146)
gear(x - 38, 432, 30, 10, 5)
text(x + 30, 426, "PyTorch", "lbl")
text(x + 30, 444, "original", "sm")
text(x, 492, "reproducible env", "sm")
text(x, 510, "feature checklist", "sm")

# ================================================================ 04 execution route
x = TX[4]
station(x, TT, "04", "EXECUTION ROUTE", "melt it, cast it for .NET", TR,
        ["export graph → ONNX", "or ORT GenAI (decoders)", "or port the architecture", "→ unsupported-feature list"])
box(x - 92, 270, 184, 26)
box(x - 82, 296, 18, 230)
box(x + 64, 296, 18, 230)
a('<g>')
box(x - 22, 296, 44, 44)
box(x - 56, 340, 112, 26, stroke=AMBER)
text(x, 358, "ONNX die", "amb")
a('<animateTransform attributeName="transform" type="translate" values="0 0;0 0;0 120;0 120;0 0" keyTimes="0;.45;.6;.75;1" dur="3.2s" repeatCount="indefinite"/>')
a('</g>')
a(f'<rect x="{x - 48}" y="500" width="96" height="8" fill="{AMBER}" opacity="0">'
  f'<animate attributeName="opacity" values="0;0;.8;0;0" keyTimes="0;.55;.62;.8;1" dur="3.2s" repeatCount="indefinite"/></rect>')
box(x + 96, 316, 96, 120)
text(x + 144, 334, "NOT YET", "red")
for i, item in enumerate(["op X", "feat Y", "dtype Z"]):
    yy = 358 + i * 24
    a(f'<line x1="{x + 106}" y1="{yy - 4}" x2="{x + 116}" y2="{yy - 4}" stroke="{RED}" stroke-width="2"/>')
    text(x + 122, yy, item, "sm", "start")
text(x + 144, 454, "listed", "sm")

# ================================================================ 05 .NET behavior
x = TX[5]
station(x, TT, "05", ".NET BEHAVIOR", "fit the C# drivetrain", TR,
        ["Starlings.Tokenizers (pinned)", "processors · orchestration", "decoding · typed results", "no Python at runtime"])
box(x - 130, 290, 260, 176)
for i, (lab, rev) in enumerate([("tokenize", False), ("process", True), ("decode", False)]):
    gx = x - 80 + i * 80
    gear(gx, 362, 34, 11, 4, reverse=rev)
    text(gx, 424, lab, "sm")
text(x, 312, "Starlings.Models.<X>", "lbl")
text(x, 452, "→ typed results", "amb")
a(f'<line x1="{x}" y1="466" x2="{x}" y2="500" stroke="{CYAN}" stroke-width="3"/>')
box(x - 18, 498, 36, 14)

# ================================================================ 06 verify fidelity
x = BX[6]
station(x, BT, "06", "VERIFY FIDELITY", "gauge it against the master", BR,
        ["model-pipeline evaluate", "PyTorch → Python ORT → C#", "first divergent stage named", "→ signed-off report"])
box(x - 135, 786, 270, 128)
gauges = [("tokens", "G1"), ("export", "Py ORT"), ("C# ids", "same"), ("C# text", "e2e")]
for i, (lab, sub) in enumerate(gauges):
    cx, cy = x - 100 + i * 67, 826
    a(f'<circle cx="{cx}" cy="{cy}" r="25" fill="{BG}" stroke="{CYAN}" stroke-width="2"/>')
    a(f'<path d="M {cx - 18} {cy + 6} A 19 19 0 0 1 {cx + 18} {cy + 6}" fill="none" stroke="{DIM}" stroke-width="4"/>')
    a(f'<line x1="{cx - 2}" y1="{cy - 19}" x2="{cx + 2}" y2="{cy - 19}" stroke="{GREEN}" stroke-width="4"/>')
    t0 = 0.02 + i * 0.1
    kt = f"0;{t0:.2f};{t0 + .06:.2f};{t0 + .1:.2f};{t0 + .13:.2f};.9;1"
    a(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - 19}" stroke="{AMBER}" stroke-width="3" stroke-linecap="round">'
      f'<animateTransform attributeName="transform" type="rotate" '
      f'values="-75 {cx} {cy};-75 {cx} {cy};28 {cx} {cy};-9 {cx} {cy};0 {cx} {cy};0 {cx} {cy};-75 {cx} {cy}" '
      f'keyTimes="{kt}" dur="6s" repeatCount="indefinite"/></line>')
    text(cx, cy + 44, lab, "sm")
    lamp_cycle(cx, cy + 62, 6, t0 + .13, 6, off=.9)
a(f'<circle cx="{x - 118}" cy="944" r="7" fill="{RED}"/>')
text(x - 104, 948, "fail → minimal repro + taxonomy", "red", "start")
text(x - 104, 968, "2 failed repairs → maintainer", "red", "start")
text(x - 104, 988, "export · tokenizer · precision…", "sm", "start")

# ================================================================ 07 optimize
x = BX[7]
station(x, BT, "07", "OPTIMIZE", "tune only what you measured", BR,
        ["profile measured bottlenecks", "model-pipeline bench / summarize", "quantized = its own variant", "→ rerun parity + quality"])
a(f'<circle cx="{x - 50}" cy="880" r="56" fill="{BG}" stroke="{CYAN}" stroke-width="2"/>')
box(x - 60, 812, 20, 12)
for k in range(12):
    ang = k * math.pi / 6
    a(f'<line x1="{x - 50 + 46 * math.sin(ang):.1f}" y1="{880 - 46 * math.cos(ang):.1f}" '
      f'x2="{x - 50 + 52 * math.sin(ang):.1f}" y2="{880 - 52 * math.cos(ang):.1f}" stroke="{TEXT2}" stroke-width="2"/>')
a(f'<line x1="{x - 50}" y1="880" x2="{x - 50}" y2="838" stroke="{AMBER}" stroke-width="3" stroke-linecap="round">'
  f'<animateTransform attributeName="transform" type="rotate" from="0 {x - 50} 880" to="360 {x - 50} 880" dur="2s" repeatCount="indefinite"/></line>')
text(x - 50, 906, "p50 p95", "sm")
box(x + 22, 800, 110, 120, stroke=AMBER)
text(x + 77, 820, "SIDING", "ambb")
a(f'<rect x="{x + 63}" y="838" width="28" height="28" rx="4" fill="{AMBER}" opacity=".55"/>')
text(x + 77, 886, "fp16 / int8", "sm")
text(x + 77, 904, "own profile", "sm")
a(f'<line x1="{x + 77}" y1="922" x2="{x + 77}" y2="1002" stroke="{AMBER}" stroke-width="2" stroke-dasharray="6 5"/>')

# ================================================================ 08 integrate
x = BX[8]
station(x, BT, "08", "INTEGRATE", "fit the sockets people use", BR,
        ["Extensions.AI · ML.NET adapters", "samples · packaging · docs", "consumer + schema tests", "→ install-from-package demo"])
box(x - 160, 770, 320, 20)
for i, lab in enumerate(["Extensions.AI", "ML.NET", "samples"]):
    px = x - 106 + i * 106
    a(f'<line x1="{px}" y1="790" x2="{px}" y2="810" stroke="{CYAN}" stroke-width="2"/>')
    box(px - 50, 810, 100, 40)
    text(px, 835, lab, "sm")
    a(f'<rect x="{px - 6}" y="850" width="12" height="10" fill="{CYAN2}"/>')
a('<g>')
a(f'<line x1="{x}" y1="870" x2="{x}" y2="920" stroke="{CYAN}" stroke-width="4"/>')
a(f'<rect x="{x - 14}" y="920" width="28" height="14" fill="{CYAN2}"/>')
a('<animateTransform attributeName="transform" type="translate" values="0 0;0 0;0 50;0 50;0 0" keyTimes="0;.4;.55;.7;1" dur="2.8s" repeatCount="indefinite"/>')
a('</g>')

# ================================================================ 09 qualify release
x = BX[9]
station(x, BT, "09", "QUALIFY", "8 inspections, all must pass", BR,
        ["gates G0–G7 · platform matrix", "1-hour soak (Starlings.Soak.Tool)", "clean install from packages", "qualification.yml · CPU + CUDA"])
a(f'<path d="M {x - 128} 1006 V 770 H {x + 128} V 1006 H {x + 106} V 848 H {x - 106} V 1006 Z" fill="{PANEL}" stroke="{CYAN}" stroke-width="2"/>')
gnames = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
gdesc = ["prov", "tok", "num", "task", "res", "perf", "dist", "intg"]
for i, g in enumerate(gnames):
    gx = x - 98 + i * 28
    lamp_cycle(gx, 790, 9, (0.4 + i * 0.55) / 9, 9, off=.93)
    text(gx, 816, g, "sm")
stable_on = (0.4 + 8 * 0.55) / 9
anim = (f'values="{DIM};{DIM};{GREEN};{GREEN};{DIM}" keyTimes="0;{stable_on:.3f};{stable_on + 0.005:.3f};.93;1" '
        f'dur="9s" repeatCount="indefinite"')
a(f'<rect x="{x - 44}" y="862" width="88" height="28" rx="3" fill="{BG}" stroke="{DIM}" stroke-width="2">'
  f'<animate attributeName="stroke" {anim}/></rect>')
a(f'<text x="{x}" y="881" class="lbl" text-anchor="middle" style="fill:{DIM}">STABLE<animate attributeName="fill" {anim}/></text>')
for i in range(5):
    lx2 = x - 80 + i * 40
    a(f'<line x1="{lx2}" y1="900" x2="{lx2}" y2="1002" stroke="{GREEN}" stroke-width="2" stroke-dasharray="4 5" opacity=".35">'
      f'<animate attributeName="opacity" values=".1;.6;.1" dur="1.4s" begin="{i * 0.2:.1f}s" repeatCount="indefinite"/></line>')
a(f'<circle cx="{x - 160}" cy="790" r="18" fill="{BG}" stroke="{CYAN}" stroke-width="2"/>')
a(f'<line x1="{x - 160}" y1="790" x2="{x - 160}" y2="777" stroke="{CYAN2}" stroke-width="2">'
  f'<animateTransform attributeName="transform" type="rotate" from="0 {x - 160} 790" to="360 {x - 160} 790" dur="4s" repeatCount="indefinite"/></line>')
text(x - 160, 826, "weekly", "sm")

# ================================================================ 10 publish
x = BX[10]
station(x, BT, "10", "PUBLISH", "a human turns the key", BR,
        ["maintainer approves the tag", "release.yml: drill → OIDC push", "packages · catalog · notes", "weights stay on Hugging Face"])
box(x - 60, 772, 150, 84)
text(x + 15, 792, "MAINTAINER KEY", "sm")
a(f'<circle cx="{x - 20}" cy="824" r="17" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>')
a(f'<g><rect x="{x - 23}" y="809" width="6" height="30" rx="2" fill="{AMBER}"/>'
  f'<animateTransform attributeName="transform" type="rotate" values="-45 {x - 20} 824;-45 {x - 20} 824;45 {x - 20} 824;45 {x - 20} 824;-45 {x - 20} 824" '
  f'keyTimes="0;.4;.5;.9;1" dur="5s" repeatCount="indefinite"/></g>')
a(f'<circle cx="{x + 46}" cy="824" r="9" fill="{DIM}"><animate attributeName="fill" values="{DIM};{DIM};{GREEN};{GREEN};{DIM}" '
  f'keyTimes="0;.49;.5;.9;1" dur="5s" repeatCount="indefinite"/></circle>')
text(x + 46, 850, "ai-v*", "sm")
box(x - 60, 868, 150, 126)
text(x + 15, 888, "4 IDENTITIES", "lbl")
for i, ident in enumerate(["impl  semver", "bundle  digests", "profile  ORT+EP", "evidence  G0–G7"]):
    text(x - 50, 910 + i * 21, ident, "sm", "start")
# warehouse (Hugging Face) holds the weights
a(f'<path d="M 598 812 L 640 778 L 682 812 V 870 H 598 Z" fill="{PANEL}" stroke="{AMBER}" stroke-width="2"/>')
text(640, 830, "HF", "ambb")
text(640, 848, "weights", "sm")
text(640, 864, "store", "sm")

# road + truck to the customer site
a(f'<line x1="900" y1="1190" x2="584" y2="1190" stroke="{CYAN}" stroke-width="2" stroke-dasharray="14 10"/>')
text(740, 1220, "packages via nuget.org", "sm")
a('<defs><clipPath id="road"><rect x="584" y="1140" width="320" height="80"/></clipPath></defs>')
a('<g clip-path="url(#road)"><g>')
a(f'<rect x="-46" y="-30" width="62" height="34" fill="{PANEL}" stroke="{AMBER}" stroke-width="2"/>')
a(f'<path d="M -46 4 V -18 H -64 L -72 -6 V 4 Z" fill="{PANEL}" stroke="{AMBER}" stroke-width="2"/>')
a(f'<circle cx="-56" cy="8" r="6" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>')
a(f'<circle cx="2" cy="8" r="6" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>')
a('<text x="-15" y="-9" class="sm" text-anchor="middle" style="font-size:9px">NuGet</text>')
a('<animateMotion path="M 980 1180 H 570" dur="6s" repeatCount="indefinite"/>')
a('</g></g>')

# ================================================================ 11 maintain
box(1300, 1160, 560, 150, stroke=GREEN)
text(1320, 1192, "11  MAINTAIN", "ttl", "start")
text(1320, 1216, "the line watches itself", "ana", "start")
for i, ln in enumerate(["detect upstream + dependency changes",
                        "rerun only the stages a change invalidates",
                        "weekly CPU qualification · per-release CUDA",
                        "tokenizer fixes go upstream in parallel"]):
    text(1320, 1242 + i * 19, ln, "repo", "start")
a(f'<circle cx="1820" cy="1200" r="16" fill="{BG}" stroke="{GREEN}" stroke-width="2"/>')
a(f'<line x1="1820" y1="1200" x2="1820" y2="1188" stroke="{GREEN}" stroke-width="2">'
  f'<animateTransform attributeName="transform" type="rotate" from="0 1820 1200" to="360 1820 1200" dur="3s" repeatCount="indefinite"/></line>')

# ================================================================ your app (runtime)
a(f'<rect x="30" y="700" width="550" height="620" fill="none" stroke="{CYAN}" stroke-width="2" stroke-dasharray="10 6"/>')
text(54, 734, "YOUR APP  (runtime)", "ttl", "start")
text(54, 758, "the part runs on-site, offline", "ana", "start")
box(334, 780, 226, 104, stroke=AMBER)
text(447, 802, "ACQUISITION", "lbl")
text(447, 822, "explicit download, once", "sm")
text(447, 840, "hashes verified · lock", "sm")
text(447, 858, "no self-update", "sm")
text(447, 876, "Starlings.Acquisition", "sm")
a(f'<line x1="596" y1="830" x2="564" y2="830" stroke="{AMBER}" stroke-width="3" stroke-dasharray="8 6" marker-end="url(#aa)">'
  f'<animate attributeName="stroke-dashoffset" values="0;-28" dur=".9s" repeatCount="indefinite"/></line>')
box(50, 780, 262, 150)
gear(100, 846, 36, 12, 3)
text(212, 808, "ONNX RUNTIME", "lbl")
text(212, 826, "in-process", "sm")
a(f'<rect x="150" y="842" width="124" height="28" rx="14" fill="{BG}" stroke="{CYAN}" stroke-width="2"/>')
text(182, 861, "CPU", "sm")
text(242, 861, "CUDA", "sm")
a(f'<circle cx="182" cy="856" r="13" fill="none" stroke="{AMBER}" stroke-width="2">'
  f'<animate attributeName="cx" values="182;182;242;242;182" keyTimes="0;.45;.5;.95;1" dur="8s" repeatCount="indefinite"/></circle>')
text(212, 896, "no silent fallback", "sm")
text(212, 914, "Starlings.Runtime.Onnx", "sm")
a(f'<line x1="334" y1="830" x2="316" y2="830" stroke="{CYAN}" stroke-width="2" marker-end="url(#ac)"/>')
for i, lab in enumerate(["typed API", "Extensions.AI", "ML.NET"]):
    sx = 50 + i * 178
    a(f'<line x1="{sx + 80}" y1="932" x2="{sx + 80}" y2="958" stroke="{CYAN}" stroke-width="2" marker-end="url(#ac)"/>')
    box(sx, 960, 160, 40)
    text(sx + 80, 985, lab, "sm")
for i, lab in enumerate(["your .NET app", "your service", "your agent"]):
    sx = 50 + i * 178
    a(f'<line x1="{sx + 80}" y1="1000" x2="{sx + 80}" y2="1044" stroke="{AMBER}" stroke-width="2"/>')
    a(f'<rect x="{sx + 10}" y="1044" width="140" height="34" rx="17" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>')
    text(sx + 80, 1066, lab, "amb")
    a(f'<circle cx="{sx + 80}" cy="1022" r="4" fill="{AMBER}">'
      f'<animate attributeName="cy" values="1004;1040" dur="1s" begin="{i * 0.33:.2f}s" repeatCount="indefinite"/></circle>')
text(54, 1124, "diagnostics report all 4 identities", "repo", "start")
text(54, 1146, "offline · no Python sidecar · no cloud API", "ana", "start")
text(54, 1170, "dotnet add package Starlings.Models.*", "repo", "start")

# ================================================================ conveyor belt (over machine feet)
a(f'<path d="{BELT}" fill="none" stroke="{CYAN}" stroke-width="28" stroke-linejoin="round"/>')
a(f'<path d="{BELT}" fill="none" stroke="{PANEL}" stroke-width="23" stroke-linejoin="round"/>')
a(f'<path d="{BELT}" fill="none" stroke="#1f5a85" stroke-width="3" stroke-dasharray="14 10">'
  f'<animate attributeName="stroke-dashoffset" values="0;-24" dur=".22s" repeatCount="indefinite"/></path>')
for cx in (380, 740, 1100, 1460, 1820):
    a(f'<polygon points="{cx - 6},{TOP - 8} {cx + 6},{TOP} {cx - 6},{TOP + 8}" fill="{CYAN2}"/>')
for cx in (2240, 1910, 1575, 1250, 930):
    a(f'<polygon points="{cx + 6},{BOT - 8} {cx - 6},{BOT} {cx + 6},{BOT + 8}" fill="{CYAN2}"/>')
a(f'<polygon points="{TURN - 8},{(TOP + BOT) // 2 - 6} {TURN},{(TOP + BOT) // 2 + 6} {TURN + 8},{(TOP + BOT) // 2 - 6}" fill="{CYAN2}"/>')

# ================================================================ the model parts riding the belt
# Each look is clipped to a fixed zone of the belt; nothing is toggled on a timer.
DUR = 36
REST = f'<rect x="0" y="600" width="{W}" height="700"/>'  # vertical run + bottom row


def top_from(x0):
    return f'<rect x="{x0}" y="400" width="{W - x0}" height="200"/>'


def bot_upto(x1):
    return f'<rect x="0" y="900" width="{x1}" height="200"/>'


ZONES = {
    "raw": f'<rect x="0" y="400" width="{TX[4]}" height="200"/>',
    "check": top_from(TX[1]) + REST,
    "serial": top_from(TX[2]) + REST,
    "cast": top_from(TX[4]) + REST,
    "core": top_from(TX[5]) + REST,
    "pass": bot_upto(BX[6]),
    "plug": bot_upto(BX[8]),
    "stable": bot_upto(BX[9]),
}
LOOKS = {
    "raw": f'<polygon points="-14,-8 -4,-15 10,-12 15,0 9,13 -6,14 -15,5" fill="{ORE}" stroke="{AMBER}" stroke-width="2"/>',
    "serial": f'<rect x="-8" y="-3" width="16" height="6" fill="{WHITE}"/>',
    "cast": f'<rect x="-14" y="-14" width="28" height="28" rx="4" fill="{AMBER}"/><rect x="-7" y="-7" width="14" height="14" rx="2" fill="#7a4c0f"/>',
    "core": f'<circle r="5" fill="{CYAN2}"/>',
    "check": f'<polyline points="-13,-18 -9,-14 -3,-22" fill="none" stroke="{GREEN}" stroke-width="2.5"/>',
    "pass": f'<circle cx="13" cy="-13" r="5.5" fill="{GREEN}" stroke="{BG}" stroke-width="1.5"/>',
    "plug": f'<rect x="-26" y="-5" width="12" height="10" fill="{CYAN2}"/>',
    "stable": f'<rect x="-19" y="-19" width="38" height="38" rx="6" fill="none" stroke="{GREEN}" stroke-width="2.5"/>',
}
a('<defs>' + ''.join(f'<clipPath id="z-{n}">{z}</clipPath>' for n, z in ZONES.items()) + '</defs>')
for name, look in LOOKS.items():
    a(f'<g clip-path="url(#z-{name})">')
    for k in range(5):
        begin = f"-{k * DUR / 5:.1f}s"
        a(f'<g><animateMotion path="{BELT}" dur="{DUR}s" begin="{begin}" repeatCount="indefinite" calcMode="paced"/>'
          f'<g transform="translate(0,-26)">{look}</g></g>')
    a('</g>')

# parts enter from the discover pedestal and finish inside the release crate
box(TX[0] - 40, 408, 80, 118)
text(TX[0], 444, "HF", "sm")
text(TX[0], 462, "GitHub", "sm")
text(TX[0], 480, "issues", "sm")
a(f'<rect x="598" y="988" width="84" height="60" fill="#2a2116" stroke="{AMBER}" stroke-width="2"/>')
a(f'<rect x="612" y="998" width="26" height="38" fill="{TEXT}" opacity=".9"/>')
for i in range(3):
    a(f'<line x1="616" y1="{1008 + i * 9}" x2="634" y2="{1008 + i * 9}" stroke="{BG}" stroke-width="2"/>')
text(662, 1022, "nupkg", "sm")

# ================================================================ title block
a(f'<rect x="1900" y="1226" width="470" height="94" fill="{PANEL}" stroke="{CYAN}" stroke-width="1.5"/>')
a(f'<line x1="1900" y1="1252" x2="2370" y2="1252" stroke="{CYAN}" stroke-width="1"/>')
text(1912, 1244, "DWG  STARLING FOUNDRY · MACHINE VIEW", "lbl", "start")
text(1912, 1272, "SRC  docs/plan/fast-faithful-ai-dotnet.md §8–10", "sm", "start")
text(1912, 1290, "DATE  2026-10-19", "sm", "start")
text(1912, 1308, "SCALE  analogy, not to scale", "sm", "start")

a("</svg>")
ARGS.out.write_text("\n".join(out) + "\n")
print(ARGS.out)
