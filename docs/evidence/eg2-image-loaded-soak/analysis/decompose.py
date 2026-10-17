"""Fitted growth (post-warmup, MiB) of each component of the loaded (or given phase) dumps.

usage: python3 -I decompose.py <phase> <x.smaps.jsonl>...
"""

import gzip
import json
import sys


def classify(m):
    name = m["name"]
    start, end = (int(x, 16) for x in m["range"].split("-"))
    if name == "[heap]":
        return "heap"
    if name:
        return "named"
    if start % (64 << 20) == 0 and end - start <= (64 << 20):
        return "arena"
    return "mmap"


def fit(ts, vs):
    n = len(ts)
    if n < 2:
        return 0.0
    mx = sum(ts) / n
    my = sum(vs) / n
    sxx = sum((t - mx) ** 2 for t in ts)
    sxy = sum((t - mx) * (v - my) for t, v in zip(ts, vs))
    return (sxy / sxx) * (ts[-1] - ts[0]) if sxx else 0.0


def main(phase, path):
    opener = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(line) for line in opener(path, "rt")]
    rows = [r for r in rows if r["phase"] == phase and r["t_s"] >= 720]
    comps = {"anon": [], "heap": [], "arena": [], "mmap": [], "gc_com": [], "mmap-gc": []}
    ts = [r["t_s"] for r in rows]
    for r in rows:
        cls = {"heap": 0, "arena": 0, "mmap": 0, "named": 0}
        for m in r["maps"]:
            cls[classify(m)] += m["anon_kb"] / 1024
        gc = r.get("gc_kb", {}).get("committed", 0) / 1024
        comps["anon"].append(r["anon_kb"] / 1024)
        comps["heap"].append(cls["heap"])
        comps["arena"].append(cls["arena"])
        comps["mmap"].append(cls["mmap"])
        comps["gc_com"].append(gc)
        comps["mmap-gc"].append(cls["mmap"] - gc)
    h = len(ts) // 2
    out = []
    for k, vs in comps.items():
        g = fit(ts, vs)
        g1 = fit(ts[: h + len(ts) % 2], vs[: h + len(ts) % 2])
        g2 = fit(ts[h:], vs[h:])
        out.append(f"{k}={g:6.1f} [{g1:6.1f},{g2:6.1f}] range {min(vs):.0f}-{max(vs):.0f}")
    print(path.split("/")[-2] if "/" in path else path)
    for o in out:
        print("   ", o)


for p in sys.argv[2:]:
    main(sys.argv[1], p)
