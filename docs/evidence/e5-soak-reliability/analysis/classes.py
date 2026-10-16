"""Per phase: fitted growth over the post-warmup window of anon and the mapping classes (MiB)."""

import gzip
import json
import sys


def classify(m):
    name = m["name"]
    s, e = (int(x, 16) for x in m["range"].split("-"))
    if name == "[heap]":
        return "heap"
    if name:
        return "named"
    if s % (64 << 20) == 0 and (e - s) <= (64 << 20):
        return "arena"
    return "mmap"


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx * (xs[-1] - xs[0])


for path in sys.argv[1:]:
    op = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(line) for line in op(path, "rt")]
    print(path)
    for phase in ("loaded", "unloaded"):
        pr = [r for r in rows if r.get("phase", "unloaded") == phase]
        if not pr:
            continue
        T = pr[-1]["t_s"]
        pr = [r for r in pr if r["t_s"] >= 0.2 * T and r["cycle"] >= 2]
        xs = [r["t_s"] for r in pr]
        series = {k: [] for k in ("anon", "heap", "arena", "mmap", "named", "fordblks", "uordblks")}
        for r in pr:
            c = {"heap": 0, "arena": 0, "mmap": 0, "named": 0}
            for m in r["maps"]:
                c[classify(m)] += m["anon_kb"] / 1024
            series["anon"].append(r["anon_kb"] / 1024)
            for k in c:
                series[k].append(c[k])
            mi = r.get("mallinfo2_kb", {})
            series["fordblks"].append(mi.get("fordblks", 0) / 1024)
            series["uordblks"].append(mi.get("uordblks", 0) / 1024)
        h = len(xs) // 2
        print(" ", phase, len(xs))
        for k, v in series.items():
            print(f"    {k:9s} {fit(xs, v):6.1f} [{fit(xs[:h], v[:h]):6.1f},{fit(xs[h:], v[h:]):6.1f}] range {min(v):.0f}-{max(v):.0f}")
