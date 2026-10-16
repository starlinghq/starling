"""Summarize a soak .smaps.jsonl: per phase, per dump, anon totals, mallinfo2 and mapping classes (MiB)."""

import gzip
import json
import sys


def classify(m):
    name = m["name"]
    start, end = (int(x, 16) for x in m["range"].split("-"))
    size = end - start
    if name == "[heap]":
        return "heap"
    if name:
        return "named"
    # glibc non-main arenas are 64 MiB-aligned heaps of at most 64 MiB
    if start % (64 << 20) == 0 and size <= (64 << 20):
        return "arena"
    return "mmap"


def main(path):
    opener = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(line) for line in opener(path, "rt")]
    for phase in ("loaded", "unloaded"):
        pr = [r for r in rows if r.get("phase", "unloaded") == phase]
        if not pr:
            continue
        print(f"== {phase}")
        print(" cycle    t   anon  heap arena  mmap named | uord  ford hblkhd arena_mi")
        for r in pr:
            cls = {"heap": 0, "arena": 0, "mmap": 0, "named": 0}
            for m in r["maps"]:
                cls[classify(m)] += m["anon_kb"]
            mi = r.get("mallinfo2_kb", {})
            f = lambda kb: f"{kb / 1024:5.0f}"  # noqa: E731
            print(
                f"{r['cycle']:6d} {r['t_s']:5.0f} {f(r['anon_kb'])} {f(cls['heap'])} {f(cls['arena'])} {f(cls['mmap'])} {f(cls['named'])} |"
                f" {f(mi.get('uordblks', 0))} {f(mi.get('fordblks', 0))} {f(mi.get('hblkhd', 0))} {f(mi.get('arena', 0))}"
            )


for p in sys.argv[1:]:
    print(p)
    main(p)
