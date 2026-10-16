import json
import sys

# smaps_attr.py <smaps.jsonl>: classify anonymous mappings per dump; first, middle and last dumps.
# glibc non-main arenas: 64 MiB-aligned start, size <= 64 MiB. [heap]: main arena. Others: large (GC, mmapped chunks).


def classify(m):
    a, b = (int(x, 16) for x in m["range"].split("-"))
    size = b - a
    if m["name"] == "[heap]":
        return "glibc main heap"
    if m["name"].startswith("["):
        return m["name"]
    if a % (64 << 20) == 0 and size <= (64 << 20):
        return "glibc arena (64 MiB aligned)"
    if size >= (256 << 20):
        return "large reservation (>=256 MiB, e.g. GC)"
    return "other anonymous"


rows = [json.loads(l) for l in open(sys.argv[1])]
for r in (rows[0], rows[len(rows) // 2], rows[-1]):
    agg = {}
    for m in r["maps"]:
        k = classify(m)
        t = agg.setdefault(k, [0, 0, 0])
        t[0] += 1
        t[1] += m["anon_kb"]
        t[2] += m["anon_huge_kb"]
    print(f"cycle {r['cycle']} t={r['t_s']:.0f}s total anon {r['anon_kb'] // 1024} MiB, huge {r['anon_huge_kb'] // 1024} MiB")
    for k, (n, an, hu) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:40} n={n:3d} anon {an // 1024:6d} MiB huge {hu // 1024:6d} MiB")
