"""Compare the anonymous mappings of two dumps of a soak .smaps.jsonl (first and last of a phase)."""

import json
import sys

path, phase = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(path) if json.loads(line).get("phase", "unloaded") == phase]
a, b = rows[int(sys.argv[3]) if len(sys.argv) > 3 else 0], rows[-1]


def maps(r):
    return {m["range"]: m for m in r["maps"]}


ma, mb = maps(a), maps(b)
print(f"cycle {a['cycle']} -> {b['cycle']}")
for k in sorted(set(ma) | set(mb), key=lambda k: int(k.split("-")[0], 16)):
    s, e = (int(x, 16) for x in k.split("-"))
    x = ma.get(k, {}).get("anon_kb", 0) / 1024
    y = mb.get(k, {}).get("anon_kb", 0) / 1024
    name = (mb.get(k) or ma.get(k))["name"]
    if abs(y - x) > 4 or max(x, y) > 100:
        print(f"{k} size {(e - s) / 2**20:9.1f} MiB {name:20s} {x:7.1f} -> {y:7.1f}")
