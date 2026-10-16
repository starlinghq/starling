"""Per dump: number of 64 MiB-aligned (glibc arena) mappings with >2 MiB anon resident, and their resident MiB."""

import json
import sys

path, phase = sys.argv[1], sys.argv[2]
for line in open(path):
    r = json.loads(line)
    if r.get("phase", "unloaded") != phase:
        continue
    ar = []
    for m in r["maps"]:
        s, e = (int(x, 16) for x in m["range"].split("-"))
        if not m["name"] and s % (64 << 20) == 0 and e - s <= (64 << 20):
            ar.append(round(m["anon_kb"] / 1024))
    print(r["cycle"], len(ar), sum(ar), sorted(ar, reverse=True))
