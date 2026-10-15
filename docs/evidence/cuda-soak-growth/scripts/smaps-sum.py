"""smaps-sum.py <pid> <t_s>: one JSON line summarising /proc/<pid>/smaps and status by mapping class (sizes in KiB).

Classes: heap ([heap], glibc main arena), arena (anonymous rw mapping starting on a 64 MiB boundary whose reservation,
with the PROT_NONE tail that follows it, spans 64 MiB: a glibc secondary arena heap), anon (other anonymous mappings:
large malloc chunks served by mmap, the .NET GC heap, driver host allocations), nvidia (/dev/nvidia* device
mappings), stack, and one entry per mapped file (by basename) for file-backed mappings.
"""
import json
import os
import sys

pid, t = sys.argv[1], float(sys.argv[2])
MB64 = 64 << 20
maps = []
cur = None
with open(f"/proc/{pid}/smaps") as f:
    for line in f:
        parts = line.split()
        if "-" in parts[0] and len(parts) >= 5 and not parts[0].endswith(":"):
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            cur = {"lo": lo, "hi": hi, "perm": parts[1], "path": " ".join(parts[5:]) if len(parts) > 5 else "", "rss": 0, "anon": 0}
            maps.append(cur)
        elif parts[0] == "Rss:":
            cur["rss"] = int(parts[1])
        elif parts[0] == "Anonymous:":
            cur["anon"] = int(parts[1])

out = {}


def add(cls, m):
    e = out.setdefault(cls, {"rss": 0, "anon": 0, "n": 0})
    e["rss"] += m["rss"]
    e["anon"] += m["anon"]
    e["n"] += 1


arenas = 0
for i, m in enumerate(maps):
    p = m["path"]
    if p == "[heap]":
        add("heap", m)
    elif p.startswith("[stack"):
        add("stack", m)
    elif p.startswith("/dev/nvidia"):
        add("nvidia", m)
    elif p == "" or p.startswith("[anon"):
        if m["perm"].startswith("---"):
            continue
        span = m["hi"] - m["lo"]
        nxt = maps[i + 1] if i + 1 < len(maps) else None
        if nxt and nxt["lo"] == m["hi"] and nxt["perm"].startswith("---") and nxt["path"] == "":
            span += nxt["hi"] - nxt["lo"]
        if m["lo"] % MB64 == 0 and span == MB64:
            add("arena", m)
            arenas += 1
        else:
            add("anon", m)
    elif p.startswith("["):
        add(p, m)
    else:
        add("file:" + os.path.basename(p.replace(" (deleted)", "")), m)

status = {}
with open(f"/proc/{pid}/status") as f:
    for line in f:
        k, _, v = line.partition(":")
        if k in ("Threads", "RssAnon", "RssFile", "RssShmem", "VmRSS", "VmData", "VmLck", "VmPin"):
            status[k] = v.strip()

big = {k: v for k, v in out.items() if not k.startswith("file:") or v["anon"] >= 1024 or v["rss"] >= 65536}
print(json.dumps({"t": t, "status": status, "arenas": arenas, "classes": big}, separators=(",", ":")))
