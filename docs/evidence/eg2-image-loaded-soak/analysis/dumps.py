"""Per dump of a soak .smaps.jsonl: mapping classes, the GC's committed memory, and glibc's per-arena malloc_info (MiB).

usage: python3 -I dumps.py <x.smaps.jsonl[.gz]> [loaded|unloaded]

Columns: anon (all anonymous resident), heap (the main glibc heap, [heap]), arena (64 MiB-aligned mappings: glibc
thread arenas), mmap (other anonymous mappings: mmapped malloc chunks and the GC's regions); gc_com, gc_heap, gc_frag
(GCMemoryInfo: committed, heap size, fragmented); mmapped (malloc_info: bytes and count of mmapped chunks); main
arena: memory from the system and free bytes (free includes the top chunk; freed pages that malloc_trim returned are
counted here but are not resident); thread arenas with more than 4 MiB: system/free.
"""

import gzip
import json
import sys
import xml.etree.ElementTree as ET

M = 1 << 20


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


def arenas(xml):
    root = ET.fromstring(xml)
    out = []
    for h in root.findall("heap"):
        tot = {t.get("type"): int(t.get("size")) for t in h.findall("total")}
        sysm = {t.get("type"): int(t.get("size")) for t in h.findall("system")}
        out.append((int(h.get("nr")), sysm.get("current", 0), tot.get("fast", 0) + tot.get("rest", 0)))
    mm = {t.get("type"): (int(t.get("count")), int(t.get("size"))) for t in root.findall("total")}
    return out, mm.get("mmap", (0, 0))


def main(path, phase=None):
    opener = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(line) for line in opener(path, "rt")]
    print(" phase    cyc  anon  heap arena  mmap | gc_com gc_heap gc_frag | mmapped(n) | main: sys  free | thread arenas: sys/free")
    for r in rows:
        if phase and r["phase"] != phase:
            continue
        cls = {"heap": 0, "arena": 0, "mmap": 0, "named": 0}
        for m in r["maps"]:
            cls[classify(m)] += m["anon_kb"]
        gc = r.get("gc_kb", {})
        line = f"{r['phase']:8s} {r['cycle']:4d} {r['anon_kb'] / 1024:5.0f} {cls['heap'] / 1024:5.0f} {cls['arena'] / 1024:5.0f} {cls['mmap'] / 1024:5.0f} |"
        line += f" {gc.get('committed', 0) / 1024:6.0f} {gc.get('heap_size', 0) / 1024:7.0f} {gc.get('fragmented', 0) / 1024:7.0f} |"
        if "malloc_info" in r:
            a, (n, sz) = arenas(r["malloc_info"])
            main_ = next(x for x in a if x[0] == 0)
            line += f" {sz / M:5.0f}({n}) | {main_[1] / M:5.0f} {main_[2] / M:5.0f} |"
            line += " " + " ".join(f"{s / M:.0f}/{f / M:.0f}" for nr, s, f in a if nr != 0 and s > 4 * M)
        print(line)


main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
