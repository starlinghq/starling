import json
import statistics
import sys

# warmcmp.py <a.jsonl> <b.jsonl>...: median ms per (path, workload, concurrency) for each file's warm run(s)
tables = []
for p in sys.argv[1:]:
    t = {}
    for line in open(p):
        d = json.loads(line)
        if "workload" in d and "ms" in d:
            t.setdefault((d["path"], d["workload"], d["concurrency"]), []).append(statistics.median(d["ms"]))
    tables.append({k: statistics.median(v) for k, v in t.items()})
keys = sorted(set().union(*tables))
print("path/workload/c | " + " | ".join(sys.argv[1:]))
for k in keys:
    vals = [t.get(k) for t in tables]
    ratio = vals[1] / vals[0] if len(vals) > 1 and vals[0] and vals[1] else None
    print(f"{k[0]} {k[1]} c={k[2]} | " + " | ".join(f"{v:.2f}" if v else "-" for v in vals) + (f" | ratio {ratio:.3f}" if ratio else ""))
