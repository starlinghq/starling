"""Summarizes the spike's runs.jsonl: outcomes, argument validity and latency per condition (python3 -I summarize.py runs.jsonl)."""
import collections, json, os, statistics, sys

runs = [json.loads(l) for l in open(sys.argv[1])]
# The runner records args_match = true for correct calls on cases without labelled arguments, so the match column is
# restricted to the cases that have labels (73 of 100).
fixtures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "cases.jsonl")
labelled = {c["id"] for c in (json.loads(l) for l in open(fixtures) if l.strip()) if c.get("args")}
by = collections.defaultdict(list)
for r in runs:
    by[r["condition"]].append(r)

def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)

def decode_rate(r):
    n = r["completion_tokens"]
    return (n - 1) / ((r["total_ms"] - r["ttft_ms"]) / 1000) if n > 1 and r["total_ms"] > r["ttft_ms"] else None

order = [c for c in ["free-all", "guided-all", "free-all-repeat", "free-subset", "guided-subset", "free-subset-repeat"] if c in by]
print("| condition | n | correct tool | wrong tool | invalid name | malformed | no call | args valid (of calls to offered tools) | labelled args match (of correct, labelled cases) |")
print("|---|---|---|---|---|---|---|---|---|")
for c in order:
    rs = by[c]
    o = collections.Counter(r["outcome"] for r in rs)
    valid = [r["args_valid"] for r in rs if r["args_valid"] is not None]
    match = [r["args_match"] for r in rs if r["args_match"] is not None and r["id"] in labelled]
    print(f"| {c} | {len(rs)} | {o['correct']} | {o['wrong_tool']} | {o['invalid_name']} | {o['malformed']} | {o['no_call']} | {sum(valid)}/{len(valid)} | {sum(match)}/{len(match)} |")

print()
print("Invalid names (condition: case id -> name):")
for c in order:
    bad = [(r["id"], r["called"]) for r in by[c] if r["outcome"] in ("invalid_name", "malformed", "no_call")]
    print(f"  {c}: {bad}")

print()
print("| condition | TTFT p50 | TTFT p90 | total p50 | total p90 | completion tokens p50 | decode tok/s p50 | decode tok/s p90 | ms/token p50 |")
print("|---|---|---|---|---|---|---|---|---|")
for c in order:
    rs = by[c]
    ttft = [r["ttft_ms"] for r in rs]; tot = [r["total_ms"] for r in rs]
    toks = [r["completion_tokens"] for r in rs]
    rate = [x for x in (decode_rate(r) for r in rs) if x]
    mspt = [r["total_ms"] / r["completion_tokens"] for r in rs if r["completion_tokens"]]
    print(f"| {c} | {pct(ttft,.5):.1f} | {pct(ttft,.9):.1f} | {pct(tot,.5):.1f} | {pct(tot,.9):.1f} | {pct(toks,.5):.0f} | {pct(rate,.5):.1f} | {pct(rate,.9):.1f} | {pct(mspt,.5):.2f} |")

print()
print("Paired ratios (B / A (same case, same tools), p50 and p90 of per-case ratios:")
pairs = [(a, b) for a, b in (("free-all", "guided-all"), ("free-subset", "guided-subset"), ("free-all", "free-all-repeat"), ("free-subset", "free-subset-repeat")) if a in by and b in by]
for free, guided in pairs:
    f = {r["id"]: r for r in by[free]}; g = {r["id"]: r for r in by[guided]}
    ids = sorted(set(f) & set(g))
    same = [i for i in ids if f[i]["completion_tokens"] == g[i]["completion_tokens"]]
    for label, key in (("ttft", "ttft_ms"), ("total", "total_ms")):
        ratios = [g[i][key] / f[i][key] for i in ids]
        print(f"  {guided} / {free} {label}: p50 {pct(ratios,.5):.3f} p90 {pct(ratios,.9):.3f}")
    rr = [decode_rate(g[i]) / decode_rate(f[i]) for i in ids if decode_rate(g[i]) and decode_rate(f[i])]
    print(f"  {guided} / {free} decode tok/s: p50 {pct(rr,.5):.3f} p90 {pct(rr,.9):.3f}")
    rs = [g[i]["total_ms"] / f[i]["total_ms"] for i in same]
    print(f"  {guided} / {free} total, cases with equal token counts (n={len(same)}): p50 {pct(rs,.5):.3f} p90 {pct(rs,.9):.3f}")
    print(f"  identical text: {sum(1 for i in ids if f[i]['text'] == g[i]['text'])}/{len(ids)}")

print()
tempt = [r for r in by["free-all"] if r["tempting"]]
print(f"free-all on 'tempting' prompts: {collections.Counter(r['outcome'] for r in tempt)}")
