import json
import statistics
import sys

# nonthp.py <json>...: least-squares slope (MiB/h) and fitted growth over the post-warmup window of rss_anon,
# AnonHugePages and their difference, with the slope of each half, for the settled loaded/unloaded series.


def fit(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


for p in sys.argv[1:]:
    d = json.load(open(p))
    cyc = d["cycles"]
    dur = d["options"]["elapsed_s"]
    start = max(cyc[min(2, len(cyc) - 1)]["end_s"], d["options"]["warmup_fraction"] * dur)
    post = [c for c in cyc if c["end_s"] >= start]
    print(p.split("/")[-2], d["environment"].get("cpu_model"), f"{len(post)} post-warmup cycles")
    for ph in ("loaded", "unloaded"):
        xs = [c["end_s"] / 3600 for c in post]
        span = xs[-1] - xs[0]
        h = len(post) // 2
        for name, f in (("rss_anon", lambda s: s["rss_anon_mb"]), ("anon_huge", lambda s: s["anon_huge_pages_mb"]),
                        ("non_huge", lambda s: s["rss_anon_mb"] - s["anon_huge_pages_mb"])):
            ys = [f(c[ph]) for c in post]
            s = fit(xs, ys)
            print(f"  {ph:8} {name:9} median {statistics.median(ys):7.0f} slope {s:6.0f} MiB/h growth {s * span:5.0f}"
                  f" halves {fit(xs[:h], ys[:h]):6.0f} {fit(xs[h:], ys[h:]):6.0f}")
