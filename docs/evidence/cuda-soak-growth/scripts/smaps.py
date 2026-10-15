import json, sys


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx * 3600 if sxx else 0


for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if int(r["status"].get("Threads", "1")) > 1]
    if not rows:
        continue
    tmax = rows[-1]["t"]
    post = [r for r in rows if r["t"] >= 0.2 * tmax]
    print(f, "samples", len(rows), "post", len(post))
    thr = [int(r["status"]["Threads"]) for r in post]
    print("  threads min/max", min(thr), max(thr), "arenas min/max", min(r["arenas"] for r in post), max(r["arenas"] for r in post))
    keys = sorted({k for r in post for k in r["classes"]})
    q = max(1, len(post) // 4)
    for k in keys:
        ys = [r["classes"].get(k, {}).get("anon", 0) / 1024 for r in post]
        rs = [r["classes"].get(k, {}).get("rss", 0) / 1024 for r in post]
        if max(rs) < 4:
            continue
        xs = [r["t"] for r in post]
        a = sum(ys[:q]) / q
        b = sum(ys[-q:]) / q
        print(f"  {k:32s} anon first-q {a:8.1f} last-q {b:8.1f} slope {fit(xs, ys):7.0f} MiB/h | rss first-q {sum(rs[:q])/q:8.1f} last-q {sum(rs[-q:])/q:8.1f}")
    ra = [int(r["status"]["RssAnon"].split()[0]) / 1024 for r in post]
    print(f"  RssAnon first-q {sum(ra[:q])/q:.1f} last-q {sum(ra[-q:])/q:.1f} slope {fit([r['t'] for r in post], ra):.0f} MiB/h")
