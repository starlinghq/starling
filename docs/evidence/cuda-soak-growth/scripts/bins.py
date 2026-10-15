import json, sys, statistics


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx * 3600 if sxx else 0


for f in sys.argv[1:]:
    d = json.load(open(f))
    cyc = d['cycles']
    dur = cyc[-1]['end_s']
    print(f.split('/')[-1], 'cycles', len(cyc), 'dur', round(dur))
    for ser in ('unloaded', 'loaded', 'trimmed'):
        pts = [(c[ser]['t_s'], c[ser]['rss_anon_mb']) for c in cyc if c.get(ser)]
        if not pts:
            continue
        nb = 12 if dur > 1800 else 6
        w = dur / nb
        med = [round(statistics.median([y for t, y in pts if i * w <= t < (i + 1) * w] or [0])) for i in range(nb)]
        thirds = []
        for i in range(3):
            sub = [(t, y) for t, y in pts if dur * (0.2 + 0.8 * i / 3) <= t < dur * (0.2 + 0.8 * (i + 1) / 3)]
            thirds.append(round(fit([a for a, _ in sub], [b for _, b in sub])) if len(sub) > 2 else None)
        print(' ', ser, 'bin medians', med, 'post-warmup thirds MiB/h', thirds)
