import json, sys, statistics

for f in sys.argv[1:]:
    d = json.load(open(f))
    run = f.split('/')[-2]
    a = {(x['series'], x['metric']): x for x in d['analysis']}
    cyc = d['cycles']
    req = sum(c['requests'] for c in cyc)
    def s(series):
        x = a.get((series, 'rss_anon'))
        if not x:
            return '-'
        return f"{x['median_mb']:.0f} ({x['slope_mb_per_hour']:.0f}, {x['verdict']})"
    print(f"| {d['model'] if isinstance(d['model'], str) else ''} | {run} | {len(cyc)} | {req} | {d['wrong_output_count']} | {s('loaded')} | {s('unloaded')} | {s('trimmed')} | {d['verdict'][:4]} |")
