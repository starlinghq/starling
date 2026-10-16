"""cmp_nodes.py <dir A> <dir B>: first node outputs (unoptimized vision graph, frame 0) whose digests differ."""

import json
import os
import sys

import numpy as np

a, b = sys.argv[1:3]
na = json.load(open(os.path.join(a, "nodes.json")))
nb = json.load(open(os.path.join(b, "nodes.json")))
shown = 0
for i, (x, y) in enumerate(zip(na, nb)):
    if x["sha256"] != y["sha256"]:
        line = f"{i:4d} {x['op']:22s} {x['node'][:60]:60s} {x['shape']}"
        fa, fb = os.path.join(a, f"node-{i:04d}.npy"), os.path.join(b, f"node-{i:04d}.npy")
        if os.path.exists(fa) and os.path.exists(fb):
            va, vb = np.load(fa).astype(np.float64), np.load(fb).astype(np.float64)
            d = np.abs(va - vb)
            line += f" max|d| {d.max():.3e} rel {d.max() / max(np.abs(va).max(), 1e-30):.3e} differing {int((d > 0).sum())}/{d.size}"
        print(line)
        shown += 1
        if shown >= 12:
            break
print(f"{len(na)} nodes compared; first {shown} differing shown")
