"""cmp_soft.py <ref npz> name=<npy or stages-dir:row> ... : soft-token row-relative differences, pairwise vs the first
and vs the reference."""

import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from textgraph_diag import sidecar  # noqa: E402

ref = np.load(sys.argv[1])["video0_soft_tokens"]
arrs = {}
for spec in sys.argv[2:]:
    name, src = spec.split("=", 1)
    if src.endswith(".npy"):
        arrs[name] = np.load(src)
    else:
        d, row = src.rsplit(":", 1)
        arrs[name] = sidecar(d, "soft_tokens", row)
mx = np.abs(ref).max()
names = list(arrs)
for k in names:
    x = arrs[k]
    print(f"{k:24s} vs ref {np.abs(x.astype(np.float64) - ref[: len(x)]).max() / mx:.3e}", end="")
    for j in names:
        if j != k:
            y = arrs[j]
            nn = min(len(x), len(y))
            print(f"  vs {j} {np.abs(x[:nn].astype(np.float64) - y[:nn]).max() / mx:.3e}", end="")
    print()
