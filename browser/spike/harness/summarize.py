"""One line per (variant, backend) from a `drift_e5_browser.py browser` output file."""

from __future__ import annotations

import json
import sys


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as stream:
        rows = json.load(stream)
    for d in rows:
        load = d["load"]
        drift = d.get("drift_vs_reference", {})
        js = {k: round(v, 1) for k, v in d.get("latency_js_median_ms", {}).items()}
        e2e = {k: round(v, 1) for k, v in d.get("latency_e2e_median_ms", {}).items()}
        ua = (d.get("memory") or {}).get("ua_specific_bytes", 0)
        print(
            d["variant"],
            d["backend"],
            d.get("threads"),
            f"fetch {load.get('fetch_ms', 0):.0f} create {load.get('create_ms', 0):.0f}",
            f"drift {drift.get('max_abs', -1):.3g} {drift.get('min_cosine', -1):.6f}",
            f"b8 {d.get('batch8_vs_single_max_abs', -1):.3g}",
            f"first {d.get('first_run_ms') or 0:.0f}",
            js,
            e2e,
            f"rssR {d['peak_rss_renderer_bytes'] / 1e6:.0f}",
            f"rssT {d['peak_rss_tree_bytes'] / 1e6:.0f} ua {ua / 1e6:.0f}",
            load.get("error", ""),
            load.get("adapter", ""),
        )


if __name__ == "__main__":
    main()
