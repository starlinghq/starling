"""Copy conformance results into docs/evidence/browser-e5-package-<date>/ without the bulky arrays.

Keeps statistics, load stages, latency samples, runtime info, downloads per path, memory and console messages; drops
the per-case embeddings and tokenizer outputs (evaluate_e5_browser.py's report keeps every bound and statistic
computed from them) and JavaScript stack frames.

Usage:
    python collect_evidence.py --evidence docs/evidence/browser-e5-package-2026-10-23 --report report.json \
        [--extra load-average.txt ...] result1.json result2.json ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

BULKY = ("embeddings", "batch8", "generator_embeddings", "tokenizer", "e5_tokenizer")


def slim(result: dict) -> dict:
    run = result.get("run")
    if isinstance(run, dict):
        for key in BULKY:
            if key in run:
                run[key] = f"<{len(run[key])} entries, removed>"
    result["console"] = [
        "\n".join(line for line in entry.split("\n") if not line.lstrip().startswith("at "))
        for entry in result.get("console", [])
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    runs = args.evidence / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for path in args.results:
        result = slim(json.loads(path.read_text(encoding="utf-8")))
        (runs / path.name).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copyfile(args.report, args.evidence / "report.json")
    for extra in args.extra:
        shutil.copyfile(extra, args.evidence / extra.name)
    print(f"wrote {len(args.results)} runs to {runs}")


if __name__ == "__main__":
    main()
