"""Summarize an onboarding run's evaluation report for a pull request's Evidence section.

Usage: python3 skills/starling-add-model/summarize_report.py artifacts/onboarding/<name>

Groups the evaluator's checks by comparison and prints, for each, the number of checks, the largest
absolute difference, its bound and whether all passed; then counts the exact (non-numeric) checks.
"""

import collections
import glob
import json
import sys
from pathlib import Path

reports = glob.glob(sys.argv[1] + "/stages/evaluate/*/output/report.json")
if len(reports) != 1:
    sys.exit(f"expected one evaluate report under {sys.argv[1]}/stages/evaluate/, found {len(reports)}")
report = json.loads(Path(reports[0]).read_text())
numeric = collections.defaultdict(list)
rates = collections.defaultdict(list)
exact = collections.Counter()
for check in report["checks"]:
    parts = check["id"].split("/")
    key = parts[0] + "/" + parts[-1] if len(parts) >= 3 else check["id"]
    if "max_abs" in check:
        numeric[key].append((check["max_abs"], check["bound"], check["status"]))
    elif "value" in check:
        rates[key].append((check["value"], check["bound"], check["status"]))
    else:
        exact[(key, check.get("status"))] += 1

print("| Comparison | Checks | Max abs difference | Bound | All passed |")
print("| --- | ---: | ---: | ---: | --- |")
for key, rows in sorted(numeric.items()):
    worst = max(rows)
    print(f"| {key} | {len(rows)} | {worst[0]:.3e} | {worst[1]:g} | {all(s == 'passed' for *_, s in rows)} |")
if rates:
    # Bounded rates such as word error rate (ASR): the largest value against its bound.
    print()
    print("| Rate | Checks | Largest value | Bound | All passed |")
    print("| --- | ---: | ---: | ---: | --- |")
    for key, rows in sorted(rates.items()):
        worst = max(rows)
        print(f"| {key} | {len(rows)} | {worst[0]:.4f} | {worst[1]:g} | {all(s == 'passed' for *_, s in rows)} |")
print()
for (key, status), count in sorted(exact.items()):
    print(f"- {key}: {count} exact checks, {status}")
print(f"\nTotal checks {len(report['checks'])}, status {report['status']}, failed {report['failed_checks']}")
