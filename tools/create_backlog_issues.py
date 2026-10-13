#!/usr/bin/env python3
"""Turn the plan's section 18 backlog table into GitHub issues.

The plan document is the single source for the backlog. This script parses
its table and prints the issues it would create. Pass --create to open them
with the GitHub CLI (`gh`), which must be installed and authenticated.

    tools/create_backlog_issues.py                 # dry run
    tools/create_backlog_issues.py --create --repo starlinghq/starling
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent.parent / "docs" / "plan" / "fast-faithful-ai-dotnet.md"
ROW = re.compile(r"^\| (P-\d{3}) \| (.+?) \| (.+?) \| (.+?) \|$")

# Lane labels by backlog ID. IDs not listed get only the "backlog" label.
LANES = {
    "tokenizers": {"P-004", "P-023", "P-024", "P-025", "P-026", "P-029"},
    "pipeline": {"P-002", "P-003", "P-007", "P-008", "P-009", "P-018", "P-020", "P-021", "P-022", "P-030"},
    "runtime": {"P-005", "P-006", "P-010", "P-019"},
    "integration": {"P-011", "P-012", "P-013", "P-017"},
    "model:gliner2": {"P-014", "P-015"},
    "model:llm": {"P-016", "P-017", "P-029"},
    "model:kev": {"P-027"},
    "governance": {"P-001", "P-028"},
}


def parse_backlog(text: str) -> list[dict[str, str]]:
    section = text.split("## 18. Initial issue backlog", 1)[1].split("\n## ", 1)[0]
    items = []
    for line in section.splitlines():
        match = ROW.match(line.strip())
        if match:
            pid, title, deps, accept = (g.strip() for g in match.groups())
            items.append({"id": pid, "title": title, "depends": deps, "acceptance": accept})
    return items


def labels_for(pid: str) -> list[str]:
    return ["backlog"] + sorted(name for name, ids in LANES.items() if pid in ids)


def body_for(item: dict[str, str]) -> str:
    return (
        f"Backlog item **{item['id']}** from `docs/plan/fast-faithful-ai-dotnet.md` section 18.\n\n"
        f"**Depends on:** {item['depends']}\n\n"
        f"**Acceptance criteria:** {item['acceptance']}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--create", action="store_true", help="create issues with gh instead of printing them")
    parser.add_argument("--repo", default="starlinghq/starling")
    args = parser.parse_args()

    items = parse_backlog(PLAN.read_text(encoding="utf-8"))
    if len(items) != 30:
        print(f"expected 30 backlog items, parsed {len(items)}", file=sys.stderr)
        return 1

    if args.create:
        for label in ["backlog", *LANES]:
            subprocess.run(["gh", "label", "create", label, "--repo", args.repo, "--force"], check=True)

    for item in items:
        title = f"{item['id']}: {item['title']}"
        labels = labels_for(item["id"])
        if not args.create:
            print(f"{title}\n  labels: {', '.join(labels)}\n  depends: {item['depends']}")
            continue
        cmd = ["gh", "issue", "create", "--repo", args.repo, "--title", title, "--body", body_for(item)]
        for label in labels:
            cmd += ["--label", label]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
