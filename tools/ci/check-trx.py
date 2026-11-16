"""Fail when a test module ran but passed nothing: every test skipped, so a green exit code proves nothing.

`dotnet test` exits 0 when every test of a module skips (a model, fixture or FFmpeg build that is absent makes the
model tests skip with the reason). This reads the TRX reports (`-- --report-xunit-trx`) under a results directory,
one report per test module, and prints per module the passed, skipped (not executed) and failed counts. It exits 1
when any module that has tests reports 0 passed, unless the module is on the allow-list, and 2 when the directory holds
no TRX report at all (the run did not happen). Failed tests are reported too, but `dotnet test`'s own exit status
already fails a run with a failed test.

The module is the file name of the test assembly (`TestMethod/@codeBase`, without `.dll`); TRX file names carry no
module name. When a directory holds several reports of one module (a rerun), the counts are added.

Allow-list (`--allow`, default `tools/ci/trx-allow-zero.txt`): one module per line, then whitespace and the reason
it may legitimately pass nothing, for example a module whose every test needs a model that is privately hosted.
Blank lines and lines starting with `#` are ignored. A line without a reason is an error, so every exemption is
explained where it is made.

    python3 -I tools/ci/check-trx.py <results dir> [--allow <file>] [--summary <markdown file to append to>]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
DEFAULT_ALLOW = Path(__file__).resolve().parent / "trx-allow-zero.txt"


@dataclass
class Counts:
    total: int = 0
    passed: int = 0
    skipped: int = 0
    failed: int = 0
    reports: int = 0


def _int(element: ET.Element, name: str) -> int:
    try:
        return int(element.get(name, "0"))
    except ValueError:
        return 0


def module_of(root: ET.Element, report: Path) -> str:
    """The test assembly's file name without `.dll`; the report's file name when the TRX has no test definition."""
    method = root.find(".//t:TestDefinitions/t:UnitTest/t:TestMethod", NS)
    code_base = method.get("codeBase", "") if method is not None else ""
    if not code_base:
        return f"(no test definitions: {report.name})"
    name = code_base.replace("\\", "/").rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".dll") else name


def read_report(report: Path) -> tuple[str, Counts]:
    root = ET.parse(report).getroot()
    counts = Counts(reports=1)
    counters = root.find("t:ResultSummary/t:Counters", NS)
    if counters is not None:
        counts.total = _int(counters, "total")
        counts.passed = _int(counters, "passed")
        counts.skipped = _int(counters, "notExecuted")
        counts.failed = sum(_int(counters, n) for n in ("failed", "error", "timeout", "aborted"))
    else:
        # No summary: count the results themselves.
        for result in root.iterfind(".//t:Results/t:UnitTestResult", NS):
            outcome = result.get("outcome", "")
            counts.total += 1
            if outcome == "Passed":
                counts.passed += 1
            elif outcome == "NotExecuted":
                counts.skipped += 1
            else:
                counts.failed += 1
    return module_of(root, report), counts


def read_allow(path: Path | None) -> dict[str, str]:
    """Module -> reason. Raises ValueError on a line without a reason."""
    allow: dict[str, str] = {}
    if path is None or not path.is_file():
        return allow
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError(f"{path}:{number}: '{parts[0]}' has no reason; write '<module> <reason>'")
        allow[parts[0]] = parts[1].strip()
    return allow


def check(results: Path, allow: dict[str, str]) -> tuple[int, list[str]]:
    """Exit status and the report lines (a markdown table)."""
    reports = sorted(results.rglob("*.trx")) if results.is_dir() else []
    if not reports:
        return 2, [f"check-trx: no TRX report under {results}: the tests did not run"]
    modules: dict[str, Counts] = {}
    unreadable: list[str] = []
    for report in reports:
        try:
            name, counts = read_report(report)
        except ET.ParseError as error:
            unreadable.append(f"{report}: {error}")
            continue
        total = modules.setdefault(name, Counts())
        total.total += counts.total
        total.passed += counts.passed
        total.skipped += counts.skipped
        total.failed += counts.failed
        total.reports += counts.reports
    lines = ["| Module | Passed | Skipped | Failed | Verdict |", "| --- | ---: | ---: | ---: | --- |"]
    bad: list[str] = []
    for name in sorted(modules):
        c = modules[name]
        if c.total == 0:
            verdict = "no tests"
        elif c.failed:
            verdict = "FAILED (failed tests)"
            bad.append(name)
        elif c.passed == 0 and name in allow:
            verdict = f"allowed to pass nothing: {allow[name]}"
        elif c.passed == 0:
            verdict = "FAILED (0 passed: every test skipped)"
            bad.append(name)
        else:
            verdict = "ok"
        lines.append(f"| `{name}` | {c.passed} | {c.skipped} | {c.failed} | {verdict} |")
    for entry in unreadable:
        lines.append(f"\ncheck-trx: unreadable TRX report {entry}")
    unused = sorted(set(allow) - set(modules))
    if unused:
        lines.append(f"\ncheck-trx: allow-list entries with no report here (not an error): {', '.join(unused)}")
    status = 1 if bad or unreadable else 0
    if bad:
        lines.append(f"\ncheck-trx: {len(bad)} module(s) failed: {', '.join(bad)}")
    else:
        lines.append(f"\ncheck-trx: {len(modules)} module(s), every module with tests passed at least one test")
    return status, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="directory searched recursively for *.trx")
    parser.add_argument("--allow", type=Path, default=DEFAULT_ALLOW, help="allow-list file (default %(default)s)")
    parser.add_argument("--summary", type=Path, help="also append the table to this markdown file")
    args = parser.parse_args(argv)
    try:
        allow = read_allow(args.allow)
    except ValueError as error:
        print(f"check-trx: {error}", file=sys.stderr)
        return 2
    status, lines = check(args.results, allow)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as out:
            out.write("\n" + text)
    return status


if __name__ == "__main__":
    sys.exit(main())
