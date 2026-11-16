"""Tests of tools/ci/check-trx.py: per-module counts from TRX reports, the 0-passed rule and the allow-list.

The TRX reports are written into temporary directories in the shape `dotnet test -- --report-xunit-trx` produces
(Microsoft.Testing.Platform's xUnit TRX: TestDefinitions/UnitTest/TestMethod/@codeBase names the test assembly,
ResultSummary/Counters holds the counts).

    python3 -I -m unittest discover -s tools/tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "check-trx.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_trx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_trx = _load()


def _trx(module: str, passed: int, skipped: int, failed: int = 0, summary: bool = True) -> str:
    dll = f"/work/ai-dotnet/tests/{module}/bin/Release/net10.0/{module}.dll"
    outcomes = ["Passed"] * passed + ["NotExecuted"] * skipped + ["Failed"] * failed
    results = "".join(
        f'<UnitTestResult testName="{module}.T.Case{i}" outcome="{o}" testId="id{i}" executionId="id{i}" />'
        for i, o in enumerate(outcomes)
    )
    definitions = "".join(
        f'<UnitTest name="{module}.T.Case{i}" id="id{i}" storage="{dll}"><Execution id="id{i}" />'
        f'<TestMethod codeBase="{dll}" className="{module}.T" name="Case{i}" /></UnitTest>'
        for i in range(len(outcomes))
    )
    counters = (
        f'<ResultSummary outcome="Completed"><Counters total="{len(outcomes)}" executed="{passed + failed}" '
        f'passed="{passed}" failed="{failed}" error="0" timeout="0" aborted="0" notExecuted="{skipped}" />'
        "</ResultSummary>"
        if summary
        else ""
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<TestRun id="run" name="fixture" xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">'
        f"<Results>{results}</Results><TestDefinitions>{definitions}</TestDefinitions>{counters}</TestRun>"
    )


class CheckTrxTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.results = self.root / "results"
        self.results.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            status = check_trx.main([str(self.results), "--allow", str(self.root / "none.txt"), *args])
        return status, out.getvalue()

    def test_every_module_with_a_pass_is_ok(self):
        self.write("a.trx", _trx("Fixture.Unit.Tests", 12, 0))
        self.write("b.trx", _trx("Fixture.Models.Tests", 3, 9))
        status, out = self.run_main()
        self.assertEqual(status, 0, out)
        self.assertIn("| `Fixture.Models.Tests` | 3 | 9 | 0 | ok |", out)

    def test_a_module_whose_every_test_skipped_fails(self):
        self.write("a.trx", _trx("Fixture.Unit.Tests", 12, 0))
        self.write("b.trx", _trx("Fixture.Models.Tests", 0, 9))
        status, out = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("FAILED (0 passed: every test skipped)", out)
        self.assertIn("1 module(s) failed: Fixture.Models.Tests", out)

    def test_the_allow_list_exempts_a_module_with_its_reason(self):
        self.write("b.trx", _trx("Fixture.Models.Tests", 0, 9))
        allow = self.root / "allow.txt"
        allow.write_text("# comment\n\nFixture.Models.Tests   every test needs a privately hosted model\n", encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = check_trx.main([str(self.results), "--allow", str(allow)])
        self.assertEqual(status, 0, out.getvalue())
        self.assertIn("allowed to pass nothing: every test needs a privately hosted model", out.getvalue())

    def test_an_allow_list_line_without_a_reason_is_refused(self):
        self.write("b.trx", _trx("Fixture.Models.Tests", 0, 9))
        allow = self.root / "allow.txt"
        allow.write_text("Fixture.Models.Tests\n", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            status = check_trx.main([str(self.results), "--allow", str(allow)])
        self.assertEqual(status, 2)
        self.assertIn("has no reason", err.getvalue())

    def test_failed_tests_fail_the_module(self):
        self.write("a.trx", _trx("Fixture.Unit.Tests", 5, 0, failed=1))
        status, out = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("FAILED (failed tests)", out)

    def test_no_report_is_its_own_status(self):
        status, out = self.run_main()
        self.assertEqual(status, 2)
        self.assertIn("no TRX report", out)

    def test_reports_of_one_module_are_added_and_nested_directories_are_searched(self):
        self.write("E5/a.trx", _trx("Fixture.Models.Tests", 0, 4))
        self.write("E5/rerun/b.trx", _trx("Fixture.Models.Tests", 2, 2))
        status, out = self.run_main()
        self.assertEqual(status, 0, out)
        self.assertIn("| `Fixture.Models.Tests` | 2 | 6 | 0 | ok |", out)

    def test_results_are_counted_when_the_summary_is_missing(self):
        self.write("a.trx", _trx("Fixture.Models.Tests", 0, 3, summary=False))
        status, out = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("| `Fixture.Models.Tests` | 0 | 3 | 0 |", out)

    def test_the_summary_file_gets_the_table(self):
        self.write("a.trx", _trx("Fixture.Unit.Tests", 1, 0))
        summary = self.root / "summary.md"
        status, _ = self.run_main("--summary", str(summary))
        self.assertEqual(status, 0)
        self.assertIn("| `Fixture.Unit.Tests` | 1 | 0 | 0 | ok |", summary.read_text(encoding="utf-8"))

    def test_the_checked_in_allow_list_parses(self):
        check_trx.read_allow(check_trx.DEFAULT_ALLOW)


if __name__ == "__main__":
    unittest.main()
