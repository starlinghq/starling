"""Per run: durations (s) and start offsets of selected Qwen35.Tests tests, from TRX files under dl/<run>/qualification-model-tests."""

import glob
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

ns = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
names = [
    "OperationalTests.TimedOutRequestsReportCancellationLatencyThenReuse",
    "CpuModelTests.CancellationBetweenStepsThenReuse",
    "CpuModelTests.CachedDecodingEqualsAFreshPrefill",
    "CpuModelTests.StreamingReassemblesNonAsciiOutput",
    "CpuModelTests.ConcurrentRequestsAreIndependent",
    "OperationalTests.SaturatedRequestsAllCompleteWithTheirSingleRequestIds",
]
short = ["TimedOut", "CancelSteps", "Cached", "Streaming", "Concurrent", "Saturated"]


def dur(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


base = sys.argv[1]
print("run          " + " ".join(f"{s:>16s}" for s in short))
for run in sorted(os.listdir(base)):
    found = {}
    first = None
    for path in glob.glob(f"{base}/{run}/qualification-model-tests/*.trx"):
        root = ET.parse(path).getroot()
        for r in root.iterfind(".//t:UnitTestResult", ns):
            n = r.get("testName")
            if ".Qwen35.Tests." not in n:
                continue
            st = datetime.fromisoformat(r.get("startTime")[:26])
            first = st if first is None or st < first else first
            for full, s in zip(names, short):
                if n.endswith(full):
                    found[s] = (dur(r.get("duration")), st, r.get("outcome")[0])
    if not found:
        continue
    cells = []
    for s in short:
        if s in found:
            d, st, o = found[s]
            cells.append(f"{d:6.1f}s@{(st - first).total_seconds():5.0f}{o}")
        else:
            cells.append(" " * 16)
    print(f"{run} " + " ".join(f"{c:>16s}" for c in cells))
