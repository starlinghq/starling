"""Python text semantics GLiNER2 depends on, as tables for the C# unit tests (backlog P-015).

GLiNER2's word splitter is a Python `re` pattern (`WhitespaceTokenSplitter`, IGNORECASE) and words are lower-cased
with `str.lower()`. This script records, for every Unicode scalar value, what the reference interpreter does:

    space               re `\\s` (str pattern)
    word                re `\\w` (str pattern)
    az_ignorecase       re `[a-z]` with IGNORECASE, beyond ASCII letters
    literal_ignorecase  the letters of `https://` and `www.` with IGNORECASE, beyond ASCII
    lower               `str.lower()` where it changes the character
    sigma_ignorable     skipped by `str.lower()`'s Final_Sigma context check (Case_Ignorable)
    sigma_cased         not skipped and counts as cased in that check
    unassigned          general category Cn in this interpreter's Unicode database

Run it with the reference interpreter (the P-014/P-015 environment, CPython 3.11, Unicode 14.0.0):

    python make_python_text_tables.py > python311-text-semantics.json
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata

SCALARS = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]


def ranges(predicate) -> list[list[int]]:
    out: list[list[int]] = []
    for c in SCALARS:
        if predicate(c):
            if out and out[-1][1] == c - 1:
                out[-1][1] = c
            else:
                out.append([c, c])
    return out


def sigma_last(s: str) -> bool:
    """Final form for a trailing capital sigma (the lower-cased prefix may change length, the sigma stays last)."""
    return s.lower()[-1] == "ς"


def sigma_second(x: str) -> bool:
    """Final form for "AΣ" followed by x."""
    return ("AΣ" + x).lower()[1] == "ς"


def main() -> int:
    space = re.compile(r"\s")
    word = re.compile(r"\w")
    az = re.compile(r"[a-z]", re.IGNORECASE)
    letters = sorted(set("https://www."))
    tables = {
        "generator": "ai-dotnet/tests/Starlings.Models.Gliner2.Tests/Data/make_python_text_tables.py",
        "python": sys.version.split()[0],
        "unicode": unicodedata.unidata_version,
        "space": ranges(lambda c: space.fullmatch(chr(c)) is not None),
        "word": ranges(lambda c: word.fullmatch(chr(c)) is not None),
        "az_ignorecase": [c for c in SCALARS if c >= 0x80 and az.fullmatch(chr(c))],
        "literal_ignorecase": {
            ch: [c for c in SCALARS if c >= 0x80 and re.fullmatch(re.escape(ch), chr(c), re.IGNORECASE)]
            for ch in letters
        },
        "lower": [[c, [ord(x) for x in chr(c).lower()]] for c in SCALARS if chr(c).lower() != chr(c)],
        # "xΣ": final iff x is cased and not ignorable. "AxΣ": final iff x is ignorable or cased.
        # "AΣx": final iff x is ignorable or not cased. Together they separate the three behaviours that matter.
        "sigma_ignorable": ranges(
            lambda c: c != 0x3A3
            and not sigma_last(chr(c) + "Σ")
            and sigma_last("A" + chr(c) + "Σ")
            and sigma_second(chr(c))
        ),
        "sigma_cased": ranges(lambda c: c != 0x3A3 and sigma_last(chr(c) + "Σ")),
        "unassigned": ranges(lambda c: unicodedata.category(chr(c)) == "Cn"),
    }
    json.dump(tables, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
