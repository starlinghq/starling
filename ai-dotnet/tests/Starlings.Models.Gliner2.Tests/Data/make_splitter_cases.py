"""Adversarial inputs for GLiNER2's word splitter, with the reference output (backlog P-015).

The P-014 corpus did not cover adversarial splitter input (very long URLs, `@` runs, e-mail backtracking, IGNORECASE
folds, astral letters, combining marks, Python-only whitespace). This script runs GLiNER2's own
`WhitespaceTokenSplitter` (GLiNER2 source 55656fbf, installed from `model-pipeline/reference/gliner2/requirements.txt`)
on the strings below, after GLiNER2's text normalization, and records the words and offsets in Unicode scalar values:

    python make_splitter_cases.py > splitter-cases.json
"""

from __future__ import annotations

import json
import sys

from gliner2.processing.word_splitter import WhitespaceTokenSplitter

CASES = {
    "email-backtracking": "Mail a.b-c@x.y.co.uk, x@y.z, x@y.zz, x@.co, a@b.c.de-f and x@y-z.co9",
    "email-ignorecase-folds": "ſam@KELVIN.ſe İnfo@ıdea.com and K@x.io",
    "url-prefix-folds": "httpſ://x.org/p?q=1 HTTPS://A.B WwW.example.com www. http:// https://x",
    "url-long": "see https://example.com/" + "a/b-c_d" * 300 + "?q=1#frag, then stop",
    "handle-runs": "@@user_1 @ @@ @_x @-y @1 end",
    "hyphen-underscore": "state-of-the-art foo_bar -x x- a--b a-_b __init__ 3-4 x-𝐀",
    "astral-letters": "𝐀𝐁𝐂 𝟏𝟐 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 😀 🇫🇷 𐐀𐐨",
    "combining-marks": "école café ñ a⃝ ẍy",
    "python-whitespace": "a\u001cb\u001dc\u001ed\u001fe\u0085f g h　i​j᠎k",
    "numbers-and-scripts": "Ⅻ ½ ² ٣ ۴ 五 Ωmega ΟΔΟΣ ΣΑΣ ﬁne ǅemal İstanbul",
    "punctuation": "...!?()[]{}<>\"'`~#$%^&*=+|\\/:;,",
    "ends-exclamation": "Wow!",
    "ends-space": "Boston ",
    "only-space": "   ",
    "empty": "",
}


def normalize(text: str) -> str:
    """GLiNER2 processor.py:537."""
    if not text:
        return "."
    return text if text[-1] in ".!?" else text + "."


def main() -> int:
    splitter = WhitespaceTokenSplitter()
    rows = []
    for case_id, text in CASES.items():
        norm = normalize(text)
        words = [{"word": w, "start": s, "end": e} for w, s, e in splitter(norm)]
        rows.append({"id": case_id, "text": text, "normalized_text": norm, "words": words})
    json.dump(
        {
            "generator": "ai-dotnet/tests/Starlings.Models.Gliner2.Tests/Data/make_splitter_cases.py",
            "python": sys.version.split()[0],
            "cases": rows,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=1,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
