"""GLiNER2 extraction cases for two capabilities the accepted corpus does not cover: `count_gate_closed` and
`label_descriptions`.

The accepted corpus (`model-pipeline/reference/corpus/gliner2-extraction-corpus-v1.jsonl`) has one description case
and no case where GLiNER2's count gate closes (`count_pred` argmax 0: no entities for the whole schema). This script
writes cases in the same format (id, category, text, labels, threshold, optional descriptions) from public-domain
text only: paragraphs of two Project Gutenberg files pinned by SHA-256 (the files of
`model-pipeline/proposed/embeddinggemma-2/long-input/texts.json`: Alice's Adventures in Wonderland, eBook #11; the
United States Constitution, eBook #5), whitespace collapsed. Labels and descriptions are this repository's.

The gate-closed cases pair a paragraph with labels it does not contain. They were chosen from a scan of 230
paragraphs x 8 label sets with the pinned GLiNER2 (586 of 1,840 closed); each case kept here closed with a margin
of at least 1 between count logit 0 and the largest other logit. Whether the gate closes is GLiNER2's output, recorded
by `run_gliner2_extraction.py`, not an assumption of this script.

Usage:
    python make_gliner2_capability_cases.py --sources <dir with pg11.txt and pg5.txt> --out <jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    "alice": (
        "pg11.txt",
        "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
        "\nCHAPTER I.\n",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
    ),
    "constitution": (
        "pg5.txt",
        "d177168e3f4898213dced30fbbe60d74ad08e884ed5dd86bff94a1b8836b5982",
        "THE CONSTITUTION OF THE UNITED STATES OF AMERICA, 1787",
        "William Jackson, Secretary",
    ),
}

# (id, source, paragraph prefix, labels, descriptions or None)
GATE_CLOSED = [
    (
        "gate-closed-chemical-president",
        "constitution",
        "Section 3. He shall from time to time",
        ["chemical compound"],
        None,
    ),
    ("gate-closed-protein-gene-labor", "constitution", "No person held to Service or Labor", ["protein", "gene"], None),
    (
        "gate-closed-contact-new-states",
        "constitution",
        "Section 3. New States may be admitted",
        ["email address", "phone number", "url"],
        None,
    ),
    (
        "gate-closed-software-senator",
        "constitution",
        "No Senator or Representative shall, during the Time",
        ["programming language", "software library"],
        None,
    ),
    (
        "gate-closed-medical-impeachment",
        "constitution",
        "Section 4. The President, Vice President and all civil",
        ["disease", "medication", "symptom"],
        None,
    ),
    (
        "gate-closed-finance-rules",
        "constitution",
        "Each house may determine the Rules of its Proceedings",
        ["stock ticker", "currency amount"],
        None,
    ),
    ("gate-closed-spacecraft-citizens", "constitution", "Section 2. The Citizens of each State", ["spacecraft"], None),
    ("gate-closed-medical-falling", "alice", "Presently she began again.", ["disease", "medication", "symptom"], None),
    (
        "gate-closed-contact-party",
        "alice",
        "They were indeed a queer-looking party",
        ["email address", "phone number", "url"],
        None,
    ),
    ("gate-closed-protein-gene-fury", "alice", "“Fury said to a mouse,", ["protein", "gene"], None),
    (
        "gate-closed-finance-mouse",
        "alice",
        "“Would it be of any use, now,” thought Alice",
        ["stock ticker", "currency amount"],
        None,
    ),
    (
        "gate-closed-description-protein",
        "constitution",
        "No person held to Service or Labor",
        ["protein"],
        {"protein": "a biological macromolecule made of amino acids"},
    ),
]

DESCRIPTIONS = [
    (
        "descriptions-party",
        "alice",
        "They were indeed a queer-looking party",
        ["person", "animal"],
        {"person": "a named human character", "animal": "a kind of animal, such as a bird or a mouse"},
    ),
    (
        "descriptions-falling",
        "alice",
        "Presently she began again.",
        ["person", "location"],
        {"person": "a named human being", "location": "a country, city or other geographic place"},
    ),
    (
        "descriptions-caterpillar",
        "alice",
        "“You!” said the Caterpillar contemptuously.",
        ["character"],
        {"character": "a named character in the story"},
    ),
    (
        "descriptions-mouse",
        "alice",
        "“Would it be of any use, now,” thought Alice",
        ["person", "animal", "language"],
        {"person": "a named human being", "animal": "an animal", "language": "a human language"},
    ),
    (
        "descriptions-opening",
        "alice",
        "Alice was beginning to get very tired",
        ["person", "object"],
        {"person": "a named human being", "object": "a physical thing that can be held"},
    ),
    (
        "descriptions-president",
        "constitution",
        "Section 3. He shall from time to time",
        ["government body", "official title"],
        {
            "government body": "a legislative, executive or judicial institution",
            "official title": "the title of a public office",
        },
    ),
    (
        "descriptions-impeachment",
        "constitution",
        "Section 4. The President, Vice President and all civil",
        ["official title", "crime"],
        {"official title": "the title of a public office", "crime": "an offence punishable by law"},
    ),
    (
        "descriptions-partial-senator",
        "constitution",
        "No Senator or Representative shall, during the Time",
        ["government body", "official title", "time period"],
        {
            "government body": "a legislative, executive or judicial institution",
            "official title": "the title of a public office",
        },
    ),
    (
        "descriptions-citizens",
        "constitution",
        "Section 2. The Citizens of each State",
        ["legal right", "government body"],
        {
            "legal right": "a privilege or immunity guaranteed by law",
            "government body": "a legislative, executive or judicial institution",
        },
    ),
    (
        "descriptions-new-states",
        "constitution",
        "Section 3. New States may be admitted",
        ["government body", "location"],
        {
            "government body": "a legislative, executive or judicial institution",
            "location": "a state, territory or other geographic place",
        },
    ),
]


def paragraphs(sources: Path) -> dict[str, list[str]]:
    out = {}
    for key, (name, want, start, end) in SOURCES.items():
        data = (sources / name).read_bytes()
        have = hashlib.sha256(data).hexdigest()
        if have != want:
            raise SystemExit(f"{name}: SHA-256 {have}, expected {want}")
        text = data.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
        text = text[text.index(start) : text.index(end)]
        out[key] = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paras = paragraphs(args.sources)
    rows = []
    for category, cases in (("count-gate-closed", GATE_CLOSED), ("descriptions", DESCRIPTIONS)):
        for case_id, source, prefix, labels, descriptions in cases:
            matches = [p for p in paras[source] if p.startswith(prefix)]
            if len(matches) != 1:
                raise SystemExit(f"{case_id}: {len(matches)} paragraphs start with {prefix!r}")
            row = {"id": case_id, "category": category, "text": matches[0], "labels": labels, "threshold": 0.5}
            if descriptions:
                row["descriptions"] = descriptions
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
