"""EmbeddingGemma 2 image and video retrieval set selection: the filters and the dataset build (no network)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from model_pipeline import evaluate_task_quality as tq

SCRIPT = Path(__file__).resolve().parents[1] / "reference" / "embeddinggemma2" / "select_eg2_retrieval.py"


def _module():
    spec = importlib.util.spec_from_file_location("select_eg2_retrieval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib imports only
    return module


sel = _module()


def page(
    title="File:A.jpg", *, license="pd", mime="image/jpeg", size=1000, width=100, height=100, seconds=None, desc=""
):
    meta = {
        "License": {"value": license},
        "LicenseShortName": {"value": license},
        "ImageDescription": {"value": desc or "A <b>red</b> fox &amp; a hound in the snow."},
    }
    info = {
        "descriptionurl": "https://commons.wikimedia.org/wiki/" + title,
        "timestamp": "2020-01-01T00:00:00Z",
        "url": "https://upload.wikimedia.org/x/" + title[5:],
        "sha1": "0" * 40,
        "size": size,
        "width": width,
        "height": height,
        "mime": mime,
        "extmetadata": meta,
        "metadata": [] if seconds is None else [{"name": "playtime_seconds", "value": seconds}],
    }
    return {"title": title, "imageinfo": [info]}


def test_query_text_cleans_and_cuts():
    assert sel.query_text("A <b>red</b> fox &amp; a hound.") == "A red fox & a hound."
    assert sel.query_text("&lt;a href=x&gt;link&lt;/a&gt; to the photo of a fox") == "link to the photo of a fox"
    assert sel.query_text("too short") is None
    long = " ".join(["word"] * 100)
    cut = sel.query_text(long)
    assert len(cut) <= sel.MAX_QUERY and long.startswith(cut) and not cut.endswith(" ")


def test_record_filters():
    assert sel.record(page(), "image")["description"] == "A red fox & a hound in the snow."
    assert sel.record(page(license="cc-by-sa-4.0"), "image") is None
    assert sel.record(page(license="cc-by-4.0"), "image") is not None
    assert sel.record(page(mime="image/gif"), "image") is None
    assert sel.record(page(size=sel.MAX_BYTES["image"] + 1), "image") is None
    assert sel.record(page(width=10_000, height=5_001), "image") is None
    assert sel.record(page(desc="short"), "image") is None
    video = {"title": "File:V.webm", "mime": "video/webm"}
    assert sel.record(page(**video, seconds="30.5"), "video")["duration_s"] == 30.5
    assert sel.record(page(**video, seconds="61"), "video") is None
    assert sel.record(page(**video), "video") is None
    assert sel.record(page("File:V.ogv", mime="application/ogg", seconds="10"), "video") is None


def test_build_writes_an_evaluable_media_set(tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    selected, entries = [], []
    for i, desc in enumerate(["A red fox in the snow at dawn.", "A lighthouse on a rocky coast."]):
        rec = sel.record(page(f"File:F{i}.jpg", desc=desc), "image")
        selected.append(rec)
        data = f"bytes {i}".encode()
        (media_dir / f"commons-image-{i:02d}.jpg").write_bytes(data)
        entries.append(
            {
                "id": f"commons-image-{i:02d}",
                "file": f"commons-image-{i:02d}.jpg",
                "title": rec["title"],
                "source": rec["description_url"],
                "url": rec["url"],
                "revision_timestamp": rec["revision_timestamp"],
                "sha1": rec["sha1"],
                "bytes": len(data),
                "license": "Public domain",
                "license_page_templates": ["PD-old"],
                "artist": "someone",
                "credit": "",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    media_list = tmp_path / "image-media.json"
    media_list.write_text(
        json.dumps({"seed": 1, "queried": "2026-10-15T00:00:00Z", "images": {"selected": selected}}), encoding="utf-8"
    )
    manifest = {"media_list_sha256": hashlib.sha256(media_list.read_bytes()).hexdigest(), "images": entries}
    (media_dir / "images.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "set"
    args = argparse.Namespace(media=media_list, fetched=media_dir, out=out, name="test-set")
    assert sel.build(args) == 0
    data = tq.load_retrieval_set(out)
    assert data.modality == "image"
    assert data.queries == {
        "q-commons-image-00": "A red fox in the snow at dawn.",
        "q-commons-image-01": "A lighthouse on a rocky coast.",
    }
    assert data.qrels == {"q-commons-image-00": {"commons-image-00": 1}, "q-commons-image-01": {"commons-image-01": 1}}
    # A media file that no longer matches its manifest is refused.
    (media_dir / "commons-image-01.jpg").write_bytes(b"other")
    with pytest.raises(SystemExit, match="SHA-256"):
        sel.build(args)
