"""Reference runner for google/embeddinggemma-2, long inputs (2,049 to 8,192 tokens and just over). CPU, fp32.

The admission record lists inputs between 2,049 and 8,192 tokens as untested. This runner proposes fixtures for that
range: text only (several lengths up to exactly 8,192, with and without the `query` and `document` prompts),
interleaved text and images, and text with videos at the 32-frame limit, plus inputs just over 8,192 tokens that the
C# package must reject (upstream does not truncate or reject; their upstream embedding is recorded for information).

Text comes from two public-domain Project Gutenberg plain-text files (`--sources`, pinned by SHA-256 below; the
Gutenberg header and footer are dropped) and from the repository's M0 tokenizer corpus. Each case's text is the
longest prefix of its source passage whose rendered input has exactly the target token count, counted by
sentence-transformers' own preprocessing (prompt, chat template, media placeholders expanded); media placeholders are
placed between paragraphs. Images come from the accepted image set (`make_eg2_images.py`, `--images`) and videos from
the accepted video set (`make_eg2_videos.py`, `--videos`), both checked against their pinned SHA-256.

Environment: requirements-video.txt (the requirements.txt environment plus torchcodec 0.17.0 on the pinned LGPL
FFmpeg 8.1 of `fetch_eg2_ffmpeg.sh`, on LD_LIBRARY_PATH), so one environment serves text, image and video rows.

Writes into `--out` (small; fixture candidates): long-embeddings.jsonl (one row per case: text, prompt, media, rendered
ids, per image and video the processor digests and soft-token digests, the fp32 embedding, `expect` embed or reject),
long-reference.json (stability: padded batches vs single rows, eager vs sdpa, a repeated run; tokenizer.json ids vs
the rendered ids of text rows), long-environment.json, texts.json (sources, licenses, cleaning, digests) and
tensors.json. Writes into `--tensors` (large; never committed): one `.npz` per row (token embeddings, soft tokens,
embedding).

Usage (LD_LIBRARY_PATH=<LGPL FFmpeg 8 lib dir>):
    python run_embeddinggemma2_long.py --model-dir <snapshot> --revision <sha> --sources <dir with pg5.txt, pg11.txt>
        --corpus model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl --images <dir> --videos <dir>
        --out <dir> --tensors <dir> [--threads 4]
"""

from __future__ import annotations

import eg2_video

eg2_video.pin_torchcodec_ffmpeg()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchcodec._core as tc_core  # noqa: E402
from eg2_video import VIDEO_TOKEN  # noqa: E402
from PIL import Image  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

from model_pipeline import fixtures  # noqa: E402

REPO = "google/embeddinggemma-2"
LIMIT = 8192
IMAGE_TOKEN = 258880
PROMPTS = {"none": "", "query": "task: search result | query: ", "document": "title: none | text: "}
PACKAGES = ["torch", "torchvision", "torchcodec", "transformers", "sentence-transformers", "tokenizers", "numpy"]
ASSETS = ["config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json", "model.safetensors"]

# Public-domain sources (Project Gutenberg plain text, UTF-8). The files are pinned by SHA-256 as downloaded on
# 2026-10-12; Gutenberg may re-issue them, so the rows carry the exact text used.
SOURCES = {
    "alice": {
        "file": "pg11.txt",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "sha256": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
        "title": "Alice's Adventures in Wonderland (Lewis Carroll, 1865), Project Gutenberg eBook #11",
        "license": "Public domain (author died 1898; Project Gutenberg eBook in the US public domain). "
        "The Project Gutenberg header, footer and license text are removed, as the Gutenberg license allows "
        "for public-domain works.",
        "start": "\nCHAPTER I.\n",
        "end": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "constitution": {
        "file": "pg5.txt",
        "url": "https://www.gutenberg.org/cache/epub/5/pg5.txt",
        "sha256": "d177168e3f4898213dced30fbbe60d74ad08e884ed5dd86bff94a1b8836b5982",
        "title": "The United States Constitution (1787), Project Gutenberg eBook #5",
        "license": "Public domain (a 1787 text; Project Gutenberg eBook in the US public domain). "
        "The Project Gutenberg header, notes, footer and license text are removed.",
        "start": "THE CONSTITUTION OF THE UNITED STATES OF AMERICA, 1787",
        "end_inclusive": "William Jackson, Secretary",
    },
}
IMAGES = {
    "photo-chelsea": "chelsea.png",
    "photo-coffee": "coffee.png",
    "photo-camera": "camera.png",
    "photo-astronaut": "astronaut.png",
    "photo-rocket": "rocket.jpg",
    "photo-hubble-deep-field": "hubble_deep_field.jpg",
    "synth-shapes": "synth-shapes.png",
    "synth-text": "synth-text.png",
}
SIX = ["photo-chelsea", "photo-coffee", "photo-camera", "photo-astronaut", "photo-rocket", "photo-hubble-deep-field"]
LONG40, V33, NTSC = "h264-long-320x240-30fps-40s", "h264-33s-160x120-30fps", "h264-ntsc-640x480-29.97fps-12s"

# Cases: id, source, start chapter (Alice) or None, prompt, target tokens (None: whole passage), media and the number
# of paragraphs between media placeholders, expectation.
# media: list of ("image", id) or ("video", id) in placeholder order.
CASES = [
    ("txt-alice-2600-none", "alice", "I", "none", 2600, [], 0, "embed"),
    ("txt-alice-4096-query", "alice", "III", "query", 4096, [], 0, "embed"),
    ("txt-alice-6144-document", "alice", "V", "document", 6144, [], 0, "embed"),
    ("txt-alice-8191-document", "alice", "VII", "document", 8191, [], 0, "embed"),
    ("txt-alice-8192-none", "alice", "I", "none", 8192, [], 0, "embed"),
    ("txt-alice-8192-query", "alice", "IX", "query", 8192, [], 0, "embed"),
    ("txt-constitution-document", "constitution", None, "document", None, [], 0, "embed"),
    ("txt-corpus-cycle-8192-none", "corpus", None, "none", 8192, [], 0, "embed"),
    ("txt-alice-8193-none", "alice", "I", "none", 8193, [], 0, "reject"),
    # The text of txt-alice-8192-none with the `document` prompt: the prompt tokens count toward the limit.
    ("txt-alice-8192-text-document-prompt", "same-as:txt-alice-8192-none", None, "document", None, [], 0, "reject"),
    ("mm-alice-6-images-6000-none", "alice", "II", "none", 6000, [("image", i) for i in SIX], 3, "embed"),
    (
        "mm-alice-8-images-8192-document",
        "alice",
        "IV",
        "document",
        8192,
        [("image", i) for i in [*SIX, "synth-shapes", "synth-text"]],
        4,
        "embed",
    ),
    ("mm-alice-6-images-8193-none", "alice", "II", "none", 8193, [("image", i) for i in SIX], 3, "reject"),
    ("mm-alice-video32-7000-none", "alice", "VI", "none", 7000, [("video", LONG40)], 6, "embed"),
    ("mm-alice-video32-8192-query", "alice", "VIII", "query", 8192, [("video", V33)], 5, "embed"),
    (
        "mm-alice-two-videos-image-8192-document",
        "alice",
        "X",
        "document",
        8192,
        [("video", LONG40), ("image", "photo-astronaut"), ("video", NTSC)],
        4,
        "embed",
    ),
    ("mm-alice-video32-8193-none", "alice", "VI", "none", 8193, [("video", LONG40)], 6, "reject"),
]


def sha(a) -> str:
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(diff) -> dict:
    a = np.abs(np.asarray(diff, dtype=np.float64)).ravel()
    return {"max_abs": float(a.max()) if a.size else 0.0, "mean_abs": float(a.mean()) if a.size else 0.0}


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def load(model_dir: Path, attn: str | None = None) -> SentenceTransformer:
    model_kwargs = {"dtype": torch.float32}
    if attn:
        model_kwargs["attn_implementation"] = attn
    st = SentenceTransformer(str(model_dir), device="cpu", model_kwargs=model_kwargs)
    st.eval()
    return st


def clean_source(name: str, raw: bytes) -> str:
    """Gutenberg text without header, footer or license: UTF-8, BOM dropped, CRLF to LF, passage only."""
    spec = SOURCES[name]
    text = raw.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    start = text.index(spec["start"])
    if "end" in spec:
        body = text[start : text.index(spec["end"], start)]
    else:
        end = text.index(spec["end_inclusive"], start) + len(spec["end_inclusive"])
        body = text[start:end]
    return body.strip()


def paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in text.split("\n\n") if p.strip()]


def chapter_paragraphs(alice: str, chapter: str) -> list[str]:
    marker = f"CHAPTER {chapter}.\n"
    return paragraphs(alice[alice.index(marker) :])


def corpus_passage(corpus: Path, repeat: int = 12) -> str:
    """The corpus's single-text cases (non-empty), one per line, cycled `repeat` times."""
    cases = [c for c in fixtures.load_corpus(corpus) if "pair" not in c and c["text"]]
    return "\n".join([c["text"] for c in cases] * repeat)


def compose(paras: list[str], media: list[tuple[str, str]], gap: int) -> tuple[str, int]:
    """Paragraphs joined by blank lines, a placeholder paragraph after every `gap` paragraphs until all media are
    placed; returns the text and the index after the last placeholder (cuts must not reach before it)."""
    out, placed, floor = [], 0, 0
    for k, p in enumerate(paras):
        out.append(p)
        if placed < len(media) and (k + 1) % gap == 0:
            out.append("<|image|>" if media[placed][0] == "image" else "<|video|>")
            placed += 1
            floor = len("\n\n".join(out))
    if placed < len(media):
        raise SystemExit("passage too short for its media")
    return "\n\n".join(out), floor


def build_item(text: str, media: list[tuple[str, str]], images: dict, videos: dict):
    if not media:
        return text
    item: dict = {"text": text}
    imgs = [images[m[1]]() for m in media if m[0] == "image"]
    vids = [str(videos[m[1]]) for m in media if m[0] == "video"]
    if imgs:
        item["image"] = imgs if len(imgs) > 1 else imgs[0]
    if vids:
        item["video"] = vids if len(vids) > 1 else vids[0]
    return item


def token_count(st: SentenceTransformer, item, prompt: str) -> int:
    kwargs = {"prompt": prompt} if prompt else {}
    return int(st.preprocess([item], **kwargs)["input_ids"].shape[1])


def cut_to_target(st, text: str, floor: int, target: int, media, images, videos, prompt: str) -> str:
    """Longest prefix (cut before a whitespace character, never before `floor`) with exactly `target` tokens;
    falls back to any code-point cut near the boundary when no whitespace cut hits the target exactly."""

    def count(n: int) -> int:
        return token_count(st, build_item(text[:n], media, images, videos), prompt)

    cuts = [i for i in range(max(floor, 1), len(text)) if text[i].isspace() and not text[i - 1].isspace()]
    cuts.append(len(text))
    if count(cuts[-1]) < target:
        raise SystemExit(f"passage reaches only {count(cuts[-1])} tokens, target {target}")
    lo, hi = 0, len(cuts) - 1  # smallest cut index with count >= target
    while lo < hi:
        mid = (lo + hi) // 2
        if count(cuts[mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    for k in range(lo, min(lo + 40, len(cuts))):
        c = count(cuts[k])
        if c == target:
            # Longest whitespace cut with the target count.
            while k + 1 < len(cuts) and count(cuts[k + 1]) == target:
                k += 1
            return text[: cuts[k]]
        if c > target + 8:
            break
    start = cuts[max(lo - 1, 0)]
    for n in range(start, min(start + 200, len(text))):
        if count(n) == target:
            return text[:n]
    raise SystemExit(f"no cut gives exactly {target} tokens")


class Capture:
    """Forward hooks: vision tower (pooled), embed_vision (soft tokens, one call per modality), language model."""

    def __init__(self, model: torch.nn.Module):
        self.data: dict[str, list[torch.Tensor]] = {}
        self.handles = [
            self._hook(model.embed_vision, "soft_tokens"),
            self._hook(model.language_model, "token_embeddings"),
        ]

    def _hook(self, module, name):
        def fn(_m, _i, out):
            t = out if isinstance(out, torch.Tensor) else out.last_hidden_state
            self.data.setdefault(name, []).append(t.detach().clone())

        return module.register_forward_hook(fn)

    def close(self):
        for h in self.handles:
            h.remove()


def render(tok, ids: list[int]) -> str:
    """Decoded sequence with each run of `<|video|>` or `<|image|>` written once with its length."""
    names = {VIDEO_TOKEN: "<|video|>", IMAGE_TOKEN: "<|image|>"}
    parts, chunk, k = [], [], 0
    while k < len(ids):
        if ids[k] in names:
            if chunk:
                parts.append(tok.decode(chunk, skip_special_tokens=False))
                chunk = []
            j = k
            while j < len(ids) and ids[j] == ids[k]:
                j += 1
            parts.append(f"{names[ids[k]]}x{j - k}")
            k = j
            continue
        chunk.append(ids[k])
        k += 1
    if chunk:
        parts.append(tok.decode(chunk, skip_special_tokens=False))
    return "".join(parts)


def video_record(path: Path) -> dict:
    d = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=0)
    m = d.metadata
    indices = eg2_video.sample_indices(m.num_frames, m.average_fps, m.duration_seconds)
    frames = d.get_frames_at(indices=indices).data.contiguous()
    return {
        "metadata": {"num_frames": m.num_frames, "average_fps": m.average_fps, "duration_seconds": m.duration_seconds},
        "sampled_indices": indices,
        "frames_sha256": sha(frames.permute(0, 2, 3, 1)),
        "decoded": {"height": int(frames.shape[2]), "width": int(frames.shape[3])},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tensors.mkdir(parents=True, exist_ok=True)

    # Sources.
    texts, source_records = {}, {}
    for name, spec in SOURCES.items():
        raw = (args.sources / spec["file"]).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec["sha256"]:
            raise SystemExit(f"{spec['file']}: SHA-256 {digest}, expected {spec['sha256']}")
        texts[name] = clean_source(name, raw)
        source_records[name] = {k: v for k, v in spec.items() if k not in ("start", "end", "end_inclusive")} | {
            "cleaning": "UTF-8, BOM dropped, CRLF to LF; passage from "
            + json.dumps(spec["start"])
            + (
                " up to " + json.dumps(spec["end"])
                if "end" in spec
                else " through " + json.dumps(spec["end_inclusive"])
            )
            + ", stripped",
            "passage_sha256": hashlib.sha256(texts[name].encode("utf-8")).hexdigest(),
            "passage_characters": len(texts[name]),
        }
    corpus_text = corpus_passage(args.corpus)
    source_records["corpus"] = {
        "file": "model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl",
        "sha256": fixtures.sha256_file(args.corpus),
        "title": "M0 tokenizer corpus (this repository): single-text cases, non-empty, one per line, cycled 12 times",
        "license": "Original cases written for this repository (docs/m0/reference-runs.md); MIT (repository LICENSE)",
        "passage_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "passage_characters": len(corpus_text),
    }

    # Media, checked against the accepted manifests.
    protected = Path(__file__).resolve().parents[2] / "protected" / "embeddinggemma-2"
    image_manifest = {
        r["id"]: r for r in json.loads((protected / "image-fp32-cpu" / "images.json").read_text())["images"]
    }
    video_manifest = {
        r["id"]: r for r in json.loads((protected / "video-fp32-cpu" / "videos.json").read_text())["videos"]
    }
    images, videos = {}, {}
    for image_id, file in IMAGES.items():
        path = args.images / file
        if image_manifest[image_id]["file"] != file or fixtures.sha256_file(path) != image_manifest[image_id]["sha256"]:
            raise SystemExit(f"{path}: not the accepted image {image_id}")
        images[image_id] = lambda p=path: Image.open(p)
    for video_id in (LONG40, V33, NTSC):
        path = args.videos / video_manifest[video_id]["file"]
        if fixtures.sha256_file(path) != video_manifest[video_id]["sha256"]:
            raise SystemExit(f"{path}: not the accepted video {video_id}")
        videos[video_id] = path

    st = load(args.model_dir)
    model = st[0].model
    processor = st[0].processor
    tok = processor.tokenizer
    assert processor.video_token_id == VIDEO_TOKEN
    tj = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    video_info = {v: video_record(p) for v, p in videos.items()}

    rows, tensors, chosen = [], {}, {}
    for case_id, source, chapter, prompt_name, target, media, gap, expect in CASES:
        prompt = PROMPTS[prompt_name]
        if source.startswith("same-as:"):
            text = chosen[source.split(":", 1)[1]]
        else:
            if source == "alice":
                paras = chapter_paragraphs(texts["alice"], chapter)
            elif source == "constitution":
                paras = paragraphs(texts["constitution"])
            else:
                paras = [corpus_text]
            full, floor = compose(paras, media, gap) if media else ("\n\n".join(paras), 0)
            text = full if target is None else cut_to_target(st, full, floor, target, media, images, videos, prompt)
        chosen[case_id] = text
        item = build_item(text, media, images, videos)
        kwargs = {"prompt": prompt} if prompt else {}
        cap = Capture(model)
        features = st.preprocess([item], **kwargs)
        with torch.inference_mode():
            emb = st(features)["sentence_embedding"][0].numpy()
        cap.close()
        ids = features["input_ids"][0].tolist()
        if target is not None and len(ids) != target:
            raise SystemExit(f"{case_id}: {len(ids)} tokens, target {target}")
        if (expect == "reject") != (len(ids) > LIMIT):
            raise SystemExit(f"{case_id}: {len(ids)} tokens does not match expectation {expect}")
        arrays: dict[str, np.ndarray] = {
            "input_ids": np.array(ids, dtype=np.int64),
            "token_embeddings": cap.data["token_embeddings"][-1][0].numpy(),
            "embedding": emb,
        }
        soft_calls = [t.numpy() for t in cap.data.get("soft_tokens", [])]
        n_img = sum(1 for m in media if m[0] == "image")
        n_vid = sum(1 for m in media if m[0] == "video")
        if len(soft_calls) != (n_img > 0) + (n_vid > 0):
            raise SystemExit(f"{case_id}: {len(soft_calls)} vision calls")
        image_soft = soft_calls[0] if n_img else None
        video_soft = soft_calls[-1] if n_vid else None
        media_records = []
        img_k = vid_k = img_off = vid_off = frame_off = 0
        for kind, media_id in media:
            if kind == "image":
                pv = features["pixel_values"][img_k]
                pos = features["image_position_ids"][img_k]
                valid = int((pos[:, 0] >= 0).sum())
                n = valid // 9
                soft = image_soft[img_off : img_off + n]
                arrays[f"image{img_k}_soft_tokens"] = soft
                media_records.append(
                    {
                        "kind": "image",
                        "id": media_id,
                        "file": IMAGES[media_id],
                        "soft_tokens": n,
                        "digests": {
                            "pixel_values": sha(pv[:valid]),
                            "positions": sha(pos[:valid]),
                            "soft_tokens": sha(soft),
                        },
                    }
                )
                img_k, img_off = img_k + 1, img_off + n
            else:
                nfv = int(features["num_frames_per_video"][vid_k])
                pv = features["pixel_values_videos"][frame_off : frame_off + nfv]
                pos = features["video_position_ids"][frame_off : frame_off + nfv]
                valid = int((pos[0, :, 0] >= 0).sum())
                n = valid // 9
                soft = video_soft[vid_off : vid_off + nfv * n]
                arrays[f"video{vid_k}_soft_tokens"] = soft
                info = video_info[media_id]
                if nfv != len(info["sampled_indices"]):
                    raise SystemExit(f"{case_id}/{media_id}: {nfv} frames, sampled {len(info['sampled_indices'])}")
                media_records.append(
                    {
                        "kind": "video",
                        "id": media_id,
                        "file": video_manifest[media_id]["file"],
                        "frames": nfv,
                        "soft_tokens_per_frame": n,
                        **info,
                        "digests": {
                            "pixel_values": sha(pv[:, :valid]),
                            "positions": sha(pos[:, :valid]),
                            "soft_tokens": sha(soft),
                        },
                    }
                )
                vid_k, vid_off, frame_off = vid_k + 1, vid_off + nfv * n, frame_off + nfv
        if image_soft is not None and img_off != image_soft.shape[0]:
            raise SystemExit(f"{case_id}: {image_soft.shape[0]} image soft tokens, {img_off} placed")
        if video_soft is not None and vid_off != video_soft.shape[0]:
            raise SystemExit(f"{case_id}: {video_soft.shape[0]} video soft tokens, {vid_off} placed")
        if sum(1 for i in ids if i == IMAGE_TOKEN) != img_off or sum(1 for i in ids if i == VIDEO_TOKEN) != vid_off:
            raise SystemExit(f"{case_id}: placeholder token counts differ from soft-token counts")
        np.savez(args.tensors / f"{case_id}.npz", **arrays)
        tensors[case_id] = {
            n: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a)} for n, a in arrays.items()
        }
        row = {
            "id": case_id,
            "kind": "text" if not media else "multimodal",
            "expect": expect,
            "source": source,
            "start_chapter": chapter,
            "prompt": prompt_name,
            "prompt_text": prompt,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "media": media_records,
            "tokens": len(ids),
            "image_token_count": img_off,
            "video_token_count": vid_off,
            "rendered": render(tok, ids),
            "input_ids": ids,
            "embedding": fixtures.float32_list(emb),
            "norm": float(np.linalg.norm(emb)),
            "tensors_sha256": {n: v["sha256"] for n, v in tensors[case_id].items()},
        }
        if not media:
            row["tokenizer_json_ids_equal"] = tj.encode(prompt + text).ids == ids
        rows.append(row)
        print(case_id, len(ids), expect, [(m["kind"], m["id"]) for m in media_records], flush=True)

    fixtures.write_jsonl(args.out / "long-embeddings.jsonl", rows)
    fixtures.write_json(args.out / "tensors.json", tensors)
    fixtures.write_json(args.tensors / "tensors.json", tensors)
    fixtures.write_json(args.out / "texts.json", source_records)
    by_id = {r["id"]: r for r in rows}
    emb_by_id = {r["id"]: np.array(r["embedding"], np.float32) for r in rows}

    reference = {
        "reference": "sentence-transformers 6.1.0 modules, torchcodec 0.17.0 decoding (FFmpeg "
        + tc_core.get_ffmpeg_library_versions()["ffmpeg_version"]
        + f"), PyTorch fp32 CPU ({args.threads} threads), sdpa attention, one input per call, full checkpoint",
        "rows": len(rows),
        "tokens": {r["id"]: r["tokens"] for r in rows},
        "text_rows_tokenizer_json_ids_equal": all(r.get("tokenizer_json_ids_equal", True) for r in rows),
        "upstream_truncates_or_rejects_over_8192": False,
        "nan_or_inf_rows": [r["id"] for r in rows if not np.isfinite(np.array(r["embedding"])).all()],
    }

    def item_of(case_id: str):
        r = by_id[case_id]
        return build_item(r["text"], [(m["kind"], m["id"]) for m in r["media"]], images, videos)

    def check(name: str, fn) -> None:
        try:
            reference[name] = fn()
        except Exception as exc:  # recorded, not fatal: the rows above are the fixtures
            reference[name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(name, reference[name], flush=True)

    def batch_vs_single(pair: list[str]) -> dict:
        with torch.inference_mode():
            batched = st.encode([item_of(c) for c in pair], batch_size=2, convert_to_numpy=True)
        return {"rows": pair} | stats(batched - np.stack([emb_by_id[c] for c in pair]))

    def repeat(case_id: str) -> dict:
        with torch.inference_mode():
            again = st.encode([item_of(case_id)], prompt=by_id[case_id]["prompt_text"], convert_to_numpy=True)[0]
        return stats(again - emb_by_id[case_id])

    if not args.skip_stability:
        check("text_padded_batch2_vs_single", lambda: batch_vs_single(["txt-alice-2600-none", "txt-alice-8192-none"]))
        check(
            "multimodal_padded_batch2_vs_single",
            lambda: batch_vs_single(["mm-alice-6-images-6000-none", "mm-alice-video32-7000-none"]),
        )
        check("repeat_txt-alice-8192-query", lambda: repeat("txt-alice-8192-query"))
        eager = load(args.model_dir, attn="eager")

        def eager_vs_sdpa(case_id: str) -> dict:
            with torch.inference_mode():
                e = eager.encode([item_of(case_id)], convert_to_numpy=True)[0]
            return stats(e - emb_by_id[case_id]) | {"cosine": cosine(e, emb_by_id[case_id])}

        for c in ["txt-alice-8192-none", "mm-alice-video32-7000-none"]:
            check(f"eager_vs_sdpa_{c}", lambda c=c: eager_vs_sdpa(c))
        del eager
    fixtures.write_json(args.out / "long-reference.json", reference)

    env = fixtures.environment(PACKAGES)
    env["torch_threads"] = torch.get_num_threads()
    env["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    env["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    env["torchcodec_ffmpeg"] = tc_core.get_ffmpeg_library_versions()
    fixtures.write_json(
        args.out / "long-environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_long.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": REPO, "revision": args.revision},
            "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
            "images": {i: {"file": f, "sha256": image_manifest[i]["sha256"]} for i, f in IMAGES.items()},
            "videos": {v: {"file": p.name, "sha256": video_manifest[v]["sha256"]} for v, p in videos.items()},
            "environment": env,
        },
    )
    print(json.dumps({k: v for k, v in reference.items() if k != "tokens"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
