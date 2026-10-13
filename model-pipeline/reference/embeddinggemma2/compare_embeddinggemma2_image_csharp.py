"""Image-path spike: inputs for and comparison of the throwaway C# preprocessing prototype.

Subcommands:

- `dump-rgb --images <dir> --out <dir>`: PIL's decoded RGB for every image of `images.json` (`Image.open`, then
  `convert("RGB")` unless the mode is RGB, as transformers' `convert_to_rgb`), one raw `<id>.rgb` file each and an
  `index.json` with sizes. This is the "pre-decoded RGB" input the C# resampler starts from.
- `compare --images <dir> --rgb <dir> --csharp <dir> --tensors <dir> --embeddings <jsonl> --out <json>
   [--runs name=dir ...]`: compares the C# outputs with the reference:
  * decoders (`<csharp>/decode/<decoder>/<id>.rgb`) against PIL's RGB: exact, or max and mean absolute
    difference in levels and the fraction of differing bytes;
  * resampler (`<csharp>/resize/{torch,pil}/<id>.rgb`) against the reference's resized uint8 image (torchvision
    backend) and against PIL's resize (transformers PIL backend, recomputed here);
  * full pipelines (`--runs`: C# `embed` output directories) against the reference embeddings, pixel values and soft
    tokens.

Usage: see the spike report, docs/m0/embeddinggemma2-image-spike.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def dump_rgb(images: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((images / "images.json").read_text(encoding="utf-8"))
    index = {}
    for r in manifest["images"]:
        im = Image.open(images / r["file"])
        rgb = im if im.mode == "RGB" else im.convert("RGB")
        a = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        (out / f"{r['id']}.rgb").write_bytes(a.tobytes())
        index[r["id"]] = {"width": a.shape[1], "height": a.shape[0], "mode": im.mode}
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")


def level_stats(got: np.ndarray, ref: np.ndarray) -> dict:
    if got.shape != ref.shape:
        return {"shape_mismatch": [list(got.shape), list(ref.shape)]}
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    return {
        "exact": bool((d == 0).all()),
        "max_abs_levels": int(d.max()),
        "mean_abs_levels": float(d.mean()),
        "differing_fraction": float((d > 0).mean()),
    }


def pil_resize(rgb: np.ndarray, th: int, tw: int) -> np.ndarray:
    if rgb.shape[:2] == (th, tw):
        return rgb
    return np.asarray(Image.fromarray(rgb, "RGB").resize((tw, th), Image.Resampling.BICUBIC))


def compare(args) -> dict:
    manifest = json.loads((args.images / "images.json").read_text(encoding="utf-8"))
    rgb_index = json.loads((args.rgb / "index.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in manifest["images"]]
    report: dict = {"decoders": {}, "resampler": {}, "pipelines": {}}

    def rgb_of(path: Path, w: int, h: int) -> np.ndarray:
        return np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(h, w, 3)

    dec_index_path = args.csharp / "decode" / "index.json"
    dec_index = json.loads(dec_index_path.read_text()) if dec_index_path.exists() else {}
    for image_id in ids:
        ref_meta = rgb_index[image_id]
        ref = rgb_of(args.rgb / f"{image_id}.rgb", ref_meta["width"], ref_meta["height"])
        npz = np.load(args.tensors / f"{image_id}.npz")
        resized_ref = npz["image0_resized_uint8"]
        th, tw = resized_ref.shape[:2]
        rows = {}
        for dec, meta in dec_index.get(image_id, {}).items():
            if "error" in meta:
                rows[dec] = {"error": meta["error"]}
                continue
            got = rgb_of(args.csharp / "decode" / dec / f"{image_id}.rgb", meta["width"], meta["height"])
            entry = {"decode_vs_pil_rgb": level_stats(got, ref), "note": meta.get("note")}
            if tuple(meta["target"]) == (th, tw):
                resized = rgb_of(args.csharp / "decode" / f"{dec}-torch" / f"{image_id}.rgb", tw, th)
                entry["decode_and_resize_vs_reference_resized"] = level_stats(resized, resized_ref)
            else:
                entry["decode_and_resize_vs_reference_resized"] = {"target_mismatch": [meta["target"], [th, tw]]}
            rows[dec] = entry
        report["decoders"][image_id] = rows

        res = {"target": [th, tw], "resize_skipped": list(ref.shape[:2]) == [th, tw]}
        pil = pil_resize(ref, th, tw)
        res["pil_backend_vs_reference"] = level_stats(pil, resized_ref)
        for q in ("torch", "pil"):
            p = args.csharp / "resize" / q / f"{image_id}.rgb"
            if p.exists():
                got = rgb_of(p, tw, th)
                res[f"csharp_{q}_vs_reference"] = level_stats(got, resized_ref)
                res[f"csharp_{q}_vs_pil_resize"] = level_stats(got, pil)
        report["resampler"][image_id] = res

    ref_rows = {}
    for line in args.embeddings.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ref_rows[r["id"]] = r
    for spec in args.runs or []:
        name, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        run = json.loads((run_dir / "run.json").read_text())
        order = list(run["rows"].keys())
        emb = np.fromfile(run_dir / "embeddings.f32", dtype=np.float32).reshape(len(order), 768)
        soft_all = np.fromfile(run_dir / "soft_tokens.f32", dtype=np.float32)
        pix_all = np.fromfile(run_dir / "pixel_values.f32", dtype=np.float32)
        soft_off = pix_off = 0
        per_row, worst = {}, {"embedding_max_abs": 0.0, "min_cosine": 1.0}
        for i, row_id in enumerate(order):
            ref = np.array(ref_rows[row_id]["embedding"], dtype=np.float32)
            d = float(np.abs(emb[i] - ref).max())
            cos = float(np.dot(emb[i].astype(np.float64), ref) / np.linalg.norm(emb[i]) / np.linalg.norm(ref))
            entry = {"embedding_max_abs": d, "cosine": cos}
            npz = np.load(args.tensors / f"{row_id}.npz")
            k = 0
            pix_d, soft_d, pix_exact = 0.0, 0.0, True
            while f"image{k}_soft_tokens" in npz:
                s_ref = npz[f"image{k}_soft_tokens"]
                p_ref = npz[f"image{k}_pixel_values"]
                s = soft_all[soft_off : soft_off + s_ref.size].reshape(s_ref.shape)
                p = pix_all[pix_off : pix_off + p_ref.size].reshape(p_ref.shape)
                soft_off += s_ref.size
                pix_off += p_ref.size
                if p.shape == p_ref.shape:
                    pix_d = max(pix_d, float(np.abs(p - p_ref).max()))
                    pix_exact &= bool(np.array_equal(p, p_ref))
                    soft_d = max(soft_d, float(np.abs(s - s_ref).max()))
                k += 1
            if k:
                entry.update(
                    {"pixel_values_max_abs": pix_d, "pixel_values_exact": pix_exact, "soft_tokens_max_abs": soft_d}
                )
            per_row[row_id] = entry
            worst["embedding_max_abs"] = max(worst["embedding_max_abs"], d)
            worst["min_cosine"] = min(worst["min_cosine"], cos)
        report["pipelines"][name] = {
            "source": run["source"],
            "worst": worst,
            "rows": per_row,
            "timing": {r: v for r, v in run["rows"].items()},
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-rgb")
    d.add_argument("--images", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--images", type=Path, required=True)
    c.add_argument("--rgb", type=Path, required=True)
    c.add_argument("--csharp", type=Path, required=True)
    c.add_argument("--tensors", type=Path, required=True)
    c.add_argument("--embeddings", type=Path, required=True)
    c.add_argument("--runs", nargs="*")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "dump-rgb":
        dump_rgb(args.images, args.out)
    else:
        report = compare(args)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
