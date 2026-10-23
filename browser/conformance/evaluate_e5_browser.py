"""Check e5 browser conformance results against the committed browser tolerances.

Inputs: result files of run_conformance.py (one variant and provider each), the protected fixtures of
multilingual-e5-small/onnx-fp32-cpu (the reference of every browser profile), and the protected
tolerances.yaml of the profile the run reports (onnx-fp32-wasm, onnx-fp32-webgpu, onnx-int8embed-wasm,
onnx-int8embed-webgpu). For browser_vs_native_same_graph, the same graph runs on Python ONNX Runtime (CPU,
batch 1, the fixture ids) with the library's pooling.

Every bound in the tolerance file is reported as passed, failed or not evaluated; a bound nobody checks is never
silently satisfied. task_metric_before_preview needs a P-030 held-out query set and is not evaluated here.

Usage:
    python evaluate_e5_browser.py --graphs <dir with model-fp32.onnx, model-int8-embed.onnx> --out report.json \
        result1.json result2.json ...
Needs numpy and onnxruntime 1.30.0 (model-pipeline/reference/encoder/requirements-browser.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTECTED = REPO / "model-pipeline" / "protected" / "multilingual-e5-small"
FIXTURES = PROTECTED / "onnx-fp32-cpu"
GRAPHS = {"fp32": "model-fp32.onnx", "int8embed": "model-int8-embed.onnx"}


def read_tolerances(path: Path) -> dict:
    """Strict reader for the browser tolerance files: comments, two-level mappings, scalars and inline lists."""
    root: dict = {}
    section: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"{path}: cannot read line {raw!r}")
        value = value.strip()
        if indent == 0:
            if value:
                root[key] = value
                section = None
            else:
                section = root.setdefault(key, {})
        elif indent == 2 and section is not None:
            section[key] = scalar(value)
        else:
            raise ValueError(f"{path}: unexpected structure at {raw!r}")
    return root


def scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        return [v.strip() for v in value[1:-1].split(",") if v.strip()]
    try:
        return float(value)
    except ValueError:
        return value


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]  # not splitlines(): texts hold U+2028 and friends


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float64)
    mean = (hidden.astype(np.float64) * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
    return (mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def native_embeddings(graph: Path, rows: list[dict]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    out = {}
    for row in rows:
        ids = np.asarray([row["input_ids"]], dtype=np.int64)
        mask = np.ones_like(ids)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(["last_hidden_state"], feeds)[0]
        out[row["id"]] = pooled(hidden, mask)[0]
    return out


def max_abs(a: dict[str, list], b: dict[str, np.ndarray]) -> float:
    return float(
        max(np.abs(np.asarray(v, dtype=np.float64) - np.asarray(b[k], dtype=np.float64)).max() for k, v in a.items())
    )


def bound(name: str, observed, limit, ok: bool) -> dict:
    return {"bound": name, "observed": observed, "limit": limit, "status": "passed" if ok else "failed"}


def evaluate(result: dict, graphs: Path, native_cache: dict) -> dict:
    run = result["run"]
    if "error" in run:
        return {"profile": None, "label": result["label"], "error": run["error"].splitlines()[0]}
    profile = run["profile"]
    tolerances = read_tolerances(PROTECTED / profile.split("/", 1)[1] / "tolerances.yaml")
    if tolerances.get("profile") != profile:
        raise ValueError(f"tolerance file names {tolerances.get('profile')}, run reports {profile}")
    reference_rows = jsonl(FIXTURES / "embeddings.jsonl")
    reference = {r["id"]: np.asarray(r["embedding"], dtype=np.float64) for r in reference_rows}
    tokenizer_rows = {r["id"]: r for r in jsonl(FIXTURES / "tokenizer.jsonl")}
    checks: list[dict] = []
    discrete = tolerances.get("discrete", {})
    floats = tolerances.get("float", {})

    # Discrete: the e5 pipeline tokenizer against the ids of the protected embeddings, and the owned tokenizer
    # against the tokenizer fixtures (ids, attention mask, UTF-16 offsets).
    e5_ids = sum(
        1
        for t in run["e5_tokenizer"]
        if t["input_ids"] != next(r for r in reference_rows if r["id"] == t["id"])["input_ids"]
    )
    e5_mask = sum(1 for t in run["e5_tokenizer"] if any(m != 1 for m in t["attention_mask"]))
    tok_ids = sum(1 for t in run["tokenizer"] if t["ids"] != tokenizer_rows[t["id"]]["encoding"]["ids"])
    tok_mask = sum(
        1 for t in run["tokenizer"] if t["attention_mask"] != tokenizer_rows[t["id"]]["encoding"]["attention_mask"]
    )
    tok_offsets = sum(
        1
        for t in run["tokenizer"]
        if t["offsets_utf16"] != tokenizer_rows[t["id"]]["encoding"]["offsets"]["utf16-code-units"]
    )
    cases = f"{len(run['e5_tokenizer'])} e5 inputs, {len(run['tokenizer'])} tokenizer cases"
    if discrete.get("token_ids") == "exact":
        checks.append(
            bound("discrete.token_ids", f"{e5_ids + tok_ids} mismatches ({cases})", "exact", e5_ids + tok_ids == 0)
        )
    if discrete.get("attention_mask") == "exact":
        checks.append(
            bound(
                "discrete.attention_mask",
                f"{e5_mask + tok_mask} mismatches ({cases})",
                "exact",
                e5_mask + tok_mask == 0,
            )
        )
    if discrete.get("offsets") == "exact":
        checks.append(
            bound(
                "discrete.offsets",
                f"{tok_offsets} mismatches (utf16, {len(run['tokenizer'])} cases)",
                "exact",
                tok_offsets == 0,
            )
        )

    got = {k: np.asarray(v, dtype=np.float64) for k, v in run["embeddings"].items()}
    diffs = np.concatenate([np.abs(got[k] - reference[k]) for k in reference])
    cosines = {
        k: float(got[k] @ reference[k] / (np.linalg.norm(got[k]) * np.linalg.norm(reference[k]))) for k in reference
    }
    worst = min(cosines, key=cosines.get)
    finite = all(np.isfinite(v).all() for v in got.values()) and all(
        np.isfinite(np.asarray(v)).all() for v in run["batch8"].values()
    )
    stats = {
        "max_abs": float(diffs.max()),
        "mean_abs": float(diffs.mean()),
        "p99_abs": float(np.percentile(diffs, 99)),
        "min_cosine": cosines[worst],
        "min_cosine_case": worst,
        "cases": len(reference),
    }
    if "embedding_max_abs" in floats:
        checks.append(
            bound(
                "float.embedding_max_abs",
                stats["max_abs"],
                floats["embedding_max_abs"],
                stats["max_abs"] <= floats["embedding_max_abs"],
            )
        )
    if "min_cosine" in floats:
        checks.append(
            bound(
                "float.min_cosine",
                stats["min_cosine"],
                floats["min_cosine"],
                stats["min_cosine"] >= floats["min_cosine"],
            )
        )
    batch = max_abs(run["batch8"], got)
    key = "padded_batch_vs_single_embedding_max_abs"
    if key in floats:
        checks.append(bound(f"float.{key}", batch, floats[key], batch <= floats[key]))
    if floats.get("nan_or_inf") == "fail":
        checks.append(bound("float.nan_or_inf", "none" if finite else "present", "fail", finite))

    same_graph = tolerances.get("browser_vs_native_same_graph", {})
    # Always reported (backend drift alone); a bound only where the profile's file sets one.
    graph = graphs / GRAPHS[result["variant"]]
    if graph not in native_cache:
        native_cache[graph] = native_embeddings(graph, reference_rows)
    vs_native = max_abs(run["embeddings"], native_cache[graph])
    stats["vs_native_same_graph_max_abs"] = vs_native
    if "embedding_max_abs" in same_graph:
        checks.append(
            bound(
                "browser_vs_native_same_graph.embedding_max_abs",
                vs_native,
                same_graph["embedding_max_abs"],
                vs_native <= same_graph["embedding_max_abs"],
            )
        )
    for name, limit in tolerances.get("task_metric_before_preview", {}).items():
        checks.append(
            {
                "bound": f"task_metric_before_preview.{name}",
                "observed": None,
                "limit": limit,
                "status": "not evaluated",
                "reason": "needs a P-030 held-out query set (gates Preview, not Experimental)",
            }
        )

    # Harness check, not a tolerance-file key: the text path (owned tokenizer + encoder through IEmbeddingGenerator, all
    # texts in one call, so padded batches) must equal the id path within the profile's padded-batch bound. It counts
    # toward "passed" like the protected bounds.
    generator_vs_ids = max_abs(run["generator_embeddings"], got)
    gen_limit = floats.get("padded_batch_vs_single_embedding_max_abs", 1e-5)
    checks.append(bound("harness.generator_vs_ids_max_abs", generator_vs_ids, gen_limit, generator_vs_ids <= gen_limit))
    return {
        "profile": profile,
        "label": result["label"],
        "browser": result.get("browser_version"),
        "cross_origin_isolated": result.get("cross_origin_isolated"),
        "onnxruntime_web": run["onnxruntime_web"],
        "runtime": run["runtime"],
        "statistics": stats,
        "generator_vs_ids_max_abs": generator_vs_ids,
        "checks": checks,
        "passed": all(c["status"] == "passed" for c in checks if c["status"] != "not evaluated"),
        "load": run["load"],
        "first_run_ms": run["first_run_ms"],
        "latency_median_ms": {k: v["median"] for k, v in run["latency_ms"].items()},
        "peak_rss_renderer_bytes": result.get("peak_rss_renderer_bytes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    cache: dict = {}
    reports = [
        evaluate(json.loads(p.read_text(encoding="utf-8")), args.graphs, cache) | {"file": p.name} for p in args.results
    ]
    args.out.write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")
    for r in reports:
        if "error" in r:
            print(f"{r['file']}: ERROR {r['error']}")
            continue
        s = r["statistics"]
        print(
            f"{r['profile']:45} {r['label']:7} {'PASS' if r['passed'] else 'FAIL'} max_abs {s['max_abs']:.2e} "
            f"min_cos {s['min_cosine']:.7f} vs_native {s['vs_native_same_graph_max_abs']:.1e} "
            f"gen_vs_ids {r['generator_vs_ids_max_abs']:.1e}"
        )
        for c in r["checks"]:
            if c["status"] != "passed":
                print(f"    {c['status']}: {c['bound']} observed {c['observed']} limit {c['limit']}")


if __name__ == "__main__":
    main()
