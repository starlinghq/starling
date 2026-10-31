"""Compile a synthetic intervention profile into the LFM3-1.7B fp32 graph (evidence and test setup, not a real profile).

Usage::

    python -m model_pipeline.interventions.lfm3_synthetic \\
        --snapshot <LFM3-1.7B-ONNX snapshot at c680759a> --cache <cache root> [--profile-dir <dir>]

The snapshot is the directory holding ``onnx/model.onnx``, its four ``model.onnx_data*`` files and ``tokenizer.json``
(for example ``$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77``). The cache root must
be on the snapshot's file system: the derived graph reaches the weights through hard links. The command writes a
profile of seeded random orthonormal directions (no behavioural meaning), compiles it with ``get_or_compile`` and
prints the cache entry directory, which ``STARLING_LFM3_INTERVENTION`` takes in the .NET tests.

The external data SHA-256 values are the fp32 asset lock's, so a cold compile hashes the 7 GB of weights once (inside
the compiler) instead of twice.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .cache import CacheEntry, get_or_compile
from .injection_sites import LFM3_FP32_GRAPH_SHA256, LFM3_HIDDEN_SIZE
from .synthetic import SyntheticSite, write_synthetic_profile

LFM3_FP32_GRAPH = "onnx/model.onnx"

# The fp32 asset lock's external data files (Lfm3ModelAssets.Fp32Lock), relative to the snapshot root.
LFM3_FP32_EXTERNAL_SHA256 = {
    "onnx/model.onnx_data": "bf44fe98d130ad2bca747e9b62f28b18a3277d45bf1367da15a478538c849057",
    "onnx/model.onnx_data_1": "bb631a02237e4f8a491e12b25c333849046332b6db60fe0a2120f278c982fa76",
    "onnx/model.onnx_data_2": "ad39ed4b378c9c1b34c1a3cdb879880c7e110061070ff29fca118e8dd0b48010",
    "onnx/model.onnx_data_3": "1538827aa1c279442457028c4e3d784b7ea82bcf03abd4b55e45fa4facf9fe62",
}

SYNTHETIC_PROFILE_ID = "lfm3-synthetic-test"
SYNTHETIC_LAYERS = (8, 16)
SYNTHETIC_RANK = 8
SYNTHETIC_SEED = 20261031


def write_lfm3_synthetic_profile(directory: str | Path, *, seed: int = SYNTHETIC_SEED):
    """``profile.json`` plus directions for residual_post after layers 8 and 16, rank 8, scales 1."""
    return write_synthetic_profile(
        directory,
        profile_id=SYNTHETIC_PROFILE_ID,
        base_model_sha256=LFM3_FP32_GRAPH_SHA256,
        sites=[SyntheticSite(layer, LFM3_HIDDEN_SIZE, SYNTHETIC_RANK) for layer in SYNTHETIC_LAYERS],
        seed=seed,
    )


def compile_lfm3_synthetic(snapshot: str | Path, cache_root: str | Path, profile_dir: str | Path) -> CacheEntry:
    profile = write_lfm3_synthetic_profile(profile_dir)
    return get_or_compile(
        cache_root=cache_root,
        model="lfm3",
        base_dir=snapshot,
        graph=LFM3_FP32_GRAPH,
        profile=profile,
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="LFM3-1.7B-ONNX snapshot root (c680759a)")
    parser.add_argument("--cache", required=True, type=Path, help="intervention cache root (same file system)")
    parser.add_argument(
        "--profile-dir", type=Path, help="where to write the synthetic profile (default: <cache>/profiles/...)"
    )
    args = parser.parse_args(argv)
    profile_dir = args.profile_dir or args.cache / "profiles" / SYNTHETIC_PROFILE_ID
    entry = compile_lfm3_synthetic(args.snapshot, args.cache, profile_dir)
    print(entry.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
