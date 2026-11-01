"""The evaluator's decode loop on the real LFM3-1.7B fp32 derived graph (env-gated).

- ``STARLING_LFM3_SNAPSHOT``: the LFM3-1.7B-ONNX snapshot at c680759a.
- ``STARLING_LFM3_INTERVENTION_CACHE``: a cache root on the snapshot's file system (the "concise" profile is compiled
  into it if absent).

On the CPU provider, the loop over the concise derived graph at strength 0 must give the greedy ids that the LFM3
onboarding evidence recorded for the original graph (``docs/evidence/lfm3-onboarding-2026-10-28``, ids the .NET
implementation also produced), for every single-turn case without thinking. Prefill logits must agree with the
loop's first step, and a nonzero strength must change them. One fp32 session (about 7 GB), CPU, under a minute.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from model_pipeline.interventions.lfm3_synthetic import LFM3_FP32_EXTERNAL_SHA256, LFM3_FP32_GRAPH  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONCISE = REPO / "model-pipeline/interventions/profiles/lfm3-1.7b/concise"
SNAPSHOT = os.environ.get("STARLING_LFM3_SNAPSHOT")
CACHE = os.environ.get("STARLING_LFM3_INTERVENTION_CACHE")

pytestmark = pytest.mark.skipif(
    not SNAPSHOT or not CACHE or not Path(SNAPSHOT, LFM3_FP32_GRAPH).is_file(),
    reason="set STARLING_LFM3_SNAPSHOT and STARLING_LFM3_INTERVENTION_CACHE",
)


def _cases():
    corpus = REPO / "ai-dotnet/tests/Starlings.Models.Lfm3.Tests/Data/template-reference.jsonl"
    ids = REPO / "docs/evidence/lfm3-onboarding-2026-10-28/onnx-generated-ids.jsonl"
    expected = {d["id"]: d["generated_ids"] for d in map(json.loads, ids.read_text(encoding="utf-8").splitlines())}
    cases = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        (c["id"], c["prompt_ids"], c["max_new_tokens"], expected[c["id"]])
        for c in cases
        if not c["enable_thinking"] and [m["role"] for m in c["messages"]] == ["user"]
    ]


def test_the_decode_loop_reproduces_the_onboarding_greedy_ids_at_strength_zero():
    from model_pipeline.interventions.cache import get_or_compile
    from model_pipeline.interventions.evaluate import Lfm3OrtModel, kl_per_position

    entry = get_or_compile(
        cache_root=CACHE,
        model="lfm3",
        base_dir=SNAPSHOT,
        graph=LFM3_FP32_GRAPH,
        profile=CONCISE / "profile.json",
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
    )
    model = Lfm3OrtModel(entry.graph_path, entry.manifest.strength_input.name, provider="cpu")
    cases = _cases()
    assert len(cases) >= 8
    for case_id, prompt_ids, max_new, expected in cases:
        out, finished = model.generate(prompt_ids, 0.0, max_new)
        assert out == expected, case_id
        assert finished == (len(expected) < max_new), case_id
    case_id, prompt_ids, _, expected = cases[0]
    logits = model.logits(prompt_ids, 0.0)
    assert logits.shape[0] == len(prompt_ids)
    assert int(np.argmax(logits[-1])) == expected[0]
    assert float(kl_per_position(logits, model.logits(prompt_ids, 0.0)).max()) == 0.0
    assert float(kl_per_position(logits, model.logits(prompt_ids, -2.0)).mean()) > 1e-4
