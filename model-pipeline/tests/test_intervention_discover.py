"""discover-sites (model_pipeline.interventions.discover): reviewable site-mapping proposals from the graph structure.

The synthetic tests always run. The real-graph tests need ``STARLING_LFM3_SNAPSHOT`` (the LFM3-1.7B-ONNX snapshot at
c680759a); they read only the 275 KB graph files, never the weights. The fp16 test also needs
``onnx/model_fp16.onnx`` in the snapshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from test_interventions import HIDDEN, LAYERS, _build_graph, _mapping, _sha, _write_base  # noqa: E402

from model_pipeline import cli  # noqa: E402
from model_pipeline.interventions import discover  # noqa: E402
from model_pipeline.interventions.discover import (  # noqa: E402
    DEFAULT_LAYER_PATTERN,
    PROPOSAL_KIND,
    SiteDiscoveryError,
    compare_with_mapping,
    discover_sites,
    proposal_json,
)
from model_pipeline.interventions.injection_sites import (  # noqa: E402
    LFM3_FP16_MAPPING,
    LFM3_FP32_MAPPING,
    MappedSite,
    SiteMapping,
)

SNAPSHOT = os.environ.get("STARLING_LFM3_SNAPSHOT")


# --------------------------------------------------------------------------------------------------------------------
# The compiler tests' residual MLP (layer<i>/ names; layer 0's residual also feeds a graph-output tap)


@pytest.fixture
def mlp_graph(tmp_path: Path) -> Path:
    return _write_base(tmp_path / "mlp", _build_graph())


def test_mlp_proposal_reproduces_the_hand_written_mapping(mlp_graph):
    proposal = discover_sites(mlp_graph, layer_pattern=r"^layer(\d+)/")
    assert proposal["kind"] == PROPOSAL_KIND
    assert proposal["graph"]["sha256"] == _sha(mlp_graph)
    assert proposal["proposed"] == {"target": "residual_post", "precision": "fp32", "hidden_size": HIDDEN, "layers": 3}
    assert compare_with_mapping(proposal, _mapping(_sha(mlp_graph))) == []
    assert [s["tensor"] for s in proposal["sites"]] == [f"layer{i}/residual_post" for i in range(LAYERS)]


def test_a_reader_outside_the_next_layer_is_listed_for_review_not_proposed(mlp_graph):
    proposal = discover_sites(mlp_graph, layer_pattern=r"^layer(\d+)/")
    first = proposal["sites"][0]
    assert first["consumers"] == ["layer1/norm", "layer1/residual_add"]
    assert first["other_readers"] == ["tap0"]
    assert any("tap0" in w for w in proposal["warnings"])
    assert all(not s["other_readers"] for s in proposal["sites"][1:])


def test_no_external_data_is_read(mlp_graph):
    before = discover_sites(mlp_graph, layer_pattern=r"^layer(\d+)/")
    (mlp_graph.parent / "model.onnx_data").unlink()
    assert discover_sites(mlp_graph, layer_pattern=r"^layer(\d+)/") == before


def test_the_default_pattern_finds_nothing_in_other_naming_and_says_so(mlp_graph):
    proposal = discover_sites(mlp_graph)
    assert proposal["sites"] == []
    assert any("--layer-pattern" in w for w in proposal["warnings"])


# --------------------------------------------------------------------------------------------------------------------
# An LFM3-shaped graph: /model/layers.L/ names, two residual adds per layer, a KV-like graph output per layer, and a
# final norm the exporter numbers like a layer (/model/layers.N/final_norm_layernorm).

N = 4
H = 8


def _lfm3_like(dtype=np.float32, *, extra_add_layer: int | None = None, subgraph_reader: bool = False):
    elem = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
    rng = np.random.default_rng(5)
    init = [numpy_helper.from_array(rng.standard_normal((11, H)).astype(dtype), "embed")]
    nodes = [helper.make_node("Gather", ["embed", "input_ids"], ["/model/embed_tokens/Gather/output_0"],
                              name="/model/embed_tokens/Gather")]  # fmt: skip
    outputs = []
    h = "/model/embed_tokens/Gather/output_0"
    for i in range(N):
        p = f"/model/layers.{i}"
        for w in ("w_op", "w_ffn"):
            init.append(numpy_helper.from_array((rng.standard_normal((H, H)) / np.sqrt(H)).astype(dtype), f"{p}/{w}"))
        nodes += [
            helper.make_node("Mul", [h, h], [f"{p}/operator_layernorm/output_0"],
                             name=f"{p}/operator_layernorm/LayerNorm"),
            helper.make_node("MatMul", [f"{p}/operator_layernorm/output_0", f"{p}/w_op"], [f"{p}/op/output_0"],
                             name=f"{p}/op/MatMul"),
            helper.make_node("Identity", [f"{p}/op/output_0"], [f"present.{i}.key"], name=f"{p}/attn/present"),
            helper.make_node("Add", [h, f"{p}/op/output_0"], [f"{p}/Add_1/output_0"], name=f"{p}/Add_1"),
            helper.make_node("MatMul", [f"{p}/Add_1/output_0", f"{p}/w_ffn"], [f"{p}/ffn/output_0"],
                             name=f"{p}/ffn/MatMul"),
            helper.make_node("Add", [f"{p}/Add_1/output_0", f"{p}/ffn/output_0"], [f"{p}/Add_2/output_0"],
                             name=f"{p}/Add_2"),
        ]  # fmt: skip
        outputs.append(helper.make_tensor_value_info(f"present.{i}.key", elem, ["batch", "sequence", H]))
        if i == extra_add_layer:
            nodes.append(helper.make_node("Add", [h, h], [f"{p}/Add_3/output_0"], name=f"{p}/Add_3"))
            nodes.append(helper.make_node("Identity", [f"{p}/Add_3/output_0"], ["tap"], name="/model/tap"))
            outputs.append(helper.make_tensor_value_info("tap", elem, ["batch", "sequence", H]))
        h = f"{p}/Add_2/output_0"
    nodes += [
        helper.make_node("Identity", [h], ["/model/logits_to_keep/output_0"], name="/model/logits_to_keep/Slice"),
        helper.make_node("Mul", ["/model/logits_to_keep/output_0", "/model/logits_to_keep/output_0"],
                         [f"/model/layers.{N}/final_norm_layernorm/output_0"],
                         name=f"/model/layers.{N}/final_norm_layernorm/LayerNorm"),
        helper.make_node("MatMul", [f"/model/layers.{N}/final_norm_layernorm/output_0", "embed_t"], ["logits"],
                         name="/lm_head/MatMul"),
    ]  # fmt: skip
    init.append(numpy_helper.from_array(rng.standard_normal((H, 11)).astype(dtype), "embed_t"))
    if subgraph_reader:
        site = "/model/layers.1/Add_2/output_0"
        branch = helper.make_graph(
            [helper.make_node("Identity", [site], ["branch_out"], name="branch_id")],
            "branch",
            [],
            [helper.make_tensor_value_info("branch_out", elem, None)],
        )
        init.append(numpy_helper.from_array(np.asarray(True), "cond"))
        nodes.append(
            helper.make_node("If", ["cond"], ["if_out"], name="/model/probe/If", then_branch=branch, else_branch=branch)
        )
        outputs.append(helper.make_tensor_value_info("if_out", elem, None))
    graph = helper.make_graph(
        nodes,
        "lfm3-like",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"])],
        [helper.make_tensor_value_info("logits", elem, ["batch", "sequence", 11]), *outputs],
        init,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def _expected_lfm3_like(precision: str, sha: str) -> SiteMapping:
    sites = []
    for i in range(N):
        nxt = f"/model/layers.{i + 1}"
        consumers = (
            (f"{nxt}/operator_layernorm/LayerNorm", f"{nxt}/Add_1") if i + 1 < N else ("/model/logits_to_keep/Slice",)
        )
        sites.append(MappedSite(i, "residual_post", f"/model/layers.{i}/Add_2/output_0", H, consumers))
    return SiteMapping("lfm3-like", f"lfm3-like/{precision}/1", sha, precision, tuple(sites))


@pytest.mark.parametrize("dtype,precision", [(np.float32, "fp32"), (np.float16, "fp16")])
def test_lfm3_like_graph(tmp_path, dtype, precision):
    path = _write_base(tmp_path / precision, _lfm3_like(dtype))
    proposal = discover_sites(path)
    assert proposal["warnings"] == []
    assert compare_with_mapping(proposal, _expected_lfm3_like(precision, _sha(path))) == []
    assert {s["dtype"] for s in proposal["sites"]} == {"float32" if precision == "fp32" else "float16"}
    # The final norm numbered like a layer is not a decoder layer: listed, not proposed.
    assert [u["layer"] for u in proposal["unmatched_layers"]] == [N]
    assert proposal["unmatched_layers"][0]["leaving"][0]["op_type"] == "Mul"
    # Evidence: the Add-only path through the layer, from the tensor entering it.
    first, second = proposal["sites"][:2]
    assert first["enters_from"] == ["/model/embed_tokens/Gather/output_0"]
    assert first["residual_path"] == ["/model/layers.0/Add_1", "/model/layers.0/Add_2"]
    assert second["enters_from"] == ["/model/layers.0/Add_2/output_0"]
    # The KV-like graph outputs leave each layer but are not Add outputs.
    assert all(s["other_readers"] == [] and not s["graph_output"] for s in proposal["sites"])


def test_two_residual_candidates_in_a_layer_are_ambiguous(tmp_path):
    path = _write_base(tmp_path / "amb", _lfm3_like(extra_add_layer=2))
    proposal = discover_sites(path)
    assert [s["layer"] for s in proposal["sites"]] == [0, 1, 3]
    assert {u["layer"]: u["reason"] for u in proposal["unmatched_layers"]}[2].startswith("several")
    joined = "\n".join(proposal["warnings"])
    assert "layer 2: several Add outputs" in joined
    assert "not contiguous" in joined
    # Layer 3's residual path starts from layer 2's output, not from the last proposed site (layer 1).
    assert "layer 3: no Add-only path from the previous site" in joined
    # Layer 1's next proposed layer is 3, so layer 2's readers are not proposed.
    assert proposal["sites"][1]["consumers"] == []


def test_a_subgraph_reader_is_warned_about_and_not_proposed(tmp_path):
    path = _write_base(tmp_path / "if", _lfm3_like(subgraph_reader=True))
    proposal = discover_sites(path)
    site = proposal["sites"][1]
    assert site["consumers"] == ["/model/layers.2/operator_layernorm/LayerNorm", "/model/layers.2/Add_1"]
    joined = "\n".join(proposal["warnings"])
    assert "/model/probe/If" in joined and "read inside subgraphs" in joined


def test_compare_reports_each_difference(tmp_path):
    path = _write_base(tmp_path / "cmp", _lfm3_like())
    proposal = discover_sites(path)
    good = _expected_lfm3_like("fp32", _sha(path))
    sites = list(good.sites)
    sites[1] = MappedSite(1, "residual_post", sites[1].tensor, H, tuple(reversed(sites[1].consumers)))
    sites[2] = MappedSite(2, "residual_post", "/model/layers.2/Add_1/output_0", H, sites[2].consumers)
    bad = SiteMapping("lfm3-like", "lfm3-like/fp16/1", "0" * 64, "fp16", tuple(sites[:3]))
    diffs = compare_with_mapping(proposal, bad)
    assert any(d.startswith("graph SHA-256") for d in diffs)
    assert any(d.startswith("precision fp32") for d in diffs)
    assert any(d.startswith("site (1, 'residual_post'): consumers") for d in diffs)
    assert any(d.startswith("site (2, 'residual_post'): tensor") for d in diffs)
    assert any(d == "site (3, 'residual_post'): proposed, not in the mapping" for d in diffs)


def test_proposal_is_deterministic_json(tmp_path):
    path = _write_base(tmp_path / "det", _lfm3_like())
    a, b = proposal_json(discover_sites(path)), proposal_json(discover_sites(path))
    assert a == b and a.endswith("\n")
    assert json.loads(a)["status"].startswith("unreviewed proposal")


def test_bad_inputs(tmp_path):
    path = _write_base(tmp_path / "bad", _lfm3_like())
    with pytest.raises(SiteDiscoveryError, match="needs a group"):
        discover_sites(path, layer_pattern=r"layers\.\d+/")
    with pytest.raises(SiteDiscoveryError, match="not a regular expression"):
        discover_sites(path, layer_pattern="(")
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"\xff\x00not a model")
    with pytest.raises(SiteDiscoveryError, match="not an ONNX model"):
        discover_sites(junk)


def test_nothing_turns_a_proposal_into_a_registered_mapping():
    # The proposal is for a reviewer; the registry holds only hand-written mappings.
    assert not any(name.startswith(("load", "mapping_from", "register")) for name in dir(discover))


# --------------------------------------------------------------------------------------------------------------------
# CLI


def test_cli_writes_the_proposal_and_compares(tmp_path, monkeypatch, capsys):
    path = _write_base(tmp_path / "cli", _lfm3_like())
    mapping = _expected_lfm3_like("fp32", _sha(path))
    monkeypatch.setattr(discover, "SITE_MAPPINGS", {mapping.base_graph_sha256: mapping})
    out = tmp_path / "proposal.json"
    assert cli.main(["discover-sites", str(path), "--out", str(out), "--compare", mapping.version]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == discover_sites(path)
    assert "reproduces lfm3-like/fp32/1 exactly" in capsys.readouterr().err

    other = SiteMapping("x", "x/fp32/1", mapping.base_graph_sha256[::-1], "fp32", mapping.sites[:2])
    monkeypatch.setattr(discover, "SITE_MAPPINGS", {other.base_graph_sha256: other})
    assert cli.main(["discover-sites", str(path), "--out", str(out), "--compare", "x/fp32/1"]) == 1
    assert "differs from x/fp32/1" in capsys.readouterr().err
    assert cli.main(["discover-sites", str(path), "--compare", "nope/1"]) == 2
    assert cli.main(["discover-sites", str(tmp_path / "missing.onnx")]) == 2


def test_cli_prints_to_stdout_and_matches_the_module_defaults(tmp_path, capsys):
    path = _write_base(tmp_path / "stdout", _lfm3_like())
    assert cli.main(["discover-sites", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["proposed"]["layers"] == N
    assert discover.main([str(path), "--out", str(tmp_path / "p.json")]) == 0
    parser = cli.argparse.ArgumentParser()
    cli._discover_arguments(parser)
    assert parser.parse_args(["g"]).layer_pattern == DEFAULT_LAYER_PATTERN


# --------------------------------------------------------------------------------------------------------------------
# The real LFM3 graphs (structure only)


def _real_graph(name: str) -> Path:
    path = Path(SNAPSHOT or "", "onnx", name)
    if not SNAPSHOT or not path.is_file():
        pytest.skip(f"set STARLING_LFM3_SNAPSHOT to the LFM3-1.7B-ONNX snapshot (c680759a) with onnx/{name}")
    return path


@pytest.mark.parametrize(
    "name,mapping,dtype",
    [("model.onnx", LFM3_FP32_MAPPING, "float32"), ("model_fp16.onnx", LFM3_FP16_MAPPING, "float16")],
)
def test_the_real_lfm3_graph_regenerates_its_mapping_exactly(name, mapping, dtype):
    proposal = discover_sites(_real_graph(name))
    assert compare_with_mapping(proposal, mapping) == []
    assert proposal["warnings"] == []
    assert [s["layer"] for s in proposal["sites"]] == list(range(24))
    assert {s["dtype"] for s in proposal["sites"]} == {dtype}
    # Field for field, consumers in the mapping's order.
    assert [
        (s["layer"], s["target"], s["tensor"], s["hidden_size"], tuple(s["consumers"])) for s in proposal["sites"]
    ] == [(m.layer, m.target, m.tensor, m.hidden_size, m.consumers) for m in mapping.sites]
    # Layer 24 is the exporter's final norm, not a decoder layer.
    assert [u["layer"] for u in proposal["unmatched_layers"]] == [24]
    assert proposal["sites"][0]["enters_from"] == ["/model/embed_tokens/Gather/output_0"]
