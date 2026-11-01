"""Generate the tiny weight-free ONNX graphs used by the ai-dotnet tests.

The graphs have no initializers (no weights); they let session, disposal, concurrency, cancellation, batching and
pooling tests run in CI without the e5 model. *.onnx files are not committed (plan section 9), so the serialized
bytes are embedded as Base64 in TestModels.g.cs.

Regenerate with onnx 1.23.1 from this directory:
    python make_test_models.py > TestModels.g.cs
"""

import base64

import onnx
from onnx import TensorProto, helper


def finish(graph: onnx.GraphProto, props: dict[str, str] | None = None) -> bytes:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="starlings-test")
    model.ir_version = 10
    if props:
        helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def add_model() -> bytes:
    """z = x + y, float [batch, 4]."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", 4])
    graph = helper.make_graph([helper.make_node("Add", ["x", "y"], ["z"])], "add", [x, y], [z])
    return finish(graph, {"purpose": "runtime-tests"})


def add_with_strength_model() -> bytes:
    """The shape of a derived intervention graph of Add: s = x + y; z = s - intervention_strength * s.

    One extra float [1] input, appended after the base inputs, as the intervention compiler adds it; strength 0 gives
    exactly the Add graph's output.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    strength = helper.make_tensor_value_info("intervention_strength", TensorProto.FLOAT, [1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", 4])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["sum"]),
        helper.make_node("Mul", ["sum", "intervention_strength"], ["delta"]),
        helper.make_node("Sub", ["sum", "delta"], ["z"]),
    ]
    graph = helper.make_graph(nodes, "add-intervention", [x, y, strength], [z])
    return finish(graph, {"purpose": "runtime-tests"})


def slow_model(steps: int = 300) -> bytes:
    """x [n, n] -> Tanh(MatMul(h, h)) repeated; long enough to cancel between nodes."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", "n"])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", "n"])
    nodes = []
    prev = "x"
    for i in range(steps):
        nodes.append(helper.make_node("MatMul", [prev, prev], [f"m{i}"]))
        name = "y" if i == steps - 1 else f"t{i}"
        nodes.append(helper.make_node("Tanh", [f"m{i}"], [name]))
        prev = name
    return finish(helper.make_graph(nodes, "slow", [x], [out]))


def fake_e5(with_token_types: bool) -> bytes:
    """e5 signature with a closed-form hidden state that mean pooling can be checked against.

    last_hidden_state[b, t, :] = [input_ids, 1000 * token_type_ids, 1, 0.5 * attention_mask]
    (component 1 is 0 when the graph has no token_type_ids input).
    """
    dims = ["batch_size", "sequence_length"]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, dims),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, dims),
    ]

    def const(name: str, value: float) -> onnx.NodeProto:
        return helper.make_node(
            "Constant", [], [name], value=helper.make_tensor(name + "_v", TensorProto.FLOAT, [], [value])
        )

    nodes = [
        helper.make_node("Constant", [], ["axes"], value=helper.make_tensor("axes_v", TensorProto.INT64, [1], [2])),
        const("zero", 0.0),
        const("one", 1.0),
        const("half", 0.5),
        const("thousand", 1000.0),
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["mask_f", "half"], ["mask_term"]),
        helper.make_node("Mul", ["ids_f", "zero"], ["zeros"]),
        helper.make_node("Add", ["zeros", "one"], ["ones"]),
    ]
    if with_token_types:
        inputs.append(helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, dims))
        nodes += [
            helper.make_node("Cast", ["token_type_ids"], ["tt_f"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["tt_f", "thousand"], ["tt_term"]),
        ]
        tt = "tt_term"
    else:
        tt = "zeros"
    features = ["ids_f", tt, "ones", "mask_term"]
    for i, f in enumerate(features):
        nodes.append(helper.make_node("Unsqueeze", [f, "axes"], [f"c{i}"]))
    nodes.append(helper.make_node("Concat", [f"c{i}" for i in range(4)], ["last_hidden_state"], axis=2))
    out = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch_size", "sequence_length", 4])
    return finish(helper.make_graph(nodes, "fake_e5", inputs, [out]))


def fake_gliner2() -> bytes:
    """GLiNER2 route A signature with closed-form outputs.

    count_logits[b] = [0, 1, 0, ..., 0] (the gate is always open)
    span_logits[b, m, w, k] = WIDTH[k] + 0.001 * word_index[b, w]   for every label m
    WIDTH = [4, -4, -12, ...]: single-word spans score about 0.982, two-word spans about 0.018, wider ones less.
    input_ids and attention_mask only contribute zeros (so every input is used).
    """
    width = [4.0 - 8.0 * k for k in range(8)]
    count_row = [0.0] * 20
    count_row[1] = 1.0
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence"]),
        helper.make_tensor_value_info("word_index", TensorProto.INT64, ["batch", "words"]),
        helper.make_tensor_value_info("schema_index", TensorProto.INT64, ["batch", "schema_slots"]),
    ]

    def const(name: str, dims: list[int], values: list, dtype=TensorProto.FLOAT) -> onnx.NodeProto:
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, dims, values))

    nodes = [
        const("width", [1, 1, 1, 8], width),
        const("count_row", [1, 20], count_row),
        const("scale", [], [0.001]),
        const("zero", [], [0.0]),
        const("ax1", [1], [1], TensorProto.INT64),
        const("ax13", [2], [1, 3], TensorProto.INT64),
        const("ax23", [2], [2, 3], TensorProto.INT64),
        const("one_i", [1], [1], TensorProto.INT64),
        const("big_i", [1], [2**62], TensorProto.INT64),
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["ids_f", "mask_f"], ["seq_f"]),
        helper.make_node("Mul", ["seq_f", "zero"], ["seq_zero"]),
        helper.make_node("ReduceSum", ["seq_zero", "ax1"], ["row_zero"], keepdims=1),
        helper.make_node("Add", ["row_zero", "count_row"], ["count_logits"]),
        helper.make_node("Cast", ["word_index"], ["wi_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["wi_f", "scale"], ["wi_term"]),
        helper.make_node("Unsqueeze", ["wi_term", "ax13"], ["wi4"]),
        helper.make_node("Cast", ["schema_index"], ["si_f"], to=TensorProto.FLOAT),
        helper.make_node("Slice", ["si_f", "one_i", "big_i", "ax1"], ["labels_f"]),
        helper.make_node("Mul", ["labels_f", "zero"], ["labels_zero"]),
        helper.make_node("Unsqueeze", ["labels_zero", "ax23"], ["labels4"]),
        helper.make_node("Add", ["wi4", "labels4"], ["base"]),
        helper.make_node("Add", ["base", "width"], ["span_logits"]),
    ]
    outputs = [
        helper.make_tensor_value_info("count_logits", TensorProto.FLOAT, ["batch", 20]),
        helper.make_tensor_value_info("span_logits", TensorProto.FLOAT, ["batch", "labels", "words", 8]),
    ]
    return finish(helper.make_graph(nodes, "fake_gliner2", inputs, outputs))


def fake_embeddinggemma2() -> bytes:
    """EmbeddingGemma 2 text-path signature with closed-form outputs.

    token_embeddings[b, t, d] = input_ids[b, t] * (1 + d / 768) + attention_mask[b, t] * cos(d)   (768 wide)
    last_hidden_state[b, t, d] = input_ids[b, t]                                               (512 wide)
    """
    import math

    dims = ["batch", "sequence"]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, dims),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, dims),
    ]

    def const(name: str, shape: list[int], values: list, dtype=TensorProto.FLOAT) -> onnx.NodeProto:
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, shape, values))

    nodes = [
        const("axes", [1], [2], TensorProto.INT64),
        const("w", [768], [1.0 + d / 768.0 for d in range(768)]),
        const("v", [768], [math.cos(d) for d in range(768)]),
        const("ones512", [512], [1.0] * 512),
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axes"], ["ids3"]),
        helper.make_node("Unsqueeze", ["mask_f", "axes"], ["mask3"]),
        helper.make_node("Mul", ["ids3", "w"], ["id_term"]),
        helper.make_node("Mul", ["mask3", "v"], ["mask_term"]),
        helper.make_node("Add", ["id_term", "mask_term"], ["token_embeddings"]),
        helper.make_node("Mul", ["ids3", "ones512"], ["last_hidden_state"]),
    ]
    outputs = [
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", 512]),
        helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, ["batch", "sequence", 768]),
    ]
    return finish(helper.make_graph(nodes, "fake_embeddinggemma2", inputs, outputs))


def fake_embeddinggemma2_vision() -> bytes:
    """EmbeddingGemma 2 vision-graph signature with closed-form outputs (slots = patches / 9).

    soft_tokens[b, s, :] = mean(pixel_values[b, 9s:9s+9, :])          (512 wide, the same value in every column)
    pooled[b, s, :] = soft_tokens[b, s, 0]                              (768 wide)
    soft_token_mask[b, s] = pixel_position_ids[b, 9s, 0] >= 0
    """
    inputs = [
        helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", "patches", 768]),
        helper.make_tensor_value_info("pixel_position_ids", TensorProto.INT64, ["batch", "patches", 2]),
    ]

    def const(name: str, shape: list[int], values: list, dtype=TensorProto.FLOAT) -> onnx.NodeProto:
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, shape, values))

    nodes = [
        const("shape_pv", [3], [0, -1, 9 * 768], TensorProto.INT64),
        const("shape_pos", [3], [0, -1, 18], TensorProto.INT64),
        const("axis2", [1], [2], TensorProto.INT64),
        const("zero_idx", [1], [0], TensorProto.INT64),
        const("zero_i64", [], [0], TensorProto.INT64),
        const("ones512", [512], [1.0] * 512),
        const("ones768", [768], [1.0] * 768),
        helper.make_node("Reshape", ["pixel_values", "shape_pv"], ["pv_slots"]),
        helper.make_node("ReduceMean", ["pv_slots"], ["mean"], axes=[2], keepdims=1),
        helper.make_node("Mul", ["mean", "ones512"], ["soft_tokens"]),
        helper.make_node("Mul", ["mean", "ones768"], ["pooled"]),
        helper.make_node("Reshape", ["pixel_position_ids", "shape_pos"], ["pos_slots"]),
        helper.make_node("Gather", ["pos_slots", "zero_idx"], ["first_x"], axis=2),
        helper.make_node("Squeeze", ["first_x", "axis2"], ["first_x2"]),
        helper.make_node("GreaterOrEqual", ["first_x2", "zero_i64"], ["soft_token_mask"]),
    ]
    outputs = [
        helper.make_tensor_value_info("soft_tokens", TensorProto.FLOAT, ["batch", "slots", 512]),
        helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["batch", "slots", 768]),
        helper.make_tensor_value_info("soft_token_mask", TensorProto.BOOL, ["batch", "slots"]),
    ]
    return finish(helper.make_graph(nodes, "fake_embeddinggemma2_vision", inputs, outputs))


def fake_embeddinggemma2_multimodal() -> bytes:
    """EmbeddingGemma 2 multimodal text-graph signature with closed-form outputs.

    token_embeddings[b, t, d] = input_ids[b, t] * (1 + d / 768) + attention_mask[b, t] * cos(d)
                                + multimodal_mask[b, t] * multimodal_embeds[b, t, 0] * 1000      (768 wide)
    last_hidden_state[b, t, d] = input_ids[b, t]                                                  (512 wide)
    """
    import math

    dims = ["batch", "sequence"]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, dims),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, dims),
        helper.make_tensor_value_info("multimodal_embeds", TensorProto.FLOAT, ["batch", "sequence", 512]),
        helper.make_tensor_value_info("multimodal_mask", TensorProto.BOOL, dims),
    ]

    def const(name: str, shape: list[int], values: list, dtype=TensorProto.FLOAT) -> onnx.NodeProto:
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(name + "_v", dtype, shape, values))

    nodes = [
        const("axes", [1], [2], TensorProto.INT64),
        const("first", [1], [0], TensorProto.INT64),
        const("w", [768], [1.0 + d / 768.0 for d in range(768)]),
        const("v", [768], [math.cos(d) for d in range(768)]),
        const("k", [768], [1000.0] * 768),
        const("ones512", [512], [1.0] * 512),
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["multimodal_mask"], ["mm_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axes"], ["ids3"]),
        helper.make_node("Unsqueeze", ["mask_f", "axes"], ["mask3"]),
        helper.make_node("Unsqueeze", ["mm_f", "axes"], ["mm3"]),
        helper.make_node("Gather", ["multimodal_embeds", "first"], ["e0"], axis=2),
        helper.make_node("Mul", ["e0", "mm3"], ["e0m"]),
        helper.make_node("Mul", ["e0m", "k"], ["mm_term"]),
        helper.make_node("Mul", ["ids3", "w"], ["id_term"]),
        helper.make_node("Mul", ["mask3", "v"], ["mask_term"]),
        helper.make_node("Add", ["id_term", "mask_term"], ["text_terms"]),
        helper.make_node("Add", ["text_terms", "mm_term"], ["token_embeddings"]),
        helper.make_node("Mul", ["ids3", "ones512"], ["last_hidden_state"]),
    ]
    outputs = [
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", 512]),
        helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, ["batch", "sequence", 768]),
    ]
    return finish(helper.make_graph(nodes, "fake_embeddinggemma2_multimodal", inputs, outputs))


def emit(doc: str, name: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return (
        f"    /// <summary>{doc}</summary>\n"
        f"    public static readonly byte[] {name} = Convert.FromBase64String(\n"
        f'        "{b64}");\n'
    )


parts = [
    emit("z = x + y, float [batch, 4].", "Add", add_model()),
    emit(
        "Derived-intervention shape of Add: z = (x + y) - intervention_strength * (x + y); intervention_strength float [1].",
        "AddWithStrength",
        add_with_strength_model(),
    ),
    emit("300 x Tanh(MatMul(h, h)) on float [n, n].", "Slow", slow_model()),
    emit(
        "e5 signature; hidden[b, t] = [id, 1000 * token_type_id, 1, 0.5 * mask].",
        "FakeE5",
        fake_e5(True),
    ),
    emit(
        "e5 signature without token_type_ids; hidden[b, t] = [id, 0, 1, 0.5 * mask].",
        "FakeE5NoTokenTypes",
        fake_e5(False),
    ),
    emit(
        "GLiNER2 route A signature; count_logits = [0, 1, 0, ...]; span_logits[b, m, w, k] = 4 - 8k + 0.001 * word_index[b, w].",
        "FakeGliner2",
        fake_gliner2(),
    ),
    emit(
        "EmbeddingGemma 2 text-path signature; token_embeddings[b, t, d] = id * (1 + d / 768) + mask * cos(d); last_hidden_state = id (512 wide).",
        "FakeEmbeddingGemma2",
        fake_embeddinggemma2(),
    ),
    emit(
        "EmbeddingGemma 2 vision-graph signature; soft_tokens[b, s, :] = mean(pixel_values[b, 9s:9s+9, :]); mask = position x of patch 9s >= 0.",
        "FakeEmbeddingGemma2Vision",
        fake_embeddinggemma2_vision(),
    ),
    emit(
        "EmbeddingGemma 2 multimodal text-graph signature; token_embeddings = id * (1 + d / 768) + mask * cos(d) + mm_mask * mm_embeds[..., 0] * 1000.",
        "FakeEmbeddingGemma2Multimodal",
        fake_embeddinggemma2_multimodal(),
    ),
]
print("// <auto-generated> by make_test_models.py (onnx 1.23.1). Weight-free test graphs. </auto-generated>")
print("namespace Starlings.TestSupport;")
print()
print("internal static class TestModels")
print("{")
print("\n".join(parts), end="")
print("}")
