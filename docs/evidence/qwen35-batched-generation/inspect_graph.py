"""Print the op inventory and the inputs of the stateful ops of an ONNX graph (no weights loaded)."""

import collections
import sys

import onnx

model = onnx.load(sys.argv[1], load_external_data=False)
g = model.graph
print("graph inputs:")
for i in g.input[:12]:
    dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
    print("  ", i.name, dims)
print("  ...", len(g.input), "inputs total")
for i in g.input:
    if ".conv" in i.name or ".recurrent" in i.name:
        dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        print("  ", i.name, dims)
        break
for i in g.input:
    if ".recurrent" in i.name:
        dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        print("  ", i.name, dims)
        break
ops = collections.Counter((n.domain, n.op_type) for n in g.node)
print("ops:", sorted(ops.items(), key=lambda kv: -kv[1])[:40])
seen = set()
for n in g.node:
    if n.op_type in ("LinearAttention", "CausalConvWithState", "GroupQueryAttention") and n.op_type not in seen:
        seen.add(n.op_type)
        print(n.op_type, n.domain)
        print("   inputs:", list(n.input))
        print("   outputs:", list(n.output))
        print("   attrs:", [(a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute])
# what consumes attention_mask (transitively, first hops)
consumers = [n for n in g.node if "attention_mask" in n.input]
print("direct attention_mask consumers:", [(n.op_type, n.name) for n in consumers])
