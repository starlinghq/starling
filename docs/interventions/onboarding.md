# Onboarding a model's site mapping

Status: **experimental**. Code: `model-pipeline/src/model_pipeline/interventions/discover.py` (the `discover-sites` command) and `injection_sites.py` (the checked-in mappings). Tests: `model-pipeline/tests/test_intervention_discover.py`, `test_intervention_lfm3.py`.

The compiler only intervenes in a graph whose exact SHA-256 has a site mapping in `injection_sites.SITE_MAPPINGS` ([compiler.md](compiler.md), "Site mappings"). It never searches an unknown graph. `discover-sites` helps a person write a mapping for a new graph: it reads the graph's structure, proposes one residual-stream tensor per decoder layer and lists the nodes that read it, and writes the result as a JSON **proposal**. A proposal is never loaded at runtime, and nothing in the package turns one into a registered mapping. A reviewer checks it and then writes the mapping by hand, keyed to the asset hash and versioned.

## Steps

1. **Pin the graph.** Use the exact `model.onnx` the model's asset lock records (for .NET models, `<Model>ModelAssets.cs`). Download it at the pinned revision and compare its SHA-256 with the lock. The proposal records the hash it read, and the mapping is keyed to that hash. A different export, revision or precision is a different graph and needs its own mapping.

2. **Generate the proposal.** This reads only the graph file (about 275 KB for LFM3), never the external weights:

   ```sh
   model-pipeline discover-sites <snapshot>/onnx/model.onnx --out proposal.json
   # or: python -m model_pipeline.interventions.discover <graph> --out proposal.json
   ```

   Decoder layers are found from node names. The default `--layer-pattern` is `(?:^|/)layers\.(\d+)/`, which matches the `/model/layers.L/...` naming of the LFM3 export. If another exporter names nodes differently, pass a regular expression whose first group is the layer index; the compiler tests' toy graph uses `^layer(\d+)/`. If nothing matches, the proposal has no sites and its warning names `--layer-pattern`.

3. **Review the proposal.** For each decoder layer, the proposal's `sites[]` gives the candidate `tensor`, its `dtype`, `shape` and `hidden_size`, the `producer` node, the proposed `consumers`, any `other_readers`, and the evidence (`residual_path`, `enters_from`). The rules behind a proposal are in the module docstring of `discover.py`. Check every item of the proposal's `review_checklist`:
   - `graph.sha256` is the lock's hash.
   - There is one site per decoder layer, numbered `0..N-1`, and N is the model's layer count. `unmatched_layers` holds only indices that are not decoder layers. In both LFM3 graphs, index 24 is the exporter's final norm, `/model/layers.24/final_norm_layernorm`, whose output leaves the "layer" through a norm, not an `Add`.
   - Each tensor is the residual stream *after* the layer. `residual_path` goes through the layer's residual adds, and `enters_from` is the previous site, or the embedding output for layer 0. The dtype and hidden size match the model config.
   - `consumers` are every reader that must see the intervened value: the next layer's readers, and after the last layer the reader that leads to the final norm and LM head. Any reader in `other_readers` keeps the original value unless you add it, so decide for each one. A graph output that exposes a site tensor always keeps the original value.
   - No warning is left unexplained. Warnings flag an ambiguous layer (two `Add` outputs leave it), a gap in the layer numbers, a residual path that does not start at the previous site, an unknown hidden size, readers inside subgraphs (the compiler does not rewire those), and unnamed nodes (a mapping names its consumers, so unnamed readers cannot be listed).

   Also look at the graph around one or two layers, for example in Netron. Discovery is a heuristic over the graph structure; the reviewer decides.

4. **Check in the mapping.** Add a `SiteMapping` to `injection_sites.py`, next to the existing ones, and add it to `build_registry((...))`. Give it the model name, a version label `<model>/<precision>/<n>`, `base_graph_sha256` set to the lock hash, the precision, and one `MappedSite(layer, "residual_post", tensor, hidden_size, consumers)` per layer, with consumers in graph order. Write the tensor names into the source; never compute them at import from a proposal file. Keep the proposal out of the repository; the mapping and its test are the record. To change a mapping, add a new version; do not edit a published one.

5. **Test it.**
   - A model-free test that the mapping is keyed to the lock: read the hash from the asset lock source, as `test_mapping_is_keyed_to_the_fp32_lock_hash` does.
   - With the real graph (env-gated): every reader of each site tensor is a listed consumer, the producer is an `Add`, and the dtype and hidden size match (`_check_mapping_against_graph` in `test_intervention_lfm3.py`).
   - With the real graph: `discover-sites` regenerates the mapping exactly. Run `model-pipeline discover-sites <graph> --compare <version>`, which exits 1 on any difference, and add a case to `test_the_real_lfm3_graph_regenerates_its_mapping_exactly`. If you edited a proposal during review, the regenerated proposal will differ there. Record each such difference in the mapping's docstring and assert it in the test.
   - With the weights (env-gated): compile a synthetic profile (seeded random orthonormal directions; `synthetic.write_synthetic_profile`) and check that strength 0 equals the base graph's outputs, that strength 1 moves them, and that a site tensor exposed as an output matches the NumPy operator. For a float16 graph, use a float16 tolerance (see below).

6. **Enable the runtime integration separately.** A Python mapping lets the compiler build derived graphs. A .NET model integration accepts an entry only for mapping versions it lists (for LFM3, `Lfm3Interventions.Fp32SiteMappingVersion`), so a new graph or precision is a separate change with its own real-model .NET tests and capability status.

## float16 graphs

The compiler checks each site's element type. For a float16 site it inserts `Cast(y)` to float32, computes the operator in float32, and inserts `Cast(delta)` to float16 before the `Sub` (or `Add`) on the float16 stream. The profile's directions stay float32. At strength 0 the correction is exactly zero, so `y - 0 = y` and the result can differ from the base graph only through operator fusion around the extra reader. That is why the fp16 strength-0 test allows two float16 units in the last place at the logits' magnitude, `2 * 2^-10 * max|logits|`, and also requires the same argmax; the measured difference is 0. The site-tensor check allows `2 * 2^-10 * (max|y| + max|delta|)`, because the correction and the `Sub` round to float16.

## Worked example: LFM3-1.7B

| Graph | SHA-256 (lock) | Proposal | Mapping |
| --- | --- | --- | --- |
| `onnx/model.onnx` (fp32) | `add6653f…5ef77` | 24 sites, 0 warnings, layer 24 unmatched; `--compare lfm3-1.7b/fp32/1`: reproduces it exactly | `lfm3-1.7b/fp32/1` (written by hand before the tool existed) |
| `onnx/model_fp16.onnx` | `d3093d69…830fe` | 24 sites, 0 warnings, layer 24 unmatched; tensors and consumers have the same names as fp32, element type float16 | `lfm3-1.7b/fp16/1` (from this proposal, reviewed) |

The fp16 review found that the residual adds `/model/layers.L/Add_1` and `/model/layers.L/Add_2` have the same structure as in the fp32 graph. `enters_from` is the previous layer's `Add_2` output, or `/model/embed_tokens/Gather/output_0` for layer 0. There are no other readers, no subgraphs, and the hidden size is 2048. The graph's three `Cast` nodes are outside the residual path: two are in the attention-mask subgraph, and the third is `graph_output_cast0`, which casts the logits to float32. Evidence for the fp16 mapping is in [lfm3.md](lfm3.md).
