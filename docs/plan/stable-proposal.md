# Stable proposal: release sets, task datasets and budgets

Status: **proposal**, 2026-10-14. Nothing here is binding. It is the input for maintainer actions M1 (P-030 release sets) and M2 (G3 datasets and quality budgets, G5 performance budgets) in [production-readiness.md](production-readiness.md). Every file it proposes sits under `model-pipeline/proposed/stable/`, with a `PROPOSED.md` per directory naming its sources, licenses, revisions, selection script and intended destination. An accepted file moves byte-identical to `model-pipeline/protected/` in a `protected-change` pull request, as for every other proposal (`model-pipeline/proposed/README.md`).

The anti-overfitting rule (plan 10.1, 17) is kept: **no example in this proposal was written by an agent.** Every text, sentence, prompt, image and video is copied verbatim from an externally authored public dataset, pinned by full commit SHA and SHA-256 (`model-pipeline/reference/stable/sources.lock.json`), or from Wikimedia Commons by recorded file revision and SHA-1. The agent wrote only the selection procedures (`model-pipeline/reference/stable/`), each a filter plus one seeded shuffle (seed 20261014), with no choice per row. Every output regenerates byte-identical.

## 1. Datasets and licenses

Licenses were read from the dataset card front matter and text, and the repository `LICENSE` where one exists, at the pinned revision on 2026-10-14 (Hugging Face API), and from the upstream repository where the Hub copy is a loader. Cards were treated as data.

### Chosen

| Use | Dataset | Pinned revision | License | Notes |
| --- | --- | --- | --- | --- |
| Tokenizer holdouts (all four profiles), e5 and EmbeddingGemma 2 text release sets | MASSIVE 1.1 test (`AmazonScience/massive`, Hub Parquet conversion) | `ed58ac423a2f4121720918bf5301577edce4ffd3` (loader `main` `ff6bd8e4…`) | CC-BY-4.0 | 52 locales, 20+ scripts |
| same | HumanEval (`openai/openai_humaneval`) | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | MIT | code, indentation, doctests |
| same, and the Qwen release prompts | OpenAssistant oasst1 validation (`OpenAssistant/oasst1`) | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | long Markdown replies (truncation path); root prompts |
| G3 retrieval (e5, EmbeddingGemma 2), browser int8 top-10 overlap | MIRACL dev, MTEB hard-negative corpus (`mteb/MIRACLRetrievalHardNegatives`) | `332a9acb49f5e83d5397683f79d23e588f685916` | CC-BY-SA-4.0 (topics and judgments Apache-2.0 per `miracl/miracl` `5be20db9…`; passages are Wikipedia) | **share-alike, flagged** |
| G3 NER (GLiNER2), primary | WNUT-17 test (`leondz/emerging_entities_17` on GitHub; Hub loader `leondz/wnut_17` `be8fbd57…`) | `e52a2d2a71ac1a051ca2d532eef28653c5c02603` | CC-BY-4.0 (upstream README and Hub card) | user-generated text (YouTube, Stack Overflow, Twitter, Reddit) |
| G3 NER (GLiNER2), alternative | Few-NERD supervised test (`DFKI-SLT/few-nerd`) | `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b` | CC-BY-SA-4.0 | **share-alike, flagged**; Wikipedia text |
| G3 generation (Qwen) | GSM8K main test (`openai/gsm8k`) | `740312add88f781978c0658806c59bc2815b9866` | MIT | automatic exact match |
| EmbeddingGemma 2 image and video release media | Wikimedia Commons: Featured pictures; Videos by NASA | per file: revision timestamp and SHA-1 in `media.json` | public domain, CC0, CC BY 2.0/4.0 (per file, Commons metadata; **each to be confirmed by the maintainer**) | bytes never committed |

**Share-alike.** CC BY-SA does not restrict running an evaluation or publishing its scores. It does apply to the committed excerpts: `corpus.jsonl` of the MIRACL subset and `gold.jsonl` of Few-NERD stay under CC BY-SA 4.0 with attribution, inside an MIT/Apache repository, and must be marked so (dataset.json says it; a `THIRD-PARTY-NOTICES.md` entry would too). If the maintainer prefers no share-alike text in the repository, the alternative is to commit only the selection script and digests and materialize the set at evaluation time (`fetch_sources.py` + `select_*.py` regenerate it byte-identical). For NER that choice is easy (WNUT-17 is CC-BY-4.0 and is proposed as primary). For multilingual retrieval there is no clean non-SA alternative found: every permissive multilingual retrieval set checked uses Wikipedia passages.

### Rejected or flagged

| Dataset | Finding | Decision |
| --- | --- | --- |
| SciFact (BEIR, MTEB) | upstream `allenai/scifact` card: CC BY-NC 2.0; BEIR and MTEB copies label it CC-BY-SA-4.0 or unknown | **rejected** (non-commercial upstream; the downstream relabel does not change it) |
| NFCorpus (BEIR, MTEB) | `mteb/nfcorpus` states no license; `BeIR/nfcorpus` says CC-BY-SA-4.0 without an upstream statement | **rejected** (unclear) |
| ArguAna (BEIR, MTEB) | MTEB card CC-BY-SA-4.0; upstream (idebate.org via the ArguAna corpus) states no license | flagged, not used (unclear upstream) |
| NQ (`mteb/nq`) | card: CC-BY-NC-SA-3.0 | rejected (NC) |
| FiQA, TREC-COVID, Touché (MTEB copies) | unknown or no license in the card | not used |
| XPQA retrieval | CDLA-Sharing-1.0 | not used (share-alike variant, no gain over MIRACL) |
| Mintaka retrieval | CC-BY-4.0, but documents are short answer strings | not used (unrepresentative documents); a CC-BY fallback for e5 if SA is refused |
| CoNLL-2003 | card `license: other`; Reuters RCV1 text under a research agreement | **rejected** |
| OntoNotes 5 | LDC license (`other`) | **rejected** |
| WikiNEuRal, MultiNERD | CC-BY-NC-SA-4.0 | **rejected** (NC) |
| CrossNER | Hub card states no license | **rejected** (unclear) |
| ARC (`allenai/ai2_arc`) | CC-BY-SA-4.0 | not used; GSM8K (MIT) preferred for Qwen |
| Alpaca, No Robots | CC-BY-NC-4.0 | rejected for the Qwen prompts; oasst1 (Apache-2.0) used |
| databricks-dolly-15k | CC-BY-SA-3.0 | not used (oasst1 is permissive) |
| FLORES+, Tatoeba | CC-BY-SA-4.0 (gated); CC-BY-2.0 but a loader without a pinned data file | not used; MASSIVE covers more scripts under CC-BY-4.0 |

## 2. P-030 release sets

| Profile family | Proposed set | Generated here | Location |
| --- | --- | --- | --- |
| e5 (XLM-R unigram) | 232-row tokenizer holdout, also the text-encoder release set | yes: `run_e5.py` output (tokenizer, batch, embeddings, parity) | `multilingual-e5-small/tokenizer-holdout/`, `…/onnx-fp32-cpu-release/` |
| EmbeddingGemma 2 text (Gemma BPE) | 232-row holdout, also the text-encoder release set | yes: `run_embeddinggemma2.py` output; the 7 MB `embeddings.jsonl` is pinned by digest, not committed | `embeddinggemma-2/tokenizer-holdout/`, `…/onnx-fp32-cpu-release/` |
| EmbeddingGemma 2 image and video | 16 images, 6 videos from Commons | selection and verified fetch; reference fixtures need a runner option (below) | `embeddinggemma-2/media/` |
| Qwen (byte-level BPE, chat) | 232-row tokenizer holdout; 32 oasst1 release prompts | tokenizer fixtures yes (0 mismatches vs transformers); generation fixture yes (transformers CPU fp32, 5 minutes; all 32 stop at 64 tokens; smallest greedy margin 5.7e-4) | `qwen3.5-0.8b/tokenizer-holdout/`, `…/tokenizer-release/`, `…/release-prompts/` |
| GLiNER2 (DeBERTa-v3 unigram) | 232-row tokenizer holdout; 60-row extraction release corpus (first 30 rows of each NER subset) | tokenizer fixtures yes; extraction fixture: commands only | `gliner2-base-v1/tokenizer-holdout/`, `…/tokenizer-release/`, `…/ner/extraction-release-v1.jsonl` |

Each tokenizer holdout has 208 MASSIVE utterances (4 per locale x 52), 8 HumanEval prompts and 16 oasst1 assistant replies (200 to 6,000 characters). The four holdouts are disjoint, and rows found in any development corpus are excluded. Findings on the holdouts, all known behaviors already recorded for the development fixtures: e5's sentence-transformers tokenizer drops a trailing `▁` on 8 rows (fixtures follow `tokenizer.json`); GLiNER2's fast and slow DeBERTa tokenizers differ on 9 rows (fixtures follow the fast tokenizer GLiNER2 loads); EmbeddingGemma 2 and Qwen show no implementation differences. Reference stability on the holdouts is in each `PROPOSED.md`; for example e5 Python ORT vs PyTorch: embedding max abs 1.5e-7, EmbeddingGemma 2 padded batch vs single 5.4e-7. Neither suggests a tolerance change.

Runner changes in this proposal: `--batch-ids` on `run_e5.py` and `run_embeddinggemma2.py` (their padded-batch rows were hard-coded to development ids; the default is unchanged). Still needed: the same for the EmbeddingGemma 2 image and video runners, whose interleaved and query rows name development media.

## 3. G3 task datasets, reference scores and proposed budgets

Budget rule: the .NET result must be at least the upstream reference's score on the exact subset minus a stated margin. For a fidelity-equivalent profile (fp32, same graph as the reference export) the margin only absorbs float-level ranking ties and threshold flips, so it is small; quantized and approximate profiles get a task-specific budget against fp32, as plan 10.1 requires. All reference scores below were computed on CPU, fp32, with `model-pipeline/reference/stable/reference_*.py`, which score with the evaluator's own functions (`model_pipeline.evaluate_task_quality`), so the numbers are the ones `model-pipeline evaluate --g3` would print.

### 3.1 Retrieval: MIRACL dev, 8 languages (`retrieval/miracl-dev-hn-8lang-v1/`)

Selection (`select_retrieval.py`): for each of ar, de, en, es, hi, ja, ru, zh, 25 dev queries with at least one relevant passage (seeded), all their judged passages (relevance 1 and the judged negatives, which are hard negatives by construction), and 100 seeded distractors from that language's hard-negative corpus. 200 queries, 2,778 passages, 2,000 judgments; 2.5 MB. Passage text is `title + " " + text` (MTEB convention). Dataset digest (`dataset_sha256`, includes `dataset.json`): `614a8d0d4d43a3930cc7ea4fea56a9835c7db108b68b9c071de8e116995d2405`.

Cost on CPU for every query and passage: e5 3.3 minutes; EmbeddingGemma 2 88 minutes on the shared workstation under load (load average 9 to 12), which is over the "minutes, not hours" aim; section 3.1 budgets name the fallback. recall@100 is 1.0 by construction (each query's judged pool is about 10 passages) and is not proposed as a budget metric; nDCG@10 and MRR@10, which rank the relevant passages among judged hard negatives, are.

| Run | nDCG@10 | MRR@10 | recall@10 | recall@1 | top-10 overlap vs PyTorch fp32 (mean / min) |
| --- | ---: | ---: | ---: | ---: | --- |
| e5, sentence-transformers PyTorch fp32 (reference) | 0.7978 | 0.7717 | 0.9789 | 0.2885 | |
| e5, Python ONNX Runtime fp32, `onnx/model.onnx` (same graph as `onnx-fp32-cpu`) | 0.7978 | 0.7717 | 0.9789 | 0.2885 | 1.000 / 1.0 |
| e5, Python ONNX Runtime, int8-embed graph (`quantize_e5_browser.py`, SHA-256 `374a3cc0…fb19`, the browser default variant) | 0.7996 | 0.7759 | 0.9789 | 0.2910 | 0.9995 / 0.9 |
| EmbeddingGemma 2, sentence-transformers PyTorch fp32 (reference) | 0.8329 | 0.8142 | 0.9889 | 0.3229 | |

Proposed `task-quality.yaml` entries (candidate files in `retrieval/budgets/`):

| Profile | Proposed minimum | Basis |
| --- | --- | --- |
| e5 `onnx-fp32-cpu`, `onnx-fp32-cuda` | `ndcg@10: 0.792`, `mrr@10: 0.765` | reference minus 0.006 (one adjacent swap moves the mean nDCG@10 by at most about 0.002 and MRR@10 by at most 0.0025 over 200 queries, so this admits two or three); ORT fp32 equals the reference exactly, so a fidelity-equivalent .NET path should too |
| e5 `onnx-fp32-wasm`, `onnx-fp32-webgpu` | same as above | same graph and precision |
| e5 `onnx-int8embed-wasm`, `-webgpu` | `top10_overlap_vs_baseline.mean: 0.95` (kickoff decision 3, already in the protected tolerances), plus `ndcg@10: 0.78` | measured 0.9995 natively; nDCG@10 reference minus 0.02, a task-quality floor for a quantized variant. `.min` (0.9 measured) is reported, not bound |
| EmbeddingGemma 2 `onnx-fp32-cpu`, `onnx-fp32-cuda` | `ndcg@10: 0.826`, `mrr@10: 0.808` | reference minus 0.006, as for e5. The run costs about 90 minutes of CPU; a 4-language subset is the fallback if that is too slow for scheduled CI |

### 3.2 NER: WNUT-17 and Few-NERD (`gliner2-base-v1/ner/`)

Selection (`select_ner.py`): 300 sentences of 5 to 60 tokens per set, seeded; sentences without entities kept; tokens joined by single spaces; offsets in code points. Labels are each dataset's class names verbatim (WNUT-17: person, location, corporation, product, creative-work, group; Few-NERD coarse: art, building, event, location, organization, person, product, with `other` dropped). Labels are not paraphrased to suit the model: that would be tuning on the test set.

Reference: the upstream GLiNER2 package (source `55656fbf`) on PyTorch CPU fp32, `GLiNER2.extract(text, Schema().entities(labels), threshold=0.5, include_confidence=True, include_spans=True)`, the default operating point the .NET producer mirrors. About 45 s per set. Every predicted span's offsets index the row text (0 mismatches), so the gold offset unit and GLiNER2's agree.

| Set (license) | Sentences / gold entities | exact micro P / R / F1 | partial micro F1 | Digest |
| --- | --- | --- | ---: | --- |
| WNUT-17 test (CC-BY-4.0), **primary** | 300 / 262 | 0.260 / 0.599 / 0.3626 | 0.4134 | `c4322bab…9a55` |
| Few-NERD supervised test (CC-BY-SA-4.0), alternative | 300 / 694 | 0.431 / 0.710 / 0.5365 | 0.6137 | `ff30dce2…1fd9e` |

The scores are zero-shot, at the default threshold, with the datasets' own label names: high recall, low precision. They measure the port's faithfulness, which is what G3 asks of a fidelity-equivalent profile, not the best attainable NER quality.

Proposed budget (`gliner2-base-v1/ner/budgets/`), GLiNER2 `onnx-fp32-cpu` and `onnx-fp32-cuda`: WNUT-17 `exact.micro.f1: 0.352`, `partial.micro.f1: 0.403` (reference minus 0.01). The protected entity rule already allows threshold flips within 2e-5 of the threshold; on this set one extra false positive moves exact micro F1 by about 0.0004 and one lost true positive by about 0.0023, so 0.01 admits a handful of flips and flags anything larger. Few-NERD alternative: `0.526` / `0.603`.

### 3.3 Qwen generation: GSM8K (`qwen3.5-0.8b/task-quality/gsm8k-test-v1/`)

Selection (`select_generation.py`): 50 GSM8K test problems (seeded), each a single user turn with Qwen's math instruction ("Please reason step by step, and put your final answer within \boxed{}."), thinking disabled, up to 512 new tokens, greedy. Metric (`score_generation.py`, fixed before any run): exact match of the last `\boxed{}` (else the last number) against the GSM8K answer, numerically.

Reference: transformers 5.18.0 on PyTorch 2.14.1 CPU, fp32, greedy (`reference/generation/run_reference.py`; Qwen3.5's linear-attention layers on transformers' reference PyTorch kernels): **exact match 0.64 (32 of 50)**. 43 outputs stop at end of turn, 7 reach 512 tokens. 16 of the 50 outputs have at least one greedy step whose top-2 logit margin is below 0.01, so a backend with slightly different arithmetic can take another path on those rows. The run took 52 minutes on the shared workstation; the .NET ORT GenAI fp32x CPU run is expected to be of the same order (the development benchmark measured 4 to 5 decode tokens/s under load). Scores: `reference-scores/transformers-cpu-fp32.json` (task SHA-256 `c60ad0bc…4884`, results SHA-256 `a105d827…e6e5`; the results file stays outside Git).

Proposed budget:

| Profile | Minimum `exact_match` | Basis |
| --- | --- | --- |
| `ortgenai-fp32x-cpu`, `ortgenai-fp32x-cuda` | 0.60 | reference minus 2 problems (each problem is 0.02); greedy fp32 should follow the reference except at near-ties, and a flip at a near-tie can go either way |
| `ortgenai-fp16x-cuda` | 0.56 | reference minus 4 problems: an approximate variant (18/22 to 19/22 greedy agreement on the development prompts), budgeted on task quality as plan 10.1 asks |

The evaluator has no generation task yet: `model-pipeline evaluate --g3` accepts `retrieval` and `ner`, and `task-quality.yaml` has no `generation` key. This proposal does not write a key nothing reads. Accepting the Qwen budget needs an evaluator change (a `--g3 generation` task reading `prompts.jsonl` with `answer`, and the results of `Starlings.Qwen35.Fidelity`, scored as `score_generation.py` does), as a separate agent workstream.

## 4. G5 performance budgets (proposed form)

From the R2 harness and `docs/benchmarks/PROTOCOL.md`. Budgets are relative to the Python same-backend run where one exists, so they hold across machines; absolute numbers are recorded per controlled run, not budgeted. All require the controlled procedure (quiet machine, 3 warm rounds, 10 cold processes, interleaved) — none of the existing runs qualifies.

| Profile | Python same-backend runner | Proposed budget | Basis (development numbers, not G5 evidence) |
| --- | --- | --- | --- |
| e5 `onnx-fp32-cpu` | yes (`bench_e5_matched.py`) | warm median C#/Python at most 1.10 per workload (upper 95% bound at most 1.15); cold start C#/Python at most 1.0; adapter p95 overhead at most max(1 ms, 10%) of direct C#, judged on medians for CPU until a quiet machine resolves p95 | 2026-10-09: warm ratios 0.92 to 1.11 with every interval containing 1; cold start 944 ms vs 1,181 ms after the tokenizer-load fix |
| e5 `onnx-fp32-cuda` | yes | warm median C#/Python at most 1.05; adapter overhead met on the ids path | 2026-10-09: C# 0.88 to 0.99 of Python |
| e5 browser profiles | no native comparison (browser vs native ORT is a fidelity check) | per-query WASM latency at most 3x native ORT CPU on the same machine, recorded with the browser version | proposal only; no controlled browser run yet |
| EmbeddingGemma 2 text CPU/CUDA | no; feasible by factoring `bench_e5_matched.py` | once it exists: warm median C#/Python at most 1.10; until then no G5 pass | dev run 2026-10-13 under load 15 to 22 |
| EmbeddingGemma 2 image, video | no | preprocessing (decode + processor) at most 5% of `embed` at the photo workload; full `embed` C#/Python at most 1.10 once a Python runner exists | dev run: preprocess 21 ms of 16 s |
| GLiNER2 CPU/CUDA | no (needs a NumPy port of the decode) | C# `prepare` + `decode` at most 5% of `inference`; `inference` vs Python ORT on the same inputs at most 1.10 once a runner exists | dev run: prepare+decode below 4 ms at 400 words, inference more than 99% |
| Qwen `fp32x-cpu`, `fp32x-cuda` | no (needs Python `onnxruntime-genai` or the C API) | decode tokens/s at least 0.95 of the native run (plan 11); TTFT within 1.10 of native | dev run under contention: 4 to 5 tokens/s; no native baseline |
| Qwen `fp16x-cuda` | no | as fp32x CUDA | |

Regression alert for all: a repeatable median regression above 5% between releases on the same controlled machine (plan 11).

## 5. Per-profile Stable checklist

After the proposals above are accepted, each profile still needs what follows. Common to every profile: qualification CI with a Windows x64 job and scheduled runs (P-018, R6, M6); a one-hour soak per profile (G4); a second release owner (M7, `docs/support-policy.md`); NuGet publishing setup and branch protection (M5); a named consumer for G7 (P-028); the maintainer recording each gate (M10).

| Profile | Maturity now | Still missing for Stable after this proposal |
| --- | --- | --- |
| e5 `onnx-fp32-cpu` | preview | G1/G2 rerun on the release set at the current tokenizer pin; G3 run with the accepted budget; controlled G5 run; G4 soak; G7 consumer; Windows evidence. Closest to Stable. |
| e5 `onnx-fp32-cuda` | experimental | the above; G1/G2 on development fixtures passed on an RTX 3090 (2026-10-14, `docs/evidence/cuda-qualification-2026-10-14/`), so the release-set rerun is a GPU run of the same commands; CUDA rows for G3 to G7, `aot`/`extensions_ai`/`mlnet` statuses, the `PROPOSED.md` decisions (M8) |
| e5 browser (4 profiles) | no manifest | registry manifests and packages first; int8-embed overlap on the release queries (passes natively: 0.9995); browser CI (headless Chromium, WASM); WebGPU stays local evidence |
| EmbeddingGemma 2 text CPU | experimental | G1/G2 on the release set; G3 budget; Python same-backend benchmark runner; `input_2050_to_8192_tokens` (R7 W3) and `prompts_task` statuses |
| EmbeddingGemma 2 text CUDA | experimental | the above; development G1/G2 passed on CUDA (2026-10-14), so the release-set rerun needs GPU time only |
| EmbeddingGemma 2 image CPU/CUDA | experimental | runner option for the release media, then G1/G2; an image G3 task (none proposed: the evaluator has no image retrieval producer yet); `image_budget_other`, `image_decode_gif_bmp_webp` statuses; GPU time for CUDA |
| EmbeddingGemma 2 video CPU/CUDA | experimental | runner option and FFmpeg 9 adapter (R7 W4, legal D2); decoded-frame benchmark fixtures; four untested video capabilities |
| GLiNER2 CPU | experimental | G1/G2 on the tokenizer and extraction release sets; G3 budget; Python same-backend benchmark; `count_gate_closed`, `label_descriptions` statuses |
| GLiNER2 CUDA | experimental | the above plus GPU time (development G1/G2 passed on CUDA, 2026-10-14) and `cancellation`, `concurrent_calls`, `aot` statuses |
| Qwen `fp32x-cpu` | experimental | release-prompt fixture and G1/G2 on it; evaluator `generation` G3 task; native benchmark runner; long context above 1,666 tokens (R8) |
| Qwen `fp32x-cuda` | experimental | the above plus GPU time, `aot` status, the documented CUDA serialization (M9), and the missing `float.chunked_vs_one_call_prefill_logprob_max_abs` key in its protected tolerances (G2 is `not-run` without it; a maintainer `protected-change`) |
| Qwen `fp16x-cuda` | experimental | **likely to stay Preview**: greedy agreement is 18/22 to 19/22 on the development prompts (19/22 on the RTX 3090 run of 2026-10-14, divergences at margins of 0.0027 or less); a Stable claim would need a task-quality budget it meets and an explicit approximate-variant label, which plan 10.1 allows but which is a weaker contract than the fp32 profiles |

## 6. Draft notices

For `model-pipeline/THIRD-PARTY-NOTICES.md` (or the repository's notices file) if the sets are accepted:

- MASSIVE, Amazon.com, Inc., CC BY 4.0: utterances in the tokenizer holdouts.
- HumanEval, OpenAI, MIT: prompts in the tokenizer holdouts.
- OpenAssistant Conversations (oasst1), LAION and contributors, Apache-2.0: replies in the tokenizer holdouts; prompts in the Qwen release set.
- MIRACL, Zhang et al., topics and judgments Apache-2.0; Wikipedia passages CC BY-SA (via `mteb/MIRACLRetrievalHardNegatives`, CC BY-SA 4.0): the retrieval subset, which stays under CC BY-SA 4.0.
- WNUT-17 Emerging and Rare Entities, Derczynski et al., CC BY 4.0: the NER subset.
- Few-NERD, Ding et al., CC BY-SA 4.0: the alternative NER subset, which stays under CC BY-SA 4.0.
- GSM8K, OpenAI, MIT: the Qwen task set.
- Wikimedia Commons media, per file as listed in `media.json` (not redistributed).

## 7. Decisions for the maintainer

1. Accept, change or reject each set (section 2) and dataset (section 3), including the share-alike question in section 1.
2. Set the budgets (sections 3 and 4); the proposed values are starting points derived from the reference scores.
3. Approve the runner changes (`--batch-ids`; the image and video runner option still to write).
4. Commission the evaluator `generation` task and the missing Python same-backend benchmark runners as agent workstreams.
5. Confirm the Commons licenses file by file.
