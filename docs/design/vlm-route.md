# Vision-language model route

Status: **proposed**, 2026-11-01. Not a support claim, not an admission record, and not a maintainer decision. Refs #105 (Qwen3-VL-4B-Instruct), #106 (SmolDocling-256M-preview) and #107 (SmolVLM2-500M-Video-Instruct).

Inputs: the three issues and their triage comments (2026-11-01), the plan (`docs/plan/fast-faithful-ai-dotnet.md`, sections 5, 6, 8, 10 and 13), the EmbeddingGemma 2 image and video path (`Starlings.Models.EmbeddingGemma2`, `docs/m0/embeddinggemma2-image-spike.md`), the generation stack (`Starlings.Generation.OnnxGenAI`, `Starlings.Models.Qwen35`, `Starlings.Models.QwenChat`), and the two existing owned decode loops on direct ONNX Runtime (`Starlings.Models.Lfm3`, `Starlings.Models.Whisper`). Upstream configuration files and model cards were read as data.

Source tags: **[hub, 2026-11-01]** means read from the Hub repository's current `main` through the Hugging Face connector. The connector returns no commit SHAs or file digests, so every such fact must be re-read at the revision pinned at admission. **[local, 2026-11-01]** means read from the installed onnxruntime-genai 0.17.1 wheel and the `Microsoft.ML.OnnxRuntimeGenAI.Managed` 0.17.1 package on the development machine. Nothing was downloaded, exported or run for this note. It contains no parity numbers, timings or memory measurements.

## 1. Problem

Starling embeds images and video (EmbeddingGemma 2, Stable) but cannot generate text about them. The three requests ask for that:

| Issue | Model | Use |
| --- | --- | --- |
| #105 | `Qwen/Qwen3-VL-4B-Instruct` | Describe, answer questions about, and extract from images (and video). |
| #106 | `docling-project/SmolDocling-256M-preview` | Page image to DocTags (layout, tables, code, formulas, text) for document indexing. The Hub also serves it as `ds4sd/SmolDocling-256M-preview`; both ids resolve [hub, 2026-11-01]. Admission records which id and revision are pinned. |
| #107 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Describe and answer questions about images and short video clips. |

No onboarding route fits any of them. `qwen-genai-cpu-v1` is text-only. `Qwen35ChatTemplate` and `QwenChatTemplate` reject image and video content. The EmbeddingGemma 2 multimodal path produces embeddings, not tokens. The triage comments ask for one shared design that picks the first candidate and orders the rest.

## 2. Constraints

From the repository rules and the plan:

- **Owned tokenization and processing (plan 6, 6.4; gate G1).** Prompt ids come from `Starlings.Tokenizers`. Image preprocessing gets the same treatment as tokenization: a declared, tested processor with exact discrete outputs and documented continuous tolerances. A backend-owned tokenizer or image processor must be declared per profile, compared against the owned one, and labelled as a dependency. It must never become a second source of model semantics.
- **Decoded RGB at the API boundary.** EmbeddingGemma 2 takes already-decoded RGB (`EmbeddingGemma2Image.FromRgb24`), with decoding in an optional adapter package (SkiaSharp). Decoders differ by several levels on JPEG, and that moves outputs (image spike, section 6). A VLM route keeps that contract.
- **No remote code.** All three architectures are in-tree in transformers (`SmolVLMForConditionalGeneration`, `Idefics3ForConditionalGeneration`, `Qwen3VLForConditionalGeneration` [hub, 2026-11-01]), so the reference needs no `trust_remote_code`. SmolDocling's repository ships a `zero_to_fp32.py` training utility; the reference never runs it.
- **Pins.** Upstream code and weights by full commit SHA. Reference environments compiled with hashes, as in `model-pipeline/reference/*/requirements.txt`.
- **Hardware and budget (plan 13).** CPU parity first. The RTX 3090 belongs to another service. CUDA runs in this lane use the RTX 4060 Ti 16 GB (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`), and that evidence supports **Experimental** status only, never a Stable CUDA claim. Paid GPU time is still open in #210.
- **Licences [hub, 2026-11-01].** SmolVLM2 and Qwen3-VL: Apache-2.0. SmolDocling: CDLA-Permissive-2.0 on the card, to re-verify at the pinned revision. SmolDocling is a `-preview` model, so admission also checks for a stable successor before pinning.
- **Bounded authority.** New tolerances and goldens are proposed in their own change for maintainer review. This design note changes no protected file.

## 3. The candidates

From each repository's `config.json`, `preprocessor_config.json`, `processor_config.json`, `generation_config.json` and file listing [hub, 2026-11-01]:

| | SmolVLM2-500M-Video-Instruct (#107) | SmolDocling-256M-preview (#106) | Qwen3-VL-4B-Instruct (#105) |
| --- | --- | --- | --- |
| Architecture | `SmolVLMForConditionalGeneration` (`smolvlm`) | `Idefics3ForConditionalGeneration` (`idefics3`) | `Qwen3VLForConditionalGeneration` (`qwen3_vl`) |
| Vision tower | SigLIP-style ViT, hidden 768, 12 heads, patch 16, 512 x 512 input | Same shape (`idefics3_vision`, `use_base_siglip` true) | ViT, depth 24, hidden 1,024, patch 16, temporal patch 2, spatial merge 2, `deepstack_visual_indexes` [5, 11, 17] |
| Connector | Pixel shuffle, `scale_factor` 4, then a projection: 1,024 patches per 512 tile become 64 tokens (`image_seq_len` 64) | Same | Patch merger (2 x 2) to hidden 2,560, plus deepstack features from three intermediate vision layers |
| Decoder | Llama (SmolLM2 family): 32 layers, hidden 960, 15 heads, 5 KV heads, head 64, RoPE theta 100,000, 8,192 positions | Llama: 30 layers, hidden 576, 9 heads, 3 KV heads, head 64, RoPE theta 100,000, 8,192 positions | Qwen3: 36 layers, hidden 2,560, 32 heads, 8 KV heads, head 128, interleaved M-RoPE (`mrope_section` [24, 20, 20], theta 5,000,000), 262,144 positions |
| Image token | `<image>` id 49190 | id 49190 | `<|image_pad|>` 151655, `<|video_pad|>` 151656, between `vision_start` 151652 and `vision_end` 151653 |
| Vocabulary | 49,280 (eos 49279) | 49,280 | 151,936 |
| Weights | `model.safetensors` 2,029,990,624 bytes, fp32 (`torch_dtype`) | 513,028,808 bytes, bf16 | 8,875,719,344 bytes in two shards, bf16; about 17.7 GB in fp32 |
| Official ONNX | `onnx/vision_encoder`, `embed_tokens`, `decoder_model_merged`, each in fp32, fp16 and quantized variants | Same three-part layout | None |
| Video | Yes: `video_sampling` fps 1, `max_frames` 64, frames at longest edge 512 | No | Yes: `Qwen3VLVideoProcessor` |

What the table implies:

- SmolVLM2 and SmolDocling are the same route: same vision tower shape, connector, image token id, vocabulary size, preprocessing configuration (section 5) and three-part ONNX layout. They differ in decoder width and depth, in the added-token set (`added_tokens.json` is 4,739 bytes for SmolVLM2 and 3,667 bytes for SmolDocling), and in SmolDocling's DocTags output.
- Qwen3-VL is a different route. The vision features enter the decoder in two ways: they replace the `<|image_pad|>` embeddings, and deepstack adds features from vision layers 5, 11 and 17 to the hidden states of early decoder layers. Its position ids are three-dimensional (time, height, width) for image tokens. A plain embedding splice does not reproduce either of these.

## 4. Graph split options

Every VLM here has the same logical pipeline: decoded RGB, then the processor (resize, split, normalize), then the vision encoder and connector (pixels to soft tokens), then the token embedding with soft tokens spliced in at the image-token positions, then the autoregressive decoder with a KV cache, and finally the owned detokenizer. The options differ in which runtime owns which step.

### 4.1 Option A: ONNX Runtime GenAI multimodal

GenAI 0.17.1 (the pinned version) ships multimodal model types. The native library contains `phi3v`, `phi4mm`, `gemma3`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_5`, `lfm2_vl` and `mistral3`, and no `smolvlm` or `idefics3` [local, 2026-11-01]. The builder maps `Qwen3VLForConditionalGeneration` to `Qwen3VLTextModel`, a text decoder with three-dimensional `position_ids`. With `exclude_embeds` the decoder takes `inputs_embeds` and the model type becomes `qwen3_vl` [local, 2026-11-01]. The builder exports only the decoder; the vision and embedding graphs come from elsewhere.

- **SmolVLM2 and SmolDocling:** not available at the pinned version. No model type exists. The builder accepts `LlamaForCausalLM`, so a text-only SmolLM2 decoder could be built, but GenAI would not run the vision part.
- **Qwen3-VL:** a model type exists. The string `deepstack` appears neither in the 0.17.1 builder sources nor in `libonnxruntime-genai.so` [local, 2026-11-01]. Whether GenAI's `qwen3_vl` pipeline applies the deepstack features is therefore unverified, and the first spike must check it against the reference before anything relies on it.
- **Independence.** GenAI's multimodal C# path is `MultiModalProcessor` (`OgaProcessorProcessImagesAndPrompts`). It decodes images and tokenizes the prompt itself, so it is a backend-owned image processor and a backend-owned tokenizer. Under plan 6.4 both would be declared dependencies, which breaks the decoded-RGB contract above. The managed binding also exposes `Generator.SetModelInput` and `OgaCreateTensorFromBuffer` [local, 2026-11-01]. In principle these let owned code feed its own prompt ids and pixel tensors past the processor. Whether a multimodal model type accepts inputs supplied that way is unverified.
- **What it keeps:** GenAI's generation loop, sampling, CUDA kernels and llguidance guided decoding, all of which `Starlings.Generation.OnnxGenAI` already wraps.

### 4.2 Option B: split ONNX on direct ONNX Runtime with an owned decode loop

Three graphs, run by `Starlings.Runtime.Onnx`. The inputs and outputs below are what the three-part layout implies; the exact names are read from the pinned graphs at admission.

1. `vision_encoder(pixel_values, pixel_attention_mask)`, which returns the connector output: 64 soft tokens of decoder width per 512 x 512 tile.
2. `embed_tokens(input_ids)`, which returns token embeddings.
3. `decoder_model_merged(inputs_embeds, attention_mask, position_ids, past_key_values.*)`, which returns logits and `present.*`.

Owned C# code then does the splice: replace the embedding rows at positions holding token 49190 with the soft tokens in order, check that the counts match (64 per tile), run the prefill, and step. The pattern already exists twice. `Starlings.Models.Lfm3` runs a managed greedy loop over an upstream step graph with its caches. `Starlings.Models.Whisper` runs an encoder followed by an autoregressive decoder on direct ONNX Runtime. The splice is the same operation EmbeddingGemma 2 performs on its multimodal text graph, where soft tokens replace the `<|image|>` embeddings. The shared pieces in `Starlings.Generation.Abstractions` (`IGenerationSequence`, `StopSequenceMatcher`, `IncrementalDetokenizer`, `GenerationOptions`, `GenerationResult`) carry over unchanged.

- **Graph source:** the official three-part export is a ready starting point for both Smol models. It is data (ONNX graphs), not code. The route still needs a reproducible export recipe from the pinned safetensors, with a pinned exporter and environment, for two reasons: so the graph is rebuildable (plan 8, stage 4) and so the downloaded graphs can be checked against it. The admission decides which one the profile ships. If the upstream export is shipped, its revision and digests are locked as for LFM3.
- **What it loses:** GenAI's guided decoding and sampling kernels. The first profile is greedy only. Seeded sampling can reuse the `TextGenerator` sampling semantics later.
- **Qwen3-VL on Option B** needs its own export: vision tower, merger and deepstack outputs, and a decoder that takes the deepstack features and three-dimensional position ids. That is a new export, not a reuse.

### 4.3 Recommendation

**Use Option B for the Smol family.** It is the only option at the pinned GenAI version. It keeps tokenization and preprocessing owned. It reuses a route shape that already exists twice in the repository.

**Keep Option A open for Qwen3-VL**, behind a spike that answers two questions. Does the `qwen3_vl` pipeline reproduce deepstack? Can owned prompt ids and pixel tensors be supplied through `SetModelInput`? If both answers are yes, Qwen3-VL can run on GenAI with an owned processor. If either is no, Qwen3-VL goes on Option B with its own export, or stays blocked.

## 5. Image preprocessing

### 5.1 SmolVLM2 and SmolDocling (Idefics3 family)

Configuration [hub, 2026-11-01], identical in both repositories except for the processor class name (`SmolVLMImageProcessor` and `Idefics3ImageProcessor`):

- `do_convert_rgb`, then a resize so that the longest edge is at most `size.longest_edge` 2,048. Each side is then rounded to a multiple of `max_image_size.longest_edge` 512, with aspect ratio preserved, using `resample` 1. The exact rounding comes from the processor code at the pinned transformers version.
- `do_image_splitting`: the resized image is cut into 512 x 512 tiles (at most 4 x 4), plus one global image resized to 512.
- `rescale_factor` 0.00392156862745098, then normalization with mean and std 0.5 per channel, so values lie in [-1, 1].
- `do_pad`: tiles are padded to a common size and `pixel_attention_mask` marks the real pixels.
- Prompt expansion. The chat template (`chat_template.json`, identical in both repositories) emits one `<image>` per image. The processor then expands it into the tile grid: per tile `<fake_token_around_image><row_i_col_j>` followed by 64 x `<image>`, then `<fake_token_around_image><global-img>` followed by 64 x `<image>`, closed by `<fake_token_around_image>`. A 4 x 4 split therefore puts 17 x 64 = 1,088 image tokens into the 8,192-token context. The exact string, including any newlines between rows, comes from the processor code at the pinned transformers version and is a fixture, not something rebuilt from memory.

Two lessons from EmbeddingGemma 2 apply directly:

1. **Pin the processor implementation, not only the configuration.** transformers can run a PIL ("slow") or torchvision ("fast") backend for the same configuration. They produce different bytes. `resample` 1 is PIL's LANCZOS enum value, and the reference must record which backend and which filter actually ran. EmbeddingGemma 2 found that its reference ran the torchvision backend, not PIL.
2. **Port the resampler bit for bit.** In EmbeddingGemma 2, one or two levels of resampling difference on under 1% of pixels moved the embedding by up to 2.6e-3. The image spike wrote both a torchvision-exact and a Pillow-exact fixed-point variant. Whichever backend the reference uses, the C# port starts from those variants (`TorchAntialiasResampler`, and the Pillow-quantized prototype in the spike) and must equal the reference's resized bytes exactly on the fixture images. Any approximation is a G1 failure.

The fidelity criterion differs from the embedding path. A generative model is judged by the LLM rules of plan 10.1, not by an embedding cosine bound: exact prompt ids, exact pixel tensors, soft tokens within a tolerance, step-0 logits within a tolerance, greedy token sequences on stable fixtures with the first divergence and its top-1 minus top-2 margin reported, and cached against uncached decoding. Pixel differences matter only through those numbers, so exact preprocessing is the only way to keep the comparison clean.

### 5.2 Video (SmolVLM2)

`video_sampling` [hub, 2026-11-01]: 1 frame per second, at most 64 frames, each frame resized to longest edge 512 with no tile splitting, so 64 image tokens per frame. The processor adds text around the frames (an introduction line and a per-frame timestamp). That text comes from the processor code at the pinned transformers version, not from `chat_template.json`, and is to be verified and pinned as fixtures. Frame decoding and sampling reuse the EmbeddingGemma 2 approach: decoded frames at the API boundary, with `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` and `FFmpeg9` as the model of an optional decoder adapter. The sampling rule itself (which timestamps are chosen) is model-specific and gets its own fixtures.

### 5.3 Qwen3-VL

The configuration gives the bounds below. The resize rounding, the temporal repetition and the patch order belong to the Qwen2-VL processor family and must be confirmed from the processor code at the pinned revision.

Dynamic resolution: the pixel count is kept between `shortest_edge` 65,536 and `longest_edge` 16,777,216 (video: 4,096 to 25,165,824). Sides are rounded to multiples of patch 16 x merge 2 = 32. Mean and std are 0.5. Images are repeated to fill `temporal_patch_size` 2. The configuration names `Qwen2VLImageProcessorFast`, the torchvision backend. Patches are flattened in the merge order the vision tower expects, and the processor emits the grid (`image_grid_thw`) from which the M-RoPE position ids are computed. Each of those steps is a separate exact-output fixture.

## 6. Tokenizer needs

- **SmolLM2 family (SmolVLM2, SmolDocling).** A new profile in starlinghq/tokenizers. The vocabulary (49,280) and `merges.txt` are byte-level BPE, so the existing interpreter likely covers the model. It still needs a profile, conformance fixtures and a release, then a pin bump here in one change (repository CLAUDE.md). The added tokens differ between the two checkpoints, so the profile is either one per checkpoint or a shared base with per-checkpoint added tokens. That is decided when the tokenizer files are pinned and compared by SHA-256. The fixtures must cover the image-structure tokens (`<fake_token_around_image>`, `<global-img>`, every `<row_i_col_j>`, `<image>`, `<end_of_utterance>`) and user text that looks like them.
- **Chat template.** A C# renderer per checkpoint, following `Qwen35ChatTemplate` and `Lfm3ChatTemplate`. The Smol template is short (430 bytes) but has a quirk: no space after `User:` when the message starts with an image. Rendered prompts are compared against `apply_chat_template` plus processor expansion on the fixture set.
- **Qwen3-VL.** The vocabulary size (151,936) is the Qwen2/Qwen3 family's, not Qwen3.5's (248,320). Reuse of the existing `qwen3-bpe-chat` profile is claimed only if the pinned `tokenizer.json` SHA-256 matches that profile's asset, or if the profile's fixtures pass on it. The template (`chat_template.json`, 5,502 bytes) handles vision content and is a new renderer, or an extension of `QwenChatTemplate`, with vision content enabled.
- **DocTags (SmolDocling).** Not a tokenizer concern but an output parser: DocTags to a typed document (text blocks, tables, code, formulas, layout locations), with fixtures checked against `docling-core` at a pinned revision.

## 7. First candidate

**SmolVLM2-500M-Video-Instruct, image mode first, on Option B, CPU fp32.** This matches the #107 triage ordering. Reasons:

1. An official three-part ONNX export exists, so the graph split is known to be exportable before any export work starts.
2. At about 0.5B parameters (2.03 GB in fp32), CPU parity is feasible. The first profile does not depend on #210.
3. SmolDocling shares the architecture, preprocessing, image token and ONNX layout. The same route then serves #106 with a smaller decoder and an output parser.
4. Video is an extension of image mode (frames are images without splitting), on the same checkpoint.
5. Qwen3-VL needs deepstack and M-RoPE, has no official ONNX, and is about 17.7 GB in fp32. That is above the 4060 Ti's 16 GB for fp32 CUDA runs, so its parity runs need either CPU (slow) or the paid budget in #210.

Order: SmolVLM2 images, then SmolVLM2 video, then the SmolDocling candidate with the DocTags parser, then the Qwen3-VL spike.

## 8. Evidence plan

Per profile, following plan section 8 and gates G0 to G7 (phased as in plan 10):

- **G0.** Admission record (`model-pipeline/admissions/`), asset lock by full commit SHA with sizes and SHA-256, licence text at the pinned revision.
- **Reference.** A new environment under `model-pipeline/reference/` with transformers, torch, torchvision and Pillow pinned and hashed. The runner writes, per fixture: the rendered and expanded prompt, prompt ids, resized tile bytes, pixel tensors and masks, soft tokens, step-0 log-softmax, greedy output ids up to a fixed limit, and the finish reason. It reuses the shared result format of `model-pipeline/reference/generation/compare.py` where it fits.
- **Fixtures.** Reuse the image spike's photos, which are already pinned by SHA-256 (scikit-image at `b700edc8005e4ee9e74fc83259f09e8df3213f15`). Add cases that exercise splitting: wide and tall panoramas, tiny images, exactly 512 and 2,048 edges, grayscale, palette and alpha images (with the upstream conversion rules recorded), several images per turn, multi-turn, and text that imitates image tokens. For SmolDocling, add page images under a licence that permits redistribution as fixtures.
- **G1.** Exact prompt ids, exact expansion string, exact resized bytes and pixel tensors.
- **G2.** Soft tokens and step-0 logits within tolerances proposed from reference self-agreement (CPU threads, and CPU against CUDA on the 4060 Ti), in a separate maintainer-reviewed change. Greedy sequences: exact match counts, first divergence, margin at the divergence, cached against uncached.
- **G3.** Task behaviour on a small held-out set: captioning and question answering for SmolVLM2, DocTags structure against `docling-core` for SmolDocling. Every advertised feature passes, or the claim is narrowed.
- **G4 to G7.** As for LFM3: cancellation between steps, disposal, concurrent requests without state leakage, clean-install and offline runs, and an `IChatClient` adapter mapping image content.
- **Failure taxonomy (plan 8):** processor, tokenizer, export/operator, precision, decoding, provider. After two failed repair cycles on the same failure, stop and report with a minimal reproducer.
- **Maturity.** CPU evidence supports Experimental. CUDA evidence from the RTX 4060 Ti supports Experimental only, and any evidence README names the GPU. Stable needs every gate in full and the maintainer's decision.

## 9. Checklist (one PR each)

1. Admission record for SmolVLM2-500M-Video-Instruct: pins, licence, scope (images first, greedy, CPU fp32), and the decision on the upstream export versus an own export.
2. starlinghq/tokenizers: SmolLM2-family profile with fixtures, including the image-structure tokens. Release, then raise the pin here and the manifests' `processing.tokenizer.version` in one change.
3. Reference environment and runner, the fixture set, and a proposed tolerance file in a separate maintainer-reviewed change.
4. C# Idefics3/SmolVLM image processor (resize with the exact resampler, splitting, rescale and normalize, padding and mask, prompt expansion), with bit-exact tests against the reference bytes.
5. Vision encoder and connector wrapper on `Starlings.Runtime.Onnx`, with soft-token parity tests.
6. New package (working name `Starlings.Models.SmolVlm`): embedding splice, owned greedy decode loop with KV cache, chat template renderer, asset locks, capabilities with explicit statuses (video, sampling, guidance, CUDA untested or unsupported until evidenced).
7. model-pipeline route (working name `smolvlm-onnx-cpu-v1`) and its plugin in `onboarding_plugins.py`, a `generation` evaluator extension for image inputs, and an Experimental registry profile with its evidence record.
8. `IChatClient` adapter mapping `DataContent` images to decoded RGB through the optional decoder adapter, with unsupported features reported as unsupported.
9. Video mode: frame sampling fixtures, the frame prompt text, and a decoder adapter. A separate capability status.
10. SmolDocling candidate on the same route: admission (including a check for a stable successor), its tokenizer profile or added-token delta, and a profile.
11. DocTags parser with `docling-core` fixtures. Its own capability status.
12. Qwen3-VL spike: GenAI `qwen3_vl` deepstack check and `SetModelInput` feasibility (Option A), against an Option B export. The result decides whether #105 proceeds, and on which route.
13. CUDA on the RTX 4060 Ti for each profile above (Experimental evidence only).

## 10. Open questions for the maintainer

- Ship the upstream ONNX export (as LFM3 does) or only an export reproduced from the pinned safetensors?
- One package per family (`Starlings.Models.SmolVlm` covering SmolVLM2 and SmolDocling) or one per model?
- The Qwen3.5-0.8B family carries a `vision_config` (a 12-layer ViT, from the Base checkpoint's `config.json`, `model-pipeline/admissions/kev.md`); the chat checkpoint is already pinned and Stable for text, and GenAI 0.17.1 has `qwen3_5` model types. It is outside the three issues, but it could later give a VLM profile that reuses an existing tokenizer profile and template renderer. Track it as a separate request?
- Paid GPU budget for Qwen3-VL fp32 parity (#210).
