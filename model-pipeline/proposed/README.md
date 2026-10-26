# Proposed fixtures

Each subdirectory holds fixtures or tolerance text that an implementation job proposes and the maintainer has not accepted yet. The layout mirrors `model-pipeline/protected/`, and each directory has a `PROPOSED.md` that names the reference source, the environment and the decisions for the reviewer. If a proposal is accepted, the maintainer moves it byte-identical to the same path under `protected/` in a `protected-change` pull request and removes it here.

Nothing here is binding, and nothing here changes a tolerance, gate or capability status. `model-pipeline evaluate` reads bounds only from `model-pipeline/protected/` (plan 10.1 and 12). Some ai-dotnet tests read the protected copy of a fixture when it exists and the proposal otherwise, so a proposal can be exercised before it is accepted; those tests never treat it as a release set.

| Directory | Proposal | Item |
| --- | --- | --- |
| `qwen3.5-0.8b/template/` | Chat-template corpus, 85 cases rendered by the pinned upstream renderer (P-029) | R8 |
| `qwen3.5-0.8b/long-context/` | Long-context evidence above 1,666 tokens (fp32x CPU) | R8 |
| `qwen3-0.6b/template/` | Chat-template corpus for the QwenChat Qwen3 profile, 54 cases (46 renderings, tools included) rendered by the pinned upstream renderer | QwenChat tool calling |
| `qwen2.5-0.5b-instruct/template/` | Chat-template corpus for the QwenChat Qwen2.5 profile, 49 cases (42 renderings, tools included) rendered by the pinned upstream renderer | QwenChat tool calling |
| `embeddinggemma-2/long-input/` | Long-input fixtures, 2,049 to 8,192 tokens (text, interleaved text and images, video at the 32-frame limit, rejections just over 8,192) | R7 W3 |
| `embeddinggemma-2/task-prompts/` | All 20 named prompts on four public texts (capability `prompts_task`) | Capability closure |
| `embeddinggemma-2/image-extra/` | Image budgets 70/140/560/1,120 and GIF/BMP/WebP decoding digests | Capability closure |
| `embeddinggemma-2/video-edge/` | Edge-case videos: rotation -90 and 180, HDR, interlaced, variable frame rate (torchcodec frames) | Capability closure |
| `gliner2-base-v1/capability-cases/` | 22 public-text cases: label descriptions and a closed count gate | Capability closure |
