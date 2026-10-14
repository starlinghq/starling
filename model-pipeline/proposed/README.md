# Proposed fixtures

Each subdirectory holds fixtures or tolerance text that an implementation job proposes and the maintainer has not accepted yet. The layout mirrors `model-pipeline/protected/`, and each directory has a `PROPOSED.md` that names the reference source, the environment and the decisions for the reviewer. If a proposal is accepted, the maintainer moves it byte-identical to the same path under `protected/` in a `protected-change` pull request and removes it here.

Nothing here is binding, and nothing here changes a tolerance, gate or capability status. `model-pipeline evaluate` reads bounds only from `model-pipeline/protected/` (plan 10.1 and 12). Some ai-dotnet tests read the protected copy of a fixture when it exists and the proposal otherwise, so a proposal can be exercised before it is accepted; those tests never treat it as a release set.

| Directory | Proposal | Item |
| --- | --- | --- |
| `qwen3.5-0.8b/template/` | Chat-template corpus, 85 cases rendered by the pinned upstream renderer (P-029) | R8 |
| `qwen3.5-0.8b/long-context/` | Long-context evidence above 1,666 tokens (fp32x CPU) | R8 |
| `embeddinggemma-2/decoder-ffmpeg9/` | The `decoder_ffmpeg9` tolerance section for the FFmpeg 9 video adapter | R7 W4 |
| `embeddinggemma-2/long-input/` | Long-input fixtures, 2,049 to 8,192 tokens (text, interleaved text and images, video at the 32-frame limit, rejections just over 8,192) | R7 W3 |
