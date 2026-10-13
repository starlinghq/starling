# Proposed evidence (not protected)

Fixtures and tolerance text proposed by an agent for the maintainer to accept. Nothing here is read by `model-pipeline evaluate` or by the ai-dotnet tests as a bound or a fixture: the evaluator reads only `model-pipeline/protected/` (plan 10.1 and 12), and agents cannot write there. Each directory has a `PROPOSED.md` with the sources, the reference environment and what the maintainer would commit, and where.

| Directory | Proposal | Item |
| --- | --- | --- |
| `embeddinggemma-2/decoder-ffmpeg9/` | The `decoder_ffmpeg9` tolerance section for the FFmpeg 9 video adapter | R7 W4 |
| `embeddinggemma-2/long-input/` | Long-input fixtures, 2,049 to 8,192 tokens (text, interleaved text and images, video at the 32-frame limit, rejections just over 8,192) | R7 W3 |

Once a proposal is accepted, the maintainer moves it under `model-pipeline/protected/` in a `protected-change` pull request and removes it here.
