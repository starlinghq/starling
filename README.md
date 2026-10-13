# Starling

Maintained .NET implementations of deep learning models and LLMs. Each supported model ships with a .NET API and reproducible evidence of what works, how fast it runs, and which upstream behavior it preserves.

Status: preview. The tokenizer is released as [`Starlings.Tokenizers`](https://www.nuget.org/packages/Starlings.Tokenizers) from [starlinghq/tokenizers](https://github.com/starlinghq/tokenizers); the model packages here are release candidates, not yet published. The execution plan is [`docs/plan/fast-faithful-ai-dotnet.md`](docs/plan/fast-faithful-ai-dotnet.md) and open kickoff decisions are in [`docs/kickoff.md`](docs/kickoff.md).

The experimental [five-model CPU onboarding workflow](model-pipeline/FIVE_MODELS.md) runs two
Qwen chat models, two BERT embedding models, and a BERT reranker, with verified resume and optional
package-only consumer evidence. Its fresh references and comparison policies remain proposals.

## Layout

The tokenizer has its own repository, [starlinghq/tokenizers](https://github.com/starlinghq/tokenizers), and is consumed here as the pinned `Starlings.Tokenizers` package. The other planned products live here as top-level directories until one of them needs its own owner or release lifecycle.

| Directory | Planned repository | Contents |
| --- | --- | --- |
| [`.github/`](.github) | `.github` | Charter, contribution and support policy, security reporting, issue templates, CODEOWNERS, CI. |
| [`ai-dotnet/`](ai-dotnet) | `ai-dotnet` | .NET libraries, backend adapters, model implementations, samples, benchmarks. |
| [`model-pipeline/`](model-pipeline) | `model-pipeline` | Manifest schema, model registry, admission records, reference runners, evaluation, protected fixtures. |
| [`browser/`](browser) | | The WebAssembly spike (e5 in the browser). |
| [`docs/`](docs) | | Plan, kickoff decisions, backlog source, evidence and release records. |
| [`tools/`](tools) | | Repository checks and maintenance scripts. |

## Build and test

Requires the .NET 10 SDK (pinned in `global.json`) and Python 3.11+.

```sh
dotnet test --solution ai-dotnet/AI.slnx
python3 -I tools/check-boundary.py

cd model-pipeline
pip install -e '.[dev]'
python3 -m pytest
ruff check .
model-pipeline validate registry
```

## License

Original code is licensed under the MIT License or the Apache License, Version 2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)), copyright StarlingHq. Model weights, tokenizer assets, and datasets keep their own terms; the code license does not cover them.
