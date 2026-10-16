# Charter

## Purpose

Make .NET a dependable place to use modern AI. A model supported here comes with a usable .NET API and reproducible evidence of what works, how fast it runs, and which upstream behavior it preserves.

## Scope

- .NET implementations and integrations for deep learning models, LLMs, and tokenizers.
- An owned tokenizer library (`Starlings.Tokenizers`) with its own release cadence.
- The delivery pipeline that pins, verifies, benchmarks, and releases each supported combination.

- A serving engine for NVIDIA GPUs, [Roost](https://github.com/starlinghq/roost), in its own repository. It may vendor and patch upstream GPU kernels behind a C ABI (pinned by full SHA, never written from scratch without a recorded need), and serves Starling models only under the profiles and fidelity gates of this repository.

Out of scope until a separate use case, owner, and budget exist: general-purpose training, distributed serving (multi-node or tensor/pipeline parallelism, including in Roost), multimodal generation, and GPU kernels outside Roost.

### Roost (amended for R-001, proposed 2026-10-16)

| Item | Value |
| --- | --- |
| Use case and consumer | .NET applications that run Starling models on a local or single-host NVIDIA GPU, in-process through `IChatClient` (Preview 1) and through an OpenAI-compatible server. Named consumer: **open, for the maintainer to name before this amendment merges** (R-001 requires one). |
| Kernel-layer owner | Mark, until a CUDA engineer is named. The owner approves every upstream kernel SHA bump and patch set. |
| Budget lines | Local GPU hours, measured by Roost's `tools/gpu-lease` ledger, inside the monthly compute budget in [kickoff](../docs/kickoff.md). Cloud GPU for Roost evidence (RunPod pods, and the H100 hours per release cycle for the M2 comparison) only within an amount recorded there; none is approved by this amendment. |
| Boundary | Roost consumes this repository's packages, registry, protected fixtures and evaluator; its profiles pass the same gates. No Roost code lands in this repository. |

## Roles

| Area | Owner |
| --- | --- |
| Technical lead, API owner, final release reviewer | Mark |
| Core API | Mark |
| ONNX Runtime backend | Mark |
| Tokenizer implementation, conformance, release lane, upstream contributions | Mark |
| Release pipeline | Mark |
| Second release owner | Open, required before broad support commitments |

Every stable model family has a named owner recorded in its manifest. A family without an owner is Experimental.

## Release authority

Only a human maintainer approves a release. Implementation agents open draft changes and produce evidence. They cannot approve their own change, hold publishing credentials, or change protected fixtures, tolerances, or release evaluation sets. See [CONTRIBUTING.md](CONTRIBUTING.md#agent-contributions).

## Maturity states

| Status | Meaning |
| --- | --- |
| Experimental | Investigation or incomplete feature set; no production support claim. |
| Preview | Declared scope passes meaningful tests; API or platform support may still change. |
| Stable | Documented contract, qualification gates G0 to G7, owner, compatibility matrix, and regression support. |
| Maintenance | Existing pinned combinations remain supported; new upstream features are not pursued. |
| Deprecated | Replacement and migration information published; support end date stated. |

Status applies per model profile and hardware combination.

## Decisions

New public APIs, backends, model families, and fidelity policy changes need a short design proposal (an issue using the design proposal template). Routine fixes need an issue plus tests.
