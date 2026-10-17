# Charter

## Purpose

Make .NET a dependable place to use modern AI. A model supported here comes with a usable .NET API and reproducible evidence of what works, how fast it runs, and which upstream behavior it preserves.

## Scope

- .NET implementations and integrations for deep learning models, LLMs, and tokenizers.
- An owned tokenizer library (`Starlings.Tokenizers`) with its own release cadence.
- The delivery pipeline that pins, verifies, benchmarks, and releases each supported combination.

- [Roost](https://github.com/starlinghq/roost), a serving engine for NVIDIA GPUs in its own repository, under the amendment below. It consumes this repository's packages, registry, protected fixtures and evaluator.

Out of scope until a separate use case, owner, and budget exist: general-purpose training and multimodal generation. Distributed serving and bespoke GPU kernels are out of scope for this repository's libraries and in scope for Roost, in the Roost repository only, under the amendment below.

## Roost amendment (proposed 2026-10-17, for the maintainer)

Status: proposal for the maintainer's review under [CONTRIBUTING.md](CONTRIBUTING.md#agent-contributions). Nothing here is adopted until the maintainer approves the pull request. Entries marked *for the maintainer* are deliberately blank.

**Goal (build plan decision 18, set by the maintainer on 2026-10-17).** Roost is a full replacement for vLLM on NVIDIA GPUs: every serving capability in the parity matrix (plan section 13.1), at or above vLLM on a published scorecard (plan section 14.3), with fidelity evidence vLLM does not publish and a fault model vLLM does not have. Multi-LoRA and user control over the kernels that run a model are first-class goals. The plan is [`docs/plan/build-plan.md`](https://github.com/starlinghq/roost/blob/ff072f8d745abb58c226a43d99ae79334cd78cdb/docs/plan/build-plan.md) in `starlinghq/roost` at commit `ff072f8d745abb58c226a43d99ae79334cd78cdb`.

**In scope for Roost, and only in its repository:**

- Distributed serving: tensor, pipeline, data and expert parallelism, multi-node serving, and disaggregated prefill and decode.
- Bespoke GPU kernels: CUDA C++ kernels written and maintained by Roost (fused and persistent kernels, collectives), alongside vendored upstream kernels pinned by full commit SHA. Built kernel binaries are release assets, never committed.
- A server with OpenAI-compatible and related routes, and an in-process API for `Starlings.Generation`.

**Excluded (build plan section 13.2):**

- Non-NVIDIA backends (ROCm, TPU, Gaudi, Neuron, XPU) and CPU serving.
- Training and fine-tuning. In-place weight update for reinforcement-learning loops is in scope.
- Audio input and output, until Starling admits an audio model.
- Ray orchestration.
- Python extension points of vLLM.
- Loading formats Starling does not lock (for example GGUF).
- AutoRound weight formats and 2-bit KV quantization, until a Starling profile needs them.

**Unchanged by the amendment.** Roost's model profiles pass the gates of this repository. It is proposed that their manifests, fixtures and tolerances live in this repository's registry and `protected/` directory under its protected-evidence workflow; that is plan decision 4, which is still open, and nothing here settles it. No Roost code lands in this repository. No throughput claim is published for a Roost profile before it passes the fidelity gates.

| Item | Value |
| --- | --- |
| Named consumer and use case | *For the maintainer.* |
| Kernel-layer owner | *For the maintainer.* The plan proposes two CUDA roles (kernels and worker; acceleration) and, from M4, a distributed-systems role (decision 19); none is filled. |
| Budget: local GPU hours per milestone | *For the maintainer.* Measured by Roost's `tools/gpu-lease` ledger. The recorded cap is 40 GPU-hours a month ([kickoff](../docs/kickoff.md)). |
| Budget: rented single-GPU hours per milestone | *For the maintainer.* Plan decision 16. |
| Budget: rented multi-GPU hours per milestone (NVLink pods, and a multi-node rental) | *For the maintainer.* Plan decision 20. |

Amounts are recorded in [kickoff](../docs/kickoff.md) when the maintainer sets them. This amendment approves no spend.

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
