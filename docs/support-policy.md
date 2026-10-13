# Support policy

Status: **draft for maintainer review** (workstream R10, `docs/plan/production-readiness.md`). This document states what each maturity label commits the project to. It does not change any label: no profile is Stable today, and maturity changes only when the maintainer records gate results (M10). Where this document proposes a number or a rule that the plan does not fix, it says so and lists it under [open questions](#open-questions).

Sources: plan sections 9, 10, 14 and 16 (`docs/plan/fast-faithful-ai-dotnet.md`), `.github/SUPPORT.md`, `.github/SECURITY.md`, `docs/kickoff.md`.

## What is supported

Support attaches to a **combination**, not to a model name. A combination is the four identities of plan section 9:

1. the package version (`Starlings.Models.E5` `0.1.0-preview.5`);
2. the model bundle: the asset lock the package ships (repository, full commit SHA, file sizes and SHA-256), and the owned tokenizer package version in `processing.tokenizer.version`;
3. the runtime profile: backend and native package versions, provider, precision, OS and architecture;
4. the evidence: fixtures, tolerances, task-quality sets and benchmark protocol that qualified it.

The registry manifest (`model-pipeline/registry/<family>/<profile>/manifest.yaml`) records 2 to 4, and the release record (`docs/releases/ai-<version>.md`) ties them to 1. A combination that no manifest and release record describe is unsupported, whatever the label of its parts.

Each capability and each platform has its own status: `supported`, `unsupported`, `experimental` or `untested`. A missing value never means supported. A profile's maturity is the ceiling: an `experimental` capability inside a Stable profile carries no Stable commitment.

## Maturity levels in practice

Response and window figures are internal operating targets, not commercial SLAs (plan section 16). Support covers staffed work only. The windows marked *proposed* are R10 proposals awaiting the maintainer.

| Maturity | What a user can rely on | API and behavior changes | Fixes go to | Support window |
| --- | --- | --- | --- | --- |
| **Experimental** | Nothing beyond the recorded evidence. The profile may change or be removed in any release. | Any release, noted in the release record. | The next release only, best effort. | None. |
| **Preview** | The declared scope passes meaningful tests (plan 14): G0, G1, G2 and G6 in full, G3 to G5 and G7 as checklists. Published bytes behind a pinned identity never change. | Allowed between preview versions, with release notes and a migration note. | The next preview version. | Until the next preview version (*proposed*). Earlier previews stay installable and their locks stay resolvable, but they get no fixes. |
| **Stable** | The documented contract (inputs, outputs, limits, unsupported features) in the package README, every gate G0 to G7 passed in full, a named owner, rows in the compatibility matrix, and regression support (below). | Breaking API changes only in a new major version. An output-changing fix (tokenizer, template, preprocessing) ships as a new qualified profile and a release note, even with no public C# change (plan 14). | The latest minor version of the current major. Security and correctness fixes also go to the previous minor version. | The previous minor version gets security and correctness fixes for 6 months after the next minor ships (*proposed*). |
| **Maintenance** | Existing pinned combinations stay supported: security and correctness fixes, and the scheduled qualification keeps running. New upstream features are not pursued (plan 14). | None except fixes. | Patch versions of the maintained line. | Until `maintenance.support_ends` in the manifest, at least 6 months after entering Maintenance (*proposed*). |
| **Deprecated** | A replacement (`maintenance.replaced_by`), migration information, and a support end date (`maintenance.support_ends`) are published (plan 14). | None. | Security fixes only, until the end date. | Until `support_ends`. For a profile that was Stable, at least 6 months after the deprecation notice (*proposed*). |

A package and the profiles it serves can carry different labels. Today every ai-dotnet package is versioned `0.1.0-preview.N` and labeled Preview, while only `multilingual-e5-small/onnx-fp32-cpu` has profile maturity `preview`; every other profile is `experimental`. The package README states both.

### Promotion and demotion

- **Promotion** happens only when the maintainer records gate results in the manifest (M10). Agents produce evidence and tooling; they never set a gate to `passed` or raise maturity. The manifest validator refuses `stable` while any gate is not `passed` or any capability is `untested` (`model-pipeline/src/model_pipeline/manifest.py`). The catalog's Stable readiness column lists what still fails (draft PR #4).
- **Demotion** from Stable to Preview or Experimental needs a release note and a visible catalog change. It never rewrites what an earlier release promised: the earlier release record stays as published.

## Security and correctness issues

A security or correctness problem marks a release **affected**. It never silently rewrites the user's dependency lock (plan 14). In practice:

1. **Published bytes are immutable.** A package version, an asset lock and a hosted graph are never replaced behind an existing identity. A fix is a new package version and, when bundle files change, a new lock.
2. **Advisory record.** Each problem gets one record, published as a GitHub security advisory for vulnerabilities (`.github/SECURITY.md`) or a regression issue for correctness. It names the affected identities: package version range, profile IDs, lock commit and file digests, tokenizer version, provider and platform. It also states severity, workaround, the fixed version, and the evidence of the fix. A record under `docs/advisories/` is proposed so the catalog can link it (*proposed*; see open questions).
3. **Manifest and catalog.** The affected profile gets a `maintenance.known_issues` entry that links the record. The generated catalog shows it. The schema has no field for affected version ranges yet (*open question*).
4. **NuGet.** Affected package versions are marked deprecated on nuget.org (reason "critical bugs" or "other", with the fixed version as the alternative). They are not deleted. Deprecation and unlisting leave existing `packages.lock.json` restores working, so a user's build does not change until they choose to update.
5. **Runtime behavior does not change.** The runtime is local and offline after explicit acquisition, and an inference call never downloads (plan 14). Nothing phones home to check advisories. Users learn about advisories through NuGet's deprecation warnings at restore time, the catalog, and the advisory itself.
6. **Withdrawn upstream assets.** If an upstream repository deletes a pinned revision, or hosting has to be withdrawn (for example for legal reasons), the old lock can no longer download. `ModelAssetNotFoundException` says so explicitly, and the record names the replacement. Already-acquired caches keep working offline. The project cannot guarantee that third-party revisions stay available.

## Tokenizer bumps

Fixed by the plan (section 14):

- The owned tokenizer `Starlings.Tokenizers` is released independently. Each production model lock records the qualified tokenizer version (`processing.tokenizer.version`), and the pin lives in `ai-dotnet/Directory.Packages.props`.
- A newer tokenizer release is not automatically certified for existing bundles.
- A tokenizer change that alters outputs for a profile requires a new qualified model profile and a release note, even when no public C# method changes.

**Open for the maintainer (M8):** when a tokenizer bump leaves a profile's outputs unchanged, what must happen before the manifest records the new version? Two options:

| Option | What runs | Cost | Risk |
| --- | --- | --- | --- |
| A. Rerun model G1/G2 | `model-pipeline evaluate --require-gates G1,G2` for every profile on the new tokenizer (what R4, draft PR #6, did at `0.3.0-preview.1`: every CPU profile passed on the development fixtures). | One evaluator run per profile. CUDA profiles need GPU time (M6). | Low. Catches interactions the conformance suite does not exercise (offsets, batching, prompt prefixes). |
| B. Accept the tokenizer conformance suite | The tokenizer repository's conformance fixtures for the profile's `profile_id` pass unchanged, and the release notes say which profiles they cover. | None in this repository. | A component the profile uses could change in a way the fixtures do not cover. This is acceptable only if the suite covers every component the profile's `tokenizer.json` uses. |

A middle path is possible: B for patch releases whose diff does not touch a component the profile uses, A otherwise. Until the maintainer decides, this document makes no rule. The repository's practice so far is a tokenizer-suite check at each bump, plus one model rerun (R4) at `0.3.0-preview.1`.

## Compatibility matrix

The compatibility matrix is generated, never hand-kept: `model-pipeline catalog` writes `docs/catalog/compatibility-matrix.md`, `docs/catalog/README.md` and `docs/catalog/catalog.json` from the manifests' `compatibility`, `capabilities`, `execution` and `qualification` fields (P-021, draft PR #4). CI fails when the committed output is stale. A Stable support claim covers exactly the matrix rows with status `supported` for that profile. `unrecorded` and `untested` cells are not claims. Today no row is `supported`: Linux x64 is `experimental` for every profile, and Windows x64 is `untested`.

Matrix rows beyond Linux x64 need qualification CI (P-018, workstream R6), with a Windows x64 job and scheduled runs.

## Regression support

Regression support applies to **Stable** profiles only (`.github/SUPPORT.md`). Earlier maturities get best-effort fixes.

A regression is any of the following, on a supported matrix row:

- an output outside the profile's protected tolerances or task-quality budget;
- a G5 benchmark beyond the profile's performance budget;
- a G4 operational failure: a leak, a hang, unbounded growth, or state leakage between requests;
- an installation or offline-execution failure (G6).

Commitments:

1. **Intake.** Reports use the regression template (`.github/ISSUE_TEMPLATE/regression.yml`), with the four identities from the diagnostics and a reproducer. Triage, including affected profiles, severity and workaround, has an internal target of 5 working days (*proposed*; `.github/SECURITY.md` uses the same figure for acknowledgement).
2. **Detection without reports.** A scheduled qualification run (R6) restores the pinned models from a cache, runs the model tests, the protected evaluator and a short soak for every Stable profile, and uploads evidence. A failure there is handled like a report. The cadence is to be set with R6; weekly is *proposed*. **Without R6 there is no regression support, and so no Stable profile.**
3. **Fix.** The fix ships as a new patch version. It reruns every gate the change affects (the scoping table in `docs/plan/maintained-update-drill.md`), and the release record links the evidence. If the fix changes outputs, it is a new qualified profile (plan 14).
4. **Regression test.** The reproducer becomes a test. A new case in protected fixtures is proposed in a separate change, with its reference source, and committed by the maintainer (plan 10.1). Agents never edit `model-pipeline/protected/` or tolerances.
5. **Affected releases** are marked as in [Security and correctness issues](#security-and-correctness-issues).

## Owner

| Role | Holder | Status |
| --- | --- | --- |
| Accountable maintainer: technical lead, API owner and final release reviewer | Mark | Settled (`docs/kickoff.md`) |
| Owner of every registry profile (`maintenance.owner`) | Mark | Recorded in every manifest |
| Second release owner | **open** | M7. Plan 16 requires a second trusted human release owner before broad support is promised. Protected branches and the `release` environment must not let an implementation agent approve its own change. |
| First named consumer | **open** | M7, P-028. Needed for G7 (representative downstream use). |

The single-owner gap is a Stable blocker in `docs/plan/production-readiness.md` (M7). Until a second release owner is named, any Stable promotion carries a bus factor of one. The maintainer decides whether that blocks the first Stable label (see open questions).

## Open questions

These are for the maintainer. None is settled by this document.

1. **M8, tokenizer bumps:** option A, B or the middle path (see [Tokenizer bumps](#tokenizer-bumps)).
2. **Support windows:** confirm or change the proposed figures: 6 months for the previous Stable minor, at least 6 months for Maintenance and Deprecated, the 5-working-day regression triage target, and a weekly scheduled qualification.
3. **Stable and 1.0:** does a Stable profile require its package to leave `0.x` (`1.0.0`), or can a `0.x` package carry a Stable profile?
4. **Second release owner (M7):** does the first Stable promotion (e5 `onnx-fp32-cpu`) wait for a second release owner, or ship with a recorded single-owner exception?
5. **Advisory records:** use GitHub advisories only, or also `docs/advisories/` records that the catalog links? Should the manifest schema gain a field for affected version ranges?
6. **Old and new profile identities** after an upstream update: see `docs/plan/maintained-update-drill.md`, open questions.
