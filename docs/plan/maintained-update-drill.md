# Maintained update and rollback drill

Status: **design, for maintainer review.** It covers backlog P-020 ("Add upstream change detection and scoped revalidation") and P-022 ("Execute first maintained update and rollback drill"), part of milestone M3 (plan section 17: "One upstream change end to end ..."). It was written in workstream R10 (`docs/plan/production-readiness.md`). This document only designs the drills; none has been run. Only the detector in section 3 exists, as code.

Status (2026-10-18): the drills are still not run. The detector now runs weekly as job `upstream` of `.github/workflows/qualification.yml`; it differs from the proposed schedule in section 3: the workflow token is read-only and cannot open an issue, so a newer upstream revision fails the job, and the failed run is the notification. Packages are now published: ai-dotnet `0.2.0` is on nuget.org (2026-10-17, `docs/releases/ai-0.2.0.md`), so the rollback drill's deprecation step (section 7) can act on a real feed.

Exit criteria from the backlog:

- **P-020:** model and tokenizer changes create useful proposals and rerun all affected gates. Upstream PR status cannot block an owned release.
- **P-022:** old and new profiles both remain identifiable. Regression and migration information is visible.

## 1. What already exists

The drills reuse the acquisition and lock mechanics as they are. No new mechanism is needed to keep old and new combinations side by side.

| Mechanism | Where | Property the drills rely on |
| --- | --- | --- |
| Manifest pins | `origin.code`, `origin.weights` (full 40-character SHA, schema `gitSha`), `origin.files` (path, size, SHA-256) | The upstream identity is exact, and branch or tag names are rejected. |
| Maintenance fields | `maintenance.latest_known_upstream_revision`, `latest_verified_upstream_revision`, `last_checked`, `review_date`, `replaced_by`, `support_ends` | *Known* (seen upstream) and *verified* (qualified here) are already separate fields. |
| Asset lock in the package | `ModelAssetLock` (`E5ModelAssets.Lock` and others): repository, commit, files with size and SHA-256; JSON schema 2 adds explicit per-file `url` for graphs hosted outside the upstream repository | Each package version carries exactly one lock. A user who pins the package version pins the bundle. |
| Cache layout | `<root>/<owner>/<name>/<commit>/<path>`, partial files under `<root>/.downloads/` (`Starlings.Acquisition`) | Two revisions of the same repository coexist in one cache, and neither overwrites the other. |
| Offline resolution | `OpenLocal` verifies and never downloads. `DownloadAsync` is the only network call. | A rollback works offline when the old snapshot is still cached. |
| Failure types | `ModelAssetNotFoundException` (revision or file removed upstream), `ModelAssetIntegrityException` (digest mismatch) | A withdrawn or altered upstream revision fails with an explicit error, never silently. |
| Tokenizer pin | `ai-dotnet/Directory.Packages.props` (exact) and `processing.tokenizer.version` in each manifest, changed together (CLAUDE.md) | Tokenizer releases are independent of model releases. |
| Release record | `docs/releases/ai-<version>.md`: packages, locks, checks, known issues | Ties a package version to its locks and evidence. |
| Catalog | `model-pipeline catalog` (draft PR #4): profiles, gates, compatibility matrix, generated from the manifests | The public view of each identity's status. |
| Templates | `.github/ISSUE_TEMPLATE/upstream-change.yml`, `regression.yml` | Intake for upstream changes and regressions. |

## 2. Change sources

| Source | Examples | Detected by |
| --- | --- | --- |
| Upstream model repository on the Hub | new commit to `intfloat/multilingual-e5-small`, `google/embeddinggemma-2`, `fastino/gliner2-base-v1`, `Qwen/Qwen3.5-0.8B` | `model-pipeline upstream` (section 3) |
| Upstream code repository | `fastino-ai/GLiNER2` (preprocessing semantics), `microsoft/onnxruntime-genai` (builder) | By hand for now; the detector reports these as `unchecked` |
| Execution runtime | `Microsoft.ML.OnnxRuntime` 1.30.0, `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 | By hand (NuGet). A future step could compare `Directory.Packages.props` with the NuGet v3 API |
| Owned tokenizer | `Starlings.Tokenizers` releases (starlinghq/tokenizers) | The tokenizer repository's release notes; the pin is raised by hand |
| Starling-built artifacts | re-exported ONNX graphs (GLiNER2, EmbeddingGemma 2), the Qwen `ortgenai` build | Our own change, so no detection needed; a new artifact means a new digest and a new lock |
| Reference stack | transformers, tokenizers, sentence-transformers, torch versions in the reference runners | By hand when fixtures are regenerated (maintainer only) |

## 3. Upstream change detection (implemented: `model-pipeline upstream`)

```sh
cd model-pipeline
model-pipeline upstream registry [--json report.json] [--responses recorded.json] [--date YYYY-MM-DD]
```

- For every non-synthetic manifest, the detector takes `origin.code` and `origin.weights`, and checks each distinct huggingface.co model URL against `GET /api/models/<repo>?blobs=true` (the default branch's `sha` and file listing).
- The **tracked** source is the one whose pinned revision equals `maintenance.latest_verified_upstream_revision`. Its upstream `sha` is also compared with `latest_known_upstream_revision`: a revision that is neither pinned nor known is **new since known**.
- When a revision moved, the detector fetches the listing at the pinned revision too. It compares the two listings by Git blob id, which covers LFS and non-LFS files. Each added, removed or modified file is then classified (section 4) and flagged as locked (it is in `origin.files`) or not.
- Each changed source gets the gates to rerun (section 4). If the pinned listing cannot be fetched, the scope is every gate.
- Output: a Markdown report on stdout, and optionally `starlings-upstream-check/1` JSON. Exit 0 means nothing new, 1 means at least one new-since-known revision, 2 means bad input or a failed fetch (a private or gated repository without `HF_TOKEN`, or a network error).
- **Read-only.** The detector never edits a manifest, never moves a pin and never changes a gate. It prints the proposed manifest edit (`latest_known_upstream_revision`, `last_checked`) for a pull request.
- `HF_TOKEN` is sent only to the Hub API endpoint. Non-Hub sources are reported as `unchecked`.
- **Offline mode:** `--responses` reads recorded API responses (`{repo: {"main" | <sha>: <model info> | {"error": ...}}}`) and never opens a connection. The tests (`model-pipeline/tests/test_upstream.py`) use only recorded responses and fail if anything calls `urlopen`.

Live run on 2026-10-13 (not committed as evidence): every Hub source in the registry was `current`, with no revision new since known. The GLiNER2 code source (GitHub) and the onnxruntime-genai builder source were `unchecked`.

**Proposed schedule** (R6 owns `.github/`, and the maintainer approves workflow changes): run weekly in the scheduled qualification workflow with a read-only token. On exit 1, open an issue from the upstream-change template with the report attached. The workflow never pushes manifest changes. Triage target: 2 working days (`.github/SUPPORT.md`).

## 4. Scoped revalidation

A detected change becomes a proposal: a pull request that records `latest_known_upstream_revision` and `last_checked`, together with the scope below. Pins and `latest_verified_upstream_revision` move only after the scoped gates pass on the new revision and the maintainer records them.

| Changed locked file | Kind | Gates to rerun | Why |
| --- | --- | --- | --- |
| `README.md`, `LICENSE*`, other `*.md` | card | G0 | Rights, terms and attribution can change (e5's MIT is stated only in the card). No behavior change. |
| `tokenizer*.json`, `*.model`, `special_tokens_map.json`, `vocab*`, `merges.txt`, `added_tokens.json`, `chat_template*`, `*processor_config.json` | tokenizer | G0, G1, G2, G3, G6 | Discrete outputs change, so new fixtures follow (maintainer). The tokenizer asset digest is part of the owned profile, and the lock changes, so G6. |
| `*.safetensors`, `*.onnx`, `*.onnx_data`, `*.bin`, `*.gguf`, `*.pt` | weights | G0, G2, G3, G4, G5, G6 | Numbers, quality, memory and speed can all move. Tokenization does not. |
| any other locked file (`config.json`, `1_Pooling/config.json`, ...) | config | G0, G1, G2, G3, G6 | Pooling, normalization, prompts and special-token settings feed both processing and numerics. |
| only files outside the lock | none | none | Record the revision as known. The pin can stay. |
| listing unavailable | unknown | G0 to G7 | Nothing is known about the change. |

Changes the detector does not see are scoped by the same rule, by hand:

| Change | Gates to rerun |
| --- | --- |
| Runtime version (ONNX Runtime, GenAI) | G2, G4, G5, G6; G7 when an adapter's dependency moves |
| Upstream code that we port (GLiNER2 preprocessing) | G1, G2, G3 |
| Re-exported or rebuilt graph (new digest) | G0, G2, G3, G5, G6 |
| Owned tokenizer, outputs change for the profile | new qualified profile: G1, G2, G3, G6 (plan 14) |
| Owned tokenizer, outputs unchanged | open (M8, `docs/support-policy.md`) |

G7 is rerun whenever a change alters an adapter's observable behavior. A Stable profile also reruns the short soak (G4) after any change, because the scheduled qualification does so anyway.

**Owned releases are not blocked upstream (P-020).** When a defect needs an upstream fix (tokenizers, onnxruntime-genai), the owned fix ships in `Starlings.Tokenizers` or our adapter first, with its own qualified profile. The upstream report or PR is tracked separately (P-026 patch ledger). Its merge status gates nothing here.

## 5. Identity of old and new profiles

The P-022 criterion is that both profiles remain identifiable. With the existing mechanics:

- The **old combination** stays fully identified by its package version, the lock inside it, the release record and the Git tag. Nothing about it is edited after release. The catalog keeps showing it until the manifest says otherwise.
- The **new combination** gets a new package version, a new lock (new commit and digests), and a release record that names both identities and links the migration note.
- The **registry** has one manifest per profile ID, so the old state needs a place. Decided (#206, open question 1; 2026-11-01 by the maintainer's delegate (agent), revisitable): while the new revision is being qualified, it gets its own profile directory, `<profile>-r<N>` (for example `multilingual-e5-small/onnx-fp32-cpu-r2`), at `experimental`. The old profile is untouched. When the new one reaches the old one's maturity, the old manifest takes `maturity: maintenance` (or `deprecated`) with `maintenance.replaced_by` and `support_ends`. Both remain in the catalog.

## 6. The update drill (P-022, first half)

Purpose: run one real change end to end through detection, proposal, scoped revalidation, release candidate and catalog, and measure the maintainer hours it takes (plan 17 asks for a steady-state figure).

**Choosing the change.** Prefer a real upstream Hub revision for a pinned model, if one appears. None existed on 2026-10-13. Otherwise use, in order:

1. a runtime bump for e5 `onnx-fp32-cpu` (the next ONNX Runtime release after 1.30.0): a real change, scoped to G2, G4, G5, G6;
2. an owned tokenizer patch release that changes no outputs for the profile (exercises M8);
3. a recorded synthetic upstream change, through `--responses`, to exercise detection and the proposal only. It cannot complete the drill, because nothing real gets qualified.

**Steps** (agent work in draft PRs; the maintainer records results):

| # | Step | Output |
| --- | --- | --- |
| 1 | Detect: `model-pipeline upstream registry --json` (or the runtime release notes) | Report; upstream-change issue |
| 2 | Propose: a PR setting `latest_known_upstream_revision` and `last_checked`, with the scope from section 4 | Draft PR |
| 3 | Lock: a new `ModelAssetLock` for the new revision (JSON form via `Starlings.Acquisition.Tool`, then `download` and `verify` with it); new `origin.files` digests; for a converted model, rerun the reproducible export and record the new graph digest | New lock and manifest (new profile directory per section 5) |
| 4 | Reference: the maintainer regenerates fixtures where discrete outputs changed (agents do not author protected sets) | Protected fixtures PR (`protected-change` label) |
| 5 | Revalidate the scoped gates: `model-pipeline evaluate --require-gates ...`, the G3 evaluator, the R1 soak, the R2 harness, the release-candidate drill | Evidence under `docs/evidence/p022-<profile>-<date>/` |
| 6 | Release candidate: new package version whose release record lists the old and new identities, the evidence, known issues and a migration example (package version, lock, any API or output difference) | `docs/releases/ai-<version>.md` |
| 7 | Catalog: `model-pipeline catalog` | Catalog and matrix show both profiles |
| 8 | Record: the maintainer copies gate results, moves `latest_verified_upstream_revision`, and sets the maturity of old and new | Maintainer PR |

**Pass criteria:** every scoped gate has evidence; the old profile's evidence is untouched; the catalog shows both identities; the release record has the migration example; maintainer hours are recorded.

## 7. The rollback drill (P-022, second half)

Purpose: show that a user can return to the previous combination without help, and that the maintainer can mark the new one affected without rewriting anything.

**User side** (scripted, extending the release-candidate drill in a clean consumer project):

1. Install the new package version and acquire its assets; run the sample; record the diagnostics (all four identities).
2. Pin the previous package version in the consumer (`packages.lock.json` regenerated by restore). With the old snapshot still cached, `OpenLocal` resolves it **offline**, and the sample's outputs equal the old release's recorded evidence (G2 fixtures from ids). Both snapshots stay side by side in the cache (`<owner>/<name>/<commit>/`).
3. Clear the cache and acquire the old lock again. The download succeeds while the old upstream revision exists. Removing it is simulated with a test endpoint that returns 404, which must yield `ModelAssetNotFoundException` naming the revision.
4. Diagnostics after rollback report the old identities exactly.

**Maintainer side:**

1. Publish the advisory and the `known_issues` entry for the new version, as in `docs/support-policy.md`, and regenerate the catalog.
2. Deprecate the new package version on the feed (in the drill, on the local feed or as a documented dry run, since nothing is published before M5). Verify that a locked restore of either version still works.
3. Confirm that no published byte, lock or release record changed (`git diff` of `docs/releases/` and the lock sources is empty, apart from the advisory link).

**Pass criteria:** rollback with no source checkout, offline when cached; outputs match the old evidence; both identities visible in diagnostics and catalog; the explicit error for a removed upstream revision; no rewritten lock.

## 8. Not covered yet

- Detection of GitHub code sources and NuGet runtime versions. Both are by hand.
- Automatic issue creation and the scheduled run (R6, with maintainer approval of `.github/`).
- A check that the *pinned* revision still exists when the default branch has not moved. A force-push or deletion upstream would show up today only at the next download. Fetching the pinned revision on every run would close this, at one extra API call per source.
- Hosted Starling repositories (`Starlinghq/qwen3.5-0.8b-ortgenai`) are not upstream and are not checked. They change only by our own commits.

## 9. Open questions for the maintainer

1. **Old and new profile identity in the registry:** settled 2026-11-01 (#206, decided 2026-11-01 by the maintainer's delegate (agent), revisitable): the `<profile>-r<N>` revision directory of section 5 (as `<model>/<profile>-r<N>`), at `experimental` until it reaches the old profile's maturity; then the old manifest moves to `maintenance` (or `deprecated`) with `maintenance.replaced_by` and `support_ends`, and both stay in the catalog. Rejected: a single manifest relying on release records, which would hide an older pinned profile's gate record from the current catalog, short of P-022's "both remain identifiable". Not chosen: the revision in the profile ID.
2. **First drill change:** accept the order in section 6 (a real Hub change, else the e5 ONNX Runtime bump, else a tokenizer patch)?
3. **M8:** the tokenizer-bump rule that decides the scope of option 2 in section 6. Settled 2026-11-01 (#202): the middle path in `docs/support-policy.md` ("Tokenizer bumps").
4. **Upstream deletion risk:** mirror the bytes of Stable bundles (as release assets) so a rollback survives an upstream deletion, or accept the risk and document it? Mirroring is redistribution, so it needs a rights check per model (G0).
5. **Detector cadence and token:** weekly, with a read-only `HF_TOKEN` that has no write scope? This is set in R6.
