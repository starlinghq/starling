# Protected evidence

Reference fixtures, golden outputs, and tolerance files per model profile (plan section 10.1). Changes need maintainer review through CODEOWNERS.

Implementation jobs read these files and never edit them to make a candidate pass. New edge cases are proposed in a separate PR that names the reference source.

Tolerances are set before optimization starts.

## Release sets (P-030)

Release evaluation sets live in a family-level `release/` directory (`<family>/release/<set>/`) and, for sets shared across families, in `release-shared/`. No development run reads them: the evaluator, the fidelity tools and the tests address a release set only by its explicit path, and each profile directory keeps its development fixtures. A profile's G3 budget is `<family>/<profile>/task-quality.yaml`, bound to a dataset by its digest.

The plan's ideal is a holdout that implementation jobs cannot read. A public repository cannot offer that, so these sets rely on four other properties instead: the data is externally authored (public benchmarks and corpora with verified licenses, selected by a pinned, seeded procedure; nothing hand-written); sets and budgets were committed before any .NET evaluation ran on them; budgets come from the upstream reference's score on the exact set; and changes need a `protected-change` pull request that cannot also change implementation code. The maintainer rotates the sets: once a set's failures have driven implementation changes it is development data, and a new set replaces it. Each set's `SOURCE.md` records its sources, licenses, revisions and selection script.

Files here are byte-exact (`-text` in `.gitattributes`).
