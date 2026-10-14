# Proposed fixtures

Each subdirectory holds fixtures that an implementation job proposes and the maintainer has not accepted yet. The layout mirrors `model-pipeline/protected/`, and each directory has a `PROPOSED.md` that names the reference source, the environment and the decisions for the reviewer. If a proposal is accepted, the maintainer moves it byte-identical to the same path under `protected/`. Tests read the protected copy when it exists and the proposal otherwise. Nothing here is binding, and nothing here changes a tolerance, gate or capability status.
