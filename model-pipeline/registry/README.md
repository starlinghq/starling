# Model registry

One directory per model family, one subdirectory per profile, each holding a `manifest.yaml` that validates against [`../schemas/model-manifest.schema.json`](../schemas/model-manifest.schema.json).

```
registry/
  <family>/
    <profile>/
      manifest.yaml
```

The public catalog is generated from these manifests (P-021); there is no second hand-maintained support list. Every capability carries an explicit status, and a missing value never implies support.

Validate with `model-pipeline validate registry`.

Quote every digest and revision in YAML. An unquoted all-digit SHA parses as a number and fails validation.
