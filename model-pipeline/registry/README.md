# Model registry

One directory per model family, one subdirectory per profile, each holding a `manifest.yaml` that validates against [`../schemas/model-manifest.schema.json`](../schemas/model-manifest.schema.json).

```
registry/
  <family>/
    <profile>/
      manifest.yaml
```

The public catalog, [`docs/catalog/`](../../docs/catalog/README.md), is generated from these manifests by `model-pipeline catalog` (P-021); there is no second hand-maintained support list. Every capability carries an explicit status, and a missing value never implies support.

Validate with `model-pipeline validate registry`. After changing a manifest, regenerate the catalog with `model-pipeline catalog`; CI fails when it is stale.

Quote every digest and revision in YAML. An unquoted all-digit SHA parses as a number and fails validation.
