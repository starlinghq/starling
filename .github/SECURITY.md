# Security policy

Report vulnerabilities privately through GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"). Do not open a public issue.

Include the affected package and version, the model bundle identity if relevant, and a minimal reproducer. We aim to acknowledge reports within 5 working days. This is an internal operating target, not a commercial SLA.

## Supported versions

| Version | Security fixes |
| --- | --- |
| `0.3.0` (ai-dotnet `Starlings.*`, current release, 2026-10-19) | Yes |
| `0.2.0` (2026-10-17) | No patches: fixes ship in the next version. Upgrade to `0.3.0`. |
| Earlier versions (`0.1.0-preview.*`) | No patches: a preview's fixes ship in the next version, not in the preview itself ([support policy](../docs/support-policy.md#maturity-levels-in-practice), Preview row). Upgrade to `0.3.0`. |

While the packages are `0.x` and there is a single release owner, fixes ship only in the latest minor line, with no back-port to the previous minor ([support policy](../docs/support-policy.md#maturity-levels-in-practice), Stable row). A fix ships as a new package version; published versions are never replaced, and affected versions are deprecated on nuget.org, not deleted ([support policy](../docs/support-policy.md#security-and-correctness-issues)).

Each published advisory also gets a reviewed record under `docs/advisories/` in this repository (created with the first advisory). It names the affected identities, including the affected NuGet version range, and the fixed version; the model catalog and the affected profiles' `maintenance.known_issues` link it ([support policy](../docs/support-policy.md#security-and-correctness-issues), step 2). `Starlings.Tokenizers` (`0.3.0` with 0.2.0) is released from [starlinghq/tokenizers](https://github.com/starlinghq/tokenizers).

Areas of particular interest: native interop and memory ownership, model bundle extraction and loading (path traversal, digest bypass), and anything that causes a network call during normal inference.
