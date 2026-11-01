# Support

Support covers staffed work only. Response goals below are internal operating targets, not SLAs.

| Request | Where | Goal |
| --- | --- | --- |
| Regression in a released profile | Regression report template | Triage (reproducer, affected profiles, severity, and workaround recorded) within 5 working days |
| Upstream change to an admitted model family | Upstream change template | Triage within 2 working days |
| New model | Model request template | Feasibility and effort decision within 10 working days when capacity exists |
| Questions | GitHub Discussions | Best effort |

Check the maturity status of a model profile before depending on it. Only Stable profiles carry regression support, backed by the weekly CPU qualification run and the per-release GPU run.

Support windows ([support policy](../docs/support-policy.md#maturity-levels-in-practice)):

- While the packages are `0.x` and there is a single release owner, fixes ship only in the latest minor version; there is no back-port window for the previous minor. From `1.0`, or once a second release owner exists, the previous minor version gets security and correctness fixes for 6 months after the next minor ships.
- A profile in Maintenance stays supported until its `maintenance.support_ends` date, at least 6 months after it entered Maintenance.
- A Stable profile that is deprecated keeps security fixes for at least 6 months after the deprecation notice.
