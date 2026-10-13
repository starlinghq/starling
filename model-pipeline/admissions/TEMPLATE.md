# Admission: <model family>

Copy to `admissions/<family>.md`. Plan section 7 and pipeline stage 1.

| Field | Value |
| --- | --- |
| Candidate | Hub ID or upstream URL |
| Code revision | Full commit SHA |
| Weight revision | Full commit SHA |
| Requested by / demand evidence | |
| Proposed owner | |
| Proposed maturity at first release | Experimental / Preview |
| Decision | Admitted / Blocked / Deferred |
| Decided by and date | |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | | |
| Weights | | |
| Tokenizer assets | | |
| Evaluation data | | |

A missing access or redistribution right blocks admission regardless of score.

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | | |
| Backend feasibility | 20% | | |
| Reusable architecture coverage | 20% | | |
| Access and licensing clarity | 15% | | |
| Testability | 10% | | |
| Maintenance burden (5 = low) | 10% | | |
| **Weighted total** | | | |

## Execution route

Backend, why, and what it cannot provide (hidden states, logits, external token IDs, template control).

## Tokenizer profile

Algorithm, asset format, `tokenizer.json` components used, and whether an existing owned profile covers it.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |

## Hardware and compute estimate

## Open risks
