# Classifier-Evidence Disagreement Analysis (RQ4 and RQ6)

## RQ4 Result

- Controlled cases: 30
- Cases with decisive evidence labels: 15
- Cases with limited/conflicting/insufficient evidence: 15
- Classifier/evidence disagreements: 8
- Disagreement rate among comparable cases: 53.33%

A disagreement means the saved classifier's real/fake prediction conflicts with
a fact-check verdict or with support from at least two trusted sources.

## RQ6 Presentation Impact

A model-only interface would show 30 binary outputs. The separated
policy marks 23 cases as manual-review or unverified instead
of presenting every output with the same confidence. This is a system-policy
evaluation, not a human-user study.

## Scope

This analysis uses controlled evidence fixtures and the actual saved classifier. It does not measure changing live-search accuracy or user trust.
