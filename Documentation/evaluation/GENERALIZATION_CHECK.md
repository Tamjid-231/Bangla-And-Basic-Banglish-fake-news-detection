# Controlled Generalization Check (RQ5)

## Result

- In-sample held-out article accuracy: 94.08%
- Controlled short-claim accuracy: 46.67%
- Accuracy drop: 47.42 percentage points
- Controlled cases: 15
- Exact overlap with train/validation text: 0

## Interpretation

The large drop shows that strong random-split article performance does not
transfer reliably to short, differently written claims. This supports cautious
website wording and separate evidence display.

## Scope

This is a small controlled short-claim distribution shift, not a publisher-based, time-based, or independently collected external dataset.
