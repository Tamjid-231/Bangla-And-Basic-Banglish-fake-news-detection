# Banglish Benchmark

## Dataset

- Controlled held-out cases: 250
- Fake: 125
- Real: 125
- Labels are inherited from BanFakeNews-2.0.
- Banglish text uses curated natural-style Roman Bangla spellings.

## Results

| Condition | Accuracy | Macro F1 | Fake F1 | ms/example | Model MB |
|---|---:|---:|---:|---:|---:|
| A. Raw Banglish + Hybrid SVM | 50.40% | 34.22% | 66.84% | 0.11 | 3.99 |
| B. Rule-normalized Banglish + Hybrid SVM | 50.40% | 34.22% | 66.84% | 0.08 | 3.99 |
| C. XLM-R zero-shot transformer | not run | not run | not run | not run | not run |

## Interpretation

Rule normalization changed macro F1 by +0.00 percentage points and triggered for 9/250 controlled cases. This is an experimental comparison, not proof of performance on all Banglish writing.

## Limitation

This benchmark measures a controlled transliterated subset. It must not be presented as performance on naturally written Banglish social-media data.
