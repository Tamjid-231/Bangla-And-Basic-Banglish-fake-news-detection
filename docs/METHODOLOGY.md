# Methodology

## Data preparation

`training/train_model.py` reads the BanFakeNews-2.0 CSV, validates the `Headline`, `Content`, and `Label` columns, combines each headline with up to the first 3,000 content characters, cleans the text, removes duplicates, and records every removal count.

The final data contains 57,908 usable rows: 9,393 fake and 48,515 real. A stratified 70/15/15 split with random state 42 is used for training, validation, and testing.

## Classical model comparison

The same split is used for Multinomial Naive Bayes, Logistic Regression, word-only Linear SVM, character-only Linear SVM, and Hybrid Linear SVM. Validation macro-F1 is the selection metric because the dataset is imbalanced. Accuracy, macro-F1, class-specific scores, confusion matrices, timing, and model size are preserved in `evaluation/retraining/`.

The Hybrid Linear SVM achieved the strongest validation macro-F1 in the original reproducible run and was selected for deployment. Word unigrams/bigrams and character-within-word trigrams are combined. The validation-selected raw threshold is aligned to a deployed threshold of 0.50.

## Banglish comparison

The controlled benchmark contains 250 balanced cases: 125 fake and 125 real. The saved report compares raw Banglish + Hybrid SVM and rule-normalized Banglish + Hybrid SVM. An optional XLM-R zero-shot condition is implemented in the benchmark script but is marked not run in the current report. Completed conditions include accuracy, macro-F1, fake-class F1, confusion matrices, inference time, and model size.

This benchmark is controlled and transliterated. It is useful for a repeatable comparison but does not represent every naturally written Banglish spelling.

## Web-evidence evaluation

The evidence engine is evaluated separately from the classifier. Thirty deterministic cases test source classification, textual matching, fact-check priority, and evidence status decisions. The set contains 15 Bangla and 15 English cases and covers supporting, limited, conflicting, false fact-check, true fact-check, and insufficient evidence.

This evaluation measures decision logic, not the changing retrieval quality of a live search engine.

## Disagreement and distribution shift

`evaluation/evaluate_disagreement.py` compares the actual deployed classifier with decisive evidence on the controlled evidence cases. `evaluation/evaluate_generalization.py` evaluates 15 differently distributed short claims with no development-text overlap. The short-claim check is intentionally reported as small and controlled, not as a complete publisher- or time-based external study.
